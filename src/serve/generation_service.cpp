#include "serve/generation_service.h"

#include "product/media_acquire/acquire.h"
#include "serve/console_log.h"
#include "serve/tool_call_parser.h"
#include "serve/translate.h"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

namespace ninfer::serve {

struct RequestCapacity {
    explicit RequestCapacity(std::size_t limit) : maximum(limit) {}

    std::mutex mutex;
    std::size_t active = 0;
    const std::size_t maximum;
};

struct RequestLifetime {
    RequestLifetime(std::shared_ptr<RequestCapacity> owner,
                    std::chrono::steady_clock::time_point begin,
                    std::chrono::steady_clock::time_point limit)
        : capacity(std::move(owner)), started(begin), deadline(limit) {}

    ~RequestLifetime() {
        std::lock_guard lock(capacity->mutex);
        --capacity->active;
    }

    std::shared_ptr<RequestCapacity> capacity;
    std::chrono::steady_clock::time_point started;
    std::chrono::steady_clock::time_point deadline;
};

ApiError request_error_to_api_error(const ninfer::RequestError& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::RequestErrorKind::ContextLengthExceeded:
        error.status = 400;
        error.code   = "context_length_exceeded";
        break;
    case ninfer::RequestErrorKind::MediaBudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::RequestErrorKind::Overloaded:
        error.param.clear();
        error.status = 429;
        error.type   = "rate_limit_error";
        error.code   = "server_overloaded";
        break;
    case ninfer::RequestErrorKind::QueueTimeout:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::RequestErrorKind::Cancelled:
        error.status = 499;
        error.type   = "request_cancelled";
        error.code   = "client_disconnected";
        break;
    case ninfer::RequestErrorKind::Unavailable:
        error.param.clear();
        error.status = 503;
        error.type   = "server_error";
        error.code   = "service_unavailable";
        break;
    case ninfer::RequestErrorKind::InvalidStructuredOutput:
        error.param  = "response_format";
        error.status = 400;
        error.code   = "invalid_response_format";
        break;
    }
    return error;
}

namespace {

using Clock = std::chrono::steady_clock;

[[noreturn]] void throw_preparation_cancelled();

[[noreturn]] void throw_media_error(const ninfer::product::media_acquire::Error& exception) {
    ApiError error;
    error.param   = "messages";
    error.message = exception.what();
    switch (exception.kind()) {
    case ninfer::product::media_acquire::ErrorKind::BudgetExceeded:
        error.status = 400;
        error.code   = "media_budget_exceeded";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteUnavailable:
        error.status = 502;
        error.type   = "server_error";
        error.code   = "media_fetch_failed";
        break;
    case ninfer::product::media_acquire::ErrorKind::RemoteTimeout:
        error.status = 504;
        error.type   = "server_error";
        error.code   = "media_fetch_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::DeadlineExceeded:
        error.status = 503;
        error.type   = "server_error";
        error.code   = "request_queue_timeout";
        break;
    case ninfer::product::media_acquire::ErrorKind::Cancelled:
        throw_preparation_cancelled();
    }
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_invalid_input(const std::exception& exception,
                                      const char* code = "invalid_media") {
    ApiError error;
    error.status  = 400;
    error.param   = "messages";
    error.code    = code;
    error.message = exception.what();
    throw ApiException(std::move(error));
}

[[noreturn]] void throw_preparation_cancelled() {
    ApiError error;
    error.status  = 499;
    error.type    = "request_cancelled";
    error.code    = "client_disconnected";
    error.message = "client disconnected during media preparation";
    throw ApiException(std::move(error));
}

ninfer::OwnedMedia acquire_media(const ContentPart& part, Clock::time_point deadline,
                                 const std::function<bool()>& is_cancelled,
                                 std::size_t& remaining_bytes) {
    if (remaining_bytes == 0) {
        throw_media_error(ninfer::product::media_acquire::Error(
            ninfer::product::media_acquire::ErrorKind::BudgetExceeded,
            "request media exceeds aggregate byte limit"));
    }
    ninfer::product::media_acquire::Policy policy;
    policy.max_bytes    = std::min(policy.max_bytes, remaining_bytes);
    policy.deadline     = deadline;
    policy.is_cancelled = is_cancelled;
    std::vector<std::uint8_t> source_bytes;
    try {
        source_bytes = ninfer::product::media_acquire::acquire_bytes(part.source, policy);
    } catch (const ninfer::product::media_acquire::Error& exception) {
        throw_media_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }

    remaining_bytes -= source_bytes.size();
    ninfer::OwnedMedia media;
    media.kind =
        part.kind == ContentKind::Image ? ninfer::MediaKind::Image : ninfer::MediaKind::Video;
    media.media_type = part.source.media_type;
    switch (part.source.kind) {
    case ninfer::product::media_acquire::SourceKind::Path:
    case ninfer::product::media_acquire::SourceKind::Url:
        media.source_name = part.source.value;
        break;
    case ninfer::product::media_acquire::SourceKind::Data:
        media.source_name = "inline-data";
        break;
    case ninfer::product::media_acquire::SourceKind::Bytes:
        media.source_name = "inline-bytes";
        break;
    }
    media.bytes = std::move(source_bytes);
    return media;
}

[[noreturn]] void throw_request_error(const ninfer::RequestError& exception) {
    throw ApiException(request_error_to_api_error(exception));
}

void check_preparation_control(Clock::time_point deadline,
                               const std::function<bool()>& is_cancelled) {
    if (is_cancelled && is_cancelled()) { throw_preparation_cancelled(); }
    if (Clock::now() >= deadline) {
        throw_request_error(ninfer::RequestError(RequestErrorKind::QueueTimeout,
                                                 "inference request expired during preparation"));
    }
}

class ServiceOutputSink final : public ninfer::OutputSink {
public:
    ServiceOutputSink(const StreamSink& sink, bool filter_tool_calls)
        : sink_(&sink), filter_tool_calls_(filter_tool_calls) {}

    void publish(ninfer::OutputDelta delta) override {
        if (delta.text.empty()) { return; }
        if (delta.channel == ninfer::OutputChannel::Reasoning) {
            if (sink_->on_reasoning) { sink_->on_reasoning(delta.text); }
        } else {
            std::string visible =
                filter_tool_calls_ ? tool_filter_.feed(delta.text) : std::move(delta.text);
            publish_content(visible);
        }
    }

    std::size_t finish(bool is_tool_call_response) {
        if (filter_tool_calls_) { publish_content(tool_filter_.finish(is_tool_call_response)); }
        return content_bytes_;
    }

private:
    void publish_content(const std::string& text) {
        if (text.empty() || !sink_->on_content) { return; }
        sink_->on_content(text);
        content_bytes_ += text.size();
    }

    const StreamSink* sink_ = nullptr;
    bool filter_tool_calls_ = false;
    ToolCallStreamFilter tool_filter_;
    std::size_t content_bytes_ = 0;
};

} // namespace

GenerationService::GenerationService(ServeOptions options, LoadProgress load_progress)
    : options_(std::move(options)) {
    ninfer::EngineOptions engine_options;
    engine_options.artifact_path            = options_.artifact_path;
    engine_options.device                   = options_.device;
    engine_options.tp                       = options_.tp;
    engine_options.pp                       = options_.pp;
    engine_options.pipeline_chunks          = options_.pipeline_chunks;
    engine_options.devices                  = options_.devices;
    engine_options.storage_device           = options_.storage_device;
    engine_options.max_context              = options_.max_context;
    engine_options.rope_mode                = options_.rope_mode;
    engine_options.yarn_factor              = options_.yarn_factor;
    engine_options.yarn_origin              = options_.yarn_origin;
    engine_options.kv_capacity              = options_.kv_capacity;
    engine_options.max_concurrency          = options_.max_concurrency;
    engine_options.max_pending_requests     = options_.max_pending_requests;
    engine_options.pending_timeout_ms       = options_.pending_timeout_ms;
    engine_options.prefill_chunk            = options_.prefill_chunk;
    engine_options.prefill_decode_rounds    = options_.prefill_decode_rounds;
    engine_options.kv_cache                 = options_.kv_cache;
    engine_options.ram_kv                   = options_.ram_kv;
    engine_options.prefix_cache_tokens      = options_.prefix_cache_tokens;
    engine_options.volta_long_attention     = options_.volta_long_attention;
    engine_options.volta_fast_decode     = options_.volta_fast_decode;
    engine_options.enable_vision            = options_.enable_vision;
    engine_options.vision_device = options_.vision_device;
    engine_options.vision_max_tokens        = options_.vision_max_tokens;
    engine_options.use_cuda_graph           = options_.use_cuda_graph;
    engine_options.speculative              = options_.speculative;
    engine_options.media_cache_bytes        = options_.media_cache_bytes;
    engine_options.media_live_bytes         = options_.media_live_bytes;
    engine_options.media_preprocess_threads = options_.media_preprocess_threads;
    engine_options.load_progress            = std::move(load_progress);
    engine_              = std::make_unique<ninfer::Engine>(std::move(engine_options));
    prompt_capabilities_ = engine_->prompt_capabilities();
    if (options_.reasoning_effort &&
        !prompt_capabilities_.reasoning_effort.supports(*options_.reasoning_effort)) {
        throw std::invalid_argument("--reasoning-effort is not exposed by the loaded chat template");
    }
    request_capacity_    = std::make_shared<RequestCapacity>(
        static_cast<std::size_t>(options_.max_concurrency) + options_.max_pending_requests);
    engine_options_ = engine_->options();
    load_ = engine_->load_summary();
    sampling_defaults_ = engine_->sampling_defaults();
}

namespace {
ninfer::RuntimeStats add_execution_stats(ninfer::RuntimeStats current,
                                         const ninfer::RuntimeStats& completed) {
    current.computed_prefill_tokens += completed.computed_prefill_tokens;
    current.committed_decode_tokens += completed.committed_decode_tokens;
    current.decode_rounds += completed.decode_rounds;
    current.decode_row_rounds += completed.decode_row_rounds;
    current.decode_wall_seconds += completed.decode_wall_seconds;
    current.speculative_drafted_tokens += completed.speculative_drafted_tokens;
    current.speculative_accepted_tokens += completed.speculative_accepted_tokens;
    return current;
}

[[noreturn]] void residency_error(int status, std::string code, std::string message) {
    ApiError error;
    error.status = status;
    error.type = "server_error";
    error.code = std::move(code);
    error.message = std::move(message);
    throw ApiException(std::move(error));
}

void check_host_memory(std::uint64_t required) {
    std::ifstream input("/proc/meminfo");
    std::string key, unit;
    std::uint64_t kib = 0;
    while (input >> key >> kib >> unit) {
        if (key != "MemAvailable:") { continue; }
        constexpr std::uint64_t reserve = 1024ULL * 1024 * 1024;
        if (kib * 1024 < required + reserve) {
            residency_error(507, "insufficient_host_memory",
                            "not enough available host RAM for the model plus 1 GiB headroom");
        }
        break;
    }
}
} // namespace

ModelResidency GenerationService::model_residency() const {
    std::lock_guard lock(request_capacity_->mutex);
    auto result = residency_;
    result.retained_host_bytes = host_model_.bytes();
    result.active_requests = request_capacity_->active;
    return result;
}

void GenerationService::update_load_progress(std::string_view phase, std::uint64_t done,
                                              std::uint64_t total) {
    std::lock_guard lock(request_capacity_->mutex);
    residency_.phase = phase;
    residency_.done = done;
    residency_.total = total;
}

ninfer::RuntimeStats GenerationService::runtime_stats() const {
    std::lock_guard lock(request_capacity_->mutex);
    return residency_.state == "gpu"
        ? add_execution_stats(engine_->runtime_stats(), completed_stats_) : completed_stats_;
}

ninfer::MemorySummary GenerationService::memory_summary() const {
    std::lock_guard lock(request_capacity_->mutex);
    return residency_.state == "gpu" ? engine_->memory_summary() : ninfer::MemorySummary{};
}

ninfer::MediaCacheSummary GenerationService::media_cache_summary() const {
    std::lock_guard lock(request_capacity_->mutex);
    return residency_.state == "gpu" ? engine_->media_cache_summary() : ninfer::MediaCacheSummary{};
}

void GenerationService::offload_to_memory() {
    ninfer::RuntimeStats previous;
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (residency_.state == "ram") { return; }
        if (residency_.state != "gpu" || request_capacity_->active != 0) {
            residency_error(409, "model_busy", "wait for active requests or the model transition to finish");
        }
        previous = completed_stats_;
        completed_stats_ = add_execution_stats(engine_->runtime_stats(), completed_stats_);
        residency_ = ModelResidency{.state = "offloading", .phase = "retain_host"};
    }
    try {
        if (!host_model_) {
            check_host_memory(std::filesystem::file_size(options_.artifact_path));
            auto model = engine_->retain_model_in_memory(LoadProgress{
                .callback = [this](auto phase, auto done, auto total) {
                    update_load_progress(phase, done, total);
                }});
            std::lock_guard lock(request_capacity_->mutex);
            host_model_ = std::move(model);
        }
        update_load_progress("release_gpu", 0, 0);
        engine_.reset();
        std::lock_guard lock(request_capacity_->mutex);
        residency_ = ModelResidency{.state = "ram", .phase = "ready"};
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        completed_stats_ = previous;
        residency_ = ModelResidency{};
        throw;
    }
}

void GenerationService::load_from_memory() {
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (residency_.state == "gpu") { return; }
        if (residency_.state != "ram") {
            residency_error(409, "model_busy", "a model transition is already in progress");
        }
        residency_ = ModelResidency{.state = "loading", .phase = "load_host"};
    }
    try {
        auto options = engine_options_;
        options.load_progress.callback = [this](auto phase, auto done, auto total) {
            update_load_progress(phase, done, total);
        };
        auto engine = std::make_unique<ninfer::Engine>(std::move(options), host_model_);
        std::lock_guard lock(request_capacity_->mutex);
        engine_ = std::move(engine);
        residency_ = ModelResidency{};
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        residency_ = ModelResidency{.state = "ram", .phase = "ready"};
        throw;
    }
}

std::shared_ptr<RequestLifetime> GenerationService::acquire_request_lifetime() const {
    const auto started = Clock::now();
    {
        std::lock_guard lock(request_capacity_->mutex);
        if (residency_.state != "gpu") {
            residency_error(503, "model_not_loaded", "model is in host RAM or changing residency; load it into VRAM first");
        }
        if (request_capacity_->active >= request_capacity_->maximum) {
            throw_request_error(ninfer::RequestError(RequestErrorKind::Overloaded,
                                                     "inference request queue is full"));
        }
        ++request_capacity_->active;
    }
    try {
        return std::make_shared<RequestLifetime>(
            request_capacity_, started,
            started + std::chrono::milliseconds(options_.pending_timeout_ms));
    } catch (...) {
        std::lock_guard lock(request_capacity_->mutex);
        --request_capacity_->active;
        throw;
    }
}

PreparedRequest GenerationService::prepare(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    PreparedRequest prepared;
    ninfer::RequestOptions request_options = to_request_options(request, options_);
    prepared.include_usage                 = request.include_usage;
    prepared.tool_capable                  = request.uses_tools() || request.has_tool_history();
    prepared.tool_name_max_length          = request.tool_name_max_length;
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    prepared.enable_thinking                   = semantics.enable_thinking;
    prepared.preserve_thinking                 = semantics.preserve_thinking;
    prepared.preserve_thinking_semantic_change = request.preserve_thinking_semantic_change;
    const bool request_has_media               = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    prepared.lifetime = acquire_request_lifetime();

    try {
        const auto acquisition_started = Clock::now();
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, prepared.lifetime->deadline, is_cancelled,
                                     remaining_media_bytes);
            });
        prepared.acquisition_seconds =
            std::chrono::duration<double>(Clock::now() - acquisition_started).count();
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = prepared.lifetime->deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        ninfer::PreparedPrompt prompt = engine_->prepare(std::move(input), control);
        check_preparation_control(prepared.lifetime->deadline, is_cancelled);
        prepared.prompt_tokens = static_cast<int>(prompt.summary().prompt_tokens);
        prepared.preparation   = prompt.preparation_stats();
        prepared.prepare_seconds =
            std::chrono::duration<double>(Clock::now() - prepared.lifetime->started).count();
        prepared.generation = engine_->submit(std::move(prompt), std::move(request_options),
                                              control);
        prepared.sampling   = prepared.generation.resolved_sampling();
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }
    return prepared;
}

int GenerationService::count_prompt_tokens(const GenerationRequest& request,
                                           std::function<bool()> is_cancelled) const {
    const auto lifetime = acquire_request_lifetime();
    const bool request_has_media = request.media_item_count() != 0;
    if (request_has_media && !options_.enable_vision) {
        const std::invalid_argument error("Vision is disabled for this server");
        throw_invalid_input(error, "vision_disabled");
    }
    const Clock::time_point deadline =
        Clock::now() + std::chrono::milliseconds(options_.pending_timeout_ms);
    const ResolvedPromptSemantics semantics =
        resolve_prompt_semantics(request, options_, prompt_capabilities_);
    try {
        std::size_t remaining_media_bytes =
            std::min(options_.max_request_bytes, ninfer::kMaximumPromptMediaBytes);
        ninfer::PromptInput input =
            to_prompt_input(request, semantics, [&](const ContentPart& part) {
                return acquire_media(part, deadline, is_cancelled, remaining_media_bytes);
            });
        check_preparation_control(deadline, is_cancelled);
        const PreparationControl control{
            .deadline     = deadline,
            .cancellation = CancellationView(is_cancelled),
        };
        const int prompt_tokens =
            static_cast<int>(engine_->count_tokens(std::move(input), control));
        check_preparation_control(deadline, is_cancelled);
        return prompt_tokens;
    } catch (const ApiException&) { throw; } catch (const ninfer::RequestError& exception) {
        throw_request_error(exception);
    } catch (const std::invalid_argument& exception) { throw_invalid_input(exception); }
}

GenerationOutcome GenerationService::run(PreparedRequest& prepared, const StreamSink* sink,
                                         std::function<bool()> is_cancelled) {
    std::unique_ptr<ServiceOutputSink> output_sink;
    if (sink != nullptr) {
        output_sink = std::make_unique<ServiceOutputSink>(*sink, prepared.tool_capable);
    }
    ninfer::OutputSink* public_sink = output_sink.get();
    ninfer::CancellationView cancellation;
    if (is_cancelled || (sink != nullptr && sink->is_cancelled)) {
        cancellation = ninfer::CancellationView([external = std::move(is_cancelled), sink]() {
            return (external && external()) ||
                   (sink != nullptr && sink->is_cancelled && sink->is_cancelled());
        });
    }

    ninfer::GenerationResult result;
    try {
        result = prepared.generation.wait(public_sink, cancellation);
    } catch (const ninfer::RequestError& exception) { throw_request_error(exception); }
    GenerationOutcome outcome;
    outcome.text              = std::move(result.content);
    outcome.reasoning         = std::move(result.reasoning);
    outcome.prompt_tokens     = static_cast<int>(result.prompt.prompt_tokens);
    outcome.completion_tokens = static_cast<int>(result.generated_token_ids.size());
    outcome.reasoning_tokens  = static_cast<int>(result.reasoning_tokens);
    outcome.finish_reason     = result.finish_reason;

    outcome.metrics.prepare_seconds = prepared.prepare_seconds;
    outcome.metrics.ttft_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.first_token_seconds - result.timings.prepare_seconds);
    outcome.metrics.vision_seconds  = result.timings.vision_seconds;
    outcome.metrics.prefill_seconds = result.timings.prefill_seconds;
    outcome.metrics.decode_seconds  = result.timings.decode_seconds;
    outcome.metrics.total_seconds =
        prepared.prepare_seconds +
        std::max(0.0, result.timings.total_seconds - result.timings.prepare_seconds);
    outcome.metrics.prefix_cache_hit_tokens     = result.reused_prompt_tokens;
    outcome.metrics.prefix_reuse_path           = result.prefix_reuse_path;
    outcome.metrics.speculative_backend         = result.speculative.backend;
    outcome.metrics.speculative_draft_window    = result.speculative.draft_window;
    outcome.metrics.speculative_rounds          = result.speculative.rounds;
    outcome.metrics.speculative_draft_tokens    = result.speculative.drafted_tokens;
    outcome.metrics.speculative_accepted_tokens = result.speculative.accepted_tokens;
    outcome.metrics.speculative_fallback_steps  = result.speculative.fallback_steps;
    outcome.metrics.speculative_accepted_per_position =
        std::move(result.speculative.accepted_per_position);

    bool is_tool_call_response = false;
    if (prepared.tool_capable) {
        ParsedToolCallOutput parsed =
            parse_qwen_tool_call_output(outcome.text, prepared.tool_name_max_length);
        outcome.text          = std::move(parsed.content);
        is_tool_call_response = parsed.is_tool_call_response;
        if (is_tool_call_response) { outcome.tool_calls = std::move(parsed.tool_calls); }
    }
    if (output_sink) {
        outcome.streamed_content_bytes = output_sink->finish(is_tool_call_response);
    }
    return outcome;
}

void GenerationService::warmup() {
    try {
        GenerationRequest request;
        ChatTurn turn;
        turn.role = ChatRole::User;
        ContentPart content;
        content.kind     = ContentKind::Text;
        content.text     = "hi";
        content.type_raw = "text";
        turn.content.push_back(std::move(content));
        request.messages.push_back(std::move(turn));
        request.max_tokens       = 4;
        request.max_tokens_set   = true;
        PreparedRequest prepared = prepare(request);
        run(prepared, nullptr);
    } catch (const std::exception& exception) {
        write_console_log(ConsoleLogLevel::Warning,
                          std::string("warmup failed (continuing): ") + exception.what());
    }
}

} // namespace ninfer::serve
