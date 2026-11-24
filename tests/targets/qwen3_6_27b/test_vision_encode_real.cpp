// Vision-tower-only check on a real artifact: the smallest harness that reaches the production
// VisionContext::encode without loading the language model.
//
// It binds only the frontend resources and the vision tensors of a .ninfer artifact (everything
// else is consumed validate-only, so no other payload is read), decodes real frames from a video
// clip, runs them through the real frontend processor, and encodes each item with the
// VisionContext compiled into ninfer_engine. The final [5120, V] embeddings are written to a file
// so that the GPU schedule can be judged against the independent host FP32 schedule:
//
//   test gpu  <artifact> <clip> <out> [frames] [width] [height]   production GPU schedule
//   test cpu  <artifact> <clip> <out> [frames] [width] [height]   host FP32 reference schedule
//   test compare <actual> <reference>                             per-item report and verdict
//
// `compare` fails on any non-finite value or an item whose cosine similarity to the reference is
// below kMinimumCosine. That is a gross bound for a broken schedule (aliased workspace, wrong
// route), not a rounding criterion: the two schedules round differently at every BF16 Op boundary
// of 27 blocks, and each Op is qualified against its own oracle elsewhere.
//
// With no arguments this is a ctest entry that skips (77): it needs an artifact and a clip.

#include "targets/qwen3_6_27b/impl/variant.h"

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/vision_context.h"

#include "artifact/materializer.h"
#include "artifact/typed_binding.h"
#include "media/decode/decode.h"
#include "targets/qwen3_6/impl/vision/host.h"
#include <ninfer/targets/qwen3_6/frontend.h>
#include <ninfer/targets/qwen3_6/frontend_resources.h>
#include <ninfer/targets/qwen3_6/prepared_prompt.h>
#include <ninfer/targets/qwen3_6/vision_control.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

using namespace ninfer;
namespace qwen    = ninfer::targets::qwen3_6;
namespace variant = ninfer::targets::qwen3_6_27b::detail;
namespace runtime = ninfer::targets::qwen3_6::detail::qwen3_6_27b_runtime;

constexpr int kOutputHidden     = 5120;
constexpr double kMinimumCosine = 0.999;

std::vector<std::uint8_t> read_bytes(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    return std::vector<std::uint8_t>(std::istreambuf_iterator<char>(in), {});
}

// Box-free nearest resample: the point is real image content at a chosen patch grid, not
// resampling quality. The frontend processor still applies its own resize and normalization.
std::vector<std::uint8_t> to_ppm(const media::decode::Image& image, int width, int height) {
    if (width <= 0 || height <= 0) {
        width  = image.width;
        height = image.height;
    }
    const std::string header =
        "P6\n" + std::to_string(width) + " " + std::to_string(height) + "\n255\n";
    std::vector<std::uint8_t> out(header.begin(), header.end());
    out.reserve(out.size() + static_cast<std::size_t>(width) * height * 3);
    for (int y = 0; y < height; ++y) {
        const int sy = std::min(image.height - 1, y * image.height / height);
        for (int x = 0; x < width; ++x) {
            const int sx = std::min(image.width - 1, x * image.width / width);
            const std::uint8_t* pixel =
                image.rgb.data() + (static_cast<std::size_t>(sy) * image.width + sx) * 3;
            out.insert(out.end(), pixel, pixel + 3);
        }
    }
    return out;
}

PromptInput prompt_for(const std::string& clip, int frames, int width, int height) {
    const std::vector<std::uint8_t> bytes = read_bytes(clip);
    // One frame per second of clip, a handful at most; the first `frames` of them are used.
    const media::decode::Video video =
        media::decode::decode_video(bytes, media::decode::Policy{}, 1.0, 4, 8);
    if (static_cast<int>(video.frames.size()) < frames) {
        throw std::runtime_error("clip yielded fewer frames than requested");
    }
    PromptInput input;
    input.options.enable_thinking = false;
    ChatMessage message;
    message.role = ChatRole::User;
    for (int frame = 0; frame < frames; ++frame) {
        MessagePart part;
        part.kind              = MessagePartKind::Media;
        part.media.kind        = MediaKind::Image;
        part.media.media_type  = "image/x-portable-pixmap";
        part.media.source_name = "frame" + std::to_string(frame) + ".ppm";
        part.media.bytes =
            to_ppm(video.frames[static_cast<std::size_t>(frame)], width, height);
        message.parts.push_back(std::move(part));
    }
    message.parts.push_back({.kind = MessagePartKind::Text, .text = "Describe the scene."});
    input.messages.push_back(std::move(message));
    return input;
}

void write_embeddings(const std::string& path,
                      const std::vector<std::vector<std::uint16_t>>& items) {
    std::ofstream out(path, std::ios::binary);
    if (!out) { throw std::runtime_error("cannot write " + path); }
    const auto count = static_cast<std::uint32_t>(items.size());
    out.write(reinterpret_cast<const char*>(&count), sizeof(count));
    for (const auto& item : items) {
        const auto words = static_cast<std::uint64_t>(item.size());
        out.write(reinterpret_cast<const char*>(&words), sizeof(words));
        out.write(reinterpret_cast<const char*>(item.data()),
                  static_cast<std::streamsize>(item.size() * sizeof(std::uint16_t)));
    }
}

std::vector<std::vector<std::uint16_t>> read_embeddings(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { throw std::runtime_error("cannot open " + path); }
    std::uint32_t count = 0;
    in.read(reinterpret_cast<char*>(&count), sizeof(count));
    std::vector<std::vector<std::uint16_t>> items(count);
    for (auto& item : items) {
        std::uint64_t words = 0;
        in.read(reinterpret_cast<char*>(&words), sizeof(words));
        item.resize(words);
        in.read(reinterpret_cast<char*>(item.data()),
                static_cast<std::streamsize>(words * sizeof(std::uint16_t)));
    }
    if (!in) { throw std::runtime_error("truncated embedding dump " + path); }
    return items;
}

float bf16_to_float(std::uint16_t bits) {
    const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16;
    float value;
    std::memcpy(&value, &wide, sizeof(value));
    return value;
}

// The materializer refuses a plan with nothing on the device, which is what the host reference
// would otherwise be. Park the smallest non-vision tensor there; it is never read back.
void place_smallest_text_tensor(const artifact::Reader& reader, artifact::Binder& binder) {
    const artifact::TensorDescriptor* smallest = nullptr;
    for (const artifact::ObjectDescriptor& object : reader.objects()) {
        const auto* tensor = std::get_if<artifact::TensorDescriptor>(&object);
        if (tensor == nullptr || tensor->name.starts_with("vision/")) { continue; }
        if (smallest == nullptr || tensor->bytes < smallest->bytes) { smallest = tensor; }
    }
    if (smallest == nullptr) { throw std::runtime_error("artifact has no non-vision tensor"); }
    binder.materialize_on_device(binder.require_tensor(smallest->name, smallest->format,
                                                       smallest->layout, smallest->shape));
}

int run_encode(bool cpu, const std::string& artifact_path, const std::string& clip,
               const std::string& out_path, int frames, int width, int height) {
    using artifact::NumericFormat;
    const artifact::TensorPlacement placement =
        cpu ? artifact::TensorPlacement::Host : artifact::TensorPlacement::Device;

    const artifact::Reader reader{std::filesystem::path(artifact_path)};
    artifact::Binder binder(reader, 1);
    const auto frontend_plan = qwen::bind_frontend_resources(binder);
    const auto backbone      = qwen::bind_vision_backbone(binder, placement);
    const auto merger_input  = qwen::bind_vision_merger_input(binder, placement);
    const auto merger_fc2    = artifact::bind_tensor(binder, "vision/merger/fc2",
                                                     NumericFormat::W8G32_F16S,
                                                     {kOutputHidden, 4608}, placement);
    const auto merger_fc2_bias = artifact::bind_tensor(binder, "vision/merger/fc2_bias",
                                                       NumericFormat::BF16, {kOutputHidden},
                                                       placement);
    const auto merger_norm = qwen::bind_vision_merger_norm(binder, placement);
    if (cpu) { place_smallest_text_tensor(reader, binder); }
    // The language model, MTP and draft tensors are neither placed nor read.
    binder.validate_unconsumed_matching("");
    const artifact::MaterializationPlan plan = binder.finish();

    DeviceContext device(0);
    artifact::MaterializedArtifact materialized = artifact::materialize(reader, plan, device);
    std::cout << "materialized vision tower: device_bytes="
              << materialized.stats().device_capacity_bytes
              << " host_bytes=" << materialized.stats().retained_host_bytes << '\n';

    qwen::FrontendOptions frontend_options;
    frontend_options.max_context              = 32'768;
    frontend_options.vision_max_tokens        = 8'192;
    frontend_options.media_preprocess_threads = 2;
    const qwen::Frontend frontend = qwen::make_frontend(
        qwen::take_frontend_resources(materialized, frontend_plan), frontend_options);
    const qwen::PreparedPrompt prompt = frontend.prepare(prompt_for(clip, frames, width, height));
    const qwen::PreparedPromptData& data = qwen::PreparedPromptAccess::view(prompt);
    const qwen::VisionControl control    = qwen::build_vision_control(data);
    if (control.items.size() != static_cast<std::size_t>(frames)) {
        throw std::runtime_error("frontend produced an unexpected number of vision items");
    }

    std::vector<std::vector<std::uint16_t>> embeddings;
    if (cpu) {
        const auto host =
            qwen::materialize_host_vision(materialized, backbone, merger_input, merger_norm,
                                          merger_fc2, merger_fc2_bias, kOutputHidden);
        for (std::size_t index = 0; index < control.items.size(); ++index) {
            const auto start = std::chrono::steady_clock::now();
            embeddings.push_back(qwen::encode_host_vision(
                *host, data.media_payloads[index]->span(), control.items[index]));
            std::cout << "item " << index << " patches=" << control.items[index].patch_count
                      << " tokens=" << control.items[index].merged_count << " cpu_seconds="
                      << std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
                             .count()
                      << '\n';
        }
        write_embeddings(out_path, embeddings);
        return 0;
    }

    auto model = std::make_unique<variant::RuntimeModelView>();
    auto& vision = model->vision.emplace();
    vision.common =
        qwen::materialize_vision_common(materialized, backbone, merger_input, merger_norm);
    vision.merger_fc2 = artifact::materialized_weight(materialized, merger_fc2,
                                                      NumericFormat::W8G32_F16S, kOutputHidden,
                                                      4608);
    vision.merger_fc2_bias = artifact::materialized_tensor(materialized, merger_fc2_bias,
                                                           NumericFormat::BF16, {kOutputHidden});
    const runtime::schedule::VisionContext context(device, *model);

    std::size_t workspace_bytes = 0;
    std::size_t max_tokens      = 0;
    for (const auto& item : control.items) {
        workspace_bytes =
            std::max(workspace_bytes, runtime::schedule::VisionContext::workspace_bytes(item));
        max_tokens = std::max(max_tokens, item.merged_count);
    }
    DeviceArena workspace(workspace_bytes);
    DeviceBuffer output_storage(max_tokens * kOutputHidden * sizeof(std::uint16_t));
    std::cout << "vision workspace bytes=" << workspace_bytes << '\n';

    for (std::size_t index = 0; index < control.items.size(); ++index) {
        const qwen::VisionItemControl& item = control.items[index];
        const runtime::schedule::VisionItemView view{data.media_payloads[index]->span(), &item};
        Tensor output(output_storage.p, DType::BF16,
                      {kOutputHidden, static_cast<std::int32_t>(item.merged_count)});
        // Two passes: the first pays one-time kernel loading, the second is the timed one. Both
        // write the same buffer, so the dump is the second pass's output.
        double best_ms = 0.0;
        for (int pass = 0; pass < 2; ++pass) {
            // The poison fill goes to the default stream; retire it before the encode's stream.
            output_storage.fill(0xff);
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto start = std::chrono::steady_clock::now();
            context.encode(view, output, workspace);
            CUDA_CHECK(cudaDeviceSynchronize());
            const double ms = std::chrono::duration<double, std::milli>(
                                  std::chrono::steady_clock::now() - start)
                                  .count();
            best_ms = pass == 0 ? ms : std::min(best_ms, ms);
        }
        std::vector<std::uint16_t> host(item.merged_count * kOutputHidden);
        output_storage.copy_to_host(host.data(), host.size() * sizeof(std::uint16_t));
        embeddings.push_back(std::move(host));
        std::cout << "item " << index << " grid=" << item.grid.temporal << "x" << item.grid.height
                  << "x" << item.grid.width << " patches=" << item.patch_count
                  << " tokens=" << item.merged_count << " encode_ms=" << best_ms << '\n';
    }
    write_embeddings(out_path, embeddings);
    return 0;
}

int run_compare(const std::string& actual_path, const std::string& reference_path) {
    const auto actual    = read_embeddings(actual_path);
    const auto reference = read_embeddings(reference_path);
    if (actual.empty() || actual.size() != reference.size()) {
        throw std::runtime_error("embedding dumps must hold the same nonzero number of items");
    }
    int failures = 0;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto& a = actual[index];
        const auto& b = reference[index];
        if (a.size() != b.size()) { throw std::runtime_error("embedding extents differ"); }
        long double error2 = 0.0L, reference2 = 0.0L, actual2 = 0.0L, dot = 0.0L;
        double max_abs = 0.0, max_reference = 0.0;
        std::size_t non_finite = 0;
        for (std::size_t i = 0; i < a.size(); ++i) {
            const double x = bf16_to_float(a[i]);
            const double y = bf16_to_float(b[i]);
            if (!std::isfinite(x) || !std::isfinite(y)) {
                ++non_finite;
                continue;
            }
            error2 += (x - y) * (x - y);
            reference2 += y * y;
            actual2 += x * x;
            dot += x * y;
            max_abs       = std::max(max_abs, std::abs(x - y));
            max_reference = std::max(max_reference, std::abs(y));
        }
        const auto cosine =
            static_cast<double>(dot / std::sqrt(std::max(actual2 * reference2, 1.0e-60L)));
        const bool ok = non_finite == 0 && cosine >= kMinimumCosine;
        if (!ok) { ++failures; }
        std::printf("item %zu elements=%zu rel_l2=%.6g max_abs=%.6g max_reference=%.6g "
                    "cosine=%.8f non_finite=%zu %s\n",
                    index, a.size(),
                    static_cast<double>(std::sqrt(error2 / std::max(reference2, 1.0e-30L))),
                    max_abs, max_reference, cosine, non_finite, ok ? "OK" : "FAIL");
    }
    std::printf("%s vision encode real: %d of %zu items outside the bound\n",
                failures == 0 ? "OK" : "FAIL", failures, actual.size());
    return failures == 0 ? 0 : 1;
}

} // namespace

int main(int argc, char** argv) {
    try {
        const std::string mode = argc > 1 ? argv[1] : "";
        if (mode == "compare" && argc == 4) { return run_compare(argv[2], argv[3]); }
        if ((mode == "gpu" || mode == "cpu") && argc >= 5) {
            const int frames = argc > 5 ? std::stoi(argv[5]) : 1;
            const int width  = argc > 6 ? std::stoi(argv[6]) : 0;
            const int height = argc > 7 ? std::stoi(argv[7]) : 0;
            return run_encode(mode == "cpu", argv[2], argv[3], argv[4], frames, width, height);
        }
        std::cout << "SKIP: usage: gpu|cpu <artifact> <clip> <out> [frames] [width] [height] | "
                     "compare <actual> <reference>\n";
        return 77;
    } catch (const std::exception& error) {
        std::cerr << "vision encode real: " << error.what() << '\n';
        return 1;
    }
}
