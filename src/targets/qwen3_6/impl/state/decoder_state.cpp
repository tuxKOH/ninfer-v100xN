#include <ninfer/targets/qwen3_6/decoder_state.h>
#include "core/device.h"

#include <limits>
#include <stdexcept>

namespace ninfer::targets::qwen3_6 {
namespace {

std::uint32_t page_count(std::uint32_t capacity) {
    if (capacity == 0) { throw std::invalid_argument("Paged KV capacity must be positive"); }
    return 1U + (capacity - 1U) / static_cast<std::uint32_t>(kPagedKVPageSize);
}

PagedKVCacheLayout plan_cache(LayoutBuilder& builder, std::uint32_t layers, std::uint32_t capacity,
                              std::int32_t kv_heads, std::int32_t head_dim, DType dtype,
                              std::int32_t quant_group, std::int32_t table_rows,
                              std::uint32_t physical_page_groups, bool ram_kv, bool host_retrieval_keys) {
    if (layers == 0 ||
        layers > static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) ||
        kv_heads <= 0 || head_dim <= 0 || table_rows <= 0) {
        throw std::invalid_argument("Paged KV cache geometry is invalid");
    }
    const bool quantized = dtype == DType::I8;
    if ((!quantized && (dtype != DType::BF16 || quant_group != 0)) ||
        (quantized && (quant_group != kKvQuantGroup || head_dim % quant_group != 0))) {
        throw std::invalid_argument("Paged KV cache dtype or quantization is invalid");
    }

    const std::uint32_t logical_pages = page_count(capacity);
    if (!ram_kv && physical_page_groups < logical_pages) {
        throw std::invalid_argument("Paged KV physical pages are below logical capacity");
    }

    PagedKVPoolSpec pool_spec;
    pool_spec.page_group_count      = physical_page_groups;
    pool_spec.logical_page_capacity = logical_pages;
    pool_spec.table_rows            = table_rows;
    pool_spec.ram_enabled           = ram_kv;
    pool_spec.planes.reserve(static_cast<std::size_t>(layers) * (quantized ? 4ULL : 2ULL));
    for (std::uint32_t layer = 0; layer < layers; ++layer) {
        pool_spec.planes.push_back({dtype, head_dim, kv_heads, 256});
        pool_spec.planes.push_back({dtype, head_dim, kv_heads, 256});
        if (quantized) {
            pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
            pool_spec.planes.push_back({DType::FP16, head_dim / quant_group, kv_heads, 256});
        }
    }
    PagedKVCacheLayout layout{
        .pool        = plan_paged_kv_pool(builder, pool_spec),
        .layers      = layers,
        .max_context = capacity,
        .kv_heads    = kv_heads,
        .head_dim    = head_dim,
        .dtype       = dtype,
        .quant_group = quant_group,
    };
    if (ram_kv) {
        // Key summaries are written per KV page and scanned only on a retrieval
        // refresh. The hot attention KV and query statistics remain device-local.
        LayoutBuilder host;
        layout.host_retrieval_keys = host_retrieval_keys;
        layout.retrieval_keys = (host_retrieval_keys ? host : builder).add_tensor(DType::FP32,
            {head_dim, kv_heads, static_cast<int>(logical_pages), static_cast<int>(layers)}, 256, "retrieval K");
        layout.retrieval_query = builder.add_tensor(DType::FP32,
            {head_dim, kv_heads, static_cast<int>(layers)}, 256, "retrieval Q");
        layout.retrieval_fallback = builder.add_tensor(DType::FP32,
            {head_dim, kv_heads, static_cast<int>(layers)}, 256, "retrieval fallback Q");
        layout.retrieval_counts = builder.add_tensor(DType::I32, {static_cast<int>(layers)}, 256, "retrieval counts");
        layout.retrieval_span = builder.add_tensor(DType::I32, {2}, 256, "retrieval span");
        layout.retrieval_scores = builder.add_tensor(DType::FP32,
            {static_cast<int>((logical_pages + 1) / 2)}, 256, "retrieval scores");
    }
    return layout;
}

} // namespace

DecoderStateLayout plan_decoder_state(LayoutBuilder& builder, const DecoderStateSpec& spec) {
    DecoderStateLayout layout;
    layout.text_kv = plan_cache(builder, spec.full_attention_layers, spec.capacity, spec.kv_heads,
                                spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                spec.kv_table_rows, spec.text_physical_page_groups, spec.ram_kv, spec.host_retrieval_keys);
    if (spec.enable_mtp) {
        layout.mtp_kv = plan_cache(builder, spec.mtp_layers, spec.capacity, spec.kv_heads,
                                   spec.attention_head_dim, spec.kv_dtype, spec.kv_quant_group,
                                   spec.kv_table_rows, spec.mtp_physical_page_groups, spec.ram_kv, spec.host_retrieval_keys);
    }
    layout.linear_attention = plan_linear_attention_state_pool(builder, spec.linear_attention);
    layout.linear_attention.host_checkpoint_slots = spec.host_checkpoint_slots;
    return layout;
}

PagedKVCache::PagedKVCache(DeviceSpan backing, const PagedKVCacheLayout& layout)
    : pool_(backing, layout.pool), layers_(layout.layers), max_context_(layout.max_context),
      kv_heads_(layout.kv_heads), head_dim_(layout.head_dim), dtype_(layout.dtype),
      quant_group_(layout.quant_group) {
    if (layout.pool.spec.ram_enabled) {
        if (layout.host_retrieval_keys) {
            host_keys_ = std::make_unique<PinnedHostBuffer>(layout.retrieval_keys.region.bytes);
            void* mapped = nullptr;
            CUDA_CHECK(cudaHostGetDevicePointer(&mapped, host_keys_->data(), 0));
            retrieval_keys = layout.retrieval_keys.bind({mapped, host_keys_->size()});
        } else {
            retrieval_keys = layout.retrieval_keys.bind(backing);
        }
        retrieval_query = layout.retrieval_query.bind(backing);
        retrieval_fallback = layout.retrieval_fallback.bind(backing);
        retrieval_counts = layout.retrieval_counts.bind(backing);
        retrieval_span = layout.retrieval_span.bind(backing);
        retrieval_scores = layout.retrieval_scores.bind(backing);
        CUDA_CHECK(cudaMemset(retrieval_counts.data, 0, retrieval_counts.bytes()));
        CUDA_CHECK(cudaMemset(retrieval_query.data, 0, retrieval_query.bytes()));
        CUDA_CHECK(cudaMemset(retrieval_fallback.data, 0, retrieval_fallback.bytes()));
    }
}

PagedKVCacheView::PagedKVCacheView(const PagedKVCache& cache, Tensor block_table) noexcept
    : cache_(&cache), block_table_(block_table) {}

std::uint32_t PagedKVCacheView::max_context() const noexcept {
    return cache_ == nullptr ? 0 : cache_->max_context();
}

PagedKVLayerView PagedKVCacheView::layer_view(std::uint32_t layer) const {
    if (cache_ == nullptr) { throw std::logic_error("Paged KV execution view is empty"); }
    return cache_->layer_view(layer, block_table_);
}

PagedKVCacheView PagedKVCache::execution_view(const PagedKVAllocation& allocation) const {
    if (!allocation.belongs_to(pool_)) {
        throw std::invalid_argument("Paged KV allocation belongs to another cache pool");
    }
    return PagedKVCacheView(*this, allocation.block_table());
}

PagedKVLayerView PagedKVCache::layer_view(std::uint32_t layer, Tensor block_table) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool quantized     = dtype_ == DType::I8;
    const std::size_t stride = quantized ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .block_table   = block_table,
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
        .read_indices = pool_.read_indices(),
        .read_table = pool_.read_tables(),
        .read_capacity = pool_.ram_enabled() ? pool_.page_group_count() * kPagedKVPageSize : 0U,
    };
}

PagedKVBatchLayerView PagedKVCache::batch_layer_view(std::uint32_t layer) const {
    if (layer >= layers_) { throw std::out_of_range("Paged KV layer is out of range"); }
    const bool quantized     = dtype_ == DType::I8;
    const std::size_t stride = quantized ? 4ULL : 2ULL;
    const std::size_t base   = static_cast<std::size_t>(layer) * stride;
    return PagedKVBatchLayerView{
        .k_pages       = pool_.plane(base),
        .v_pages       = pool_.plane(base + 1),
        .k_scale_pages = quantized ? pool_.plane(base + 2) : Tensor(),
        .v_scale_pages = quantized ? pool_.plane(base + 3) : Tensor(),
        .block_tables  = pool_.block_tables(),
        .head_dim      = head_dim_,
        .num_kv_heads  = kv_heads_,
        .dtype         = dtype_,
        .quant_group   = quant_group_,
        .read_indices = pool_.read_indices(),
        .read_tables = pool_.read_tables(),
        .read_capacity = pool_.ram_enabled() ? pool_.page_group_count() * kPagedKVPageSize : 0U,
    };
}

std::size_t DecoderStateLayout::kv_payload_bytes() const noexcept {
    return text_kv.payload_bytes() + (mtp_kv ? mtp_kv->payload_bytes() : 0);
}

DecoderState::DecoderState(DeviceSpan backing, const DecoderStateLayout& layout)
    : text_kv(backing, layout.text_kv), linear_attention(backing, layout.linear_attention) {
    if (layout.mtp_kv) { mtp_kv.emplace(backing, *layout.mtp_kv); }
}

PagedKVCache* DecoderState::mtp_cache() noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

const PagedKVCache* DecoderState::mtp_cache() const noexcept { return mtp_kv ? &*mtp_kv : nullptr; }

} // namespace ninfer::targets::qwen3_6
