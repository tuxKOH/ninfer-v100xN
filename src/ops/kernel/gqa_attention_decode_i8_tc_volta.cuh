#pragma once

// ninfer::ops - Volta (sm_70) tensor-core GQA small-T attention partial kernel, INT8-G64 cache.
//
// This is gqa_attention_small_t_tc_volta_partial_kernel (gqa_attention_prefill_volta.cuh) with
// the KV cache dtype swapped from bf16 to the int8 + per-64-group-scale codec. The tile
// topology, QK^T/PV arithmetic, online-softmax recurrence and partial_* output format are the
// same. Two warps compute the two independent eight-key QK tiles once and share their FP32
// scores with all four PV dimension warps. Each warp retains the original softmax/PV order.
//
// Why it exists: the INT8 decode path was SIMT-only, and SIMT is what caps it. Measured at 82k,
// width 4, the INT8 SIMT kernel runs at 1.86 TFLOP/s and the bf16 tensor-core kernel at 2.71 --
// so bf16 beat int8 by 1.46x *while reading twice the bytes*, purely because it had tensor cores.
// The two properties are separable: this kernel takes the tensor-core math and keeps int8's
// halved traffic. See the V100 performance summary.
//
// The only structural difference is where the fp16 that feeds mma.sync comes from. The bf16
// kernel copies raw bf16 bytes into k_s/v_s and converts them in place, which works because bf16
// and fp16 are both two bytes. int8 codes are one byte, so an in-place expansion would clobber
// its own neighbours; instead the staging loop dequantizes through registers on the way in. That
// costs nothing here, because cp_async below sm_80 is already a synchronous load+store through
// registers (ops/common/memory.cuh) -- there is no async pipeline to break.
//
// Structurally mirrors gqa_attention_small_t_tc_partial_bf16_kernel's Ampere+ branch
// (gqa_attention_decode_bf16.cuh) -- same paged-KV addressing, GQA row mapping, and
// cache-write handling, reused via the shared helpers in gqa_attention_decode.cuh -- but
// the tile/warp topology and QK^T/PV compute core are rebuilt from scratch on Volta's
// mma.sync.m8n8k4 primitives (ops/common/volta_mma.cuh), since Volta has neither
// ldmatrix nor m16n8k16. Writes the same partial_acc/partial_m/partial_l format the
// existing reduce kernel already consumes, so no changes are needed anywhere else in the
// online-softmax merge pipeline.
//
// Why this kernel's tile topology differs from the Ampere+ kernel's (see the V100 performance summary
// for the full derivation -- this is the load-bearing design note, read it before changing
// any of the constants below):
//
//   1. Register wall. Ampere's ldmatrix-based PV tile packs 2 output rows x 2 columns per
//      thread per mma call; Volta's mma.m8n8k4 packs only 1 row x 8 columns per thread per
//      call. Holding the *entire* D=256 head-dim output resident across the whole key loop
//      (as the Ampere kernel does, ~128 registers) would cost Volta 32 chunks x 8 floats =
//      256 registers/thread for a *single* output row -- already at Volta's hard 255/thread
//      cap with nothing left for Q/K fragments or loop state. Split the head dimension
//      across four or eight warps instead of giving each warp the full D range. The TP2
//      long-context four/five-query route uses eight warps, halving each warp's PV accumulators
//      and reducing register pressure. Other shapes keep the four-warp schedule. Warps
//      reuse one FP32 QK tile, perform the same online-softmax recurrence independently,
//      and diverge only for the PV accumulate.
//
//   2. Shared-memory wall. This kernel reloads Q from shared memory every key-tile
//      iteration rather than keeping Q fragments resident in registers (that trade-off is
//      what keeps (1)'s register budget low -- persisting Q *and* a full-width accumulator
//      would both want ~128 registers, right back at the wall). That means Q must stay
//      resident in shared memory for the whole key loop, so -- unlike the Ampere kernel,
//      which stages Q into the same shared buffer it later reuses for K/V once Q has been
//      consumed into registers -- this kernel needs separate, simultaneously-live q_s/k_s/
//      v_s buffers. To fit that under the 48KB static-shared-memory default, Br is fixed at
//      one 32-row Volta tile (not scaled by warp count) and Bc is halved to 16 keys/tile;
//      q_s+k_s+v_s then costs 16KB+8KB+8KB = 32KB, comfortably under budget.
//
//   3. Row capacity. Fixing Br=32 means a single tile can't cover every (TokenTile x
//      GroupSize) row count this op needs (up to 6*8=48 for the widest GQA geometry this
//      codebase instantiates). Rather than scale Br back up (reintroducing the shared-memory
//      wall), the kernel loops over row-tile "passes": each pass restages Q for a different
//      32-row slice and re-walks the whole key range. At most 2 passes are ever needed for
//      any geometry this codebase instantiates. This trades some redundant K/V global/cache
//      reads across passes for a fixed, small, well-understood resource footprint --
//      consistent with this kernel's stated correctness-first phase 1 priority.

#include "ops/common/volta_mma.cuh"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math_constants.h>

#include <cstdint>

namespace ninfer::ops {

__device__ __forceinline__ void volta_attention_pv_fp32(float (&d)[8], const half2 (&p)[4],
                                                        const half2 (&v)[4]) {
    const auto* a = reinterpret_cast<const unsigned*>(p);
    const auto* b = reinterpret_cast<const unsigned*>(v);
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),
                   "+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
                 : "r"(a[0]),"r"(a[1]),"r"(b[0]),"r"(b[1]));
    asm volatile("mma.sync.aligned.m8n8k4.row.row.f32.f16.f16.f32 "
                 "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
                 "{%0,%1,%2,%3,%4,%5,%6,%7};"
                 : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3]),
                   "+f"(d[4]),"+f"(d[5]),"+f"(d[6]),"+f"(d[7])
                 : "r"(a[2]),"r"(a[3]),"r"(b[2]),"r"(b[3]));
}

template <typename Geometry, int TokenTile, int WarpsPerCta, bool MultiBatch, bool Masked,
          typename CacheInput, bool SharedSoftmax = false, bool Fp32Pv = false>
__launch_bounds__(WarpsPerCta * 32, 2) __global__ void gqa_attention_small_t_tc_volta_partial_i8_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* pos, std::int8_t* cache_k_i8,
    std::int8_t* cache_v_i8, __half* cache_k_scale, __half* cache_v_scale,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride, std::int32_t tokens,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity, float scale,
    GqaPartial<Fp32Pv>* partial_acc, float* partial_m, float* partial_l, bool volta_long_attention) {
    static_assert(TokenTile >= 1 && TokenTile <= 6);
    static_assert(WarpsPerCta == 4 || WarpsPerCta == 8);
    static_assert(!Fp32Pv || (SharedSoftmax && WarpsPerCta == 8));

#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    constexpr int DimSplit      = WarpsPerCta; // warps split the head dim, not the row range
    constexpr int Br            = 32;          // one Volta tile's worth of rows per pass
    constexpr int Bc            = 16;
    constexpr int D             = kGqaHeadDim;
    constexpr int Threads       = DimSplit * 32;
    constexpr int DChunks       = D / 8;           // QK^T: full head-dim contraction, unsplit
    constexpr int PVChunks      = Bc / 8;          // key sub-groups per Bc tile
    constexpr int DSlice        = D / DimSplit;    // this warp's PV output width
    constexpr int DChunksLocal  = DSlice / 8;       // this warp's resident accumulator chunks
    constexpr int PageIds       = 64;
    constexpr int Groups        = D / kGqaKvQuantGroup; // quant groups per key row
    constexpr bool InlineQK = (Geometry::QHeads == 12 || Geometry::QHeads == 6) && TokenTile == 5 &&
                              SharedSoftmax && !Fp32Pv;
    constexpr bool PrefetchKv = (Geometry::QHeads == 12 || Geometry::QHeads == 6) &&
                              TokenTile == 5 && WarpsPerCta == 8;
    constexpr int KvChunks = Bc * (D / 8) / Threads;
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(D % kGqaKvQuantGroup == 0, "head dim must divide into whole quant groups");
    static_assert(kGqaKvQuantGroup % 8 == 0,
                  "an 8-wide staging chunk must sit inside one quant group, so it needs one scale");

    // Declared fp16 (not bf16): Volta's mma.sync.m8n8k4 only accepts fp16 operands. Unlike the
    // bf16 sibling kernel, which lands raw cache bytes here and reinterprets them in place, K/V
    // arrive already dequantized to fp16 -- int8 codes are half the width of the destination, so
    // an in-place expansion would overwrite the neighbouring chunk. Q is converted inline at the
    // point of load, since it's read scalar-wise.
    // Row stride is padded, not D. The mma feed reads *down* a column of these tiles -- one
    // half2 per row at a fixed d -- so with an unpadded 512-byte row stride (D=256 halves) every
    // row lands on the same shared-memory bank, since banks wrap every 128 bytes. ncu measured
    // 724M bank conflicts against 891M wavefronts for 42.2M shared loads: ~21 replays per
    // instruction, near the 32-way worst case, and L1/TEX throughput pinned at 85% while the
    // tensor pipe idled. SmemPad shifts each row by a whole 16-byte vector so consecutive rows
    // start 4 banks apart, which keeps every 16-byte store in the staging loop aligned.
    constexpr int SmemPad    = 8;
    constexpr int SmemStride = D + SmemPad;
    static_assert(SmemPad % 8 == 0, "pad must preserve 16-byte alignment of the staging stores");
    __shared__ __align__(16) half q_s[Br * SmemStride];
    __shared__ __align__(16) half k_s[Bc * SmemStride];
    __shared__ __align__(16) half v_s[Bc * SmemStride];
    __shared__ std::int32_t physical_pages_s[PageIds];
    // Lane-major within each score component: consumers and producers access contiguous
    // banks, including the 24 live rows of a TP2 MTP4 call.
    __shared__ float scores_s[PVChunks][8][32];
    __shared__ half2 probabilities_s[SharedSoftmax ? PVChunks : 1][4][32];
    __shared__ float alpha_s[SharedSoftmax ? PVChunks : 1][32];

    const int kv_head     = static_cast<int>(blockIdx.x);
    const int split       = static_cast<int>(blockIdx.y);
    const int batch       = MultiBatch ? static_cast<int>(blockIdx.z) : 0;
    const int split_count = static_cast<int>(gridDim.y);
    const int tid         = static_cast<int>(threadIdx.x);
    const int warp        = tid >> 5;
    const int lane        = tid & 31;
    const int dim_warp    = warp; // DimSplit == WarpsPerCta: every warp is a dim-slice of the one row-tile
    int valid_tokens       = tokens;
    if constexpr (Masked) {
        const int remaining = valid_columns[batch] - column_begin;
        valid_tokens         = remaining <= 0 ? 0 : (remaining < tokens ? remaining : tokens);
    }
    const int row_count = tokens * Geometry::GroupSize;

    std::int64_t column_base = column_begin;
    if constexpr (MultiBatch) { column_base += static_cast<std::int64_t>(batch) * full_width; }
    q += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::QHeads * column_base;
    pos += column_base;
    if constexpr (CacheInput::writes_cache) {
        input.k += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
        input.v += static_cast<std::int64_t>(kGqaHeadDim) * Geometry::KVHeads * column_base;
    }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const std::int32_t* block_table =
        block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    if constexpr (MultiBatch) {
        partial_acc += static_cast<std::int64_t>(batch) * kGqaHeadDim * Geometry::QHeads * tokens *
                       split_count;
        partial_m += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
        partial_l += static_cast<std::int64_t>(batch) * Geometry::QHeads * tokens * split_count;
    }

    auto write_neutral = [&]() {
        for (int row = tid; row < row_count; row += Threads) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] =
                    -CUDART_INF_F;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = 0.0f;
            }
        }
        for (int idx = tid; idx < row_count * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head    = 0;
            int token     = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                    GqaPartial<Fp32Pv>{};
            }
        }
    };

    if (kv_head < 0 || kv_head >= Geometry::KVHeads || tokens < 1 || tokens > TokenTile ||
        split_count <= 0) {
        return;
    }
    if (valid_tokens == 0) {
        write_neutral();
        return;
    }

    const std::int32_t first_pos = pos[0];
    const std::int32_t last_pos  = pos[tokens - 1];
    if (first_pos < 0 || last_pos < 0 || last_pos >= logical_capacity) {
        write_neutral();
        return;
    }

    const int window = last_pos + 1;
    // Producer and INT8 reducer must agree on the live partials, including the T=5/6
    // window specializations; the BF16 policy can leave unread or unwritten split rows.
    const int active_split_count =
        gqa_small_t_active_splits<Geometry, true>(window, split_count, TokenTile,
                                                volta_long_attention, MultiBatch ? gridDim.z : 1);
    if (split >= active_split_count) { return; }

    const int logical_tiles = div_up(window, Bc);
    const bool tile_split   = logical_tiles >= active_split_count;
    const int units_per_split =
        tile_split ? div_up(logical_tiles, active_split_count) : div_up(window, active_split_count);
    const int split_start = split * units_per_split * (tile_split ? Bc : 1);
    const int split_limit = split_start + units_per_split * (tile_split ? Bc : 1);
    const int split_end   = (split_limit < window) ? split_limit : window;
    if (split_start >= split_end) {
        write_neutral();
        return;
    }
    const int first_tile = (split_start / Bc) * Bc;
    const int key_blocks = div_up(split_end - first_tile, Bc);
    const int first_page = first_tile >> kPagedKVPageShift;
    const int page_count = ((split_end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int page = tid; page < page_count; page += Threads) {
        physical_pages_s[page] = block_table[first_page + page];
    }

    if constexpr (CacheInput::writes_cache) {
        // Quantize this tile's new rows into the cache before any of it is staged back out.
        // One warp owns one (token, 64-group) pair so that the absmax reduction that sets the
        // group's scale is exactly a full-warp reduction: 32 lanes x 2 elements = 64 = one
        // group. This mirrors the SIMT kernel's append block (gqa_attention_decode_i8.cuh);
        // the codes have to come out bit-identical, since the two kernels share a cache.
        const int warps = Threads / 32;
        for (int pair = warp; pair < valid_tokens * Groups; pair += warps) {
            const int token    = pair / Groups;
            const int grp      = pair - token * Groups;
            const int position = pos[token];
            if (position < split_start || position >= split_end || position < 0 ||
                position >= logical_capacity) {
                continue;
            }
            int physical_page     = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
            physical_page          = __shfl_sync(FullMask, physical_page, 0);
            const int page_offset = position & kPagedKVPageMask;
            const int d0          = grp * kGqaKvQuantGroup + lane;
            const int d1          = d0 + 32;
            const std::int64_t src0 = gqa_kv_new_index<Geometry>(kv_head, d0, token);
            const std::int64_t src1 = gqa_kv_new_index<Geometry>(kv_head, d1, token);
            const float kv0         = __bfloat162float(input.k[src0]);
            const float kv1         = __bfloat162float(input.k[src1]);
            const float vv0         = __bfloat162float(input.v[src0]);
            const float vv1         = __bfloat162float(input.v[src1]);
            float kamax             = fmaxf(fabsf(kv0), fabsf(kv1));
            float vamax             = fmaxf(fabsf(vv0), fabsf(vv1));
            kamax                   = warp_max(kamax, FullMask);
            vamax                   = warp_max(vamax, FullMask);
            const __half ksh        = __float2half_rn(kamax > 0.0f ? kamax / 127.0f : 0.0f);
            const __half vsh        = __float2half_rn(vamax > 0.0f ? vamax / 127.0f : 0.0f);
            const float k_inv       = __half2float(ksh) > 0.0f ? 1.0f / __half2float(ksh) : 0.0f;
            const float v_inv       = __half2float(vsh) > 0.0f ? 1.0f / __half2float(vsh) : 0.0f;
            cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
                gqa_kv_quant_code(kv0, k_inv);
            cache_k_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
                gqa_kv_quant_code(kv1, k_inv);
            cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d0, page_offset)] =
                gqa_kv_quant_code(vv0, v_inv);
            cache_v_i8[gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d1, page_offset)] =
                gqa_kv_quant_code(vv1, v_inv);
            if (lane == 0) {
                const std::int64_t so =
                    gqa_kv_quant_scale_index<Geometry>(physical_page, kv_head, grp, page_offset);
                cache_k_scale[so] = ksh;
                cache_v_scale[so] = vsh;
            }
        }
        __syncthreads();
    }

    // One row-tile "pass" per Br=32 slice of this token tile's rows (at most 2 for any
    // geometry this codebase instantiates -- see file comment #3). Each pass restages Q
    // and re-walks the whole key range for its own 32-row slice.
    for (int row_base = 0; row_base < row_count; row_base += Br) {
        const int rows_here = (row_count - row_base < Br) ? (row_count - row_base) : Br;

        for (int idx = tid; idx < Br * D; idx += Threads) {
            const int row = idx / D;
            const int d   = idx - row * D;
            int q_head     = 0;
            int token      = 0;
            __nv_bfloat16 value = __float2bfloat16(0.0f);
            if (row < rows_here) {
                gqa_small_t_tc_row_to_qt<Geometry>(row_base + row, tokens, kv_head, q_head, token);
                if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                    value = q[gqa_q_index<Geometry>(q_head, d, token)];
                }
            }
            q_s[row * SmemStride + d] = __float2half(__bfloat162float(value));
        }
        __syncthreads();

        const int r_lo = volta_d_get_i(0) & ~2;
        const int r_hi = volta_d_get_i(0) | 2;
        int q_head_lo = 0, tok_lo = 0, q_head_hi = 0, tok_hi = 0;
        gqa_small_t_tc_row_to_qt<Geometry>(row_base + r_lo, tokens, kv_head, q_head_lo, tok_lo);
        gqa_small_t_tc_row_to_qt<Geometry>(row_base + r_hi, tokens, kv_head, q_head_hi, tok_hi);
        const int qabs_lo = r_lo < rows_here ? pos[tok_lo] : -1;
        const int qabs_hi = r_hi < rows_here ? pos[tok_hi] : -1;
        int physical_page = physical_pages_s[0];

        // acc_f[c] holds this warp's DSlice-wide PV output for head-dim chunk c (columns
        // [dim_warp*DSlice + c*8, +8)), folded across key-tiles by the online-softmax
        // alpha rescale -- see file comment #1 for why this is DChunksLocal (not DChunks)
        // wide.
        float acc_f[DChunksLocal][8];
#pragma unroll
        for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
            for (int i = 0; i < 8; ++i) { acc_f[c][i] = 0.0f; }
        }
        // Every thread's 8 D-tile registers straddle TWO distinct rows (r_lo, r_hi -- see
        // volta_mma.cuh), so their softmax state must be tracked as two independent series,
        // not collapsed into one: thread `lane` and thread `lane^2` both compute IDENTICAL
        // bm_lo/bm_hi (each is already a full 2-lane reduction), and both update m_lo/l_lo
        // and m_hi/l_hi identically from those identical inputs -- so every thread has a
        // fully correct, non-mixed view of both rows' state, and can normalize d_score's
        // row_lo group against m_lo and its row_hi group against m_hi (previously this used
        // a single shared max/alpha for both groups, silently corrupting whichever group
        // wasn't "this thread's own row" -- masked by the very first softmax update, since
        // then m_lo==m_hi==-inf makes alpha 0 regardless, but wrong from the second update
        // onward whenever the two rows' true maxima differ).
        float m_lo = -CUDART_INF_F, m_hi = -CUDART_INF_F;
        float l_lo = 0.0f, l_hi = 0.0f;

        // Keep the next tile's packed bytes in registers while the current QK/softmax/PV
        // runs. The qualified five-query TP2 route keeps the same split, QK, softmax
        // and PV order; other widths retain their measured staging schedule.
        int2 next_k[PrefetchKv ? KvChunks : 1], next_v[PrefetchKv ? KvChunks : 1];
        half next_ks[PrefetchKv ? KvChunks : 1], next_vs[PrefetchKv ? KvChunks : 1];
        auto prefetch_kv = [&](int tile) {
#pragma unroll
            for (int c = 0; c < KvChunks; ++c) {
                const int chunk = tid + c * Threads;
                const int key_l = chunk / (D / 8);
                const int d = (chunk % (D / 8)) * 8;
                const int key = tile + key_l;
                next_k[c] = make_int2(0, 0);
                next_v[c] = make_int2(0, 0);
                next_ks[c] = __float2half(0.0f);
                next_vs[c] = __float2half(0.0f);
                if (key >= split_start && key < split_end) {
                    const int page = physical_pages_s[(key >> kPagedKVPageShift) - first_page];
                    const int offset = key & kPagedKVPageMask;
                    const auto code = gqa_kv_quant_code_index<Geometry>(page, kv_head, d, offset);
                    const auto group = gqa_kv_quant_scale_index<Geometry>(
                        page, kv_head, d / kGqaKvQuantGroup, offset);
                    next_k[c] = load_vec<int2>(cache_k_i8 + code);
                    next_v[c] = load_vec<int2>(cache_v_i8 + code);
                    next_ks[c] = cache_k_scale[group];
                    next_vs[c] = cache_v_scale[group];
                }
            }
        };
        if constexpr (PrefetchKv) { prefetch_kv(first_tile); }

        for (int kb = 0; kb < key_blocks; ++kb) {
            const int k0 = first_tile + kb * Bc;
            if (kb != 0 && (k0 & kPagedKVPageMask) == 0) {
                physical_page = physical_pages_s[(k0 >> kPagedKVPageShift) - first_page];
            }

            // Stage K/V for this key tile, dequantizing int8 -> fp16 on the way in. Unlike the
            // bf16 sibling there is no second in-place conversion pass: the codes are half the
            // width of the fp16 destination, so they are widened in registers between the load
            // and the store. Every key in range is read from the cache, including this tile's
            // own new tokens -- the append block above has already written and __syncthreads()'d
            // them, so there is no need for the bf16 kernel's separate "from_new" source.
            if constexpr (PrefetchKv) {
#pragma unroll
                for (int c = 0; c < KvChunks; ++c) {
                    const int chunk = tid + c * Threads;
                    const int key_l = chunk / (D / 8);
                    const int d = (chunk % (D / 8)) * 8;
                    store_vec(&k_s[key_l * SmemStride + d],
                              gqa_kv_dequant_i8x8_f16(next_k[c], next_ks[c]));
                    store_vec(&v_s[key_l * SmemStride + d],
                              gqa_kv_dequant_i8x8_f16(next_v[c], next_vs[c]));
                }
            } else {
                for (int chunk = tid; chunk < Bc * (D / 8); chunk += Threads) {
                    const int key_l = chunk / (D / 8);
                    const int d     = (chunk - key_l * (D / 8)) * 8;
                    const int key   = k0 + key_l;
                    half* k_dst     = &k_s[key_l * SmemStride + d];
                    half* v_dst     = &v_s[key_l * SmemStride + d];
                    if (key >= split_start && key < split_end) {
                        const int page_offset = key & kPagedKVPageMask;
                        const std::int64_t code_off =
                            gqa_kv_quant_code_index<Geometry>(physical_page, kv_head, d, page_offset);
                        const std::int64_t scale_off = gqa_kv_quant_scale_index<Geometry>(
                            physical_page, kv_head, d / kGqaKvQuantGroup, page_offset);
                        const __half ks = cache_k_scale[scale_off];
                        const __half vs = cache_v_scale[scale_off];
                        store_vec(k_dst, gqa_kv_dequant_i8x8_f16_from(&cache_k_i8[code_off], ks));
                        store_vec(v_dst, gqa_kv_dequant_i8x8_f16_from(&cache_v_i8[code_off], vs));
                    } else {
                        store_vec(k_dst, make_int4(0, 0, 0, 0));
                        store_vec(v_dst, make_int4(0, 0, 0, 0));
                    }
                }
            }
            __syncthreads();
            if constexpr (PrefetchKv) {
                if (kb + 1 < key_blocks) { prefetch_kv(k0 + Bc); }
            }

            // The default five-query softmax consumes both groups on warp 0. Interleave
            // their unchanged contractions there, avoiding the shared-score handoff.
            float inline_score[InlineQK ? PVChunks : 1][8] = {};
            if constexpr (InlineQK) {
                if (warp == 0) {
#pragma unroll
                    for (int c = 0; c < DChunks; ++c) {
                        half2 qf[4];
                        volta_load_qp(qf, reinterpret_cast<const half2*>(&q_s[c * 8]),
                                      SmemStride / 2);
#pragma unroll
                        for (int sub = 0; sub < PVChunks; ++sub) {
                            half2 kf[4];
                            volta_load_k(kf, reinterpret_cast<const half2*>(
                                &k_s[sub * 8 * SmemStride + c * 8]), SmemStride / 2);
                            volta_mma_qk(inline_score[sub], qf, kf);
                        }
                    }
                }
            } else {
                if (warp < PVChunks) {
                    float d_score[8] = {};
#pragma unroll
                    for (int c = 0; c < DChunks; ++c) {
                        half2 qf[4], kf[4];
                        volta_load_qp(qf, reinterpret_cast<const half2*>(&q_s[c * 8]),
                                      SmemStride / 2);
                        volta_load_k(kf, reinterpret_cast<const half2*>(
                            &k_s[warp * 8 * SmemStride + c * 8]), SmemStride / 2);
                        volta_mma_qk(d_score, qf, kf);
                    }
#pragma unroll
                    for (int j = 0; j < 8; ++j) { scores_s[warp][j][lane] = d_score[j]; }
                }
                __syncthreads();
            }

            // The default profile retains eight-key softmax/PV association. FP32 PV merges
            // the two score groups into one sixteen-key online-softmax update.
            auto prepare_probability = [&](int sub, half2 (&p)[4]) {
                const int sub_k0 = k0 + sub * 8;
                float d_score[8];
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    d_score[j] = InlineQK ? inline_score[sub][j] : scores_s[sub][j][lane];
                }

                // --- Causal mask + scale. Row = volta_d_get_i(l) (0..31, this pass's local
                // row index); each thread's 8 registers span exactly two distinct rows
                // (l&2==0 vs l&2==2, see volta_mma.cuh), so two qabs lookups (not eight)
                // cover this thread's own rows. ---
#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const int row   = volta_d_get_i(l);
                    const int col   = volta_d_get_j(l);
                    const int key   = sub_k0 + col;
                    const bool lo   = (l & 2) == 0;
                    const int qabs  = lo ? qabs_lo : qabs_hi;
                    const bool ok =
                        row < rows_here && key >= split_start && key < split_end && key <= qabs;
                    d_score[l] = ok ? d_score[l] * scale : -CUDART_INF_F;
                }

                // --- Per-row online-softmax update. Threads tid and tid^2 together hold
                // the complete 8-key row for both rows this thread touches, so a single
                // offset-2 shfl_xor gives an exact per-row reduction. ---
                float bm_lo = fmaxf(fmaxf(d_score[0], d_score[1]), fmaxf(d_score[4], d_score[5]));
                float bm_hi = fmaxf(fmaxf(d_score[2], d_score[3]), fmaxf(d_score[6], d_score[7]));
                bm_lo        = fmaxf(bm_lo, __shfl_xor_sync(FullMask, bm_lo, 2, 32));
                bm_hi        = fmaxf(bm_hi, __shfl_xor_sync(FullMask, bm_hi, 2, 32));

                const float new_m_lo = fmaxf(m_lo, bm_lo);
                const float new_m_hi = fmaxf(m_hi, bm_hi);
                const float alpha_lo =
                    (m_lo == -CUDART_INF_F) ? 0.0f : exp2_approx((m_lo - new_m_lo) * Log2E);
                const float alpha_hi =
                    (m_hi == -CUDART_INF_F) ? 0.0f : exp2_approx((m_hi - new_m_hi) * Log2E);

#pragma unroll
                for (int l = 0; l < 8; ++l) {
                    const float new_m = ((l & 2) == 0) ? new_m_lo : new_m_hi;
                    d_score[l] = (new_m > -CUDART_INF_F && d_score[l] > -CUDART_INF_F)
                                     ? exp2_approx((d_score[l] - new_m) * Log2E)
                                     : 0.0f;
                }
                float bl_lo = d_score[0] + d_score[1] + d_score[4] + d_score[5];
                float bl_hi = d_score[2] + d_score[3] + d_score[6] + d_score[7];
                bl_lo        = bl_lo + __shfl_xor_sync(FullMask, bl_lo, 2, 32);
                bl_hi        = bl_hi + __shfl_xor_sync(FullMask, bl_hi, 2, 32);

                l_lo = l_lo * alpha_lo + bl_lo;
                l_hi = l_hi * alpha_hi + bl_hi;
                m_lo = new_m_lo;
                m_hi = new_m_hi;
                // Own-row selection: this thread's PV accumulator (acc_f) is tied to output
                // row=lane specifically, which is r_lo when lane&2==0 and r_hi otherwise --
                // see volta_mma.cuh's I_MAJOR addressing for the Q/P/PV-output tile.
                const float alpha = ((lane & 2) == 0) ? alpha_lo : alpha_hi;

                volta_softmax_to_half2(p, d_score);
                return alpha;
            };

            if constexpr (Fp32Pv) {
                if (warp == 0) {
                    float values[2][8];
#pragma unroll
                    for (int sub = 0; sub < 2; ++sub) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const float score = scores_s[sub][j][lane];
                            const int row = volta_d_get_i(j);
                            const int key = k0 + sub * 8 + volta_d_get_j(j);
                            const int qabs = (j & 2) == 0 ? qabs_lo : qabs_hi;
                            values[sub][j] = row < rows_here && key >= split_start &&
                                             key < split_end && key <= qabs
                                ? score * scale : -CUDART_INF_F;
                        }
                    }
                    float bm_lo = -CUDART_INF_F, bm_hi = -CUDART_INF_F;
#pragma unroll
                    for (int sub = 0; sub < 2; ++sub) {
                        bm_lo = fmaxf(bm_lo, fmaxf(fmaxf(values[sub][0], values[sub][1]),
                                                   fmaxf(values[sub][4], values[sub][5])));
                        bm_hi = fmaxf(bm_hi, fmaxf(fmaxf(values[sub][2], values[sub][3]),
                                                   fmaxf(values[sub][6], values[sub][7])));
                    }
                    bm_lo = fmaxf(bm_lo, __shfl_xor_sync(FullMask, bm_lo, 2, 32));
                    bm_hi = fmaxf(bm_hi, __shfl_xor_sync(FullMask, bm_hi, 2, 32));
                    const float next_m_lo = fmaxf(m_lo, bm_lo);
                    const float next_m_hi = fmaxf(m_hi, bm_hi);
                    const float a_lo = m_lo == -CUDART_INF_F ? 0.0f :
                                      exp2_approx((m_lo - next_m_lo) * Log2E);
                    const float a_hi = m_hi == -CUDART_INF_F ? 0.0f :
                                      exp2_approx((m_hi - next_m_hi) * Log2E);
                    float sum_lo = 0.0f, sum_hi = 0.0f;
#pragma unroll
                    for (int sub = 0; sub < 2; ++sub) {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            const float m = (j & 2) == 0 ? next_m_lo : next_m_hi;
                            values[sub][j] = m > -CUDART_INF_F && values[sub][j] > -CUDART_INF_F
                                ? exp2_approx((values[sub][j] - m) * Log2E) : 0.0f;
                        }
                        sum_lo += values[sub][0] + values[sub][1] + values[sub][4] + values[sub][5];
                        sum_hi += values[sub][2] + values[sub][3] + values[sub][6] + values[sub][7];
                        half2 p[4];
                        volta_softmax_to_half2(p, values[sub]);
#pragma unroll
                        for (int j = 0; j < 4; ++j) probabilities_s[sub][j][lane] = p[j];
                    }
                    sum_lo += __shfl_xor_sync(FullMask, sum_lo, 2, 32);
                    sum_hi += __shfl_xor_sync(FullMask, sum_hi, 2, 32);
                    l_lo = l_lo * a_lo + sum_lo;
                    l_hi = l_hi * a_hi + sum_hi;
                    m_lo = next_m_lo;
                    m_hi = next_m_hi;
                    alpha_s[0][lane] = (lane & 2) == 0 ? a_lo : a_hi;
                    alpha_s[1][lane] = 1.0f;
                }
                __syncthreads();
            } else if constexpr (SharedSoftmax) {
                // Each dimension warp uses the same probabilities. Publish both eight-key
                // updates in order, then let all warps consume their unchanged PV slices.
                if (warp == 0) {
#pragma unroll
                    for (int sub = 0; sub < PVChunks; ++sub) {
                        half2 p[4];
                        alpha_s[sub][lane] = prepare_probability(sub, p);
#pragma unroll
                        for (int j = 0; j < 4; ++j) { probabilities_s[sub][j][lane] = p[j]; }
                    }
                }
                __syncthreads();
            }

#pragma unroll
            for (int sub = 0; sub < PVChunks; ++sub) {
                half2 p[4];
                float alpha;
                if constexpr (SharedSoftmax) {
                    alpha = alpha_s[sub][lane];
#pragma unroll
                    for (int j = 0; j < 4; ++j) { p[j] = probabilities_s[sub][j][lane]; }
                } else {
                    alpha = prepare_probability(sub, p);
                }

#pragma unroll
                for (int c = 0; c < DChunksLocal; ++c) {
                    half2 vf[4];
                    volta_load_v(vf,
                                 reinterpret_cast<const half2*>(
                                     &v_s[sub * 8 * SmemStride + dim_warp * DSlice + c * 8]),
                                 SmemStride / 2);
                    if constexpr (Fp32Pv) {
                        if (sub == 0) {
                            const float alpha_lo = alpha_s[0][lane & ~2];
                            const float alpha_hi = alpha_s[0][lane | 2];
#pragma unroll
                            for (int j = 0; j < 8; ++j) {
                                acc_f[c][j] *= (j & 2) == 0 ? alpha_lo : alpha_hi;
                            }
                        }
                        volta_attention_pv_fp32(acc_f[c], p, vf);
                    } else {
                        half2 pv[4] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}};
                        volta_mma_pv(pv, p, vf);
#pragma unroll
                        for (int n = 0; n < 4; ++n) {
                            const float2 contrib = __half22float2(pv[n]);
                            acc_f[c][2 * n + 0] = acc_f[c][2 * n + 0] * alpha + contrib.x;
                            acc_f[c][2 * n + 1] = acc_f[c][2 * n + 1] * alpha + contrib.y;
                        }
                    }
                }
            }
            __syncthreads();
        }

        // --- Write-out: partial_m/partial_l once per row-tile (dim_warp==0 only -- every
        // dim-split warp computed the same m/l redundantly), partial_acc per dim-split
        // warp's own DSlice-wide column range. ---
        const int row      = lane;
        const float own_m = ((lane & 2) == 0) ? m_lo : m_hi;
        const float own_l = ((lane & 2) == 0) ? l_lo : l_hi;
        if (dim_warp == 0 && row < rows_here) {
            int q_head = 0;
            int token  = 0;
            gqa_small_t_tc_row_to_qt<Geometry>(row_base + row, tokens, kv_head, q_head, token);
            if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
                partial_m[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = own_m;
                partial_l[gqa_partial_stat_index<Geometry>(q_head, token, split, tokens)] = own_l;
            }
        }
        if constexpr (Fp32Pv) {
#pragma unroll
            for (int c = 0; c < DChunksLocal; ++c) {
#pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const int output_row = volta_d_get_i(j);
                    if (output_row < rows_here) {
                        int q_head = 0, token = 0;
                        gqa_small_t_tc_row_to_qt<Geometry>(row_base + output_row, tokens, kv_head,
                                                          q_head, token);
                        const int d = dim_warp * DSlice + c * 8 + volta_d_get_j(j);
                        partial_acc[gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens)] =
                            acc_f[c][j];
                    }
                }
            }
        } else {
            if (row < rows_here) {
                int q_head = 0;
                int token  = 0;
                gqa_small_t_tc_row_to_qt<Geometry>(row_base + row, tokens, kv_head, q_head, token);
                if (gqa_valid_q_head<Geometry>(kv_head, q_head)) {
#pragma unroll
                    for (int c = 0; c < DChunksLocal; ++c) {
                        const int d = dim_warp * DSlice + c * 8;
                        __nv_bfloat16 out8[8];
#pragma unroll
                        for (int i = 0; i < 8; ++i) { out8[i] = __float2bfloat16(acc_f[c][i]); }
                        const std::int64_t dst =
                            gqa_partial_acc_index<Geometry>(q_head, d, token, split, tokens);
                        store_vec(&partial_acc[dst], *reinterpret_cast<const int4*>(out8));
                    }
                }
            }
        }
        __syncthreads();
    }
#endif // !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
}

} // namespace ninfer::ops
