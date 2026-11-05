#pragma once

#include "ops/common/volta_mma.cuh"
#include "ops/kernel/gqa_attention_decode.cuh"
#include "ops/kernel/gqa_attention_kv_quant.cuh"

namespace ninfer::ops {

// SM70 TP2/TP4 five-query verifier for the optional complete-history long profile.
// Four 8-row QK warps each own their online softmax. Q fragments are distributed
// across quadpairs, then broadcast for each contraction: 32 rather than 128 Q
// registers per lane. K/V reuse the initial Q staging tile after that broadcast
// representation is resident. Eight dimension warps accumulate PV in FP32.
// This 32-key softmax/partition profile is not bit-equivalent to the 16-key TC
// verifier; both are checked against the same decoded-input attention oracle.
template <typename G, bool MultiBatch, bool Masked, typename CacheInput>
__global__ __launch_bounds__(256, 2) void gqa_attention_verify5_qpn_volta_partial_i8_kernel(
    const __nv_bfloat16* q, CacheInput input, const std::int32_t* positions,
    std::int8_t* keys, std::int8_t* values, half* key_scales, half* value_scales,
    const std::int32_t* block_tables, const std::int32_t* valid_columns,
    const std::int32_t* table_rows, std::int32_t table_stride,
    std::int32_t full_width, std::int32_t column_begin, std::int32_t logical_capacity,
    float scale, __nv_bfloat16* partial, float* partial_m, float* partial_l) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ == 700
    static_assert(G::QHeads == 12 || G::QHeads == 6);
    constexpr int T = 5, D = kGqaHeadDim, Bc = 32, Stride = D + 8, PStride = Bc + 8;
    constexpr int Rows = T * G::GroupSize, RowTiles = (Rows + 7) / 8;
    constexpr float Log2E = 1.4426950408889634074f;
    constexpr unsigned Mask = 0xffffffffu;
    __shared__ __align__(16) half tiles[2 * Bc * Stride];
    half* qs = tiles;
    half* ks = tiles;
    half* vs = ks + Bc * Stride;
    __shared__ __align__(16) half ps[RowTiles][8][PStride];
    __shared__ float alphas[RowTiles][8];
    constexpr int LongSplitPages =
        (((kGqaVoltaLongKeysPerSplit + Bc - 1) / Bc) * Bc + kPagedKVPageSize - 1) /
            kPagedKVPageSize + 1;
    constexpr int PageIds = LongSplitPages > kGqaSmallTSplitPageIds<G, Bc>
        ? LongSplitPages : kGqaSmallTSplitPageIds<G, Bc>;
    __shared__ std::int32_t pages[PageIds];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int head = blockIdx.x, split = blockIdx.y, split_count = gridDim.y;
    const int batch = MultiBatch ? blockIdx.z : 0;
    const std::int64_t column = column_begin + static_cast<std::int64_t>(batch) * full_width;
    q += column * D * G::QHeads;
    positions += column;
    if constexpr (CacheInput::writes_cache) {
        input.k += column * D * G::KVHeads;
        input.v += column * D * G::KVHeads;
    }
    if constexpr (MultiBatch) {
        partial += static_cast<std::int64_t>(batch) * D * G::QHeads * T * split_count;
        partial_m += static_cast<std::int64_t>(batch) * G::QHeads * T * split_count;
        partial_l += static_cast<std::int64_t>(batch) * G::QHeads * T * split_count;
    }
    const auto neutral = [&] {
        for (int i = tid; i < Rows * D; i += 256) {
            const int row = i / D;
            partial[gqa_partial_acc_index<G>(head * G::GroupSize + row / T,
                                             i % D, row % T, split, T)] = __float2bfloat16(0.0f);
        }
        for (int row = tid; row < Rows; row += 256) {
            const auto index = gqa_partial_stat_index<G>(head * G::GroupSize + row / T,
                                                         row % T, split, T);
            partial_m[index] = -CUDART_INF_F;
            partial_l[index] = 0.0f;
        }
    };
    int valid_tokens = T;
    if constexpr (Masked) valid_tokens = min(T, max(0, valid_columns[batch] - column_begin));
    if (valid_tokens == 0 || positions[0] < 0 || positions[T - 1] < 0 ||
        positions[T - 1] >= logical_capacity) {
        neutral();
        return;
    }
    const int window = positions[T - 1] + 1;
    const int active = gqa_small_t_active_splits<G, true>(window, split_count, T, true,
                                                          MultiBatch ? gridDim.z : 1);
    if (split >= active) return;
    const int tiles_per_split = div_up(div_up(window, Bc), active);
    const int begin = split * tiles_per_split * Bc;
    const int end = min(window, begin + tiles_per_split * Bc);
    if (begin >= end) { neutral(); return; }
    const int table_row = table_rows == nullptr ? 0 : table_rows[batch];
    const auto* table = block_tables + static_cast<std::int64_t>(table_row) * table_stride;
    const int first_page = begin >> kPagedKVPageShift;
    const int page_count = ((end - 1) >> kPagedKVPageShift) - first_page + 1;
    for (int i = tid; i < page_count; i += 256) pages[i] = table[first_page + i];
    for (int i = tid; i < RowTiles * 8 * D; i += 256) {
        const int row = i / D, dim = i % D;
        qs[row * Stride + dim] = row < Rows && row % T < valid_tokens
            ? __float2half(__bfloat162float(q[gqa_q_index<G>(head * G::GroupSize + row / T,
                                                            dim, row % T)]))
            : __ushort_as_half(0);
    }
    if constexpr (CacheInput::writes_cache) {
        for (int pair = warp; pair < valid_tokens * kGqaKvQuantGroups; pair += 8) {
            const int token = pair / kGqaKvQuantGroups, group = pair % kGqaKvQuantGroups;
            const int position = positions[token];
            if (position < begin || position >= end) continue;
            const int page = table[position >> kPagedKVPageShift];
            const int offset = position & kPagedKVPageMask;
            const int d0 = group * kGqaKvQuantGroup + lane, d1 = d0 + 32;
            const float k0 = __bfloat162float(input.k[gqa_kv_new_index<G>(head, d0, token)]);
            const float k1 = __bfloat162float(input.k[gqa_kv_new_index<G>(head, d1, token)]);
            const float v0 = __bfloat162float(input.v[gqa_kv_new_index<G>(head, d0, token)]);
            const float v1 = __bfloat162float(input.v[gqa_kv_new_index<G>(head, d1, token)]);
            const float ka = warp_max(fmaxf(fabsf(k0), fabsf(k1)), Mask);
            const float va = warp_max(fmaxf(fabsf(v0), fabsf(v1)), Mask);
            const half kscale = __float2half_rn(ka > 0.0f ? ka / 127.0f : 0.0f);
            const half vscale = __float2half_rn(va > 0.0f ? va / 127.0f : 0.0f);
            const float ki = __half2float(kscale) > 0.0f ? 1.0f / __half2float(kscale) : 0.0f;
            const float vi = __half2float(vscale) > 0.0f ? 1.0f / __half2float(vscale) : 0.0f;
            keys[gqa_kv_quant_code_index<G>(page, head, d0, offset)] = gqa_kv_quant_code(k0, ki);
            keys[gqa_kv_quant_code_index<G>(page, head, d1, offset)] = gqa_kv_quant_code(k1, ki);
            values[gqa_kv_quant_code_index<G>(page, head, d0, offset)] = gqa_kv_quant_code(v0, vi);
            values[gqa_kv_quant_code_index<G>(page, head, d1, offset)] = gqa_kv_quant_code(v1, vi);
            if (lane == 0) {
                const auto index = gqa_kv_quant_scale_index<G>(page, head, group, offset);
                key_scales[index] = kscale;
                value_scales[index] = vscale;
            }
        }
    }
    __syncthreads();
    const int qp = (lane >> 2) & 3;
    const int r = (lane & 3) + ((lane & 16) ? 4 : 0);
    // Quadpairs keep disjoint K fragments for the same eight query rows.
    // The following barrier retires every Q load before K/V overwrite this tile.
    uint2 q_registers[D / 16];
    if (warp < RowTiles) {
#pragma unroll
        for (int part = 0; part < D / 16; ++part) {
            q_registers[part] = *reinterpret_cast<const uint2*>(
                qs + (warp * 8 + r) * Stride + part * 16 + qp * 4);
        }
    }
    __syncthreads();
    float accum[RowTiles][8] = {};
    float maximum_lo = -CUDART_INF_F, maximum_hi = -CUDART_INF_F;
    float denominator_lo = 0.0f, denominator_hi = 0.0f;
    for (int tile = begin; tile < end; tile += Bc) {
        for (int chunk = tid; chunk < Bc * (D / 16); chunk += 256) {
            const int key = tile + chunk / (D / 16), dim = (chunk % (D / 16)) * 16;
            const int local = key - tile;
            if (key < end) {
                const int page = pages[(key >> kPagedKVPageShift) - first_page];
                const int offset = key & kPagedKVPageMask;
                const auto code = gqa_kv_quant_code_index<G>(page, head, dim, offset);
                const auto group = gqa_kv_quant_scale_index<G>(page, head, dim / 64, offset);
                const half kscale = key_scales[group], vscale = value_scales[group];
#pragma unroll
                for(int c=0;c<16;c+=16) {
                    const uint4 kr = __ldcs(reinterpret_cast<const uint4*>(keys + code + c));
                    const uint4 vr = __ldcs(reinterpret_cast<const uint4*>(values + code + c));
                    store_vec(ks+local*Stride+dim+c,
                        gqa_kv_dequant_i8x8_f16(make_int2(kr.x,kr.y),kscale));
                    store_vec(ks+local*Stride+dim+c+8,
                        gqa_kv_dequant_i8x8_f16(make_int2(kr.z,kr.w),kscale));
                    store_vec(vs+local*Stride+dim+c,
                        gqa_kv_dequant_i8x8_f16(make_int2(vr.x,vr.y),vscale));
                    store_vec(vs+local*Stride+dim+c+8,
                        gqa_kv_dequant_i8x8_f16(make_int2(vr.z,vr.w),vscale));
                }
            } else {
#pragma unroll
                for(int c=0;c<16;c+=8) {
                    store_vec(ks+local*Stride+dim+c,make_int4(0,0,0,0));
                    store_vec(vs+local*Stride+dim+c,make_int4(0,0,0,0));
                }
            }
        }
        __syncthreads();
        // TP4 streams one KV head per rank. Fetch the next tile while the
        // current tile's QK/softmax/PV runs, without another register buffer.
        if constexpr (G::QHeads == 6) {
            if (tid < 64) {
                const int next_key = tile + Bc + tid / 2;
                if (next_key < end) {
                    const int page = pages[(next_key >> kPagedKVPageShift) - first_page];
                    const auto code = gqa_kv_quant_code_index<G>(
                        page, head, (tid & 1) * 128, next_key & kPagedKVPageMask);
                    asm volatile("prefetch.global.L2 [%0];" : : "l"(keys + code));
                    asm volatile("prefetch.global.L2 [%0];" : : "l"(values + code));
                }
            }
        }
        if (warp < RowTiles) {
            float dot[Bc / 32][8] = {};
#pragma unroll
            for (int dim = 0; dim < D; dim += 4) {
                const auto local_q = q_registers[dim / 16];
                // Bits 0,1,4 select the row; bits 2,3 select its owning quadpair.
                const int owner = (lane & 19) | (((dim / 4) & 3) << 2);
                const auto a = make_uint2(__shfl_sync(Mask, local_q.x, owner),
                                          __shfl_sync(Mask, local_q.y, owner));
#pragma unroll
                for (int block = 0; block < Bc / 32; ++block) {
                    const auto b = *reinterpret_cast<const uint2*>(
                        ks + (block * 32 + qp * 8 + r) * Stride + dim);
                    volta_mma_qp_n(dot[block], a.x, a.y, b.x, b.y);
                }
            }
            const int row_lo = warp * 8 + ((lane & 16) ? 4 : 0) + (lane & 1);
            const int row_hi = row_lo + 2;
            const int pos_lo = row_lo < Rows && row_lo % T < valid_tokens
                ? positions[row_lo % T] : -1;
            const int pos_hi = row_hi < Rows && row_hi % T < valid_tokens
                ? positions[row_hi % T] : -1;
            float bm_lo = -CUDART_INF_F, bm_hi = -CUDART_INF_F;
#pragma unroll
            for (int block = 0; block < Bc / 32; ++block) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int col = block * 32 + qp * 8 + (i & 1) +
                                    (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
                    const int limit = (i & 2) ? pos_hi : pos_lo;
                    dot[block][i] = tile + col < end && tile + col <= limit
                        ? dot[block][i] * scale : -CUDART_INF_F;
                    if (i & 2) bm_hi = fmaxf(bm_hi, dot[block][i]);
                    else bm_lo = fmaxf(bm_lo, dot[block][i]);
                }
            }
#pragma unroll
            for (int offset = 2; offset <= 8; offset *= 2) {
                bm_lo = fmaxf(bm_lo, __shfl_xor_sync(Mask, bm_lo, offset));
                bm_hi = fmaxf(bm_hi, __shfl_xor_sync(Mask, bm_hi, offset));
            }
            const float next_lo = fmaxf(maximum_lo, bm_lo), next_hi = fmaxf(maximum_hi, bm_hi);
            const float alpha_lo = maximum_lo == -CUDART_INF_F ? 0.0f
                : exp2_approx((maximum_lo - next_lo) * Log2E);
            const float alpha_hi = maximum_hi == -CUDART_INF_F ? 0.0f
                : exp2_approx((maximum_hi - next_hi) * Log2E);
            float sum_lo = 0.0f, sum_hi = 0.0f;
#pragma unroll
            for (int block = 0; block < Bc / 32; ++block) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
                    const int col = block * 32 + qp * 8 + (i & 1) +
                                    (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
                    const float maximum = (i & 2) ? next_hi : next_lo;
                    const float p = dot[block][i] > -CUDART_INF_F
                        ? exp2_approx((dot[block][i] - maximum) * Log2E) : 0.0f;
                    ps[warp][row][col] = __float2half(p);
                    if (i & 2) sum_hi += p; else sum_lo += p;
                }
            }
#pragma unroll
            for (int offset = 2; offset <= 8; offset *= 2) {
                sum_lo += __shfl_xor_sync(Mask, sum_lo, offset);
                sum_hi += __shfl_xor_sync(Mask, sum_hi, offset);
            }
            denominator_lo = denominator_lo * alpha_lo + sum_lo;
            denominator_hi = denominator_hi * alpha_hi + sum_hi;
            maximum_lo = next_lo; maximum_hi = next_hi;
            if (qp == 0) alphas[warp][r] = (lane & 2) ? alpha_hi : alpha_lo;
        }
        __syncthreads();
#pragma unroll
        for (int group = 0; group < RowTiles; ++group) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
                accum[group][i] *= alphas[group][row];
            }
            const int dim = warp * 32 + qp * 8 + r;
#pragma unroll
            for (int key = 0; key < Bc; key += 4) {
                const auto a = *reinterpret_cast<const uint2*>(&ps[group][r][key]);
                const half2 b0 = __halves2half2(vs[key * Stride + dim], vs[(key + 1) * Stride + dim]);
                const half2 b1 = __halves2half2(vs[(key + 2) * Stride + dim], vs[(key + 3) * Stride + dim]);
                volta_mma_qp_n(accum[group], a.x, a.y, *reinterpret_cast<const unsigned*>(&b0),
                              *reinterpret_cast<const unsigned*>(&b1));
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int group = 0; group < RowTiles; ++group) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = group * 8 + ((i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1));
            const int dim = warp * 32 + qp * 8 + (i & 1) + (((lane >> 1) & 1) << 1) + ((i >> 2) << 2);
            if (row < Rows) {
                partial[gqa_partial_acc_index<G>(head * G::GroupSize + row / T,
                                                 dim, row % T, split, T)] =
                    __float2bfloat16(accum[group][i]);
            }
        }
    }
    if (warp < RowTiles && qp == 0 && warp * 8 + r < Rows) {
        const int row = warp * 8 + r;
        const auto index = gqa_partial_stat_index<G>(head * G::GroupSize + row / T,
                                                     row % T, split, T);
        partial_m[index] = (lane & 2) ? maximum_hi : maximum_lo;
        partial_l[index] = (lane & 2) ? denominator_hi : denominator_lo;
    }
#endif
}

} // namespace ninfer::ops
