#include "targets/qwen3_6/impl/vision/sm86/attention.h"

#include "core/device.h"

#include <cuda_bf16.h>
#include <math_constants.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace ninfer::targets::qwen3_6::vision_sm86 {
namespace {

constexpr int kDim = 72;
constexpr int kHeads = 16;
constexpr int kThreads = 128;
constexpr std::int64_t kTokenBytes = kDim * kHeads * 2;

struct AddressRange {
    std::uintptr_t begin;
    std::uintptr_t end;
};

AddressRange checked_range(const void* data, std::uint64_t bytes, const char* name) {
    const auto begin = reinterpret_cast<std::uintptr_t>(data);
    if (data == nullptr || bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
        throw std::invalid_argument(std::string("vision attention: invalid ") + name + " extent");
    }
    return {begin, begin + static_cast<std::uintptr_t>(bytes)};
}

bool overlaps(AddressRange a, AddressRange b) {
    return a.begin < b.end && b.begin < a.end;
}

AddressRange validate_activation(const Tensor& tensor, std::int32_t patches, const char* name) {
    if (tensor.dtype != DType::BF16 || tensor.ne[0] != kDim || tensor.ne[1] != kHeads ||
        tensor.ne[2] != patches || tensor.ne[3] != 1 || tensor.nb[0] != 2 ||
        tensor.nb[1] != kDim * 2 || tensor.nb[2] < kTokenBytes || tensor.nb[2] % 2 != 0 ||
        reinterpret_cast<std::uintptr_t>(tensor.data) % 2 != 0) {
        throw std::invalid_argument(std::string("vision attention: invalid ") + name +
                                    " shape, strides or alignment");
    }
    const auto stride = static_cast<std::uint64_t>(tensor.nb[2]);
    const auto last = static_cast<std::uint64_t>(patches - 1);
    const auto max_offset = static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
    if (last != 0 && stride > (max_offset - kTokenBytes) / last) {
        throw std::overflow_error(std::string("vision attention: ") + name + " extent overflows");
    }
    return checked_range(tensor.data, last * stride + kTokenBytes, name);
}

// These helpers are private to the SM86 backend and require Ampere instructions.
constexpr int kVisionAttentionHeadDim = kDim;
constexpr int kVisionAttentionHeads = kHeads;
constexpr int kVisionAttentionPaddedD = 128;
constexpr int kTile = 64;

__device__ __forceinline__ unsigned smem_addr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void cp_async_16(void* dst, const void* src) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(smem_addr(dst)), "l"(src));
}
__device__ __forceinline__ void cp_commit() { asm volatile("cp.async.commit_group;"); }
template <int N> __device__ __forceinline__ void cp_wait() {
    asm volatile("cp.async.wait_group %0;" :: "n"(N));
}
__device__ __forceinline__ void ldmatrix_x4(unsigned& a, unsigned& b, unsigned& c, unsigned& d,
                                            unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x4_t(unsigned& a, unsigned& b, unsigned& c, unsigned& d,
                                              unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(a), "=r"(b), "=r"(c), "=r"(d) : "r"(addr));
}
__device__ __forceinline__ void mma_bf16(float& c0, float& c1, float& c2, float& c3,
                                         unsigned a0, unsigned a1, unsigned a2, unsigned a3,
                                         unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
__device__ __forceinline__ unsigned pack_bf16x2(float lo, float hi) {
    unsigned result;
    asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(result)
                 : "r"(__float_as_uint(hi)), "r"(__float_as_uint(lo)));
    return result;
}
__device__ __forceinline__ unsigned pack_residual(float lo, float hi) {
    return pack_bf16x2(lo - __bfloat162float(__float2bfloat16_rn(lo)),
                       hi - __bfloat162float(__float2bfloat16_rn(hi)));
}
__device__ __forceinline__ float exp2_approx(float x) {
    float y;
    asm("ex2.approx.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
template <int Width> __device__ __forceinline__ float warp_max(float x, unsigned mask) {
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1)
        x = fmaxf(x, __shfl_xor_sync(mask, x, offset, Width));
    return x;
}
template <int Width> __device__ __forceinline__ float warp_sum(float x, unsigned mask) {
#pragma unroll
    for (int offset = Width / 2; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(mask, x, offset, Width);
    return x;
}
__device__ __forceinline__ int vision_attention_swz(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}
__device__ __forceinline__ unsigned vision_attention_swz_addr(unsigned base, unsigned col,
                                                               unsigned part, unsigned row) {
    return base + ((col | part) ^ row);
}

template <int Rows, int Threads>
__device__ __forceinline__ void stage_rows(__nv_bfloat16* dst, const __nv_bfloat16* src,
                                           std::int64_t first, int end, int head, int tid,
                                           std::int64_t stride_t) {
    constexpr int VecsPerRow = kVisionAttentionPaddedD / 8;
    for (int chunk = tid; chunk < Rows * VecsPerRow; chunk += Threads) {
        const int row = chunk / VecsPerRow;
        const int d = (chunk % VecsPerRow) * 8;
        __nv_bfloat16* smem = &dst[row * kVisionAttentionPaddedD + vision_attention_swz(row, d)];
        if (first + row < end && d < kDim) {
            const __nv_bfloat16* global = src + (first + row) * stride_t + head * kDim + d;
            if ((reinterpret_cast<std::uintptr_t>(global) & 15u) == 0) {
                cp_async_16(smem, global);
            } else {
#pragma unroll
                for (int i = 0; i < 8; ++i) { smem[i] = global[i]; }
            }
        } else {
#pragma unroll
            for (int i = 0; i < 8; ++i) { smem[i] = __float2bfloat16(0.0f); }
        }
    }
}

template <int Br, int Bc>
__launch_bounds__(Br * 2, 128 / Br) __global__ void vision_attention_flash_kernel(
    const __nv_bfloat16* __restrict__ q, const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v, const std::int32_t* __restrict__ bounds,
    int patches, int segments, __nv_bfloat16* __restrict__ out,
    std::int64_t q_stride_t, std::int64_t k_stride_t, std::int64_t v_stride_t) {
    static_assert(Br == 16 || Br == 32 || Br == 64);
    static_assert(Bc == 16 || Bc == 32 || Bc == 64);
    constexpr int D             = kVisionAttentionHeadDim;
    constexpr int Dp            = kVisionAttentionPaddedD;
    constexpr int Threads       = Br * 2;
    constexpr int QKNt          = Bc / 8;
    constexpr int QKKs          = 5; // ceil(72 / 16)
    constexpr int PVNt          = D / 8;
    constexpr int PVKs          = Bc / 16;
    constexpr int RowBytes      = Dp * static_cast<int>(sizeof(__nv_bfloat16));
    constexpr float ScaleLog2E  = 0.11785113019775792073f * 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    const std::int64_t block_begin = static_cast<std::int64_t>(blockIdx.x) * Br;
    const int block_end = static_cast<int>(block_begin + Br < patches ? block_begin + Br : patches);
    int low = 0;
    int high = segments;
    while (low + 1 < high) {
        const int mid = low + (high - low) / 2;
        if (bounds[mid] <= block_begin) { low = mid; } else { high = mid; }
    }
    for (int segment = low; segment < segments && bounds[segment] < block_end; ++segment) {
        const int begin = bounds[segment];
        const int end = bounds[segment + 1];
        const int q0 = static_cast<int>(block_begin > begin ? block_begin : begin);
        const int query_end = block_end < end ? block_end : end;
        const int head = static_cast<int>(blockIdx.y);
        const int tid  = static_cast<int>(threadIdx.x);
        const int warp = tid >> 5;
        const int lane = tid & 31;

        extern __shared__ __align__(16) __nv_bfloat16 shared[];
        __nv_bfloat16* q_s = shared;
        __nv_bfloat16* k_s = q_s + Br * Dp;
        __nv_bfloat16* v_s = k_s + Bc * Dp;

        const int gid       = lane >> 2;
        const int lid       = lane & 3;
        const int a_mat     = lane >> 3;
        const int a_rin     = lane & 7;
        const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
        const int b_rin     = lane & 7;
        const int b_koff    = ((lane >> 3) & 1) << 3;
        const int warp_row0 = warp * 16;

        const unsigned q_sbase     = smem_addr(q_s);
        const unsigned k_sbase     = smem_addr(k_s);
        const unsigned v_sbase     = smem_addr(v_s);
        const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * RowBytes);
        const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
        const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
        const unsigned k_lane_base = k_sbase + static_cast<unsigned>(b_rin * RowBytes) +
                                     static_cast<unsigned>((lane >> 4) * 8 * RowBytes);
        const unsigned k_as        = static_cast<unsigned>((b_koff >> 3) << 4);
        const unsigned k_r         = static_cast<unsigned>(b_rin << 4);
        const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 8 * RowBytes) +
                                     static_cast<unsigned>(b_rin * RowBytes);
        const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
        const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

        stage_rows<Br, Threads>(q_s, q, q0, query_end, head, tid, q_stride_t);

        float acc[PVNt][4];
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
#pragma unroll
            for (int item = 0; item < 4; ++item) { acc[n][item] = 0.0f; }
        }
        float m0 = -CUDART_INF_F;
        float m1 = -CUDART_INF_F;
        float l0 = 0.0f;
        float l1 = 0.0f;

        cp_commit();
        stage_rows<Bc, Threads>(k_s, k, begin, end, head, tid, k_stride_t);
        cp_commit();

        const int key_blocks = static_cast<int>((static_cast<std::int64_t>(end) - begin + Bc - 1) / Bc);
        for (int kb = 0; kb < key_blocks; ++kb) {
            const std::int64_t key0 = static_cast<std::int64_t>(begin) + static_cast<std::int64_t>(kb) * Bc;
            cp_wait<0>();
            __syncthreads();

            stage_rows<Bc, Threads>(v_s, v, key0, end, head, tid, v_stride_t);
            cp_commit();

            float score[QKNt][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
            }
            unsigned af[2][4];
            unsigned bf[2][QKNt][2];
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        vision_attention_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(
                    bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                    vision_attention_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 8 * RowBytes),
                                              0u, k_as, k_r));
            }
#pragma unroll
            for (int ks = 0; ks < QKKs; ++ks) {
                const int cur = ks & 1;
                const int nxt = cur ^ 1;
                if (ks + 1 < QKKs) {
                    const unsigned ck = static_cast<unsigned>((ks + 1) << 5);
                    ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                                vision_attention_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                    for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                        ldmatrix_x4(bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0],
                                    bf[nxt][nt2 + 1][1],
                                    vision_attention_swz_addr(
                                        k_lane_base + static_cast<unsigned>(nt2 * 8 * RowBytes), ck,
                                        k_as, k_r));
                    }
                }
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                             af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
                }
            }

            const int row0       = warp_row0 + gid;
            const int row1       = row0 + 8;
            const std::int64_t query0 = static_cast<std::int64_t>(q0) + row0;
            const std::int64_t query1 = static_cast<std::int64_t>(q0) + row1;
            const bool full_tile = static_cast<std::int64_t>(q0) + Br <= query_end && key0 + Bc <= end;
            float block_max0     = -CUDART_INF_F;
            float block_max1     = -CUDART_INF_F;
            if (full_tile) {
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    block_max0 = fmaxf(block_max0, fmaxf(score[nt][0], score[nt][1]));
                    block_max1 = fmaxf(block_max1, fmaxf(score[nt][2], score[nt][3]));
                }
            } else {
#pragma unroll
                for (int nt = 0; nt < QKNt; ++nt) {
                    const std::int64_t key_a = key0 + nt * 8 + 2 * lid;
                    const std::int64_t key_b = key_a + 1;
                    score[nt][0] = query0 < query_end && key_a < end ? score[nt][0] : -CUDART_INF_F;
                    score[nt][1] = query0 < query_end && key_b < end ? score[nt][1] : -CUDART_INF_F;
                    score[nt][2] = query1 < query_end && key_a < end ? score[nt][2] : -CUDART_INF_F;
                    score[nt][3] = query1 < query_end && key_b < end ? score[nt][3] : -CUDART_INF_F;
                    block_max0   = fmaxf(block_max0, fmaxf(score[nt][0], score[nt][1]));
                    block_max1   = fmaxf(block_max1, fmaxf(score[nt][2], score[nt][3]));
                }
            }
            block_max0 = warp_max<4>(block_max0, FullMask);
            block_max1 = warp_max<4>(block_max1, FullMask);

            const float next_m0 = fmaxf(m0, block_max0);
            const float next_m1 = fmaxf(m1, block_max1);
            const float m0_l2   = next_m0 * ScaleLog2E;
            const float m1_l2   = next_m1 * ScaleLog2E;
            const float alpha0  = exp2_approx(__fmaf_rn(m0, ScaleLog2E, -m0_l2));
            const float alpha1  = exp2_approx(__fmaf_rn(m1, ScaleLog2E, -m1_l2));

            float block_sum0 = 0.0f;
            float block_sum1 = 0.0f;
            unsigned p_frag[PVKs][4];
            unsigned p_low[PVKs][4];
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = score[nt][0] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][0], ScaleLog2E, -m0_l2))
                                      : 0.0f;
                const float p01 = score[nt][1] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][1], ScaleLog2E, -m0_l2))
                                      : 0.0f;
                const float p10 = score[nt][2] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][2], ScaleLog2E, -m1_l2))
                                      : 0.0f;
                const float p11 = score[nt][3] > -CUDART_INF_F
                                      ? exp2_approx(__fmaf_rn(score[nt][3], ScaleLog2E, -m1_l2))
                                      : 0.0f;
                block_sum0 += p00 + p01;
                block_sum1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_low[pk][0] = pack_residual(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                    p_low[pk][1] = pack_residual(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_low[pk][2] = pack_residual(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                    p_low[pk][3] = pack_residual(p10, p11);
                }
            }

            l0 = __fmaf_rn(l0, alpha0, block_sum0);
            l1 = __fmaf_rn(l1, alpha1, block_sum1);
            m0 = next_m0;
            m1 = next_m1;
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                acc[n][0] *= alpha0;
                acc[n][1] *= alpha0;
                acc[n][2] *= alpha1;
                acc[n][3] *= alpha1;
            }

            cp_wait<0>();
            __syncthreads();
            if (kb + 1 < key_blocks) {
                stage_rows<Bc, Threads>(k_s, k, key0 + Bc, end, head, tid, k_stride_t);
                cp_commit();
            }

            constexpr int PVTilePairs = (PVNt + 1) / 2;
            constexpr int PVLoads     = PVKs * PVTilePairs;
            unsigned vf[2][4];
            ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                          vision_attention_swz_addr(v_lane_base, 0u, v_as, v_r));
#pragma unroll
            for (int load = 0; load < PVLoads; ++load) {
                const int pk   = load / PVTilePairs;
                const int n2   = (load % PVTilePairs) * 2;
                const int cur  = load & 1;
                const int next = cur ^ 1;
                if (load + 1 < PVLoads) {
                    const int next_pk = (load + 1) / PVTilePairs;
                    const int next_n2 = ((load + 1) % PVTilePairs) * 2;
                    ldmatrix_x4_t(vf[next][0], vf[next][1], vf[next][2], vf[next][3],
                                  vision_attention_swz_addr(
                                      v_lane_base + static_cast<unsigned>(next_pk * 16 * RowBytes),
                                      static_cast<unsigned>(next_n2 << 4), v_as, v_r));
                }
                mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[pk][0], p_frag[pk][1],
                         p_frag[pk][2], p_frag[pk][3], vf[cur][0], vf[cur][1]);
                mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_low[pk][0], p_low[pk][1],
                         p_low[pk][2], p_low[pk][3], vf[cur][0], vf[cur][1]);
                if (n2 + 1 < PVNt) {
                    mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3],
                             p_frag[pk][0], p_frag[pk][1], p_frag[pk][2], p_frag[pk][3], vf[cur][2],
                             vf[cur][3]);
                    mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3],
                             p_low[pk][0], p_low[pk][1], p_low[pk][2], p_low[pk][3], vf[cur][2],
                             vf[cur][3]);
                }
            }
        }

        l0                 = warp_sum<4>(l0, FullMask);
        l1                 = warp_sum<4>(l1, FullMask);
        const float inv_l0 = l0 > 0.0f ? __frcp_rn(l0) : 0.0f;
        const float inv_l1 = l1 > 0.0f ? __frcp_rn(l1) : 0.0f;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            const int d0     = n * 8 + 2 * lid;
            const std::int64_t query0 = static_cast<std::int64_t>(q0) + warp_row0 + gid;
            const std::int64_t query1 = query0 + 8;
            if (query0 < query_end) {
                const std::int64_t offset =
                    (static_cast<std::int64_t>(query0) * kVisionAttentionHeads + head) * D + d0;
                out[offset] = __float2bfloat16(acc[n][0] * inv_l0);
                out[offset + 1] = __float2bfloat16(acc[n][1] * inv_l0);
            }
            if (query1 < query_end) {
                const std::int64_t offset =
                    (static_cast<std::int64_t>(query1) * kVisionAttentionHeads + head) * D + d0;
                out[offset] = __float2bfloat16(acc[n][2] * inv_l1);
                out[offset + 1] = __float2bfloat16(acc[n][3] * inv_l1);
            }
        }
        __syncthreads();
    }
}

} // namespace

std::size_t attention_workspace_bytes(std::int32_t patches, std::int32_t segments) {
    if (patches <= 0 || segments <= 0 || segments > patches) {
        throw std::invalid_argument("vision attention: invalid patch or segment count");
    }
    return 0;
}

void attention_cu_seqlens(const Tensor& q, const Tensor& k, const Tensor& v,
                          const Tensor& cu_seqlens, Tensor& out, cudaStream_t stream) {
    const std::int32_t patches = q.ne[2];
    const std::int64_t segments64 = static_cast<std::int64_t>(cu_seqlens.ne[0]) - 1;
    if (segments64 <= 0 || segments64 > patches) {
        throw std::invalid_argument("vision attention: invalid patch or segment count");
    }
    const auto segments = static_cast<std::int32_t>(segments64);
    (void)attention_workspace_bytes(patches, segments);
    const auto q_range = validate_activation(q, patches, "q");
    const auto k_range = validate_activation(k, patches, "k");
    const auto v_range = validate_activation(v, patches, "v");
    const auto out_range = validate_activation(out, patches, "out");
    if (!out.is_contiguous()) {
        throw std::invalid_argument("vision attention: output must be contiguous");
    }
    if (cu_seqlens.dtype != DType::I32 || cu_seqlens.ne[1] != 1 ||
        cu_seqlens.ne[2] != 1 || cu_seqlens.ne[3] != 1 || !cu_seqlens.is_contiguous() ||
        reinterpret_cast<std::uintptr_t>(cu_seqlens.data) % alignof(std::int32_t) != 0) {
        throw std::invalid_argument("vision attention: bounds must be contiguous I32 [S+1]");
    }
    const auto bounds_bytes = static_cast<std::size_t>(cu_seqlens.ne[0]) * sizeof(std::int32_t);
    const auto bounds_range = checked_range(cu_seqlens.data, bounds_bytes, "bounds");
    if (overlaps(out_range, q_range) || overlaps(out_range, k_range) ||
        overlaps(out_range, v_range) || overlaps(out_range, bounds_range)) {
        throw std::invalid_argument("vision attention: output must not overlap inputs");
    }
    std::vector<std::int32_t> bounds(static_cast<std::size_t>(segments) + 1);
    CUDA_CHECK(cudaMemcpyAsync(bounds.data(), cu_seqlens.data, bounds_bytes,
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (bounds.front() != 0 || bounds.back() != patches) {
        throw std::invalid_argument("vision attention: bounds must span [0, P]");
    }
    for (std::size_t i = 1; i < bounds.size(); ++i) {
        if (bounds[i] <= bounds[i - 1]) {
            throw std::invalid_argument("vision attention: bounds must be strictly increasing");
        }
    }
    const dim3 grid(static_cast<unsigned>((static_cast<std::int64_t>(patches) + kTile - 1) / kTile), kHeads);
    constexpr int shared_bytes = (kTile + 2 * kTile) * kVisionAttentionPaddedD * sizeof(__nv_bfloat16);
    vision_attention_flash_kernel<kTile, kTile><<<grid, kThreads, shared_bytes, stream>>>(
        static_cast<const __nv_bfloat16*>(q.data), static_cast<const __nv_bfloat16*>(k.data),
        static_cast<const __nv_bfloat16*>(v.data), static_cast<const std::int32_t*>(cu_seqlens.data),
        patches, segments, static_cast<__nv_bfloat16*>(out.data), q.nb[2] / 2, k.nb[2] / 2,
        v.nb[2] / 2);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace ninfer::targets::qwen3_6::vision_sm86
