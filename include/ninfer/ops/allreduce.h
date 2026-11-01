#pragma once

// ninfer::ops - two- and four-device collectives.
//
// These are the cross-device primitives tensor-parallel forward passes need: one
// summing reduction for row-parallel projections, one row gather for the vocabulary-split output
// head. Both operate on an ExecutionContext holding two or four DeviceContexts and use each
// device's own compute stream; no stream parameter is accepted, because "the stream a device
// executes on" is already a property of its DeviceContext and a second, disagreeing stream story
// would break the ordering guarantees documented below.
//
// TRANSPORT. Gather and large/fallback reduction payloads move with
// cudaMemcpyAsync(cudaMemcpyDeviceToDevice) over unified virtual
// addresses, which every 64-bit Linux CUDA context has: a device pointer already names its
// device, so this one entry point expresses a cross-device transfer as well as a local one. When
// the driver grants peer access and startup validates it, the copy is a direct device-to-device
// PCIe transfer. On this V100 topology the translated IOMMU domain prevents direct P2P; CUDA's
// UVA D2D implementation then selects its driver-managed staging path. Startup validates the
// exact UVA D2D route in either case. Callers need no peer-access branch;
// enable_peer_access() only reports whether direct peer access was qualified.
// Qualified aligned SM70 reductions use a two-epoch packet push: direct TP2
// admits 640 KiB (64 hidden rows), while TP4 and mapped-host TP2 retain 80 KiB.
// Each rank keeps its own contribution in registers and publishes a remote copy into a
// startup-owned mailbox; it never reads the peer's input allocation. After every lane arrives,
// it adds in FP32 and rounds once to BF16. Empty slots use a reserved NaN payload, preserving
// all finite input bits. Two epochs prevent reuse until the peer consumes the older packet.
// Source/output lifetime is local-stream ordered; mailbox lifetime is part of PeerEvents.
// WSL2 SM70 TP2 without P2P instead owns 320 KiB of mapped pinned host mailboxes
// and device-local epoch counters. Aligned reductions through 80 KiB use the same
// packet formula with system fences after publishing and clearing. The CPU never
// touches live mailboxes; there are no host callbacks, inter-device atomics or
// peer-device pointer accesses. Device-mailbox projection fusions stay disabled.
// Host packet polling is bounded; losing a participating device fails the CUDA
// execution instead of leaving an indefinitely resident poller. Wider/scalar
// reductions and all gathers retain the verified driver-staged transport.
// Other small qualified SM70 payloads use direct peer reads into rank-local staging. Both
// reads retire before either original operand is overwritten.
//
// The equivalent cudaMemcpyPeerAsync entry point is deliberately NOT used: it is rejected inside a
// stream capture region (cudaErrorStreamCaptureUnsupported), which would make the whole
// tensor-parallel decode program uncapturable. See src/ops/common/allreduce.cu's pull_peer().
//
// The pairwise choreography below describes TP2. TP4 waits for every other rank's
// inputs_ready before reading and every other rank's pull_done before overwriting
// a source. Both routes retain the operands until all reads retire; TP4 sums in
// global rank order so replicated residuals remain bit-identical across ranks.
// ORDERING for the DMA/direct-read routes. Every transfer is a PULL: rank r reads the peer's source into storage that rank r
// alone owns, on rank r's own stream. Nothing a rank owns is ever written by the peer's stream, so
// the classic push hazard -- the peer overwriting a staging buffer that this rank has not finished
// reading yet -- cannot exist. Each rank issues two event records and two event waits:
//
//   stream(r):  ...producer of the rank-r inputs...
//               record(inputs_ready[r])
//               wait(inputs_ready[1-r])              // peer's source is complete
//               peer source -> rank-r storage
//               record(pull_done[r])
//               wait(pull_done[1-r])                 // peer has finished reading MY source
//               ...local combine, for allreduce_sum...
//
// The two waits carry the whole correctness argument:
//
//   * wait(inputs_ready[1-r]) makes the inbound copy read a completed peer source, both within a
//     call and across calls -- in call k+1 that event is recorded after call k's local work on the
//     peer's stream, so the next pull cannot start before the previous call finished with the
//     source it reads.
//   * wait(pull_done[1-r]) is the write-after-read barrier for THIS rank's own source: it
//     guarantees the peer has finished reading rank r's buffer before rank r (or the caller, on
//     return) overwrites it. For allreduce_sum that protects the in-place combine; for both ops it
//     is what makes back-to-back calls that reuse the same buffers safe.
//
// POST-CONDITION. On return each rank's stream is ordered after the peer's read of that rank's
// inputs. Therefore an unbounded sequence of calls sharing the same buffers, staging, and
// PeerEvents needs NO host synchronization between calls -- which is exactly what a captured
// CUDA graph replays, and what a decode token's collectives do. For the 27B model that is
// 128 all-reduces (64 layers x 2 row-parallel projections: the mixer output and the MLP down
// projection) plus one allgather_rows per logit column, so 129 collectives at batch 1.
//
// Everything in a call is stream-ordered CUDA work (memcpy, event record, event wait, kernel
// launch): there is no host round trip or allocation in the hot path. The WSL
// packet route polls mapped RAM only from GPU kernels. Both routes are capturable.
// Measured: with both device streams
// enrolled in ONE capture -- the peer's stream joined to the origin's by an event fork -- the record/wait pairs
// above become graph EDGES rather than nodes, and the whole 128-collective decode program
// instantiates as a single cross-device cudaGraphExec. Event objects are caller-owned and created
// once, because cudaEventCreate is not capturable.
//
// CALLER OBLIGATION -- the legacy default stream. These ops read their inputs on
// DeviceContext::stream, which core creates with cudaStreamNonBlocking and which therefore does
// NOT implicitly synchronize with a device's legacy default stream. Work issued with the plain
// cudaMemcpy/cudaMemset/<<<...>>> forms (no stream argument) lands on that default stream and is
// unordered against these collectives. A caller that stages inputs that way must retire them
// first (a device or stream synchronize, or an event); producing the inputs on
// DeviceContext::stream needs no extra step. This trap is invisible in single-device code, where
// the default stream orders everything.
//
// After a call returns the work is enqueued but not complete; the caller synchronizes each
// device's own stream, or lets subsequent stream-ordered work depend on it, as usual.

#include "core/device.h" // DeviceContext, ExecutionContext
#include "core/arena.h"
#include "core/tensor.h"

#include <cuda_runtime.h> // cudaEvent_t
#include <memory>

#include <array>
#include <cstddef>
#include <span>

namespace ninfer::ops {

// The RMSNorm consumer of a replicated BF16 collective output. Every span has
// one entry per rank; weights are BF16 [D] and outputs are disjoint [D,T]. Outputs
// are BF16, or (published SM70 TP4 D5120, T1..8 only) the exact FP16 image of
// that BF16 result. The latter preserves the BF16 rounding before the FP16 cast.
struct ReplicatedRmsNorm {
    std::span<const Tensor> weight;
    std::span<const Tensor> output;
    float eps;
    bool unit_offset;
};

// Qualifies the actual cross-device copy route at startup. On Linux, first resolve both CUDA
// devices' PCI bus IDs to /sys/bus/pci/devices/<BDF>/iommu_group/type. DMA and DMA-FQ prevent
// PCIe P2P. An active direct NVLink mesh is checked via NVML's real remote PCI endpoints and
// does not traverse that host IOMMU; the restriction is bypassed only for this verified mesh.
// Otherwise leave peer access disabled and validate CUDA's UVA D2D fallback instead.
// When all directed pairs advertise peer access, enable them and check distinct
// distinct 16 KiB patterns with the collectives' UVA D2D API on their destination compute streams.
// Returns true only when direct peer access is enabled and both copies are exact. If direct P2P is
// unavailable or fails qualification, it is disabled and the same UVA D2D route is qualified with
// driver-managed staging; the function returns false after emitting one diagnostic. A failed UVA
// check or CUDA API error throws; inference must not continue.
//
// Call once during setup, before graph capture or concurrent inference. This function allocates
// temporary probe buffers, synchronizes each compute stream, and releases its buffers while
// restoring the caller's current device. Peer-access changes are context-level operations;
// none of this setup is graph-capturable or belongs in a hot path. A TP1 context returns false;
// TP4 qualifies every directed device pair and requires a complete direct-peer mesh.
bool enable_peer_access(const ExecutionContext& ec);

// The reusable cross-device transport: two ordering events per device, created with timing
// disabled, plus fixed SM70 TP2/TP4 packet mailboxes/epochs when direct P2P was qualified. All
// resources are allocated explicitly at startup and owned by this stream domain.
//
//   inputs_ready(r) - recorded on rank r's stream once the inputs rank r contributes are complete;
//                     the peer waits on it before reading them.
//   pull_done(r)    - recorded on rank r's stream once rank r's inbound copy has finished reading
//                     the peer's source; the peer waits on it before overwriting that source.
//
// Re-recording an event a previous call already recorded is well defined: cudaStreamWaitEvent
// captures the event's state at the time the wait is issued, and the collectives always issue a
// call's waits before the next call's records, so one instance serves an unbounded number of
// sequential calls. Instances are not thread-safe: one instance belongs to one stream pair. A
// moved-from instance holds no events and must not be passed to a collective (the ops reject it).
class PeerEvents {
public:
    enum class HostPackets { Auto, Disabled, Required };
    // direct_peer_access is the result of enable_peer_access(ec), not advertised capability.
    PeerEvents(const ExecutionContext& ec, bool direct_peer_access,
               HostPackets host_packets = HostPackets::Auto);
    ~PeerEvents();

    PeerEvents(const PeerEvents&)            = delete;
    PeerEvents& operator=(const PeerEvents&) = delete;
    PeerEvents(PeerEvents&& other) noexcept;
    PeerEvents& operator=(PeerEvents&& other) noexcept;

    [[nodiscard]] cudaEvent_t inputs_ready(int rank) const noexcept {
        return inputs_ready_[static_cast<std::size_t>(rank)];
    }

    [[nodiscard]] cudaEvent_t pull_done(int rank) const noexcept {
        return pull_done_[static_cast<std::size_t>(rank)];
    }

    [[nodiscard]] bool direct_peer_access() const noexcept { return direct_peer_access_; }
    [[nodiscard]] int ranks() const noexcept { return ranks_; }

    // WSL TP2 has no P2P. These are mapped aliases of startup-owned pinned RAM,
    // independent of the direct-device mailboxes exposed to projection fusions.
    [[nodiscard]] bool host_packets() const noexcept { return host_packets_[0] != nullptr; }
    [[nodiscard]] void* host_packet_storage(int device_rank, int owner_rank) const noexcept {
        return host_packet_aliases_[device_rank][owner_rank];
    }
    [[nodiscard]] unsigned* host_packet_epochs(int rank) const noexcept {
        return static_cast<unsigned*>(host_packet_epochs_[rank].p);
    }

    // Fixed startup-owned SM70 transport storage. TP4 has a channel per source
    // rank in each epoch; TP2 retains its single inbound channel.
    [[nodiscard]] void* packet_storage(int rank) const noexcept {
        return packets_[static_cast<std::size_t>(rank)].p;
    }
    static constexpr std::size_t kPacketMaxBytes = 81920;
    static constexpr std::size_t kPacketBlocks = kPacketMaxBytes / (16 * 64);
    static constexpr std::size_t kTp2PacketMaxBytes = 64 * 5120 * sizeof(std::uint16_t);
    // Capacity of one direct-device channel, fixed by the startup rank domain.
    // Mapped-host channels have their independent kPacketMaxBytes capacity.
    [[nodiscard]] std::size_t packet_capacity_bytes() const noexcept {
        return ranks_ == 2 ? kTp2PacketMaxBytes : kPacketMaxBytes;
    }

    // False for a moved-from instance.
    [[nodiscard]] bool live() const noexcept {
        if (ranks_ != 2 && ranks_ != 4) { return false; }
        for (int r = 0; r < ranks_; ++r) {
            if (inputs_ready_[r] == nullptr || pull_done_[r] == nullptr) { return false; }
        }
        return true;
    }

private:
    std::array<cudaEvent_t, kMaximumExecutionDevices> inputs_ready_{};
    std::array<cudaEvent_t, kMaximumExecutionDevices> pull_done_{};
    std::array<DeviceBuffer, kMaximumExecutionDevices> packets_{};
    std::array<std::unique_ptr<PinnedHostBuffer>, 2> host_packets_{};
    std::array<std::array<void*, 2>, 2> host_packet_aliases_{};
    std::array<DeviceBuffer, 2> host_packet_epochs_{};
    bool direct_peer_access_ = false;
    int ranks_ = 0;
};

/**
 * Summing all-reduce over two or four devices, in place:
 *
 *   ideal[i] = sum_r buffer_rank_r[i]   for every i,
 *
 * written back to all active buffers, which afterwards hold the identical sum.
 *
 * `buffer[r]` is a contiguous BF16 tensor resident on `ec.dev[r]`; both ranks carry the same
 * shape. `staging[r]` is BF16 scratch, also resident on `ec.dev[r]`, and
 * must not overlap `buffer[r]`; it holds either the peer contribution or the local sum, and
 * its contents after the call are unspecified. TP2 staging has the same shape as its buffer;
 * TP4 staging holds at least four full contributions. Rank r's stream is the only stream that ever
 * touches `staging[r]`.
 *
 * The oracle evaluates `ideal` in FP64 from the represented BF16 inputs; the observable BF16
 * output is that value rounded once to BF16 storage. Both DMA/local-combine and direct-peer
 * sum routes use FP32 accumulation and one round-to-nearest-even on
 * store. There is no per-call allocation; PeerEvents owns the protocol storage.
 *
 * Requires `ec.device_count` in {2,4} and live events for the same rank count. Consecutive calls need no
 * host synchronization between them.
 */
// Both spans have ec.device_count entries. TP4 staging is contiguous BF16 with at least
// 4 * buffer[r].numel() elements: the DMA route retains all contributions until
// every source read has retired, then reduces once in rank order 0,1,2,3.
// The independent oracle is the FP64 sum of all represented BF16 operands;
// TP4 accumulates in FP32 and rounds only the final sum to BF16. All ranks use
// the same association, including the direct-P2P route.
void allreduce_sum(std::span<const Tensor> buffer, std::span<const Tensor> staging,
                   const ExecutionContext& ec, const PeerEvents& events);

// In-place sum followed by RMSNorm from the represented BF16 sum. The sum
// remains an observable rounding boundary, and the independent norm oracle is
// its complete FP64 formula. Norm weights/outputs have one entry per rank;
// outputs are disjoint from the sum buffers. Qualified SM70 TP4 D5120/T1..8
// fuses transport and normalization; other shapes compose the same Ops.
void allreduce_sum_rmsnorm(std::span<const Tensor> buffer, std::span<const Tensor> staging,
                          const ReplicatedRmsNorm& norm,
                          const ExecutionContext& ec, const PeerEvents& events);

// TP2 out-of-place sum, with the same FP64 oracle and one BF16 output rounding.
// Both input/output spans contain two equally shaped contiguous BF16 tensors on their ranks.
// Inputs and outputs are disjoint; inputs remain unchanged. No per-call workspace or allocation.
// On return both streams are ordered after all source reads, including the DMA fallback,
// so callers may immediately reuse input storage without host synchronization.
void allreduce_sum_to(std::span<const Tensor> input, std::span<const Tensor> output,
                      const ExecutionContext& ec, const PeerEvents& events);

// TP2 sum followed by RMSNorm of the represented BF16 sum. The sum is an
// observable output and therefore a rounding boundary. Its exact oracle is the
// FP64 sum rounded once to BF16; the norm oracle evaluates the complete RMSNorm
// formula in FP64 from that represented sum and the BF16 weights. Input planes
// remain unchanged and retire on both streams before source reuse is allowed.
// No per-call allocation. Qualified direct SM70 TP2 D5120/T1..64 uses one fused kernel;
// other profiles compose the same two Ops.
void allreduce_sum_rmsnorm_to(std::span<const Tensor> input,
                              std::span<const Tensor> output,
                              const ReplicatedRmsNorm& norm,
                              const ExecutionContext& ec, const PeerEvents& events);

/**
 * Row gather, exact (no arithmetic). The gathered axis is `ne[1]`, so each rank
 * contributes one contiguous block of a `[C, R]` tensor (`ne[0] == C` is the row length,
 * `ne[1] == R` the row count, `ne[2] == ne[3] == 1`):
 *
 *   destination[r][c, row] = part[0][c, row]                    for row <  part[0].ne[1]
 *   destination[r][c, row] = part[1][c, row - part[0].ne[1]]    otherwise
 *
 * This is the TP2 formula; TP4 concatenates four blocks in global rank order.
 * Each device ends up holding the identical full image. `part[r]` is the row range
 * owned by `ec.dev[r]`: rank 0 owns the leading rows and rank 1 the trailing rows, and the two
 * counts must sum to the destination row count. `destination[r]` and `part[r]` are contiguous,
 * share one dtype, agree on `ne[0]`, live on `ec.dev[r]`, and must not overlap. A caller whose
 * split axis is not `ne[1]` (for example `[V, B]` logits split by vocabulary with B > 1) arranges
 * the layout so that it is; this Op relocates contiguous storage and performs no transpose.
 *
 * `destination[r]` is written only by rank r's stream: rank r copies its own block locally and
 * pulls the peer's block. The transformation is a pure relocation of storage, so it is verified by
 * exact byte comparison. The Op holds no persistent state and allocates nothing.
 *
 * Requires `ec.device_count` in {2,4} and live events for the same rank count. Consecutive calls need no
 * host synchronization between them.
 */
void allgather_rows(std::span<const Tensor> destination, std::span<const Tensor> part,
                    const ExecutionContext& ec, const PeerEvents& events);

/**
 * Exact one-way vocabulary gather used by DFlash's rank-0 selector. `destination` is the full
 * `[C,T]` image on rank 0; `part[0]` and `part[1]` are the leading/trailing `[C_r,T]` shards on
 * their owning ranks. The first/contiguous dimension is concatenated for every column. Rank 1 is
 * deliberately not written because it never consumes DFlash proposal logits.
 * The source lifetime edge still orders rank 1 after rank 0 has finished importing its shard,
 * so the same work buffers may be reused by the next graph round.
 */
void gather_columns_rank0(const Tensor& destination, std::span<const Tensor> part,
                          const ExecutionContext& ec, const PeerEvents& events);

// Merge one FP32 score and local I32 index [1,T] per rank into replicated I32 [T].
// The largest score wins; ties select the lowest global index (rank * shard_rows
// + local index). Inputs must be non-NaN; +/-infinity and signed zero are valid.
// Gather scratch is FP32/I32 [R,T] on rank zero. No input/output may overlap.
// Sources and destinations retire on all participating streams before reuse.
// Direct TP2/TP4 peer access fuses merge and publication; otherwise use UVA DMA.
// Optional rank-zero I32 id_map[device_count * shard_rows] maps the winning global
// index after tie-breaking. It is immutable until all participating streams retire.
void allreduce_argmax(std::span<const Tensor> values, std::span<const Tensor> indices,
                       std::span<const Tensor> output, Tensor& gathered_values,
                       Tensor& gathered_indices, std::int32_t shard_rows,
                       const ExecutionContext& ec, const PeerEvents& events,
                       const std::int32_t* id_map = nullptr);

// Exact relocation from rank 0 to each other active rank. Destinations are indexed by
// rank minus one and contain ec.device_count - 1 entries. All tensors have the same contiguous
// dtype and shape. SM70 TP4 direct-peer I32 payloads through 4 KiB use one origin
// publication kernel; other payloads use qualified UVA D2D transport. Every stream
// orders source/output reuse after publication, including under CUDA Graph capture.
void broadcast_rank0(const Tensor& source, std::span<const Tensor> destinations,
                     const ExecutionContext& ec, const PeerEvents& events);

} // namespace ninfer::ops
