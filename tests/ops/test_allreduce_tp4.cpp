// TP4 collectives against independent mathematical/byte-relocation oracles.
// Also replay back-to-back in-place reductions from one four-device CUDA Graph.
#include "ninfer/ops/allreduce.h"
#include "core/decode_graph.h"

#include <array>
#include <bit>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <memory>
#include <vector>

using namespace ninfer;

namespace {
struct Allocation {
    int device;
    void* pointer = nullptr;
    Allocation(int id, std::size_t bytes) : device(id) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMalloc(&pointer, bytes));
    }
    ~Allocation() { cudaSetDevice(device); cudaFree(pointer); }
};

std::uint16_t bf16(double value) {
    const auto word = std::bit_cast<std::uint32_t>(static_cast<float>(value));
    return static_cast<std::uint16_t>((word + 0x7fffU + ((word >> 16) & 1U)) >> 16);
}
double represented(std::uint16_t word) {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16);
}
void synchronize(const ExecutionContext& ec) {
    for (int r = 0; r < ec.device_count; ++r) {
        CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
        ec.dev[r]->synchronize();
    }
}

template<class Body>
DecodeGraphDefinition capture(const ExecutionContext& ec, Body body) {
    std::array<std::unique_ptr<DecodeGraphPeerBridge>, 3> bridges;
    std::array<DecodeGraphPeerCapture, 3> peers;
    for (int r = 1; r < 4; ++r) {
        bridges[r - 1] = std::make_unique<DecodeGraphPeerBridge>(ec.dev[0]->device,
                                                                 ec.dev[r]->device);
        peers[r - 1] = {bridges[r - 1].get(), ec.dev[r]->stream};
    }
    CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
    DecodeGraphDefinition definition;
    definition.capture(ec.dev[0]->stream, body, peers);
    // Exercise the production enrollment and exception-cleanup implementation.
    return definition;
}

int reduce_case(const ExecutionContext& ec, const ops::PeerEvents& events, int count) {
    constexpr int guard = 16;
    constexpr std::uint16_t sentinel = 0x5a5a;
    std::array<std::unique_ptr<Allocation>, 4> allocations, scratch;
    std::array<Tensor, 4> buffers, staging;
    std::array<std::vector<std::uint16_t>, 4> inputs;
    std::vector<std::uint16_t> expected(count), twice(count);
    for (int r = 0; r < 4; ++r) {
        allocations[r] = std::make_unique<Allocation>(ec.dev[r]->device, (count + 2 * guard) * 2);
        scratch[r] = std::make_unique<Allocation>(ec.dev[r]->device, (4 * count + 2 * guard) * 2);
        buffers[r] = Tensor(static_cast<std::uint16_t*>(allocations[r]->pointer) + guard,
                            DType::BF16, {count});
        staging[r] = Tensor(static_cast<std::uint16_t*>(scratch[r]->pointer) + guard,
                            DType::BF16, {count, 4});
        inputs[r].assign(count + 2 * guard, sentinel);
        for (int i = 0; i < count; ++i) {
            inputs[r][guard + i] = bf16((double((i * (r + 3) + r * 17) % 1021) - 510) / 64);
        }
    }
    for (int i = 0; i < count; ++i) {
        double ideal = 0;
        for (int r = 0; r < 4; ++r) { ideal += represented(inputs[r][guard + i]); }
        expected[i] = bf16(ideal);
        // The first Op's BF16 output is a public storage boundary.
        twice[i] = bf16(4 * represented(expected[i]));
    }
    const auto reset = [&] {
        for (int r = 0; r < 4; ++r) {
            CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
            CUDA_CHECK(cudaMemcpyAsync(allocations[r]->pointer, inputs[r].data(), inputs[r].size() * 2,
                                       cudaMemcpyHostToDevice, ec.dev[r]->stream));
            std::vector<std::uint16_t> empty(4 * count + 2 * guard, sentinel);
            CUDA_CHECK(cudaMemcpyAsync(scratch[r]->pointer, empty.data(), empty.size() * 2,
                                       cudaMemcpyHostToDevice, ec.dev[r]->stream));
            ec.dev[r]->synchronize();
        }
    };
    const auto verify = [&](const std::vector<std::uint16_t>& oracle) {
        int failures = 0;
        synchronize(ec);
        for (int r = 0; r < 4; ++r) {
            CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
            std::vector<std::uint16_t> actual(count + 2 * guard);
            CUDA_CHECK(cudaMemcpy(actual.data(), allocations[r]->pointer, actual.size() * 2,
                                   cudaMemcpyDeviceToHost));
            for (int i = 0; i < count; ++i) {
                if (actual[guard + i] != oracle[i]) { ++failures; break; }
            }
            for (int i = 0; i < guard; ++i) {
                if (actual[i] != sentinel || actual[count + guard + i] != sentinel) { ++failures; }
            }
            std::array<std::uint16_t, guard> edge;
            for (int offset : {0, 4 * count + guard}) {
                CUDA_CHECK(cudaMemcpy(edge.data(), static_cast<std::uint16_t*>(scratch[r]->pointer) + offset,
                                       sizeof(edge), cudaMemcpyDeviceToHost));
                for (auto word : edge) { if (word != sentinel) { ++failures; } }
            }
        }
        return failures;
    };
    reset();
    ops::allreduce_sum(buffers, staging, ec, events);
    int failures = verify(expected);
    auto graph = capture(ec, [&] {
        ops::allreduce_sum(buffers, staging, ec, events);
        ops::allreduce_sum(buffers, staging, ec, events);
    });
    DecodeGraphExecutable executable;
    executable.instantiate(graph);
    for (int replay = 0; replay < 3; ++replay) {
        reset();
        CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
        executable.launch(ec.dev[0]->stream);
        failures += verify(twice);
    }
    // Decode alternates one-token, MTP verification and partially filled blocks.
    // Reuse one transport across a long captured chain with changing message sizes.
    // The oracle applies each public BF16 output boundary independently.
    if (count == 5120 * 64) {
        const std::array<int, 6> sizes{5120, 5120 * 5, 5128, 5120 * 20, 5120 * 40, count};
        std::array<std::vector<std::uint16_t>, 4> logical = inputs;
        for (int round = 0; round < 8; ++round) {
            for (int size : sizes) {
                for (int i = 0; i < size; ++i) {
                    double sum = 0;
                    for (int r = 0; r < 4; ++r) { sum += represented(logical[r][guard + i]); }
                    for (int r = 0; r < 4; ++r) { logical[r][guard + i] = bf16(sum); }
                }
            }
        }
        std::vector<std::uint16_t> mixed_oracle(logical[0].begin() + guard,
                                                logical[0].begin() + guard + count);
        auto mixed = capture(ec, [&] {
            for (int round = 0; round < 8; ++round) {
                for (int size : sizes) {
                    auto views = buffers;
                    for (int r = 0; r < 4; ++r) {
                        views[r] = Tensor(buffers[r].data, DType::BF16, {size});
                    }
                    ops::allreduce_sum(views, staging, ec, events);
                }
            }
        });
        DecodeGraphExecutable mixed_executable;
        mixed_executable.instantiate(mixed);
        for (int replay = 0; replay < 3; ++replay) {
            reset();
            CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
            mixed_executable.launch(ec.dev[0]->stream);
            failures += verify(mixed_oracle);
        }
    }
    std::cout << "TP4 reduce count=" << count << " direct=" << events.direct_peer_access()
              << " failures=" << failures << '\n';
    return failures;
}

int norm_case(const ExecutionContext& ec, const ops::PeerEvents& events, bool unit_offset) {
    constexpr int width = 5120, rows = 64, count = width * rows;
    std::array<std::unique_ptr<Allocation>, 4> data, scratch, weights, norms;
    std::array<Tensor, 4> buffers, staging, weight, output;
    std::array<std::vector<std::uint16_t>, 4> input;
    std::vector<std::uint16_t> host_weight(width);
    for (int i = 0; i < width; ++i) host_weight[i] = bf16((i % 23 - 11) / 16.);
    for (int r = 0; r < 4; ++r) {
        data[r] = std::make_unique<Allocation>(ec.dev[r]->device, count * 2);
        scratch[r] = std::make_unique<Allocation>(ec.dev[r]->device, count * 8);
        weights[r] = std::make_unique<Allocation>(ec.dev[r]->device, width * 2);
        norms[r] = std::make_unique<Allocation>(ec.dev[r]->device, count * 2);
        buffers[r] = Tensor(data[r]->pointer, DType::BF16, {width, rows});
        staging[r] = Tensor(scratch[r]->pointer, DType::BF16, {width, rows, 4});
        weight[r] = Tensor(weights[r]->pointer, DType::BF16, {width});
        output[r] = Tensor(norms[r]->pointer, DType::BF16, {width, rows});
        input[r].resize(count);
        for (int i = 0; i < count; ++i) input[r][i] = bf16(((i * (r+3) + 17*r) % 1021 - 510) / 64.);
        CUDA_CHECK(cudaMemcpyAsync(weight[r].data, host_weight.data(), width * 2,
                                   cudaMemcpyHostToDevice, ec.dev[r]->stream));
    }
    const auto reset = [&] {
        for (int r = 0; r < 4; ++r) {
            CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
            CUDA_CHECK(cudaMemcpyAsync(buffers[r].data, input[r].data(), count * 2,
                                       cudaMemcpyHostToDevice, ec.dev[r]->stream));
        }
        synchronize(ec);
    };
    const std::array<int, 10> widths{1, 5, 8, 9, 20, 40, 64, 3, 7, 64};
    const auto run = [&] {
        for (std::size_t call = 0; call < widths.size(); ++call) {
            auto b = buffers, n = output;
            for (int r = 0; r < 4; ++r) {
                b[r] = b[r].slice(1, 0, widths[call]);
                n[r] = n[r].slice(1, 0, widths[call]);
            }
            if (call == 2) ops::allreduce_sum(b, staging, ec, events);
            else ops::allreduce_sum_rmsnorm(b, staging,
                ops::ReplicatedRmsNorm{weight, n, 1.e-6f, unit_offset}, ec, events);
        }
    };
    auto logical = input;
    for (int tokens : widths) {
        for (int i = 0; i < width * tokens; ++i) {
            double sum = 0.;
            for (int r = 0; r < 4; ++r) sum += represented(logical[r][i]);
            for (int r = 0; r < 4; ++r) logical[r][i] = bf16(sum);
        }
    }
    const auto& oracle = logical[0];
    std::vector<double> norm_oracle(count);
    for (int row = 0; row < rows; ++row) {
        double sum = 0.;
        for (int i = 0; i < width; ++i) {
            const double v = represented(oracle[row * width + i]);
            sum += v*v;
        }
        const double inv = 1. / std::sqrt(sum / width + 1.e-6);
        for (int i = 0; i < width; ++i) norm_oracle[row*width+i] =
            represented(oracle[row*width+i]) * inv *
            (represented(host_weight[i]) + (unit_offset ? 1. : 0.));
    }
    const auto verify = [&] {
        synchronize(ec);
        int failures = 0;
        for (int r = 0; r < 4; ++r) {
            CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
            std::vector<std::uint16_t> sum(count), norm(count);
            CUDA_CHECK(cudaMemcpy(sum.data(), buffers[r].data, count*2, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(norm.data(), output[r].data, count*2, cudaMemcpyDeviceToHost));
            if (sum != oracle) ++failures;
            for (int i = 0; i < count; ++i) {
                const double ideal = norm_oracle[i];
                if (std::abs(represented(norm[i]) - ideal) > 0.004 * std::abs(ideal) + 2.e-5) {
                    ++failures; break;
                }
            }
        }
        return failures;
    };
    reset(); run();
    int failures = verify();
    auto graph = capture(ec, run);
    DecodeGraphExecutable executable;
    executable.instantiate(graph);
    for (int replay = 0; replay < 3; ++replay) {
        reset();
        CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
        executable.launch(ec.dev[0]->stream);
        failures += verify();
    }
    std::cout << "TP4 mixed-width sum/RMSNorm direct=" << events.direct_peer_access()
              << " unit_offset=" << unit_offset << " failures=" << failures << '\n';
    return failures;
}

int gather_cases(const ExecutionContext& ec, const ops::PeerEvents& events) {
    constexpr int columns = 5;
    constexpr int total_rows = 30;
    std::array<std::unique_ptr<Allocation>, 4> sources, destinations;
    std::array<Tensor, 4> row_parts, row_destinations, column_parts;
    std::array<std::vector<std::int32_t>, 4> inputs;
    std::vector<std::int32_t> row_oracle, column_oracle(columns * total_rows);
    int prefix = 0;
    for (int r = 0; r < 4; ++r) {
        const int rows = 3 * (r + 1);
        inputs[r].resize(rows * columns);
        for (int row = 0; row < rows; ++row) {
            for (int c = 0; c < columns; ++c) {
                inputs[r][row * columns + c] = r * 100000 + row * 100 + c;
            }
        }
        row_oracle.insert(row_oracle.end(), inputs[r].begin(), inputs[r].end());
        // Column gather interprets exactly the same represented words as [rows, columns].
        for (int c = 0; c < columns; ++c) {
            for (int k = 0; k < rows; ++k) {
                column_oracle[c * total_rows + prefix + k] = inputs[r][c * rows + k];
            }
        }
        prefix += rows;
        sources[r] = std::make_unique<Allocation>(ec.dev[r]->device, inputs[r].size() * 4);
        destinations[r] = std::make_unique<Allocation>(ec.dev[r]->device, total_rows * columns * 4);
        row_parts[r] = Tensor(sources[r]->pointer, DType::I32, {columns, rows});
        column_parts[r] = Tensor(sources[r]->pointer, DType::I32, {rows, columns});
        row_destinations[r] = Tensor(destinations[r]->pointer, DType::I32, {columns, total_rows});
        CUDA_CHECK(cudaMemcpyAsync(sources[r]->pointer, inputs[r].data(), inputs[r].size() * 4,
                                   cudaMemcpyHostToDevice, ec.dev[r]->stream));
    }
    synchronize(ec);
    Tensor column_destination(destinations[0]->pointer, DType::I32, {total_rows, columns});
    const auto verify = [&](bool column) {
        synchronize(ec);
        int failures = 0;
        for (int r = 0; r < (column ? 1 : 4); ++r) {
            CUDA_CHECK(cudaSetDevice(ec.dev[r]->device));
            std::vector<std::int32_t> actual(total_rows * columns);
            CUDA_CHECK(cudaMemcpy(actual.data(), destinations[r]->pointer, actual.size() * 4,
                                   cudaMemcpyDeviceToHost));
            failures += actual != (column ? column_oracle : row_oracle);
        }
        return failures;
    };
    int failures = 0;
    for (bool column : {false, true}) {
        const auto body = [&] {
            for (int repetition = 0; repetition < 2; ++repetition) {
                if (column) { ops::gather_columns_rank0(column_destination, column_parts, ec, events); }
                else { ops::allgather_rows(row_destinations, row_parts, ec, events); }
            }
        };
        body();
        failures += verify(column);
        auto graph = capture(ec, body);
        DecodeGraphExecutable executable;
        executable.instantiate(graph);
        for (int replay = 0; replay < 3; ++replay) {
            CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
            executable.launch(ec.dev[0]->stream);
            failures += verify(column);
        }
    }
    std::cout << "TP4 uneven gathers failures=" << failures << '\n';
    return failures;
}

int broadcast_case(const ExecutionContext& ec, const ops::PeerEvents& events, int count) {
    constexpr int guard = 16;
    constexpr std::uint32_t sentinel = 0x5a5aa5a5;
    std::array<std::unique_ptr<Allocation>, 4> storage, history;
    std::array<Tensor, 4> views;
    const auto data = [](const auto& allocation) {
        return static_cast<std::uint32_t*>(allocation->pointer) + guard;
    };
    for (int rank = 0; rank < 4; ++rank) {
        storage[rank] = std::make_unique<Allocation>(ec.dev[rank]->device, (count + 2 * guard) * 4);
        history[rank] = std::make_unique<Allocation>(ec.dev[rank]->device, (2 * count + 2 * guard) * 4);
        views[rank] = Tensor(data(storage[rank]), DType::I32, {count});
        std::vector<std::uint32_t> initial(count + 2 * guard, sentinel);
        CUDA_CHECK(cudaMemcpy(storage[rank]->pointer, initial.data(), initial.size() * 4,
                               cudaMemcpyHostToDevice));
        initial.assign(2 * count + 2 * guard, sentinel);
        CUDA_CHECK(cudaMemcpy(history[rank]->pointer, initial.data(), initial.size() * 4,
                               cudaMemcpyHostToDevice));
    }
    const auto body = [&] {
        for (int round = 0; round < 2; ++round) {
            ops::broadcast_rank0(views[0], std::span(views).subspan(1), ec, events);
            // Consume each result on its owner's stream before the next call writes it.
            // Rank zero also overwrites the source immediately after saving the result.
            for (int rank = 0; rank < 4; ++rank) {
                CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
                CUDA_CHECK(cudaMemcpyAsync(data(history[rank]) + round * count, views[rank].data,
                                           count * 4, cudaMemcpyDeviceToDevice, ec.dev[rank]->stream));
                CUDA_CHECK(cudaMemsetAsync(views[rank].data, rank == 0 ? 0x23 : 0x44,
                                           count * 4, ec.dev[rank]->stream));
            }
        }
    };
    auto graph = capture(ec, body);
    DecodeGraphExecutable executable;
    executable.instantiate(graph);
    int failures = 0;
    for (int replay = 0; replay < 3; ++replay) {
        std::vector<std::uint32_t> expected(2 * count, 0x23232323);
        for (int i = 0; i < count; ++i) {
            expected[i] = 0x80000000U ^ (std::uint32_t(i) * 65537U + replay * 1234567U);
        }
        CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
        CUDA_CHECK(cudaMemcpyAsync(views[0].data, expected.data(), count * 4,
                                   cudaMemcpyHostToDevice, ec.dev[0]->stream));
        if (replay == 0) { body(); }
        else {
            // Replay must also order the origin after independently queued peer work.
            std::array<std::unique_ptr<DecodeGraphPeerBridge>, 3> gates;
            for (int rank = 1; rank < 4; ++rank) {
                gates[rank - 1] = std::make_unique<DecodeGraphPeerBridge>(ec.dev[0]->device,
                                                                         ec.dev[rank]->device);
                CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
                CUDA_CHECK(cudaMemsetAsync(views[rank].data, 0, count * 4, ec.dev[rank]->stream));
                gates[rank - 1]->gate_launch(ec.dev[rank]->stream, ec.dev[0]->stream);
            }
            CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
            executable.launch(ec.dev[0]->stream);
            synchronize(ec);
        }
        synchronize(ec);
        for (int rank = 0; rank < 4; ++rank) {
            CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
            std::vector<std::uint32_t> actual(2 * count);
            CUDA_CHECK(cudaMemcpy(actual.data(), data(history[rank]), actual.size() * 4,
                                   cudaMemcpyDeviceToHost));
            failures += actual != expected;
            const auto verify_guards = [&](const auto& allocation, int words) {
                std::array<std::uint32_t, guard> edge;
                for (int offset : {0, words + guard}) {
                    CUDA_CHECK(cudaMemcpy(edge.data(), static_cast<std::uint32_t*>(allocation->pointer) + offset,
                                           sizeof(edge), cudaMemcpyDeviceToHost));
                    for (auto word : edge) failures += word != sentinel;
                }
            };
            verify_guards(storage[rank], count);
            verify_guards(history[rank], 2 * count);
        }
    }
    std::cout << "TP4 exact broadcast T=" << count << " direct=" << events.direct_peer_access()
              << " failures=" << failures << '\n';
    return failures;
}

int argmax_case(const ExecutionContext& ec, const ops::PeerEvents& events, int columns, bool mapped) {
    constexpr int rounds = 16, shard = 62080, guard = 16;
    constexpr std::uint32_t sentinel = 0x5a5aa5a5;
    const std::size_t bytes = columns * sizeof(std::uint32_t);
    const auto allocate = [&](int rank, int words) {
        auto allocation = std::make_unique<Allocation>(ec.dev[rank]->device, (words + 2 * guard) * 4);
        std::vector<std::uint32_t> initial(words + 2 * guard, sentinel);
        CUDA_CHECK(cudaMemcpy(allocation->pointer, initial.data(), initial.size() * 4,
                               cudaMemcpyHostToDevice));
        return allocation;
    };
    const auto data = [](const auto& allocation) {
        return static_cast<std::uint32_t*>(allocation->pointer) + guard;
    };
    std::array<std::unique_ptr<Allocation>, 4> score, index, output, history, seed;
    std::array<Tensor, 4> scores, indices, outputs;
    std::array<std::vector<std::uint32_t>, 4> host_seed;
    for (int rank = 0; rank < 4; ++rank) {
        score[rank] = allocate(rank, columns);
        index[rank] = allocate(rank, columns);
        output[rank] = allocate(rank, columns);
        history[rank] = allocate(rank, rounds * columns);
        seed[rank] = allocate(rank, 4 * columns);
        scores[rank] = Tensor(data(score[rank]), DType::FP32, {1, columns});
        indices[rank] = Tensor(data(index[rank]), DType::I32, {1, columns});
        outputs[rank] = Tensor(data(output[rank]), DType::I32, {columns});
        host_seed[rank].resize(4 * columns);
    }
    auto sv = allocate(0, 4 * columns), si = allocate(0, 4 * columns);
    auto map_storage = allocate(0, 4 * shard);
    Tensor gathered_scores(data(sv), DType::FP32, {4, columns});
    Tensor gathered_indices(data(si), DType::I32, {4, columns});
    std::vector<std::int32_t> map(4 * shard);
    for (int i = 0; i < 4 * shard; ++i) { map[i] = (i * 97 + 13) % (4 * shard); }
    CUDA_CHECK(cudaMemcpy(data(map_storage), map.data(), map.size() * 4, cudaMemcpyHostToDevice));
    const auto invoke = [&] {
        for (int round = 0; round < rounds; ++round) {
            for (int rank = 0; rank < 4; ++rank) {
                CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
                CUDA_CHECK(cudaMemcpyAsync(scores[rank].data, data(seed[rank]) + (round % 2) * columns,
                    bytes, cudaMemcpyDeviceToDevice, ec.dev[rank]->stream));
                CUDA_CHECK(cudaMemcpyAsync(indices[rank].data, data(seed[rank]) + (2 + round % 2) * columns,
                    bytes, cudaMemcpyDeviceToDevice, ec.dev[rank]->stream));
            }
            ops::allreduce_argmax(scores, indices, outputs, gathered_scores, gathered_indices,
                shard, ec, events, mapped ? reinterpret_cast<const int*>(data(map_storage)) : nullptr);
            for (int rank = 0; rank < 4; ++rank) {
                CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
                CUDA_CHECK(cudaMemcpyAsync(data(history[rank]) + round * columns, outputs[rank].data,
                    bytes, cudaMemcpyDeviceToDevice, ec.dev[rank]->stream));
            }
        }
    };
    auto definition = capture(ec, invoke);
    DecodeGraphExecutable executable;
    executable.instantiate(definition);
    int failures = 0;
    const auto check_guards = [&](const auto& allocation, int words) {
        CUDA_CHECK(cudaSetDevice(allocation->device));
        for (int offset : {0, guard + words}) {
            std::array<std::uint32_t, guard> edge;
            CUDA_CHECK(cudaMemcpy(edge.data(), static_cast<std::uint32_t*>(allocation->pointer) + offset,
                                   sizeof(edge), cudaMemcpyDeviceToHost));
            for (auto word : edge) { failures += word != sentinel; }
        }
    };
    for (int probe = 0; probe < 3; ++probe) {
        for (int rank = 0; rank < 4; ++rank) {
            for (int mode = 0; mode < 2; ++mode) {
                for (int t = 0; t < columns; ++t) {
                    const int i = mode * columns + t;
                    float v = float((t + mode + probe + rank * 3) % 11 - 5);
                    if (t % 8 == 0) { v = mode ? -INFINITY : INFINITY; }
                    if (t % 8 == 1) { v = rank % 2 ? -0.0F : 0.0F; }
                    if (t % 8 >= 2 && t % 8 <= 5) {
                        v = rank == (t + mode + probe) % 4 ? INFINITY : -INFINITY;
                    }
                    host_seed[rank][i] = std::bit_cast<std::uint32_t>(v);
                    host_seed[rank][2 * columns + i] = (t * 11 + probe * 13 + mode * 17) % shard;
                }
            }
            CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
            CUDA_CHECK(cudaMemcpy(data(seed[rank]), host_seed[rank].data(), 4 * bytes,
                                   cudaMemcpyHostToDevice));
        }
        if (probe == 0) { invoke(); }
        else {
            CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
            executable.launch(ec.dev[0]->stream);
        }
        synchronize(ec);
        std::vector<std::int32_t> expected(rounds * columns);
        for (int round = 0; round < rounds; ++round) {
            for (int t = 0; t < columns; ++t) {
                const int i = (round % 2) * columns + t;
                double best = -INFINITY;
                int winner = 4 * shard;
                for (int rank = 0; rank < 4; ++rank) {
                    const double value = std::bit_cast<float>(host_seed[rank][i]);
                    const int global = rank * shard + host_seed[rank][2 * columns + i];
                    if (value > best || (value == best && global < winner)) {
                        best = value;
                        winner = global;
                    }
                }
                expected[round * columns + t] = mapped ? map[winner] : winner;
            }
        }
        for (int rank = 0; rank < 4; ++rank) {
            CUDA_CHECK(cudaSetDevice(ec.dev[rank]->device));
            std::vector<std::int32_t> actual(rounds * columns);
            CUDA_CHECK(cudaMemcpy(actual.data(), data(history[rank]), actual.size() * 4,
                                   cudaMemcpyDeviceToHost));
            failures += actual != expected;
            std::vector<std::uint32_t> actual_seed(4 * columns);
            CUDA_CHECK(cudaMemcpy(actual_seed.data(), data(seed[rank]), 4 * bytes, cudaMemcpyDeviceToHost));
            failures += actual_seed != host_seed[rank];
            check_guards(score[rank], columns);
            check_guards(index[rank], columns);
            check_guards(output[rank], columns);
            check_guards(history[rank], rounds * columns);
            check_guards(seed[rank], 4 * columns);
        }
        check_guards(sv, 4 * columns);
        check_guards(si, 4 * columns);
        check_guards(map_storage, 4 * shard);
    }
    CUDA_CHECK(cudaSetDevice(ec.dev[0]->device));
    std::vector<std::int32_t> actual_map(map.size());
    CUDA_CHECK(cudaMemcpy(actual_map.data(), data(map_storage), map.size() * 4, cudaMemcpyDeviceToHost));
    failures += actual_map != map;
    std::cout << "TP4 exact argmax T=" << columns << " mapped=" << mapped
              << " direct=" << events.direct_peer_access() << " failures=" << failures << '\n';
    return failures;
}

} // namespace

int main() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    if (count < 4) { std::cout << "SKIP: requires four CUDA devices\n"; return 77; }
    ExecutionContext ec({0, 1, 2, 3});
    const bool direct = ops::enable_peer_access(ec);
    int failures = 0;
    for (bool route : {false, true}) {
        if (route && !direct) { continue; }
        ops::PeerEvents events(ec, route);
        for (int size : {511, 5120, 5120 * 5, 5120 * 8, 5120 * 9, 5120 * 10,
                         5120 * 16, 5120 * 20, 5120 * 32, 5120 * 40, 5120 * 48,
                         5120 * 64, 5120 * 65, 5120 * 2560}) {
            failures += reduce_case(ec, events, size);
        }
        failures += norm_case(ec, events, false);
        failures += norm_case(ec, events, true);
        failures += gather_cases(ec, events);
        for (int columns : {1, 5, 8, 129, 1024, 1025, 5120}) {
            failures += broadcast_case(ec, events, columns);
        }
        for (int columns : {1, 5, 8, 33, 129}) {
            failures += argmax_case(ec, events, columns, false);
            failures += argmax_case(ec, events, columns, true);
        }
    }
    return failures ? 1 : 0;
}
