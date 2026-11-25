#include "ninfer/ops/token_bitmask.h"
#include "ops/op_tester.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using namespace ninfer;
using namespace ninfer::test;

namespace {

constexpr std::uint16_t kNegativeInfinity = 0xff80u;

std::vector<std::uint16_t> oracle(std::vector<std::uint16_t> logits, std::int32_t physical_rows,
                                  std::int32_t valid_rows, std::int32_t row_offset,
                                  const std::vector<std::int32_t>& constrained,
                                  const std::vector<std::uint32_t>& bitmask, std::int32_t words) {
    for (std::size_t t = 0; t < constrained.size(); ++t) {
        if (constrained[t] == 0) { continue; }
        for (std::int32_t v = 0; v < valid_rows; ++v) {
            const std::int32_t token = row_offset + v;
            const std::uint32_t word = bitmask[t * static_cast<std::size_t>(words) + token / 32];
            if (((word >> (token % 32)) & 1u) == 0u) {
                logits[t * static_cast<std::size_t>(physical_rows) + v] = kNegativeInfinity;
            }
        }
    }
    return logits;
}

// `mapped` places the constraint image in page-locked host memory, the production route.
int run_case(std::int32_t physical_rows, std::int32_t valid_rows, std::int32_t row_offset,
             std::int32_t words, std::int32_t columns, bool mapped) {
    std::vector<std::uint16_t> logits(static_cast<std::size_t>(physical_rows) * columns);
    for (std::size_t i = 0; i < logits.size(); ++i) {
        logits[i] = f32_to_bf16(-8.0F + static_cast<float>((i * 2654435761u) % 4096u) / 256.0F);
    }
    std::vector<std::int32_t> constrained(static_cast<std::size_t>(columns));
    std::vector<std::uint32_t> bitmask(static_cast<std::size_t>(columns) * words);
    for (std::int32_t t = 0; t < columns; ++t) {
        constrained[t] = t % 3 == 1 ? 0 : 1;
        for (std::int32_t w = 0; w < words; ++w) {
            const std::uint32_t mixed = static_cast<std::uint32_t>(w) * 1664525u +
                                        static_cast<std::uint32_t>(t + 1) * 1013904223u;
            // Mix fully licensed, fully rejected and sparse words.
            bitmask[static_cast<std::size_t>(t) * words + w] =
                mixed % 5 == 0 ? 0xffffffffu : mixed % 5 == 1 ? 0u : mixed;
        }
    }
    const auto expected =
        oracle(logits, physical_rows, valid_rows, row_offset, constrained, bitmask, words);

    GuardedDeviceBuffer device_logits(logits.size() * sizeof(std::uint16_t));
    device_logits.copy_from_host(logits.data(), logits.size() * sizeof(std::uint16_t));
    const std::size_t flag_bytes = constrained.size() * sizeof(std::int32_t);
    const std::size_t mask_bytes = bitmask.size() * sizeof(std::uint32_t);
    void* host_image = nullptr;
    DeviceBuffer device_flags;
    DeviceBuffer device_mask;
    const std::int32_t* flags = nullptr;
    const std::uint32_t* mask = nullptr;
    if (mapped) {
        cuda_check(cudaMallocHost(&host_image, flag_bytes + mask_bytes), "cudaMallocHost");
        std::memcpy(host_image, constrained.data(), flag_bytes);
        std::memcpy(static_cast<std::uint8_t*>(host_image) + flag_bytes, bitmask.data(), mask_bytes);
        flags = static_cast<const std::int32_t*>(host_image);
        mask  = reinterpret_cast<const std::uint32_t*>(static_cast<std::uint8_t*>(host_image) +
                                                      flag_bytes);
    } else {
        device_flags = to_device(constrained);
        device_mask  = to_device(bitmask);
        flags        = static_cast<const std::int32_t*>(device_flags.p);
        mask         = static_cast<const std::uint32_t*>(device_mask.p);
    }

    Tensor tensor(device_logits.data(), DType::BF16, {physical_rows, columns});
    ops::apply_token_bitmask(tensor, valid_rows, row_offset, flags, mask, words, nullptr);
    cuda_synchronize();
    if (host_image != nullptr) { cuda_check(cudaFreeHost(host_image), "cudaFreeHost"); }

    const auto actual = from_device<std::uint16_t>(device_logits.data(), logits.size());
    const std::string label = "token_bitmask rows=" + std::to_string(physical_rows) +
                              " valid=" + std::to_string(valid_rows) +
                              " offset=" + std::to_string(row_offset) +
                              " T=" + std::to_string(columns) + (mapped ? " mapped" : " device");
    int failures = verify_exact(label.c_str(), actual, expected);
    failures += device_logits.verify_guards(label.c_str());
    return failures;
}

} // namespace

int main() {
    if (cuda_unavailable()) {
        std::cout << "SKIP: no usable CUDA device\n";
        return 77;
    }

    constexpr std::int32_t kPhysical = 248320;
    constexpr std::int32_t kDomain   = 248077;
    constexpr std::int32_t kWords    = (kDomain + 31) / 32;
    int failures                     = 0;
    // Full-vocabulary verify/decode/prefill columns.
    for (const std::int32_t columns : {1, 5, 6, 48}) {
        failures += run_case(kPhysical, kDomain, 0, kWords, columns, true);
    }
    failures += run_case(kPhysical, kDomain, 0, kWords, 5, false);
    // Tensor-parallel vocabulary shards, including the short final shard.
    for (const std::int32_t ranks : {2, 4}) {
        const std::int32_t shard = kPhysical / ranks;
        for (std::int32_t rank = 0; rank < ranks; ++rank) {
            const std::int32_t valid = std::min(shard, kDomain - rank * shard);
            failures += run_case(shard, valid, rank * shard, kWords, 5, true);
        }
    }
    // Word-unaligned offset and a partial final word.
    failures += run_case(300, 237, 45, 10, 4, false);
    std::cout << (failures ? "FAIL" : "OK") << " token_bitmask\n";
    return failures ? 1 : 0;
}
