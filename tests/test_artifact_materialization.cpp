#include "artifact/binder.h"
#include "artifact/materializer.h"
#include "artifact/reader.h"
#include "artifact/typed_binding.h"
#include "artifact_fixture.h"
#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

constexpr std::array<std::byte, 3> kResource = {
    std::byte{1},
    std::byte{1},
    std::byte{1},
};
constexpr std::array<std::byte, 4> kTensor = {
    std::byte{2},
    std::byte{2},
    std::byte{2},
    std::byte{2},
};
constexpr std::array<std::byte, 8> kSecondTensor = {
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
    std::byte{3}, std::byte{3}, std::byte{3}, std::byte{3},
};
constexpr std::size_t kFp8TensorBytes = 260;
constexpr std::size_t kTailReadBytes  = 256 + kFp8TensorBytes;

ninfer::test::artifact_fixture::TemporaryArtifact write_fixture() {
    using Json = ninfer::test::artifact_fixture::Json;
    return ninfer::test::artifact_fixture::write_fixture(
        {
            {"identity", {{"model_id", "fixture-model"}, {"weights_id", "fixture-weights"}}},
            {"objects", Json::array({
                            {{"name", "frontend/test.json"},
                             {"kind", "resource"},
                             {"encoding", "raw-bytes-v1"},
                             {"offset", 0},
                             {"bytes", 3}},
                            {{"name", "weights/test"},
                             {"kind", "tensor"},
                             {"shape", {2}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 256},
                             {"bytes", 4}},
                            {{"name", "weights/second"},
                             {"kind", "tensor"},
                             {"shape", {4}},
                             {"format", "BF16"},
                             {"layout", "contiguous-le-v1"},
                             {"offset", 8192},
                             {"bytes", 8}},
                            {{"name", "weights/fp8"},
                             {"kind", "tensor"},
                             {"shape", {2, 4}},
                             {"format", "FP8_E4M3FN_ROW_BF16S"},
                             {"layout", "row-scale-v1"},
                             {"offset", 8448},
                             {"bytes", kFp8TensorBytes}},
                        })},
        },
        "materialization");
}

bool cuda_unavailable(cudaError_t error) {
    return error == cudaErrorNoDevice || error == cudaErrorInsufficientDriver;
}

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

} // namespace

int main() {
    try {
        auto fixture = write_fixture();
        ninfer::artifact::Reader reader(fixture.path);
        ninfer::artifact::Binder validation_binder(reader);
        const auto validated_resource = validation_binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        validation_binder.retain_on_host(validated_resource);
        constexpr std::array<std::uint64_t, 1> validated_shape = {2};
        const auto validated_only                              = validation_binder.require_tensor(
            "weights/test", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, validated_shape);
        validation_binder.validate_only(validated_only);
        constexpr std::array<std::uint64_t, 1> retained_shape = {4};
        const auto retained_tensor                            = validation_binder.require_tensor(
            "weights/second", ninfer::artifact::NumericFormat::BF16,
            ninfer::artifact::StorageLayout::ContiguousLeV1, retained_shape);
        validation_binder.materialize_on_device(retained_tensor);
        constexpr std::array<std::uint64_t, 2> fp8_shape = {2, 4};
        const auto validated_fp8                         = validation_binder.require_tensor(
            "weights/fp8", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
            ninfer::artifact::StorageLayout::RowScaleV1, fp8_shape);
        validation_binder.validate_only(validated_fp8);
        const auto validation_plan = validation_binder.finish();
        require(validation_plan.object_count == 4 && validation_plan.host_objects.size() == 1 &&
                    validation_plan.device_objects.size() == 1 &&
                    validation_plan.device_capacity_bytes[0] == kSecondTensor.size(),
                "validate-only tensor was included in the materialization plan");

        {
            ninfer::artifact::Binder primary_binder(reader, 2);
            const auto primary = ninfer::artifact::bind_tensor(
                primary_binder, "weights/test", ninfer::artifact::NumericFormat::BF16, {2},
                ninfer::artifact::TensorPlacement::PrimaryDevice);
            primary_binder.validate_unconsumed_matching();
            const auto primary_plan = primary_binder.finish();
            require(primary_plan.device_objects.size() == 1 &&
                        primary_plan.device_objects.front().object.index == primary.index &&
                        primary_plan.device_objects.front().device == 0 &&
                        primary_plan.device_capacity_bytes[0] == kTensor.size() &&
                        primary_plan.device_capacity_bytes[1] == 0,
                    "primary-only tensor was replicated or omitted");
        }
        {
            ninfer::artifact::Binder stage_binder(reader, 2);
            stage_binder.set_device_mask_resolver([](std::string_view) { return 2U; });
            const auto stage = ninfer::artifact::bind_tensor(
                stage_binder, "weights/test", ninfer::artifact::NumericFormat::BF16, {2},
                ninfer::artifact::TensorPlacement::Device);
            stage_binder.validate_unconsumed_matching();
            const auto plan = stage_binder.finish();
            require(plan.device_objects.size() == 1 &&
                        plan.device_objects.front().object.index == stage.index &&
                        plan.device_objects.front().device == 1 &&
                        plan.device_capacity_bytes[0] == 0 &&
                        plan.device_capacity_bytes[1] == kTensor.size(),
                    "pipeline tensor placement changed its bytes or resident owner");
        }
        {
            using namespace ninfer::artifact;
            Binder binder(reader, 2, false, 2);
            const auto vision = bind_tensor(binder, "weights/test", NumericFormat::BF16, {2},
                                            TensorPlacement::AuxiliaryDevice);
            const auto text = bind_tensor(binder, "weights/second", NumericFormat::BF16, {4},
                                          TensorPlacement::Device);
            binder.validate_unconsumed_matching();
            const auto plan = binder.finish();
            std::array<int, 3> vision_count{};
            std::array<int, 3> text_count{};
            for (const auto& placement : plan.device_objects) {
                if (placement.object.index == vision.index) { ++vision_count.at(placement.device); }
                if (placement.object.index == text.index) { ++text_count.at(placement.device); }
            }
            require(vision_count == std::array{0, 0, 1} && text_count == std::array{1, 1, 0},
                    "auxiliary placement mixed vision and replicated text slots");
        }
        {
            using namespace ninfer::artifact;
            Binder binder(reader, 2);
            bool rejected = false;
            try {
                (void)bind_tensor(binder, "weights/test", NumericFormat::BF16, {2},
                                  TensorPlacement::AuxiliaryDevice);
            } catch (const ArtifactError&) { rejected = true; }
            require(rejected, "auxiliary placement without an auxiliary slot was accepted");
        }
        for (const int slot : {0, 1, 3}) {
            bool rejected = false;
            try {
                ninfer::artifact::Binder binder(reader, 2, false, slot);
            } catch (const ninfer::artifact::ArtifactError&) { rejected = true; }
            require(rejected, "overlapping or non-adjacent auxiliary slot was accepted");
        }
        // A sharded placement that names no range for a device is a shard-map bug, not a request
        // to replicate: the binder must refuse it rather than quietly reserving a full copy.
        {
            ninfer::artifact::Binder sharded_binder(reader, 2);
            require(sharded_binder.device_count() == 2, "binder did not record its device count");
            sharded_binder.set_shard_resolver([](std::string_view) {
                ninfer::artifact::ShardPlacement placement;
                placement.axis = ninfer::artifact::ShardAxis::Rows;
                placement.device_ranges[0].push_back(ninfer::artifact::SliceRange{0, 2});
                return placement; // device 1 deliberately left empty
            });
            const auto sharded = sharded_binder.require_tensor(
                "weights/second", ninfer::artifact::NumericFormat::BF16,
                ninfer::artifact::StorageLayout::ContiguousLeV1, retained_shape);
            bool rejected = false;
            try {
                sharded_binder.materialize_on_device(sharded);
            } catch (const ninfer::artifact::ArtifactError&) { rejected = true; }
            require(rejected, "a sharded object with no range for a device was not rejected");
        }

        int device_count              = 0;
        const cudaError_t count_error = cudaGetDeviceCount(&device_count);
        if (cuda_unavailable(count_error)) {
            std::cout << "SKIP: no usable CUDA device\n";
            return 77;
        }
        CUDA_CHECK(count_error);
        if (device_count == 0) {
            std::cout << "SKIP: no CUDA devices\n";
            return 77;
        }

        ninfer::artifact::Binder binder(reader);

        const auto resource = binder.require_resource(
            "frontend/test.json", ninfer::artifact::ResourceEncoding::RawBytesV1);
        binder.retain_on_host(resource);
        constexpr std::array<std::uint64_t, 1> second_shape = {4};
        const auto second =
            binder.require_tensor("weights/second", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, second_shape);
        binder.materialize_on_device(second);

        // Bind in the opposite order from the artifact. Device placement order and file read order
        // are intentionally independent, exercising the direct-I/O scatter path.
        constexpr std::array<std::uint64_t, 1> tensor_shape = {2};
        const auto tensor =
            binder.require_tensor("weights/test", ninfer::artifact::NumericFormat::BF16,
                                  ninfer::artifact::StorageLayout::ContiguousLeV1, tensor_shape);
        binder.materialize_on_device(tensor);

        const auto fp8 = binder.require_tensor(
            "weights/fp8", ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S,
            ninfer::artifact::StorageLayout::RowScaleV1, fp8_shape);
        binder.materialize_on_device(fp8);

        const ninfer::artifact::MaterializationPlan plan = binder.finish();
        require(plan.object_count == 4 && plan.host_objects.size() == 1 &&
                    plan.device_objects.size() == 3 && plan.device_capacity_bytes[0] == 772,
                "binder produced the wrong materialization plan");

        ninfer::DeviceContext device(0);
        auto materialized = ninfer::artifact::materialize(reader, plan, device);

        std::array<std::byte, kTensor.size()> copied{};
        CUDA_CHECK(cudaMemcpy(copied.data(), materialized.device_data(tensor), copied.size(),
                              cudaMemcpyDeviceToHost));
        require(copied == kTensor, "device tensor payload differs from the artifact");
        std::array<std::byte, kSecondTensor.size()> second_copied{};
        CUDA_CHECK(cudaMemcpy(second_copied.data(), materialized.device_data(second),
                              second_copied.size(), cudaMemcpyDeviceToHost));
        require(second_copied == kSecondTensor,
                "second device tensor payload differs from the artifact");
        std::array<std::byte, kFp8TensorBytes> fp8_copied{};
        CUDA_CHECK(cudaMemcpy(fp8_copied.data(), materialized.device_data(fp8), fp8_copied.size(),
                              cudaMemcpyDeviceToHost));
        require(std::all_of(fp8_copied.begin(), fp8_copied.end(),
                            [](std::byte value) { return value == std::byte{4}; }),
                "FP8 device tensor payload differs from the artifact");

        const ninfer::Weight fp8_weight = ninfer::artifact::materialized_weight(
            materialized, fp8, ninfer::artifact::NumericFormat::FP8_E4M3FN_ROW_BF16S, 2, 4);
        require(fp8_weight.qtype == ninfer::QType::FP8_E4M3FN_ROW_BF16S &&
                    fp8_weight.layout == ninfer::QuantLayout::RowScale &&
                    fp8_weight.scale_dtype == ninfer::DType::BF16 && fp8_weight.n == 2 &&
                    fp8_weight.k == 4 && fp8_weight.group == 4 && fp8_weight.group_size == 4 &&
                    fp8_weight.qdata == fp8_weight.payload && fp8_weight.qhigh == nullptr &&
                    fp8_weight.scales == static_cast<const std::byte*>(fp8_weight.payload) + 256 &&
                    fp8_weight.payload_bytes == kFp8TensorBytes,
                "materialized FP8 Weight metadata is incomplete");

        const auto retained = materialized.host_bytes(resource);
        require(std::equal(retained.begin(), retained.end(), kResource.begin(), kResource.end()),
                "retained resource payload differs from the artifact");

        const auto& stats = materialized.stats();
        require(stats.tensor_count == 3 && stats.resource_count == 1 &&
                    stats.h2d_bytes == kTensor.size() + kSecondTensor.size() + kFp8TensorBytes &&
                    stats.retained_host_bytes == kResource.size() &&
                    stats.file_bytes == kResource.size() +
                                            ninfer::artifact::Reader::direct_io_alignment +
                                            kTailReadBytes,
                "materialization statistics are incomplete");
        require(materialized.device_arena().capacity() == plan.device_capacity_bytes[0] &&
                    materialized.device_arena().used() == plan.device_capacity_bytes[0],
                "materialized tensor does not own the planned device backing");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
