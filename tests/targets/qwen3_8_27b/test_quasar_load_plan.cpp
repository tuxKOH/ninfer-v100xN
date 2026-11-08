#include "artifact/binder.h"
#include "artifact/reader.h"
#include "targets/qwen3_6_27b/impl/load/bindings.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace {
using namespace ninfer::artifact;
using namespace ninfer::targets::qwen3_6_27b::detail;
using Json = nlohmann::json;

void require(bool value, const std::string& message) {
    if (!value) { throw std::runtime_error(message); }
}

std::uint64_t little_endian(const char* bytes, int count) {
    std::uint64_t result = 0;
    for (int i = 0; i < count; ++i) {
        result |= std::uint64_t(static_cast<unsigned char>(bytes[i])) << (8 * i);
    }
    return result;
}

void verify(const char* path) {
    // Independent source-directory/scalar oracle. No GPU allocation or weight conversion.
    std::ifstream source(path, std::ios::binary);
    std::array<char, 32> header{};
    source.read(header.data(), header.size());
    require(source.good() && header[7] == 3, "QUASAR test requires its registered v3 file");
    std::string json_bytes(little_endian(header.data() + 8, 8), '\0');
    source.read(json_bytes.data(), json_bytes.size());
    require(source.good(), "truncated source directory");
    const Json directory = Json::parse(json_bytes);
    const auto payload_start = (header.size() + json_bytes.size() + 4095) / 4096 * 4096;
    std::unordered_map<std::string, Json> stored;
    for (const auto& object : directory.at("objects")) {
        stored.emplace(object.at("id").get<std::string>(), object);
    }
    const auto scalar = [&](std::uint64_t offset) {
        std::array<char, 4> bytes{};
        source.seekg(payload_start + offset);
        source.read(bytes.data(), bytes.size());
        require(source.good(), "source FP32 scalar is truncated");
        return static_cast<std::uint32_t>(little_endian(bytes.data(), 4));
    };

    Reader reader(path);
    require(reader.identity().model_id == "qwen3.8-27b" &&
                reader.identity().weights_id == "quasar-nvfp4", "QUASAR identity was not retained");
    require(reader.objects().size() == 1334, "QUASAR projected inventory is incomplete");
    for (const int tp : {2, 4}) {
        for (const auto backend : {ninfer::SpeculativeBackend::None, ninfer::SpeculativeBackend::Mtp,
                                   ninfer::SpeculativeBackend::DFlash}) {
            if (tp == 4 && backend == ninfer::SpeculativeBackend::DFlash) { continue; }
            Binder binder(reader, tp);
            const auto plan = bind_artifact(binder, WeightsProfile::Qwen38QuasarNvfp4,
                {.speculative = backend,
                 .proposal_head = backend == ninfer::SpeculativeBackend::Mtp
                     ? ninfer::ProposalHead::Optimized : ninfer::ProposalHead::Full}, tp);
            int projections = 0;
            const auto check = [&](const WeightPlan& weight, const std::string& divisor) {
                const auto& descriptor = std::get<TensorDescriptor>(binder.descriptor(weight.object));
                const auto& original = stored.at(descriptor.name);
                require(weight.format == NumericFormat::NVFP4 && original.at("format") == "nvfp4",
                        descriptor.name + ": wrong weight format");
                const auto payload = reader.payload(descriptor);
                require(payload.absolute_offset == payload_start + original.at("offset").get<std::uint64_t>() &&
                            payload.data.size() == original.at("bytes").get<std::uint64_t>(),
                        descriptor.name + ": not the exact original packed payload range");
                const auto elements = descriptor.shape[0] * descriptor.shape[1];
                // All registered shapes have aligned code planes; stored E2M1/E4M3 words are
                // followed by one FP32 weight divisor. Read it directly from the original file.
                const auto expected_weight = scalar(original.at("offset").get<std::uint64_t>() +
                                                    elements / 2 + elements / 16);
                const auto expected_input = scalar(stored.at(divisor).at("offset").get<std::uint64_t>());
                require(weight.weight_scale_divisor_bits == expected_weight &&
                            weight.input_scale_divisor_bits == expected_input,
                        descriptor.name + ": scale pairing differs from source FP32 words");
                ++projections;
            };
            for (std::size_t layer = 0; layer < plan.bindings.text_layers.size(); ++layer) {
                const auto& weights = plan.bindings.text_layers[layer];
                const auto prefix = "text/layers/" + std::to_string(layer) + "/";
                if (weights.is_full_attention) {
                    check(std::get<FusedAttentionProjectionPlan>(weights.attention.projection).query_key_gate_value,
                          prefix + "attention/input_projection/input_scale_divisor");
                    check(weights.attention.output, prefix + "attention/output_projection/input_scale_divisor");
                } else {
                    check(std::get<FusedGdnInputProjectionPlan>(weights.gdn.input_projection).query_key_value_z,
                          prefix + "gdn/input_projection/input_scale_divisor");
                    check(weights.gdn.output, prefix + "gdn/output_projection/input_scale_divisor");
                    require(std::get<FusedGdnControlProjectionPlan>(weights.gdn.control_projection)
                                .a_b_projection.format == NumericFormat::BF16,
                            "QUASAR GDN controls lost the artifact's BF16 boundary");
                }
                check(weights.mlp.gate_up, prefix + "mlp/gate_up_projection/input_scale_divisor");
                check(weights.mlp.down, prefix + "mlp/down_projection/input_scale_divisor");
            }
            require(projections == 256 && plan.bindings.token_embedding.format == NumericFormat::W8G32_F16S &&
                        plan.bindings.output_head.format == NumericFormat::W8G32_F16S &&
                        plan.bindings.mtp_format == NumericFormat::W8G32_F16S,
                    "QUASAR storage profile is incomplete");
            int mtp_objects = 0;
            std::array<int, 4> dflash_objects{};
            std::array<int, 4> output_heads{};
            std::array<int, 4> mtp_heads{};
            for (const auto& placement : plan.materialization.device_objects) {
                const auto name = object_name(binder.descriptor(placement.object));
                require(!name.starts_with("vision/"),
                        "text-only QUASAR uploaded a Vision object to a GPU");
                if (name.starts_with("dflash/")) { ++dflash_objects.at(placement.device); }
                if (name.starts_with("mtp/")) { ++mtp_objects; }
                if (name == "text/output_head") { ++output_heads.at(placement.device); }
                if (name == "text/draft_head") { ++mtp_heads.at(placement.device); }
            }
            require(backend == ninfer::SpeculativeBackend::Mtp ? mtp_objects > 0 : mtp_objects == 0,
                    "MTP startup selection is incorrect");
            require(dflash_objects == (backend == ninfer::SpeculativeBackend::DFlash
                        ? std::array{66, 66, 0, 0} : std::array{0, 0, 0, 0}),
                    "DFlash startup selection is incorrect");
            for (int rank = 0; rank < tp; ++rank) {
                require(output_heads.at(rank) == 1 &&
                            mtp_heads.at(rank) == (backend == ninfer::SpeculativeBackend::Mtp ? 1 : 0),
                        "vocabulary/proposal head residency is incomplete");
            }
            require(plan.bindings.dflash.projection_format == NumericFormat::W8G32_F16S,
                    "QUASAR DFlash projections lost their W8 format");
        }
    }
    for (int mode = 0; mode < 2; ++mode) {
        Binder binder(reader, 4);
        bool rejected = false;
        try {
            (void)bind_artifact(binder, WeightsProfile::Qwen38QuasarNvfp4,
                {.vision = mode == 0, .volta_fast_decode = false,
                 .speculative = mode == 1 ? ninfer::SpeculativeBackend::DFlash
                                          : ninfer::SpeculativeBackend::Mtp}, 4);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "unsupported TP4 Vision/DFlash combination admitted");
    }
    {
        Binder binder(reader, 4);
        const auto plan = bind_artifact(binder, WeightsProfile::Qwen38QuasarNvfp4,
            {.volta_fast_decode = true, .speculative = ninfer::SpeculativeBackend::Mtp}, 4);
        require(plan.bindings.features.volta_fast_decode,
                "TP4 experimental fast-decode profile was not selected");
    }
    for (const auto device : {ninfer::VisionDevice::Gpu, ninfer::VisionDevice::Cpu}) {
    for (const auto backend : {ninfer::SpeculativeBackend::None, ninfer::SpeculativeBackend::Mtp}) {
        Binder binder(reader, 2);
        const auto plan = bind_artifact(binder, WeightsProfile::Qwen38QuasarNvfp4,
            {.vision = true, .vision_device = device, .speculative = backend}, 2);
        int vision_objects = 0;
        int host_vision_objects = 0;
        for (const auto& placement : plan.materialization.host_objects) {
            host_vision_objects += object_name(binder.descriptor(placement.object)).starts_with("vision/");
        }
        for (const auto& placement : plan.materialization.device_objects) {
            if (object_name(binder.descriptor(placement.object)).starts_with("vision/")) {
                require(placement.device == 0, "Vision must live only on the primary TP2 device");
                ++vision_objects;
            }
        }
        require((device == ninfer::VisionDevice::Gpu ? vision_objects > 0 && host_vision_objects == 0
                                                    : vision_objects == 0 && host_vision_objects > 0) &&
                !plan.bindings.features.volta_fast_decode,
                "Vision weights must reside only on the selected encoder device");
    }
    }
    {
        Binder primary_binder(reader, 2);
        const auto primary = bind_artifact(primary_binder, WeightsProfile::Qwen38QuasarNvfp4,
                                           {.vision = true}, 2);
        Binder auxiliary_binder(reader, 2, false, 2);
        const auto auxiliary = bind_artifact(auxiliary_binder, WeightsProfile::Qwen38QuasarNvfp4,
                                             {.vision = true}, 2, 1, 7);
        require(!primary.bindings.auxiliary_vision_slot &&
                    auxiliary.bindings.auxiliary_vision_slot == 2 &&
                    auxiliary.bindings.auxiliary_vision_device == 7,
                "Vision slot or physical device was not recorded");
        std::size_t vision_count = 0;
        std::size_t text_count = 0;
        std::size_t sharded_count = 0;
        for (const auto& placement : auxiliary.materialization.device_objects) {
            const auto name = object_name(auxiliary_binder.descriptor(placement.object));
            if (name.starts_with("vision/")) {
                require(placement.device == 2, "auxiliary Vision escaped its slot");
                ++vision_count;
                continue;
            }
            require(placement.device < 2, "text object was placed on the auxiliary slot");
            const auto& original = primary.materialization.device_objects;
            const auto match = std::find_if(original.begin(), original.end(), [&](const auto& item) {
                return item.object.index == placement.object.index && item.device == placement.device;
            });
            require(match != original.end() && match->bytes == placement.bytes &&
                        match->copies.size() == placement.copies.size(),
                    "auxiliary Vision changed a text replica or shard");
            for (std::size_t i = 0; i < placement.copies.size(); ++i) {
                const auto& a = placement.copies[i];
                const auto& b = match->copies[i];
                require(a.source_offset == b.source_offset && a.dest_offset == b.dest_offset &&
                            a.bytes == b.bytes, "auxiliary Vision changed a text shard range");
            }
            ++text_count;
            sharded_count += !placement.copies.empty();
        }
        std::size_t primary_text_count = 0;
        for (const auto& placement : primary.materialization.device_objects) {
            const auto name = object_name(primary_binder.descriptor(placement.object));
            if (name.starts_with("vision/")) {
                require(placement.device == 0, "ordinary Vision left the primary slot");
            } else {
                ++primary_text_count;
            }
        }
        require(vision_count > 0 && text_count == primary_text_count && sharded_count > 0,
                "Vision or text shard placement is incomplete");
    }
    for (const bool missing_slot : {false, true}) {
        Binder binder(reader, 2, false, missing_slot ? std::nullopt : std::optional<int>{2});
        bool rejected = false;
        try {
            (void)bind_artifact(binder, WeightsProfile::Qwen38QuasarNvfp4,
                                {.vision = true}, 2, 1,
                                missing_slot ? std::optional<int>{7} : std::nullopt);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "mismatched auxiliary Vision selection and slot was accepted");
    }
    {
        Binder binder(reader, 2);
        const auto plan = bind_artifact(binder, WeightsProfile::Qwen38QuasarNvfp4,
            {.volta_fast_decode = true, .speculative = ninfer::SpeculativeBackend::Mtp}, 2);
        require(plan.bindings.features.volta_fast_decode, "experimental profile was not selected");
    }
    for (int mode = 0; mode < 3; ++mode) {
        Binder binder(reader, 2);
        bool rejected = false;
        try {
            (void)bind_artifact(binder, WeightsProfile::Qwen38QuasarNvfp4,
                {.vision = true, .volta_fast_decode = mode == 0,
                 .speculative = mode == 1 ? ninfer::SpeculativeBackend::DFlash
                                          : ninfer::SpeculativeBackend::Mtp},
                mode == 2 ? 1 : 2, mode == 2 ? 2 : 1);
        } catch (const std::invalid_argument&) { rejected = true; }
        require(rejected, "unsupported experimental/Vision/DFlash/PP2 combination admitted");
    }
}
} // namespace

int main() {
    const char* path = std::getenv("NINFER_QUASAR_ARTIFACT");
    if (!path || !*path) { std::cout << "SKIP: set NINFER_QUASAR_ARTIFACT\n"; return 77; }
    try {
        verify(path);
        std::cout << "OK QUASAR exact packed ranges, 256 scale pairs, TP2/TP4 None/MTP placement, TP2 DFlash "
                     "zero Text GPU Vision objects, TP2 Vision placement and experimental admission\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
