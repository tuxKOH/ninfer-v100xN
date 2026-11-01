#pragma once

#include <ninfer/types.h>

#include <chrono>
#include <string>
#include <string_view>

namespace ninfer::targets::qwen3_6::frontend_internal {

inline void check_preparation_control(const PreparationControl& control,
                                      std::string_view stage = "preparation") {
    if (control.cancellation.requested()) {
        throw RequestError(RequestErrorKind::Cancelled,
                           "inference request was cancelled during " + std::string(stage));
    }
    if (control.deadline != std::chrono::steady_clock::time_point{} &&
        std::chrono::steady_clock::now() >= control.deadline) {
        throw RequestError(RequestErrorKind::QueueTimeout,
                           "inference request expired during " + std::string(stage));
    }
}

} // namespace ninfer::targets::qwen3_6::frontend_internal
