#pragma once

#include "types.h"
#include "async_handle.h"
#include <memory>
#include <utility>
#include <vector>

namespace vex {

class Orchestrator {
public:
    // Synchronous — blocks until complete.
    static std::pair<std::vector<LevelResult>, DecodeMetrics>
    batch_decode(const BatchConfig& config);

    // Asynchronous — returns handle immediately, decode runs on background threads.
    static std::shared_ptr<DecodeHandle>
    batch_decode_async(const BatchConfig& config);
};

} // namespace vex
