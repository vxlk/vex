#pragma once

#include "types.h"
#include <string>

namespace vex {

/// Get presentation timestamps for every frame via packet-only scan (no decode).
/// Fast and independent — only reads container-level packet metadata.
FrameTimesResult get_frame_times(const std::string& path);

}  // namespace vex
