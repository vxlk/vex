#pragma once

#include "types.h"
#include <string>

namespace vex {

/// Probe a video file: get frame timestamps, codec info, resolution, and
/// FFmpeg's auto-detected decode thread count — all from a single file open.
ProbeResult probe_video(const std::string& path);

}  // namespace vex
