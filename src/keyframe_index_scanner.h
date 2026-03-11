#pragma once

#include "types.h"

extern "C" {
#include <libavformat/avformat.h>
}

namespace vex {

// Extract keyframe index with 5-level fallback chain:
//   1. Container index (MP4 stss, AVI idx1, MKV Cues)
//   2. Packet header scan (av_read_frame + AV_PKT_FLAG_KEY)
//   3. Decode scan (decode every frame, check pict_type)
//   4. Forced interval (every 30th frame)
//   5. Skip
// Returned keyframes are sorted by byte offset ascending for sequential I/O.
KeyframeIndex scan_keyframes(AVFormatContext* fmt_ctx, int stream_index);

}  // namespace vex
