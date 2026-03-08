#pragma once

#include "types.h"
#include "hw_accel.h"

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/frame.h>
}

#include <string>

namespace vex {

class FileDecoder {
public:
    // Opens the file and initialises the video decoder.
    // hw may be nullptr for software-only decode.
    FileDecoder(const std::string& path, HWAccelContext* hw = nullptr);
    ~FileDecoder();

    FileDecoder(const FileDecoder&) = delete;
    FileDecoder& operator=(const FileDecoder&) = delete;

    bool is_valid() const { return valid_; }

    // Extract keyframe index via the fallback chain.
    KeyframeIndex scan_keyframes();

    // Seek to keyframe and decode one I-frame into out_frame (YUV420P).
    bool seek_and_decode(const KeyframeInfo& kf, AVFrame* out_frame);

    int         video_stream_index() const { return video_idx_; }
    int         source_width()       const { return width_; }
    int         source_height()      const { return height_; }
    std::string codec_name()         const { return codec_name_; }
    int64_t     file_size()          const { return file_size_; }

private:
    AVFormatContext* fmt_ctx_   = nullptr;
    AVCodecContext*  codec_ctx_ = nullptr;
    AVFrame*         hw_frame_  = nullptr;   // scratch frame for HW decode
    HWAccelContext*  hw_accel_  = nullptr;
    int              video_idx_ = -1;
    int              width_     = 0;
    int              height_    = 0;
    int64_t          file_size_ = 0;
    std::string      codec_name_;
    bool             valid_     = false;
};

} // namespace vex
