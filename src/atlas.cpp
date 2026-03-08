#include "atlas.h"
#include "encoder.h"

#include <cstring>
#include <algorithm>

namespace vex {

// ── Constructor ─────────────────────────────────────────────────────────────

AtlasBuilder::AtlasBuilder(int thumb_w, int thumb_h, int columns, int file_count, int quality)
    : thumb_w_(thumb_w),
      thumb_h_(thumb_h),
      columns_(columns),
      rows_((file_count + columns - 1) / columns),
      file_count_(file_count),
      quality_(quality),
      grid_pixel_w_(thumb_w * columns),
      grid_pixel_h_(thumb_h * rows_) {}

// ── ensure_time_step ────────────────────────────────────────────────────────

void AtlasBuilder::ensure_time_step(int time_step) {
    std::lock_guard<std::mutex> lock(ts_mutex_);
    int needed = time_step + 1;
    if (needed > static_cast<int>(time_steps_.size())) {
        size_t old_size = time_steps_.size();
        time_steps_.resize(static_cast<size_t>(needed));
        for (size_t i = old_size; i < time_steps_.size(); ++i) {
            init_time_step(time_steps_[i]);
        }
    }
}

// ── init_time_step ──────────────────────────────────────────────────────────

void AtlasBuilder::init_time_step(TimeStep& ts) {
    size_t y_size = static_cast<size_t>(grid_pixel_w_) * grid_pixel_h_;
    size_t uv_size = static_cast<size_t>(grid_pixel_w_ / 2) * (grid_pixel_h_ / 2);

    ts.y_plane.resize(y_size);
    ts.u_plane.resize(uv_size);
    ts.v_plane.resize(uv_size);

    // Black luma, neutral chroma
    std::memset(ts.y_plane.data(), 0, y_size);
    std::memset(ts.u_plane.data(), 128, uv_size);
    std::memset(ts.v_plane.data(), 128, uv_size);

    ts.count = 0;
}

// ── num_time_steps ──────────────────────────────────────────────────────────

int AtlasBuilder::num_time_steps() const {
    std::lock_guard<std::mutex> lock(ts_mutex_);
    return static_cast<int>(time_steps_.size());
}

// ── add_thumbnail ───────────────────────────────────────────────────────────

void AtlasBuilder::add_thumbnail(int time_step, int file_index, const uint8_t* y, const uint8_t* u,
                                 const uint8_t* v, int y_stride, int uv_stride) {
    // Grid position
    int col = file_index % columns_;
    int row = file_index / columns_;
    int x = col * thumb_w_;
    int y_pos = row * thumb_h_;

    // Access the time step (caller must have called ensure_time_step first)
    TimeStep& ts = time_steps_[static_cast<size_t>(time_step)];

    // Blit Y plane
    for (int r = 0; r < thumb_h_; ++r) {
        uint8_t* dst = ts.y_plane.data() + static_cast<size_t>(y_pos + r) * grid_pixel_w_ + x;
        const uint8_t* src = y + static_cast<size_t>(r) * y_stride;
        std::memcpy(dst, src, static_cast<size_t>(thumb_w_));
    }

    // Blit U plane (half dimensions)
    int half_w = thumb_w_ / 2;
    int half_h = thumb_h_ / 2;
    int half_grid_w = grid_pixel_w_ / 2;
    int half_x = x / 2;
    int half_y_pos = y_pos / 2;

    for (int r = 0; r < half_h; ++r) {
        uint8_t* dst =
            ts.u_plane.data() + static_cast<size_t>(half_y_pos + r) * half_grid_w + half_x;
        const uint8_t* src = u + static_cast<size_t>(r) * uv_stride;
        std::memcpy(dst, src, static_cast<size_t>(half_w));
    }

    // Blit V plane
    for (int r = 0; r < half_h; ++r) {
        uint8_t* dst =
            ts.v_plane.data() + static_cast<size_t>(half_y_pos + r) * half_grid_w + half_x;
        const uint8_t* src = v + static_cast<size_t>(r) * uv_stride;
        std::memcpy(dst, src, static_cast<size_t>(half_w));
    }

    ts.count++;
}

// ── compose_all ─────────────────────────────────────────────────────────────

SpriteAtlasResult AtlasBuilder::compose_all() {
    SpriteAtlasResult result{};
    result.grid_w = columns_;
    result.grid_h = rows_;
    result.thumb_w = thumb_w_;
    result.thumb_h = thumb_h_;
    result.file_count = file_count_;

    JpegEncoder encoder;

    int half_grid_w = grid_pixel_w_ / 2;

    // Offset table: T+1 entries for T time steps
    result.offsets.push_back(0);

    for (size_t i = 0; i < time_steps_.size(); ++i) {
        const TimeStep& ts = time_steps_[i];

        // Upper bound on JPEG size for the full grid
        size_t max_size = JpegEncoder::max_jpeg_size(grid_pixel_w_, grid_pixel_h_);

        // Grow blob to accommodate worst-case
        size_t old_size = result.blob.size();
        result.blob.resize(old_size + max_size);

        size_t jpeg_size = encoder.encode(ts.y_plane.data(), ts.u_plane.data(), ts.v_plane.data(),
                                          grid_pixel_w_, half_grid_w, grid_pixel_w_, grid_pixel_h_,
                                          quality_, result.blob.data() + old_size, max_size);

        if (jpeg_size == 0) {
            // Encoding failed; revert the blob growth
            result.blob.resize(old_size);
            result.offsets.push_back(static_cast<int64_t>(old_size));
            continue;
        }

        // Shrink blob to actual size
        result.blob.resize(old_size + jpeg_size);
        result.offsets.push_back(static_cast<int64_t>(old_size + jpeg_size));
    }

    return result;
}

}  // namespace vex
