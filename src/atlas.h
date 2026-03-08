#pragma once

#include "types.h"
#include <cstdint>
#include <vector>
#include <mutex>

namespace vex {

class AtlasBuilder {
public:
    AtlasBuilder(int thumb_w, int thumb_h, int columns, int file_count, int quality);

    // Ensure storage exists for the given time step (call before add_thumbnail).
    void ensure_time_step(int time_step);

    // Store a thumbnail at grid position (time_step, file_index).
    // Thread-safe for distinct (time_step, file_index) pairs.
    void add_thumbnail(int time_step, int file_index,
                       const uint8_t* y, const uint8_t* u, const uint8_t* v,
                       int y_stride, int uv_stride);

    // Composite all time steps into a SpriteAtlasResult. Single-threaded.
    SpriteAtlasResult compose_all();

    int grid_w() const { return columns_; }
    int grid_h() const { return rows_; }
    int num_time_steps() const;

private:
    struct TimeStep {
        std::vector<uint8_t> y_plane, u_plane, v_plane;
        int count = 0;
    };

    int thumb_w_, thumb_h_;
    int columns_, rows_;
    int file_count_, quality_;
    int grid_pixel_w_, grid_pixel_h_;

    mutable std::mutex ts_mutex_;
    std::vector<TimeStep> time_steps_;

    void init_time_step(TimeStep& ts);
};

} // namespace vex
