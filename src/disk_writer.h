#pragma once

#include "types.h"
#include <cstdio>
#include <string>
#include <vector>

namespace vex {

class DiskWriter {
public:
    explicit DiskWriter(const std::string& path);
    ~DiskWriter();

    DiskWriter(const DiskWriter&) = delete;
    DiskWriter& operator=(const DiskWriter&) = delete;

    bool is_valid() const { return file_ != nullptr; }

    // Append one JPEG frame. Returns false on write error.
    bool write_frame(const uint8_t* data, size_t size);

    // Flush offset table + header, close file, return DiskResult.
    DiskResult finalize(const std::vector<std::array<int64_t, 3>>& metadata,
                        const LevelConfig& config, uint64_t source_hash);

private:
    FILE* file_ = nullptr;
    std::string path_;
    std::vector<int64_t> offsets_;
    int64_t current_pos_ = 0;
};

}  // namespace vex
