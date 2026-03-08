#include "disk_writer.h"

#include <cstring>

namespace vex {

// ── Constructor ─────────────────────────────────────────────────────────────

DiskWriter::DiskWriter(const std::string& path) : path_(path), current_pos_(0) {
    file_ = fopen(path.c_str(), "wb");
    if (file_) {
        setvbuf(file_, nullptr, _IOFBF, 262144);  // 256KB write buffer
    }
}

// ── Destructor ──────────────────────────────────────────────────────────────

DiskWriter::~DiskWriter() {
    if (file_) {
        fclose(file_);
        file_ = nullptr;
    }
}

// ── write_frame ─────────────────────────────────────────────────────────────

bool DiskWriter::write_frame(const uint8_t* data, size_t size) {
    if (!file_)
        return false;

    // Record the offset of this frame
    offsets_.push_back(current_pos_);

    size_t written = fwrite(data, 1, size, file_);
    if (written != size) {
        return false;
    }

    current_pos_ += static_cast<int64_t>(size);
    return true;
}

// ── finalize ────────────────────────────────────────────────────────────────

DiskResult DiskWriter::finalize(const std::vector<std::array<int64_t, 3>>& metadata,
                                const LevelConfig& config, uint64_t source_hash) {
    DiskResult result{};
    result.cache_path = path_;

    if (!file_) {
        return result;
    }

    // Step 1: Record final offset (gives N+1 entries for N frames)
    offsets_.push_back(current_pos_);

    // Step 2: Write the offset table
    int64_t offset_table_pos = current_pos_;
    size_t offset_bytes = offsets_.size() * sizeof(int64_t);
    fwrite(offsets_.data(), sizeof(int64_t), offsets_.size(), file_);
    current_pos_ += static_cast<int64_t>(offset_bytes);

    // Step 3: Write metadata (each entry is 3 x int64_t)
    int64_t metadata_pos = current_pos_;
    if (!metadata.empty()) {
        size_t meta_bytes = metadata.size() * sizeof(std::array<int64_t, 3>);
        fwrite(metadata.data(), sizeof(std::array<int64_t, 3>), metadata.size(), file_);
        current_pos_ += static_cast<int64_t>(meta_bytes);
    }

    // Step 4: Build CacheHeader
    uint32_t frame_count = static_cast<uint32_t>(offsets_.size() - 1);

    CacheHeader header{};
    header.magic = CACHE_MAGIC;
    header.version = CACHE_VERSION;
    header.frame_count = frame_count;
    header.reserved1 = 0;
    header.offset_table_pos = offset_table_pos;
    header.metadata_pos = metadata_pos;
    header.source_hash = source_hash;
    header.width = static_cast<int32_t>(config.width);
    header.height = static_cast<int32_t>(config.height);
    header.quality = static_cast<int32_t>(config.quality);
    header.output_format = static_cast<int32_t>(config.output);
    std::memset(header.padding, 0, sizeof(header.padding));

    // Step 5: Write header
    fwrite(&header, sizeof(CacheHeader), 1, file_);
    current_pos_ += static_cast<int64_t>(sizeof(CacheHeader));

    // Step 6: Close file
    fclose(file_);
    file_ = nullptr;

    // Step 7: Build result
    result.frame_count = static_cast<int>(frame_count);
    result.total_bytes = current_pos_;
    result.metadata = metadata;

    return result;
}

}  // namespace vex
