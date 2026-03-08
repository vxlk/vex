#pragma once

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>
#include <array>
#include <optional>
#include <memory>

namespace vex {

// ── Enums ──────────────────────────────────────────────────────────────────

enum class IndexStrategy : int {
    CONTAINER_INDEX  = 0,
    PACKET_SCAN      = 1,
    DECODE_SCAN      = 2,
    FORCED_INTERVAL  = 3,
    SKIPPED          = 4,
};

enum class OutputFormat {
    JPEG_STREAM,
    SPRITE_ATLAS,
};

// ── Configuration ──────────────────────────────────────────────────────────

struct LevelConfig {
    int width           = 0;      // 0 = NATIVE (source resolution)
    int height          = 0;      // 0 = NATIVE (source resolution)
    int quality         = 100;     // JPEG quality 1–100
    OutputFormat output  = OutputFormat::JPEG_STREAM;
    bool in_memory      = true;    // false → write to cache_path on disk
    std::string cache_path;        // required when in_memory=false
    int atlas_columns   = 10;      // only for sprite_atlas
};

struct BatchConfig {
    std::vector<std::string> paths;
    std::vector<LevelConfig> levels;
    int max_threads     = 8;
    bool keyframes_only = false;
    int frame_skip      = 1;       // ignored if keyframes_only=true
    bool use_hw_accel   = true;    // false → skip GPU decode, use software only
};

// ── Keyframe index ─────────────────────────────────────────────────────────

struct KeyframeInfo {
    int64_t byte_offset  = 0;
    int64_t pts          = 0;      // in stream time_base units
    int64_t pts_ms       = 0;      // milliseconds
    int     frame_number = 0;
    int64_t size         = -1;     // -1 if unknown
};

struct KeyframeIndex {
    std::vector<KeyframeInfo> keyframes;
    IndexStrategy strategy = IndexStrategy::SKIPPED;
    int64_t scan_time_us   = 0;
};

// ── Async event ────────────────────────────────────────────────────────────

struct FrameEvent {
    int      file_index  = 0;
    int      frame_index = 0;
    int      level_index = 0;
    int64_t  pts_ms      = 0;
    uint32_t blob_offset = 0;
    uint32_t jpeg_size   = 0;
};

// ── Metrics ────────────────────────────────────────────────────────────────

struct LevelMetrics {
    int         width              = 0;
    int         height             = 0;
    int         quality            = 0;
    std::string output_format;
    int64_t     scale_us           = 0;
    int64_t     encode_us          = 0;
    int64_t     atlas_composite_us = 0;
    int64_t     atlas_encode_us    = 0;
    int64_t     output_bytes       = 0;
    int         frame_count        = 0;
    double      avg_jpeg_bytes     = 0.0;
};

struct FileStats {
    int           file_index      = 0;
    int64_t       decode_us       = 0;
    int           keyframe_count  = 0;
    int64_t       file_size_bytes = 0;
    IndexStrategy index_strategy  = IndexStrategy::SKIPPED;
    std::string   codec_name;                // e.g. "h264", "hevc", "vp9"
    int           source_width    = 0;
    int           source_height   = 0;
    bool          hw_accel_used   = false;   // true if GPU decode was used
};

struct DecodeMetrics {
    // Wall clock
    int64_t total_wall_us = 0;

    // CPU-time summed across all threads (microseconds)
    int64_t index_scan_us = 0;
    int64_t seek_us       = 0;
    int64_t decode_us     = 0;
    int64_t io_us         = 0;

    // Counts
    int files_processed   = 0;
    int keyframes_decoded = 0;
    int keyframes_skipped = 0;
    int threads_used      = 0;

    // Pipeline info
    std::string hw_accel_backend;   // "d3d11va", "cuda", "qsv", "vaapi", or ""
    std::string encoder;            // JPEG encoder name, e.g. "libturbojpeg"

    // Memory (bytes)
    int64_t peak_decode_memory  = 0;
    int64_t total_output_bytes  = 0;
    int64_t atlas_scratch_bytes = 0;

    // Per-level
    std::vector<LevelMetrics> levels;

    // Per-file
    std::vector<FileStats> file_stats;

    // Derived
    double decode_fps() const {
        return decode_us > 0 ? keyframes_decoded / (decode_us / 1e6) : 0;
    }
    double pipeline_fps() const {
        return total_wall_us > 0 ? keyframes_decoded / (total_wall_us / 1e6) : 0;
    }
    double effective_decode_us() const {
        return threads_used > 0 ? static_cast<double>(decode_us) / threads_used : 0;
    }
    double thread_utilization() const {
        return total_wall_us > 0 ? effective_decode_us() / total_wall_us : 0;
    }
};

// ── Per-file output buffer ─────────────────────────────────────────────────

struct FileBlob {
    uint8_t* data     = nullptr;
    size_t   size     = 0;       // actual bytes written
    size_t   capacity = 0;       // allocated size
    std::vector<int64_t> offsets; // byte offset of each JPEG within data

    FileBlob() = default;

    FileBlob(FileBlob&& o) noexcept
        : data(o.data), size(o.size), capacity(o.capacity),
          offsets(std::move(o.offsets)) {
        o.data = nullptr;
        o.size = 0;
        o.capacity = 0;
    }

    FileBlob& operator=(FileBlob&& o) noexcept {
        if (this != &o) {
            free(data);
            data     = o.data;
            size     = o.size;
            capacity = o.capacity;
            offsets  = std::move(o.offsets);
            o.data = nullptr;
            o.size = 0;
            o.capacity = 0;
        }
        return *this;
    }

    ~FileBlob() { free(data); }

    FileBlob(const FileBlob&)            = delete;
    FileBlob& operator=(const FileBlob&) = delete;

    bool allocate(size_t cap) {
        free(data);
        data = static_cast<uint8_t*>(malloc(cap));
        if (!data) { capacity = 0; size = 0; return false; }
        capacity = cap;
        size = 0;
        offsets.clear();
        return true;
    }

    bool ensure_remaining(size_t needed) {
        if (size + needed <= capacity) return true;
        size_t new_cap = std::max(capacity * 2, size + needed);
        uint8_t* p = static_cast<uint8_t*>(realloc(data, new_cap));
        if (!p) return false;
        data = p;
        capacity = new_cap;
        return true;
    }

    void shrink_to_fit() {
        if (size < capacity && size > 0) {
            uint8_t* p = static_cast<uint8_t*>(realloc(data, size));
            if (p) { data = p; capacity = size; }
        }
    }

    uint8_t* release() {
        uint8_t* ptr = data;
        data = nullptr;
        size = 0;
        capacity = 0;
        return ptr;
    }
};

// ── Result types ───────────────────────────────────────────────────────────

struct JpegStreamResult {
    std::vector<FileBlob>              blobs;     // one per source file
    std::vector<std::array<int64_t,3>> metadata;  // [file_index, frame_number, pts_ms]
};

struct SpriteAtlasResult {
    std::vector<uint8_t> blob;       // all atlas JPEGs concatenated
    std::vector<int64_t> offsets;    // T+1 entries
    int grid_w     = 0;
    int grid_h     = 0;
    int thumb_w    = 0;
    int thumb_h    = 0;
    int file_count = 0;
};

struct DiskResult {
    std::string cache_path;
    int     frame_count = 0;
    int64_t total_bytes = 0;
    std::vector<std::array<int64_t,3>> metadata;
};

struct LevelResult {
    OutputFormat format;
    bool in_memory;
    std::optional<JpegStreamResult>  jpeg_stream;
    std::optional<SpriteAtlasResult> sprite_atlas;
    std::optional<DiskResult>        disk;
};

// ── Disk cache format ──────────────────────────────────────────────────────

static constexpr uint32_t CACHE_MAGIC   = 0x00584556; // 'VEX\0'
static constexpr uint32_t CACHE_VERSION = 1;

#pragma pack(push, 1)
struct CacheHeader {
    uint32_t magic;
    uint32_t version;
    uint32_t frame_count;
    uint32_t reserved1;
    int64_t  offset_table_pos;
    int64_t  metadata_pos;
    uint64_t source_hash;
    int32_t  width;
    int32_t  height;
    int32_t  quality;
    int32_t  output_format;
    uint8_t  padding[8];
};
#pragma pack(pop)

static_assert(sizeof(CacheHeader) == 64, "CacheHeader must be 64 bytes");

} // namespace vex
