#pragma once

#include "types.h"
#include "virtual_blob.h"
#include <vector>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <memory>

namespace vex {

class DecodeHandle {
public:
    DecodeHandle(int num_threads, int total_files);

    // ── Worker API (called from decode threads) ─────────────────────────────

    void publish_event(int thread_id, FrameEvent evt);
    void increment_keyframes(int count = 1);
    void increment_files(int count = 1);
    void set_done();
    void register_blob(int file_index, const uint8_t* data);

    // ── Consumer API (called from Python / main thread) ─────────────────────

    std::vector<FrameEvent> drain_events();

    bool is_done() const { return done_.load(std::memory_order_acquire); }

    struct Progress {
        int keyframes_decoded;
        int files_completed;
        int total_files;
    };
    Progress progress() const;

    // Return a pointer into the file blob for a given event.
    const uint8_t* peek_jpeg_data(const FrameEvent& evt, size_t* out_size) const;

    void wait_until_done();

    // Called by orchestrator after all work finishes.
    void set_results(std::vector<LevelResult> results, DecodeMetrics metrics);
    std::pair<std::vector<LevelResult>, DecodeMetrics> get_results();

    // ── Streaming API (zero-copy views into VirtualBlob) ────────────────────

    // Register VirtualBlob pointers for streaming access.
    // blobs[level_index][file_index] = non-owning pointer to VirtualBlob.
    void register_stream_blobs(std::vector<std::vector<VirtualBlob*>> blobs);

    // Zero-copy view into the streaming JPEG buffer.
    struct StreamView {
        const uint8_t* data;  // stable base pointer (never moves)
        size_t current_size;  // atomically-published byte count
    };
    StreamView peek_stream(int file_index, int level_index) const;

    // ── Frame times streaming API ─────────────────────────────────────────
    void register_frame_times(std::vector<std::vector<double>>* times,
                              std::vector<std::unique_ptr<std::atomic<int>>>* published);

    // Returns frame times collected so far for a file (thread-safe snapshot).
    std::vector<double> peek_frame_times(int file_index) const;

    // Keep the SharedContext alive as long as this handle exists,
    // so VirtualBlob pointers remain valid.
    void set_context_keepalive(std::shared_ptr<void> ctx);

private:
    int num_threads_;
    int total_files_;

    // Per-thread event storage, cache-line padded.
    struct alignas(64) ThreadEvents {
        std::vector<FrameEvent> events;
        std::atomic<int> published{0};
        ThreadEvents() = default;
        ThreadEvents(ThreadEvents&& o) noexcept
            : events(std::move(o.events)), published(o.published.load(std::memory_order_relaxed)) {}
    };
    std::vector<ThreadEvents> thread_events_;
    std::vector<int> drain_cursors_;

    // Progress
    std::atomic<int> keyframes_decoded_{0};
    std::atomic<int> files_completed_{0};
    std::atomic<bool> done_{false};

    std::mutex done_mutex_;
    std::condition_variable done_cv_;

    // File blob pointers for peek_jpeg
    std::vector<const uint8_t*> file_blobs_;
    mutable std::mutex blobs_mutex_;

    // Final results
    std::vector<LevelResult> results_;
    DecodeMetrics metrics_;

    // Streaming: non-owning pointers to VirtualBlobs.
    // Indexed as stream_blobs_[level][file].
    std::vector<std::vector<VirtualBlob*>> stream_blobs_;

    // Prevents SharedContext (which owns the VirtualBlobs) from being
    // destroyed while this handle is alive.
    std::shared_ptr<void> context_keepalive_;

    // Frame times: non-owning pointers into SharedContext's vectors.
    // Safe because context_keepalive_ (shared_ptr<void>) prevents the
    // SharedContext from being destroyed while this handle is alive.
    // Cleared when get_results() releases context_keepalive_.
    std::vector<std::vector<double>>* frame_times_ = nullptr;
    std::vector<std::unique_ptr<std::atomic<int>>>* frame_times_published_ = nullptr;
};

}  // namespace vex
