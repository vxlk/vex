#include "async_handle.h"

#include <algorithm>

namespace vex {

// ── Constructor ─────────────────────────────────────────────────────────────

DecodeHandle::DecodeHandle(int num_threads, int total_files)
    : num_threads_(num_threads), total_files_(total_files) {
    thread_events_.resize(static_cast<size_t>(num_threads));
    drain_cursors_.resize(static_cast<size_t>(num_threads), 0);
    file_blobs_.resize(static_cast<size_t>(total_files), nullptr);
}

// ── Worker API ──────────────────────────────────────────────────────────────

void DecodeHandle::publish_event(int thread_id, FrameEvent evt) {
    auto& te = thread_events_[static_cast<size_t>(thread_id)];
    te.events.push_back(evt);
    te.published.store(static_cast<int>(te.events.size()), std::memory_order_release);
}

void DecodeHandle::increment_keyframes(int count) {
    keyframes_decoded_.fetch_add(count, std::memory_order_relaxed);
}

void DecodeHandle::increment_files(int count) {
    files_completed_.fetch_add(count, std::memory_order_relaxed);
}

void DecodeHandle::set_done() {
    done_.store(true, std::memory_order_release);
    std::lock_guard<std::mutex> lock(done_mutex_);
    done_cv_.notify_all();
}

void DecodeHandle::register_blob(int file_index, const uint8_t* data) {
    std::lock_guard<std::mutex> lock(blobs_mutex_);
    file_blobs_[static_cast<size_t>(file_index)] = data;
}

// ── Consumer API ────────────────────────────────────────────────────────────

std::vector<FrameEvent> DecodeHandle::drain_events() {
    std::vector<FrameEvent> collected;

    for (int i = 0; i < num_threads_; ++i) {
        auto& te = thread_events_[static_cast<size_t>(i)];
        int published = te.published.load(std::memory_order_acquire);
        int cursor = drain_cursors_[static_cast<size_t>(i)];

        if (published > cursor) {
            collected.insert(collected.end(), te.events.begin() + cursor,
                             te.events.begin() + published);
            drain_cursors_[static_cast<size_t>(i)] = published;
        }
    }

    return collected;
}

DecodeHandle::Progress DecodeHandle::progress() const {
    Progress p{};
    p.keyframes_decoded = keyframes_decoded_.load(std::memory_order_relaxed);
    p.files_completed = files_completed_.load(std::memory_order_relaxed);
    p.total_files = total_files_;
    return p;
}

const uint8_t* DecodeHandle::peek_jpeg_data(const FrameEvent& evt, size_t* out_size) const {
    std::lock_guard<std::mutex> lock(blobs_mutex_);
    const uint8_t* blob = file_blobs_[static_cast<size_t>(evt.file_index)];
    if (!blob) {
        if (out_size)
            *out_size = 0;
        return nullptr;
    }
    if (out_size)
        *out_size = evt.jpeg_size;
    return blob + evt.blob_offset;
}

void DecodeHandle::wait_until_done() {
    std::unique_lock<std::mutex> lock(done_mutex_);
    done_cv_.wait(lock, [this]() { return done_.load(std::memory_order_acquire); });
}

void DecodeHandle::set_results(std::vector<LevelResult> results, DecodeMetrics metrics) {
    results_ = std::move(results);
    metrics_ = std::move(metrics);
}

std::pair<std::vector<LevelResult>, DecodeMetrics> DecodeHandle::get_results() {
    // Release streaming resources — VirtualBlobs have been moved into
    // the final results, so stream pointers are now invalid.
    stream_blobs_.clear();
    context_keepalive_.reset();
    return {std::move(results_), std::move(metrics_)};
}

// ── Streaming API ───────────────────────────────────────────────────────────

void DecodeHandle::register_stream_blobs(
    std::vector<std::vector<VirtualBlob*>> blobs) {
    stream_blobs_ = std::move(blobs);
}

DecodeHandle::StreamView DecodeHandle::peek_stream(
    int file_index, int level_index) const {
    StreamView sv{nullptr, 0};
    if (level_index < 0 ||
        level_index >= static_cast<int>(stream_blobs_.size()))
        return sv;
    auto& level = stream_blobs_[static_cast<size_t>(level_index)];
    if (file_index < 0 ||
        file_index >= static_cast<int>(level.size()))
        return sv;
    VirtualBlob* vb = level[static_cast<size_t>(file_index)];
    if (!vb || !vb->data)
        return sv;
    sv.data = vb->data;
    sv.current_size = vb->published_size.load(std::memory_order_acquire);
    return sv;
}

void DecodeHandle::set_context_keepalive(std::shared_ptr<void> ctx) {
    context_keepalive_ = std::move(ctx);
}

}  // namespace vex
