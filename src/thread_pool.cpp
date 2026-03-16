#include "thread_pool.h"

namespace vex {

ThreadPool& ThreadPool::instance() {
    static ThreadPool pool;
    return pool;
}

ThreadPool::~ThreadPool() {
    shutdown();
}

void ThreadPool::worker_loop() {
    while (true) {
        WorkItem item;

        // ── Dequeue phase ──────────────────────────────────────────
        // Hold mutex_ only long enough to pop one item.  The actual
        // task execution happens outside the lock so other workers
        // (and run() callers) aren't blocked.
        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [this] { return shutdown_ || !queue_.empty(); });

            if (shutdown_)
                return;

            item = std::move(queue_.front());
            queue_.pop_front();

            // Track in-flight items so run() can size the pool correctly
            // when a new batch arrives while existing work is executing.
            active_.fetch_add(1, std::memory_order_relaxed);
        }

        // ── Execute phase ──────────────────────────────────────────
        // The task is a bound closure: [func, id]() { func(id); }
        // It may be long-running (e.g. an entire video decode).
        item.task();

        active_.fetch_sub(1, std::memory_order_relaxed);

        // ── Batch completion signalling ────────────────────────────
        // Atomically increment the batch's completed counter.  If we
        // are the last item to finish (completed reaches total), wake
        // the run() caller that is blocked on this batch's condvar.
        //
        // The acq_rel ordering ensures the run() caller sees all
        // side-effects from every item's execution before it resumes.
        if (item.batch->completed.fetch_add(1, std::memory_order_acq_rel) + 1 >=
            item.batch->total) {
            std::lock_guard<std::mutex> lock(item.batch->mutex);
            item.batch->cv.notify_one();
        }
    }
}

void ThreadPool::grow_to(int n) {
    // Caller must hold mutex_.
    while (static_cast<int>(workers_.size()) < n) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

void ThreadPool::run(std::function<void(int)> func, int count) {
    if (count <= 0)
        return;

    // Each run() call gets its own Batch — this is the key invariant
    // that makes concurrent run() calls safe.  No shared counters,
    // no shared func pointer, no cross-batch interference.
    auto batch = std::make_shared<Batch>();
    batch->total = count;

    {
        std::lock_guard<std::mutex> lock(mutex_);

        // Enqueue one WorkItem per id.  Each item captures `func` by
        // value (shared via std::function's internal refcount) and a
        // unique integer id, so items are fully independent.
        for (int i = 0; i < count; ++i) {
            queue_.push_back(WorkItem{[f = func, i]() { f(i); }, batch});
        }

        // Grow the pool so this batch can run concurrently with any
        // items that are already executing.  Without this, a second
        // batch_decode_async() call would have to wait for the first
        // batch's long-running workers to finish before its items
        // could be picked up — defeating the purpose of async decode.
        int target = active_.load(std::memory_order_relaxed) + static_cast<int>(queue_.size());
        grow_to(target);
    }

    // Wake all workers — some may be idle waiting on work_cv_.
    work_cv_.notify_all();

    // Block until every item in *this* batch is done.  This wait is
    // completely independent of other batches: it uses the Batch's
    // private mutex + condvar, so other run() callers blocking on
    // their own batches don't interfere.
    {
        std::unique_lock<std::mutex> lock(batch->mutex);
        batch->cv.wait(lock, [&batch] {
            return batch->completed.load(std::memory_order_acquire) >= batch->total;
        });
    }
}

void ThreadPool::shutdown() {
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (shutdown_)
            return;
        shutdown_ = true;
    }
    work_cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable())
            w.join();
    }
    workers_.clear();
}

}  // namespace vex
