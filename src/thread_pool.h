#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace vex {

/// Process-global persistent thread pool.
///
/// Workers stay alive across calls — only new threads are spawned when
/// capacity increases.  Safe to call run() concurrently from multiple
/// threads; each call gets independent completion tracking.
///
/// Design overview
/// ───────────────
/// Work is submitted via run(func, count), which enqueues `count`
/// independent WorkItems onto a shared deque.  Persistent worker threads
/// pull items one at a time and execute them.
///
/// Each run() call allocates its own Batch object (shared_ptr) that
/// carries an atomic completion counter and a private condvar.  Workers
/// decrement the counter after finishing an item and signal the condvar
/// when the batch is fully complete.  This means multiple concurrent
/// run() calls — from different threads — never share completion state,
/// eliminating the cross-batch corruption that caused deadlocks when the
/// pool used a single set of member variables (func_, total_, completed_).
///
/// The pool grows on demand: when a new batch arrives, run() ensures
/// there are at least (active + queued) workers so that items from the
/// new batch don't have to wait for long-running items from an earlier
/// batch to finish.
class ThreadPool {
public:
    static ThreadPool& instance();

    /// Submit work: call func(id) for id in [0, count).
    ///
    /// Blocks the caller until every item in this batch has completed.
    /// Safe to call concurrently from multiple threads — each invocation
    /// gets its own Batch with independent completion tracking.
    ///
    /// The pool is grown if necessary so that items from this batch can
    /// run concurrently with any already-active items from other batches.
    void run(std::function<void(int)> func, int count);

    /// Shut down all workers (blocks until joined).  Called automatically
    /// by the destructor.  Pending queue items are abandoned — only call
    /// this when all batches are known to have completed (e.g. at process
    /// exit via the static-singleton destructor).
    void shutdown();

    ~ThreadPool();

private:
    ThreadPool() = default;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /// Per-invocation completion state.
    ///
    /// Each run() call creates one of these so concurrent batches never
    /// interfere with each other.  The caller waits on `cv`; the last
    /// worker to finish an item from this batch signals it.
    struct Batch {
        std::atomic<int> completed{0};
        int total{0};
        std::mutex mutex;              // protects the cv wait
        std::condition_variable cv;    // signalled when completed >= total
    };

    /// A single unit of work in the queue.
    ///
    /// `task` is a bound closure (func + id).  `batch` points back to
    /// the owning Batch so the worker can signal completion.
    struct WorkItem {
        std::function<void()> task;
        std::shared_ptr<Batch> batch;
    };

    /// Main loop executed by every persistent worker thread.
    ///
    /// Workers block on work_cv_ until an item is available (or shutdown
    /// is requested).  After executing an item, the worker atomically
    /// increments the Batch's completed counter and signals the Batch's
    /// cv when the batch is fully done.
    void worker_loop();

    /// Grow the pool to at least n workers.  Caller must hold mutex_.
    void grow_to(int n);

    std::mutex mutex_;                   // protects queue_, workers_, shutdown_
    std::condition_variable work_cv_;    // workers wait on this for new items
    std::deque<WorkItem> queue_;         // pending work items (FIFO)
    std::vector<std::thread> workers_;   // persistent worker threads
    std::atomic<int> active_{0};         // items currently being executed
    bool shutdown_ = false;
};

}  // namespace vex
