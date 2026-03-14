#pragma once

#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace vex {

/// Process-global persistent thread pool.
///
/// Workers stay alive across calls — only new threads are spawned when
/// capacity increases.  Eliminates per-call std::thread creation overhead.
class ThreadPool {
public:
    static ThreadPool& instance();

    /// Submit work: call func(id) for id in [0, count).
    /// Wakes (or spawns) workers and blocks the caller until all items
    /// have completed.
    void run(std::function<void(int)> func, int count);

    /// Ensure at least n worker threads exist.
    void ensure_capacity(int n);

    /// Shut down all workers (blocks until joined).
    void shutdown();

    ~ThreadPool();

private:
    ThreadPool() = default;
    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    void worker_loop();

    std::mutex mutex_;
    std::condition_variable work_cv_;     // workers wait on this
    std::condition_variable done_cv_;     // run() waits on this

    std::vector<std::thread> workers_;

    // Current batch
    std::function<void(int)> func_;
    int next_id_ = 0;
    int total_ = 0;
    int completed_ = 0;

    bool shutdown_ = false;
};

}  // namespace vex
