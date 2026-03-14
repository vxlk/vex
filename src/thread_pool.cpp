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
        int my_id;
        std::function<void(int)> fn;

        {
            std::unique_lock<std::mutex> lock(mutex_);
            work_cv_.wait(lock, [this] { return shutdown_ || next_id_ < total_; });

            if (shutdown_)
                return;

            my_id = next_id_++;
            fn = func_;
        }

        fn(my_id);

        {
            std::lock_guard<std::mutex> lock(mutex_);
            completed_++;
            if (completed_ >= total_)
                done_cv_.notify_one();
        }
    }
}

void ThreadPool::ensure_capacity(int n) {
    std::lock_guard<std::mutex> lock(mutex_);
    while (static_cast<int>(workers_.size()) < n) {
        workers_.emplace_back(&ThreadPool::worker_loop, this);
    }
}

void ThreadPool::run(std::function<void(int)> func, int count) {
    if (count <= 0)
        return;

    ensure_capacity(count);

    {
        std::lock_guard<std::mutex> lock(mutex_);
        func_ = std::move(func);
        next_id_ = 0;
        total_ = count;
        completed_ = 0;
    }

    work_cv_.notify_all();

    {
        std::unique_lock<std::mutex> lock(mutex_);
        done_cv_.wait(lock, [this] { return completed_ >= total_; });
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
