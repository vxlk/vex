#pragma once

#include <cstdint>
#include <cstddef>
#include <atomic>
#include <vector>

namespace vex {

// Growable byte buffer backed by virtual memory reservation.
// The base pointer is stable for the lifetime of the object — it never
// moves, which makes it safe to hand out views (memoryview / numpy)
// while decode threads are still appending data.
//
// On Windows:  VirtualAlloc MEM_RESERVE + MEM_COMMIT
// On Linux:    mmap MAP_ANONYMOUS + mprotect
//
// Only physical pages that are actually written are committed, so a
// 256 MB reservation only costs address space, not RAM.

class VirtualBlob {
public:
    static constexpr size_t DEFAULT_RESERVATION = 256ULL * 1024 * 1024;  // 256 MB
    static constexpr size_t COMMIT_CHUNK        = 1ULL * 1024 * 1024;    // 1 MB

    uint8_t* data = nullptr;
    size_t size = 0;                       // bytes written (worker-only)
    size_t committed = 0;                  // bytes backed by physical pages
    size_t reserved = 0;                   // total virtual address range
    std::vector<int64_t> offsets;          // byte offset of each JPEG

    // Consumer-safe published size.  The worker stores this with
    // memory_order_release after writing JPEG data.  The consumer
    // loads it with memory_order_acquire to determine how many
    // bytes are safe to read.
    std::atomic<size_t> published_size{0};

    VirtualBlob() = default;
    ~VirtualBlob();

    VirtualBlob(VirtualBlob&& o) noexcept;
    VirtualBlob& operator=(VirtualBlob&& o) noexcept;

    VirtualBlob(const VirtualBlob&) = delete;
    VirtualBlob& operator=(const VirtualBlob&) = delete;

    // Reserve virtual address space.  Must be called before any writes.
    // Returns false on failure (out of address space / OS error).
    bool reserve(size_t reservation_size);

    // Ensure at least `needed` bytes are available past `size`.
    // Commits new pages in COMMIT_CHUNK increments.
    // Returns false if the request exceeds the reservation.
    bool ensure_remaining(size_t needed);

    // Publish the current size to consumers (atomic release).
    // Call this after writing data so peek_stream sees the new bytes.
    void publish() {
        published_size.store(size, std::memory_order_release);
    }

    // Reset to empty without releasing the reservation.
    // Used for HW retry / sequential fallback.
    void reset();

    // Release all virtual memory.
    void release();

private:
    void free_vm();

    static size_t page_size();
    static size_t round_up(size_t n, size_t alignment);
};

}  // namespace vex
