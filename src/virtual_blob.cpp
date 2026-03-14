#include "virtual_blob.h"

#include <algorithm>
#include <cstring>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace vex {

// ── Platform helpers ────────────────────────────────────────────────────────

size_t VirtualBlob::page_size() {
    static size_t ps = [] {
#ifdef _WIN32
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return static_cast<size_t>(si.dwAllocationGranularity);  // 64 KB on Windows
#else
        long p = sysconf(_SC_PAGESIZE);
        return p > 0 ? static_cast<size_t>(p) : 4096;
#endif
    }();
    return ps;
}

size_t VirtualBlob::round_up(size_t n, size_t alignment) {
    return (n + alignment - 1) & ~(alignment - 1);
}

// ── Lifecycle ───────────────────────────────────────────────────────────────

VirtualBlob::~VirtualBlob() {
    free_vm();
}

VirtualBlob::VirtualBlob(VirtualBlob&& o) noexcept
    : data(o.data),
      size(o.size),
      committed(o.committed),
      reserved(o.reserved),
      offsets(std::move(o.offsets)),
      published_size(o.published_size.load(std::memory_order_relaxed)),
      retired_(std::move(o.retired_)) {
    o.data = nullptr;
    o.size = 0;
    o.committed = 0;
    o.reserved = 0;
}

VirtualBlob& VirtualBlob::operator=(VirtualBlob&& o) noexcept {
    if (this != &o) {
        free_vm();
        data = o.data;
        size = o.size;
        committed = o.committed;
        reserved = o.reserved;
        offsets = std::move(o.offsets);
        published_size.store(o.published_size.load(std::memory_order_relaxed),
                             std::memory_order_relaxed);
        retired_ = std::move(o.retired_);
        o.data = nullptr;
        o.size = 0;
        o.committed = 0;
        o.reserved = 0;
    }
    return *this;
}

// ── Reserve / commit / grow / release ───────────────────────────────────────

bool VirtualBlob::reserve(size_t reservation_size) {
    free_vm();

    size_t ps = page_size();
    reservation_size = round_up(reservation_size, ps);

#ifdef _WIN32
    data =
        static_cast<uint8_t*>(VirtualAlloc(nullptr, reservation_size, MEM_RESERVE, PAGE_NOACCESS));
#else
    void* p = mmap(nullptr, reservation_size, PROT_NONE,
                   MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    data = (p == MAP_FAILED) ? nullptr : static_cast<uint8_t*>(p);
#endif

    if (!data)
        return false;

    reserved = reservation_size;
    committed = 0;
    size = 0;
    published_size.store(0, std::memory_order_relaxed);
    offsets.clear();
    return true;
}

bool VirtualBlob::ensure_remaining(size_t needed) {
    size_t required = size + needed;
    if (required <= committed)
        return true;

    // Need to grow: allocate a new 2x region, copy, retire the old one.
    if (required > reserved) {
        size_t new_min = std::max(reserved * 2, required);
        if (!grow(new_min))
            return false;
        // After grow, reserved/committed/data are updated.
        // Fall through to commit logic in case grow only committed
        // enough for the copied data and we need more.
        if (required <= committed)
            return true;
    }

    // Commit in COMMIT_CHUNK increments, rounded up to page size
    size_t ps = page_size();
    size_t new_committed = round_up(std::max(required, committed + COMMIT_CHUNK), ps);
    if (new_committed > reserved)
        new_committed = reserved;

    size_t commit_bytes = new_committed - committed;

#ifdef _WIN32
    void* result = VirtualAlloc(data + committed, commit_bytes, MEM_COMMIT, PAGE_READWRITE);
    if (!result)
        return false;
#else
    if (mprotect(data + committed, commit_bytes, PROT_READ | PROT_WRITE) != 0)
        return false;
#endif

    committed = new_committed;
    return true;
}

bool VirtualBlob::grow(size_t new_min_reserved) {
    size_t ps = page_size();
    size_t new_reserved = round_up(new_min_reserved, ps);

    // 1. Reserve new region
    uint8_t* new_data = nullptr;
#ifdef _WIN32
    new_data =
        static_cast<uint8_t*>(VirtualAlloc(nullptr, new_reserved, MEM_RESERVE, PAGE_NOACCESS));
#else
    void* p =
        mmap(nullptr, new_reserved, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    new_data = (p == MAP_FAILED) ? nullptr : static_cast<uint8_t*>(p);
#endif
    if (!new_data)
        return false;

    // 2. Commit pages covering existing data
    size_t new_committed = 0;
    if (size > 0) {
        new_committed = round_up(size, ps);
        if (new_committed > new_reserved)
            new_committed = new_reserved;

#ifdef _WIN32
        void* result = VirtualAlloc(new_data, new_committed, MEM_COMMIT, PAGE_READWRITE);
        if (!result) {
            VirtualFree(new_data, 0, MEM_RELEASE);
            return false;
        }
#else
        if (mprotect(new_data, new_committed, PROT_READ | PROT_WRITE) != 0) {
            munmap(new_data, new_reserved);
            return false;
        }
#endif

        // 3. Copy existing data
        std::memcpy(new_data, data, size);
    }

    // 4. Retire old region (keep mapped for outstanding memoryviews)
    if (data) {
        retired_.push_back({data, reserved});
    }

    // 5. Swap in new region
    data = new_data;
    committed = new_committed;
    reserved = new_reserved;
    // size, published_size, offsets unchanged

    return true;
}

void VirtualBlob::reset() {
    size = 0;
    published_size.store(0, std::memory_order_release);
    offsets.clear();
    free_retired();
    // Committed pages on current region stay — we'll reuse them.
}

void VirtualBlob::release() {
    free_vm();
}

void VirtualBlob::free_retired() {
    for (auto& r : retired_) {
        if (r.base) {
#ifdef _WIN32
            VirtualFree(r.base, 0, MEM_RELEASE);
#else
            munmap(r.base, r.reserved_size);
#endif
        }
    }
    retired_.clear();
}

void VirtualBlob::free_vm() {
    if (!data) {
        free_retired();
        return;
    }

#ifdef _WIN32
    VirtualFree(data, 0, MEM_RELEASE);
#else
    munmap(data, reserved);
#endif

    data = nullptr;
    size = 0;
    committed = 0;
    reserved = 0;
    published_size.store(0, std::memory_order_relaxed);
    offsets.clear();
    free_retired();
}

}  // namespace vex
