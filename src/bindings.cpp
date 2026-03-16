#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "orchestrator.h"
#include "async_handle.h"
#include "probe.h"
#include "thread_pool.h"

namespace py = pybind11;

namespace {

// ── Memory ownership: the capsule pattern ───────────────────────────────────
//
// numpy arrays need their backing memory to outlive the C++ function that
// created them.  A stack-local std::vector is destroyed on return, so we
// can't just hand numpy a pointer into it.
//
// The solution used throughout this file:
//
//   1. Move (or copy) the vector onto the heap with `new`.
//   2. Wrap the heap pointer in a py::capsule whose destructor calls
//      `delete` on it.
//   3. Pass the capsule as the numpy array's "base" object.
//
// Python's GC ref-counts the numpy array → capsule chain.  When the last
// Python reference to the array is dropped, the capsule destructor fires
// and frees the C++ vector.  No leak, no manual free, no prevent-copy —
// Python owns the memory from the moment the capsule is constructed.
//

// ── Helper: convert LevelResult vector + metrics to Python objects ──────────

py::dict metrics_to_dict(const vex::DecodeMetrics& m) {
    py::dict d;
    d["total_wall_us"] = m.total_wall_us;
    d["index_scan_us"] = m.index_scan_us;
    d["seek_us"] = m.seek_us;
    d["decode_us"] = m.decode_us;
    d["io_us"] = m.io_us;
    d["files_processed"] = m.files_processed;
    d["keyframes_decoded"] = m.keyframes_decoded;
    d["keyframes_skipped"] = m.keyframes_skipped;
    d["threads_used"] = m.threads_used;
    d["peak_decode_memory"] = m.peak_decode_memory;
    d["total_output_bytes"] = m.total_output_bytes;
    d["atlas_scratch_bytes"] = m.atlas_scratch_bytes;
    d["decode_fps"] = m.decode_fps();
    d["pipeline_fps"] = m.pipeline_fps();
    d["thread_utilization"] = m.thread_utilization();
    d["hw_accel_backend"] = m.hw_accel_backend;
    d["encoder"] = m.encoder;

    py::list level_list;
    for (const auto& lm : m.levels) {
        py::dict ld;
        ld["width"] = lm.width;
        ld["height"] = lm.height;
        ld["quality"] = lm.quality;
        ld["output_format"] = lm.output_format;
        ld["scale_us"] = lm.scale_us;
        ld["encode_us"] = lm.encode_us;
        ld["atlas_composite_us"] = lm.atlas_composite_us;
        ld["atlas_encode_us"] = lm.atlas_encode_us;
        ld["output_bytes"] = lm.output_bytes;
        ld["frame_count"] = lm.frame_count;
        ld["avg_jpeg_bytes"] = lm.avg_jpeg_bytes;
        level_list.append(ld);
    }
    d["levels"] = level_list;

    py::list file_list;
    for (const auto& fs : m.file_stats) {
        py::dict fd;
        fd["file_index"] = fs.file_index;
        fd["decode_us"] = fs.decode_us;
        fd["keyframe_count"] = fs.keyframe_count;
        fd["file_size_bytes"] = fs.file_size_bytes;
        fd["index_strategy"] = static_cast<int>(fs.index_strategy);
        fd["codec_name"] = fs.codec_name;
        fd["source_width"] = fs.source_width;
        fd["source_height"] = fs.source_height;
        fd["hw_accel_used"] = fs.hw_accel_used;
        if (!fs.frame_times.empty()) {
            // Copy (not move) — fs is a const ref.  Capsule destructor
            // deletes when Python drops the numpy array.
            auto* heap = new std::vector<double>(fs.frame_times);
            auto cap =
                py::capsule(heap, [](void* p) { delete static_cast<std::vector<double>*>(p); });
            fd["frame_times"] = py::array_t<double>({static_cast<py::ssize_t>(heap->size())},
                                                    {sizeof(double)}, heap->data(), cap);
        }
        file_list.append(fd);
    }
    d["file_stats"] = file_list;

    return d;
}

py::dict convert_sprite_atlas_result(vex::SpriteAtlasResult& sar) {
    py::dict d;

    // Move blob to heap.  Python (via capsule destructor) frees it when
    // the numpy array is garbage-collected.
    auto* blob_heap = new std::vector<uint8_t>(std::move(sar.blob));
    auto blob_cap =
        py::capsule(blob_heap, [](void* p) { delete static_cast<std::vector<uint8_t>*>(p); });
    d["blob"] = py::array_t<uint8_t>({static_cast<py::ssize_t>(blob_heap->size())}, {1},
                                     blob_heap->data(), blob_cap);

    // Same pattern for the offset table.
    auto* offs_heap = new std::vector<int64_t>(std::move(sar.offsets));
    auto offs_cap =
        py::capsule(offs_heap, [](void* p) { delete static_cast<std::vector<int64_t>*>(p); });
    d["offsets"] = py::array_t<int64_t>({static_cast<py::ssize_t>(offs_heap->size())},
                                        {sizeof(int64_t)}, offs_heap->data(), offs_cap);

    d["grid_w"] = sar.grid_w;
    d["grid_h"] = sar.grid_h;
    d["thumb_w"] = sar.thumb_w;
    d["thumb_h"] = sar.thumb_h;
    d["file_count"] = sar.file_count;
    return d;
}

py::dict convert_disk_result(vex::DiskResult& dr) {
    py::dict d;
    d["cache_path"] = dr.cache_path;
    d["frame_count"] = dr.frame_count;
    d["total_bytes"] = dr.total_bytes;

    // Heap + capsule: Python frees when the numpy array is collected.
    auto* meta_heap = new std::vector<std::array<int64_t, 3>>(std::move(dr.metadata));
    auto meta_cap = py::capsule(
        meta_heap, [](void* p) { delete static_cast<std::vector<std::array<int64_t, 3>>*>(p); });
    d["metadata"] =
        py::array_t<int64_t>({static_cast<py::ssize_t>(meta_heap->size()), py::ssize_t(3)},
                             {py::ssize_t(3 * sizeof(int64_t)), py::ssize_t(sizeof(int64_t))},
                             reinterpret_cast<const int64_t*>(meta_heap->data()), meta_cap);
    return d;
}

py::tuple convert_results(std::vector<vex::LevelResult>& results, vex::DecodeMetrics& metrics) {
    py::list result_list;

    for (auto& lr : results) {
        py::dict d;
        d["format"] =
            (lr.format == vex::OutputFormat::JPEG_STREAM) ? "jpeg_stream" : "sprite_atlas";
        d["in_memory"] = lr.in_memory;

        if (lr.jpeg_stream.has_value()) {
            auto& jsr = lr.jpeg_stream.value();

            // Each VirtualBlob produces TWO numpy arrays (data + offsets),
            // each needing independent lifetimes.  We split ownership:
            //
            //   offsets array → capsule owns a heap std::vector<int64_t>
            //                   (moved out of VirtualBlob::offsets)
            //
            //   data array   → capsule owns the VirtualBlob* itself
            //                   (destructor calls ~VirtualBlob → VirtualFree)
            //
            // Python GC can free them in any order — the offsets vector is
            // self-contained (moved out), and the VirtualBlob's VM pages
            // are released independently.
            py::list blob_list;
            py::list offsets_list;
            for (auto& blob : jsr.blobs) {
                auto* vb = new vex::VirtualBlob(std::move(blob));

                // Move offsets to their own heap vector so the numpy array
                // doesn't depend on the VirtualBlob's lifetime.
                auto* offs_heap = new std::vector<int64_t>(std::move(vb->offsets));
                auto offs_cap = py::capsule(
                    offs_heap, [](void* p) { delete static_cast<std::vector<int64_t>*>(p); });
                py::array_t<int64_t> offs({static_cast<py::ssize_t>(offs_heap->size())},
                                          {sizeof(int64_t)}, offs_heap->data(), offs_cap);
                offsets_list.append(offs);

                // Data array — capsule owns the VirtualBlob itself.
                // When Python drops the data array, capsule destructor
                // deletes vb → ~VirtualBlob → VirtualFree/munmap.
                if (vb->data && vb->size > 0) {
                    size_t sz = vb->size;
                    uint8_t* ptr = vb->data;
                    auto cap =
                        py::capsule(vb, [](void* p) { delete static_cast<vex::VirtualBlob*>(p); });
                    auto arr = py::array_t<uint8_t>({static_cast<py::ssize_t>(sz)}, {1}, ptr, cap);
                    blob_list.append(arr);
                } else {
                    // Empty blob (file failed to decode) — delete now,
                    // nothing to expose to Python.
                    delete vb;
                    blob_list.append(py::array_t<uint8_t>(0));
                }
            }

            d["blobs"] = blob_list;
            d["offsets"] = offsets_list;

            // Heap + capsule: Python frees when the numpy array is collected.
            if (!jsr.metadata.empty()) {
                auto* meta_heap = new std::vector<std::array<int64_t, 3>>(std::move(jsr.metadata));
                auto meta_cap = py::capsule(meta_heap, [](void* p) {
                    delete static_cast<std::vector<std::array<int64_t, 3>>*>(p);
                });
                d["metadata"] = py::array_t<int64_t>(
                    {static_cast<py::ssize_t>(meta_heap->size()), py::ssize_t(3)},
                    {py::ssize_t(3 * sizeof(int64_t)), py::ssize_t(sizeof(int64_t))},
                    reinterpret_cast<const int64_t*>(meta_heap->data()), meta_cap);
            } else {
                d["metadata"] = py::array_t<int64_t>(0);
            }
        }

        if (lr.sprite_atlas.has_value()) {
            d["atlas"] = convert_sprite_atlas_result(lr.sprite_atlas.value());
        }

        if (lr.disk.has_value()) {
            d["disk"] = convert_disk_result(lr.disk.value());
        }

        result_list.append(d);
    }

    py::dict metrics_dict = metrics_to_dict(metrics);
    return py::make_tuple(result_list, metrics_dict);
}

}  // anonymous namespace

// ── Python module ───────────────────────────────────────────────────────────

PYBIND11_MODULE(_vex_core, m) {
    m.doc() = "vex: high-performance video thumbnail extractor";

    // ── IndexStrategy enum ──────────────────────────────────────────────────
    py::enum_<vex::IndexStrategy>(m, "IndexStrategy")
        .value("CONTAINER_INDEX", vex::IndexStrategy::CONTAINER_INDEX)
        .value("PACKET_SCAN", vex::IndexStrategy::PACKET_SCAN)
        .value("DECODE_SCAN", vex::IndexStrategy::DECODE_SCAN)
        .value("FORCED_INTERVAL", vex::IndexStrategy::FORCED_INTERVAL)
        .value("SKIPPED", vex::IndexStrategy::SKIPPED)
        .export_values();

    // ── TimestampStrategy enum ───────────────────────────────────────────────
    py::enum_<vex::TimestampStrategy>(m, "TimestampStrategy")
        .value("SAMPLE_TABLE", vex::TimestampStrategy::SAMPLE_TABLE)
        .value("BLOCK_TIMESTAMP", vex::TimestampStrategy::BLOCK_TIMESTAMP)
        .value("PES_TIMESTAMP", vex::TimestampStrategy::PES_TIMESTAMP)
        .value("TAG_TIMESTAMP", vex::TimestampStrategy::TAG_TIMESTAMP)
        .value("FIXED_RATE", vex::TimestampStrategy::FIXED_RATE)
        .value("GENERIC_PTS", vex::TimestampStrategy::GENERIC_PTS)
        .value("LINEAR_FALLBACK", vex::TimestampStrategy::LINEAR_FALLBACK)
        .export_values();

    // ── OutputFormat enum ───────────────────────────────────────────────────
    py::enum_<vex::OutputFormat>(m, "OutputFormat")
        .value("JPEG_STREAM", vex::OutputFormat::JPEG_STREAM)
        .value("SPRITE_ATLAS", vex::OutputFormat::SPRITE_ATLAS)
        .export_values();

    // ── LevelConfig ─────────────────────────────────────────────────────────
    py::class_<vex::LevelConfig>(m, "LevelConfig")
        .def(py::init([](int width, int height, int quality, const std::string& output,
                         bool in_memory, const std::string& cache_path, int atlas_columns) {
                 vex::LevelConfig lc{};
                 lc.width = width;
                 lc.height = height;
                 lc.quality = quality;
                 lc.in_memory = in_memory;
                 lc.cache_path = cache_path;
                 lc.atlas_columns = atlas_columns;
                 if (output == "sprite_atlas") {
                     lc.output = vex::OutputFormat::SPRITE_ATLAS;
                 } else {
                     lc.output = vex::OutputFormat::JPEG_STREAM;
                 }
                 return lc;
             }),
             py::arg("width") = 0, py::arg("height") = 0, py::arg("quality") = 100,
             py::arg("output") = "jpeg_stream", py::arg("in_memory") = true,
             py::arg("cache_path") = "", py::arg("atlas_columns") = 10)
        .def_readwrite("width", &vex::LevelConfig::width)
        .def_readwrite("height", &vex::LevelConfig::height)
        .def_readwrite("quality", &vex::LevelConfig::quality)
        .def_readwrite("output", &vex::LevelConfig::output)
        .def_readwrite("in_memory", &vex::LevelConfig::in_memory)
        .def_readwrite("cache_path", &vex::LevelConfig::cache_path)
        .def_readwrite("atlas_columns", &vex::LevelConfig::atlas_columns);

    // ── batch_decode (synchronous) ──────────────────────────────────────────
    m.def(
        "batch_decode",
        [](const std::vector<std::string>& paths, const std::vector<vex::LevelConfig>& levels,
           int max_threads, bool keyframes_only, int frame_skip, bool use_hw_accel,
           bool collect_frame_times) -> py::tuple {
            vex::BatchConfig cfg{};
            cfg.paths = paths;
            cfg.levels = levels;
            cfg.max_threads = max_threads;
            cfg.keyframes_only = keyframes_only;
            cfg.frame_skip = frame_skip;
            cfg.use_hw_accel = use_hw_accel;
            cfg.collect_frame_times = collect_frame_times;

            std::vector<vex::LevelResult> results;
            vex::DecodeMetrics metrics;

            {
                py::gil_scoped_release release;
                auto pair = vex::Orchestrator::batch_decode(cfg);
                results = std::move(pair.first);
                metrics = std::move(pair.second);
            }

            // GIL is re-acquired here; safe to create Python objects
            return convert_results(results, metrics);
        },
        py::arg("paths"), py::arg("levels"), py::arg("max_threads") = 0,
        py::arg("keyframes_only") = true, py::arg("frame_skip") = 1, py::arg("use_hw_accel") = true,
        py::arg("collect_frame_times") = false,
        "Decode video keyframes synchronously. Returns (results_list, metrics_dict).");

    // ── DecodeHandle wrapper ────────────────────────────────────────────────
    py::class_<vex::DecodeHandle, std::shared_ptr<vex::DecodeHandle>>(m, "DecodeHandle")
        .def("drain_events",
             [](vex::DecodeHandle& self) -> py::list {
                 std::vector<vex::FrameEvent> events;
                 {
                     py::gil_scoped_release release;
                     events = self.drain_events();
                 }
                 py::list result;
                 for (const auto& evt : events) {
                     py::dict d;
                     d["file_index"] = evt.file_index;
                     d["frame_index"] = evt.frame_index;
                     d["level_index"] = evt.level_index;
                     d["pts_ms"] = evt.pts_ms;
                     d["blob_offset"] = evt.blob_offset;
                     d["jpeg_size"] = evt.jpeg_size;
                     result.append(d);
                 }
                 return result;
             })
        .def("peek_jpeg",
             [](vex::DecodeHandle& self, py::dict evt_dict) -> py::memoryview {
                 vex::FrameEvent evt{};
                 evt.file_index = evt_dict["file_index"].cast<int>();
                 evt.frame_index = evt_dict["frame_index"].cast<int>();
                 evt.level_index = evt_dict["level_index"].cast<int>();
                 evt.pts_ms = evt_dict["pts_ms"].cast<int64_t>();
                 evt.blob_offset = evt_dict["blob_offset"].cast<uint32_t>();
                 evt.jpeg_size = evt_dict["jpeg_size"].cast<uint32_t>();

                 size_t out_size = 0;
                 const uint8_t* data = self.peek_jpeg_data(evt, &out_size);
                 if (!data || out_size == 0) {
                     return py::memoryview::from_memory(static_cast<const void*>(""), 0);
                 }
                 return py::memoryview::from_memory(static_cast<const void*>(data),
                                                    static_cast<py::ssize_t>(out_size));
             })
        .def(
            "peek_stream",
            [](vex::DecodeHandle& self, int file_index, int level_index) -> py::object {
                auto sv = self.peek_stream(file_index, level_index);
                if (!sv.data || sv.current_size == 0) {
                    return py::none();
                }
                // Return a read-only memoryview into the stable VirtualBlob
                // memory.  Safe as long as the DecodeHandle is alive (the
                // context_keepalive shared_ptr prevents the VirtualBlob from
                // being freed).
                return py::memoryview::from_memory(static_cast<const void*>(sv.data),
                                                   static_cast<py::ssize_t>(sv.current_size));
            },
            py::arg("file_index"), py::arg("level_index"),
            "Zero-copy view into the streaming JPEG buffer.")
        .def_property_readonly("progress",
                               [](vex::DecodeHandle& self) -> py::dict {
                                   auto p = self.progress();
                                   py::dict d;
                                   d["keyframes_decoded"] = p.keyframes_decoded;
                                   d["files_completed"] = p.files_completed;
                                   d["total_files"] = p.total_files;
                                   return d;
                               })
        .def_property_readonly("done",
                               [](vex::DecodeHandle& self) -> bool { return self.is_done(); })
        .def(
            "peek_frame_times",
            [](vex::DecodeHandle& self, int file_index) -> py::object {
                std::vector<double> times;
                {
                    py::gil_scoped_release release;
                    times = self.peek_frame_times(file_index);
                }
                if (times.empty())
                    return py::none();
                // Heap + capsule: Python frees when the array is collected.
                auto* heap = new std::vector<double>(std::move(times));
                auto cap =
                    py::capsule(heap, [](void* p) { delete static_cast<std::vector<double>*>(p); });
                return py::object(py::array_t<double>({static_cast<py::ssize_t>(heap->size())},
                                                      {sizeof(double)}, heap->data(), cap));
            },
            py::arg("file_index"), "Snapshot of frame times collected so far for a file.")
        .def("result", [](vex::DecodeHandle& self) -> py::tuple {
            self.wait_until_done();
            auto pair = self.get_results();
            return convert_results(pair.first, pair.second);
        });

    // ── batch_decode_async ──────────────────────────────────────────────────
    m.def(
        "batch_decode_async",
        [](const std::vector<std::string>& paths, const std::vector<vex::LevelConfig>& levels,
           int max_threads, bool keyframes_only, int frame_skip, bool use_hw_accel,
           size_t blob_reservation, bool collect_frame_times,
           py::object probe_info_obj) -> std::shared_ptr<vex::DecodeHandle> {
            vex::BatchConfig cfg{};
            cfg.paths = paths;
            cfg.levels = levels;
            cfg.max_threads = max_threads;
            cfg.keyframes_only = keyframes_only;
            cfg.frame_skip = frame_skip;
            cfg.use_hw_accel = use_hw_accel;
            cfg.blob_reservation = blob_reservation;
            cfg.collect_frame_times = collect_frame_times;

            // Parse optional probe_info list
            if (!probe_info_obj.is_none()) {
                py::list pi_list = probe_info_obj.cast<py::list>();
                cfg.probe_info.resize(pi_list.size());
                for (size_t i = 0; i < pi_list.size(); ++i) {
                    py::object item = pi_list[i];
                    if (item.is_none())
                        continue;
                    // Accept dict or object with codec_id/width/height attrs
                    if (py::isinstance<py::dict>(item)) {
                        py::dict d = item.cast<py::dict>();
                        if (d.contains("codec_id"))
                            cfg.probe_info[i].codec_id = d["codec_id"].cast<int>();
                        if (d.contains("width"))
                            cfg.probe_info[i].width = d["width"].cast<int>();
                        if (d.contains("height"))
                            cfg.probe_info[i].height = d["height"].cast<int>();
                    } else {
                        if (py::hasattr(item, "codec_id"))
                            cfg.probe_info[i].codec_id = item.attr("codec_id").cast<int>();
                        if (py::hasattr(item, "width"))
                            cfg.probe_info[i].width = item.attr("width").cast<int>();
                        if (py::hasattr(item, "height"))
                            cfg.probe_info[i].height = item.attr("height").cast<int>();
                    }
                }
            }

            py::gil_scoped_release release;
            return vex::Orchestrator::batch_decode_async(cfg);
        },
        py::arg("paths"), py::arg("levels"), py::arg("max_threads") = 0,
        py::arg("keyframes_only") = true, py::arg("frame_skip") = 1, py::arg("use_hw_accel") = true,
        py::arg("blob_reservation") = 0, py::arg("collect_frame_times") = false,
        py::arg("probe_info") = py::none(),
        "Decode video keyframes asynchronously. Returns DecodeHandle.");

    // ── probe (standalone probe: timestamps + codec info + thread count) ────
    auto probe_impl = [](const std::string& path) -> py::dict {
        vex::ProbeResult result;
        {
            py::gil_scoped_release release;
            result = vex::probe_video(path);
        }
        py::dict d;
        // Heap + capsule: Python frees when the numpy array is collected.
        auto* heap = new std::vector<double>(std::move(result.times_sec));
        auto cap = py::capsule(heap, [](void* p) { delete static_cast<std::vector<double>*>(p); });
        d["times"] = py::array_t<double>({static_cast<py::ssize_t>(heap->size())}, {sizeof(double)},
                                         heap->data(), cap);
        d["frame_count"] = result.frame_count;
        d["duration_sec"] = result.duration_sec;
        d["fps"] = result.fps;
        d["strategy"] = static_cast<int>(result.strategy);
        d["container"] = result.container;
        d["codec"] = result.codec;
        d["codec_id"] = result.codec_id;
        d["width"] = result.width;
        d["height"] = result.height;
        d["file_size"] = result.file_size;
        d["decode_threads"] = result.decode_threads;
        return d;
    };

    m.def("probe", probe_impl, py::arg("path"),
          "Probe a video file: timestamps, codec info, resolution, decode thread count.");

    // Shutdown persistent thread pool on module unload.
    auto cleanup = py::capsule(&vex::ThreadPool::instance(),
                               [](void*) { vex::ThreadPool::instance().shutdown(); });
    m.add_object("_thread_pool_cleanup", cleanup);
}
