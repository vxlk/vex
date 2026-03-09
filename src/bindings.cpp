#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include "orchestrator.h"
#include "async_handle.h"

namespace py = pybind11;

namespace {

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
        file_list.append(fd);
    }
    d["file_stats"] = file_list;

    return d;
}

py::dict convert_sprite_atlas_result(vex::SpriteAtlasResult& sar) {
    py::dict d;

    // Move blob to heap — capsule takes ownership, zero-copy to numpy
    auto* blob_heap = new std::vector<uint8_t>(std::move(sar.blob));
    auto blob_cap =
        py::capsule(blob_heap, [](void* p) { delete static_cast<std::vector<uint8_t>*>(p); });
    d["blob"] = py::array_t<uint8_t>({static_cast<py::ssize_t>(blob_heap->size())}, {1},
                                     blob_heap->data(), blob_cap);

    // Move offsets to heap — capsule takes ownership, zero-copy
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

    // Move metadata to heap — capsule takes ownership, zero-copy
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

            // Move each VirtualBlob to the heap so a capsule can own it.
            // The capsule destructor calls ~VirtualBlob which releases
            // the VM reservation (VirtualFree / munmap).
            py::list blob_list;
            py::list offsets_list;
            for (auto& blob : jsr.blobs) {
                auto* vb = new vex::VirtualBlob(std::move(blob));

                // Offsets array — points into vb->offsets (capsule prevents
                // the VirtualBlob from being destroyed while numpy is alive)
                auto* offs_heap = new std::vector<int64_t>(std::move(vb->offsets));
                auto offs_cap = py::capsule(
                    offs_heap, [](void* p) { delete static_cast<std::vector<int64_t>*>(p); });
                py::array_t<int64_t> offs({static_cast<py::ssize_t>(offs_heap->size())},
                                          {sizeof(int64_t)}, offs_heap->data(), offs_cap);
                offsets_list.append(offs);

                // Data array — capsule owns the VirtualBlob
                if (vb->data && vb->size > 0) {
                    size_t sz = vb->size;
                    uint8_t* ptr = vb->data;
                    auto cap = py::capsule(
                        vb, [](void* p) { delete static_cast<vex::VirtualBlob*>(p); });
                    auto arr = py::array_t<uint8_t>({static_cast<py::ssize_t>(sz)}, {1}, ptr, cap);
                    blob_list.append(arr);
                } else {
                    delete vb;
                    blob_list.append(py::array_t<uint8_t>(0));
                }
            }

            d["blobs"] = blob_list;
            d["offsets"] = offsets_list;

            // Move metadata to heap — capsule ownership, zero-copy
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
           int max_threads, bool keyframes_only, int frame_skip, bool use_hw_accel) -> py::tuple {
            vex::BatchConfig cfg{};
            cfg.paths = paths;
            cfg.levels = levels;
            cfg.max_threads = max_threads;
            cfg.keyframes_only = keyframes_only;
            cfg.frame_skip = frame_skip;
            cfg.use_hw_accel = use_hw_accel;

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
        py::arg("paths"), py::arg("levels"), py::arg("max_threads") = 8,
        py::arg("keyframes_only") = true, py::arg("frame_skip") = 1, py::arg("use_hw_accel") = true,
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
        .def("peek_stream",
             [](vex::DecodeHandle& self, int file_index,
                int level_index) -> py::object {
                 auto sv = self.peek_stream(file_index, level_index);
                 if (!sv.data || sv.current_size == 0) {
                     return py::none();
                 }
                 // Return a read-only memoryview into the stable VirtualBlob
                 // memory.  Safe as long as the DecodeHandle is alive (the
                 // context_keepalive shared_ptr prevents the VirtualBlob from
                 // being freed).
                 return py::memoryview::from_memory(
                     static_cast<const void*>(sv.data),
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
        .def("result", [](vex::DecodeHandle& self) -> py::tuple {
            self.wait_until_done();
            auto pair = self.get_results();
            return convert_results(pair.first, pair.second);
        });

    // ── batch_decode_async ──────────────────────────────────────────────────
    m.def(
        "batch_decode_async",
        [](const std::vector<std::string>& paths, const std::vector<vex::LevelConfig>& levels,
           int max_threads, bool keyframes_only, int frame_skip,
           bool use_hw_accel, size_t blob_reservation) -> std::shared_ptr<vex::DecodeHandle> {
            vex::BatchConfig cfg{};
            cfg.paths = paths;
            cfg.levels = levels;
            cfg.max_threads = max_threads;
            cfg.keyframes_only = keyframes_only;
            cfg.frame_skip = frame_skip;
            cfg.use_hw_accel = use_hw_accel;
            cfg.blob_reservation = blob_reservation;

            py::gil_scoped_release release;
            return vex::Orchestrator::batch_decode_async(cfg);
        },
        py::arg("paths"), py::arg("levels"), py::arg("max_threads") = 8,
        py::arg("keyframes_only") = true, py::arg("frame_skip") = 1,
        py::arg("use_hw_accel") = true, py::arg("blob_reservation") = 0,
        "Decode video keyframes asynchronously. Returns DecodeHandle.");
}
