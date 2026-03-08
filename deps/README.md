# vex — Dependency Setup (Windows x64)

vex requires two native libraries: **FFmpeg** (shared/dev build) and **TurboJPEG** (libjpeg-turbo).
Place them in this `deps/` directory as described below.

---

## FFmpeg (shared/dev build)

The FFmpeg CLI tools in `deps/ffmpeg/bin/` are not sufficient for building vex. You need the
**shared development build** which includes headers (`.h`), import libraries (`.lib`),
and runtime DLLs (`.dll`).

### Download

1. Go to <https://github.com/BtbN/FFmpeg-Builds/releases>
2. Download **`ffmpeg-master-latest-win64-gpl-shared.zip`**
   (or the latest dated release, e.g., `ffmpeg-n7.1-latest-win64-gpl-shared.zip`)
3. Extract the archive. Inside you will find a folder like `ffmpeg-master-latest-win64-gpl-shared/`.

### Install

Copy the contents so the layout is:

```
deps/ffmpeg/
    include/
        libavcodec/
            avcodec.h
            ...
        libavformat/
            avformat.h
            ...
        libavutil/
            ...
        libswscale/
            ...
        libswresample/
            ...
    lib/
        avcodec.lib
        avformat.lib
        avutil.lib
        swscale.lib
        swresample.lib
        ...
    bin/
        avcodec-61.dll      (version number may vary)
        avformat-61.dll
        avutil-59.dll
        swscale-8.dll
        swresample-5.dll
        ...
```

The key directories from the extracted archive are:
- `include/` -- copy as-is to `deps/ffmpeg/include/`
- `lib/` -- copy as-is to `deps/ffmpeg/lib/`
- `bin/` -- copy as-is to `deps/ffmpeg/bin/`

Alternatively, set the `FFMPEG_DIR` environment variable to point to the extracted folder.

---

## TurboJPEG (libjpeg-turbo)

### Download

1. Go to <https://github.com/libjpeg-turbo/libjpeg-turbo/releases>
2. Download one of:
   - **`libjpeg-turbo-X.X.X-vc64.exe`** (Windows x64 installer), or
   - **`libjpeg-turbo-X.X.X-msvc.zip`** (portable zip)
3. Install or extract.

### Install

Copy the contents so the layout is:

```
deps/turbojpeg/
    include/
        turbojpeg.h
        jconfig.h
        ...
    lib/
        turbojpeg.lib          (dynamic) or
        turbojpeg-static.lib   (static — either works)
        ...
```

If you used the installer, the default install location is `C:\libjpeg-turbo64\`.
Copy the `include/` and `lib/` directories from there into `deps/turbojpeg/`.

Alternatively, set the `TURBOJPEG_DIR` environment variable to point to the install location.

---

## Build Commands

Once both dependencies are in place:

```bash
mkdir build && cd build
cmake .. -G "Visual Studio 17 2022" -A x64
cmake --build . --config Release
```

Or using Ninja (faster):

```bash
mkdir build && cd build
cmake .. -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build .
```

After a successful build, `python/vex/_vex_core.pyd` will be created along with the
required FFmpeg DLLs, and you can use vex from Python:

```python
import sys
sys.path.insert(0, "python")
from vex import batch_decode, LevelConfig
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| CMake says "FFmpeg not found" | Verify `deps/ffmpeg/include/libavcodec/avcodec.h` exists |
| CMake says "TurboJPEG not found" | Verify `deps/turbojpeg/include/turbojpeg.h` exists |
| Link errors for avcodec | Make sure you have the **shared** build (with `.lib` files), not just headers |
| DLL not found at runtime | The post-build step should copy DLLs; check `python/vex/` for `.dll` files |
| Python not found by CMake | Ensure Python 3.13+ is on PATH and has NumPy installed |
