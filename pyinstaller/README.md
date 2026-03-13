# PyInstaller Integration for Vex

Vex ships a PyInstaller hook that collects the compiled C++ extension (`_vex_core.pyd`) and its runtime DLLs (FFmpeg + TurboJPEG) so they are bundled into your frozen application automatically.

## Quick Start

Point PyInstaller at the hook directory — either on the command line or in your `.spec` file.

### CLI

```bash
pyinstaller --additional-hooks-dir=path/to/vex/pyinstaller  your_app.py
```

### .spec file

```python
a = Analysis(
    ["your_app.py"],
    hookspath=["path/to/vex/pyinstaller"],
    # ... your other settings ...
)
```

That's it. The hook handles DLL collection and hidden imports.

## What the Hook Does

1. **Collects binaries** — `collect_dynamic_libs("vex")` finds `_vex_core.pyd` plus the six DLLs that live next to it (`avcodec-*.dll`, `avformat-*.dll`, `avutil-*.dll`, `swscale-*.dll`, `swresample-*.dll`, `turbojpeg.dll`) and adds them to the bundle.

2. **Declares hidden imports** — `vex._vex_core` is loaded at runtime (not via a top-level `import`), so PyInstaller's static analysis can't see it. The hook adds it explicitly, along with `numpy`.

3. **Frozen-app DLL loading** — `vex.__init__` detects when it's running inside a PyInstaller bundle (`sys._MEIPASS`) and adds that directory to the Windows DLL search path so the bundled DLLs are found at import time.

## Minimal .spec Example

```python
# -*- mode: python ; coding: utf-8 -*-

a = Analysis(
    ["my_app.py"],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
    hookspath=["path/to/vex/pyinstaller"],  # <-- add this line
    # ...
)

pyz = PYZ(a.pure)

exe = EXE(pyz, a.scripts, [], name="my_app", console=True)

# For --onedir output:
coll = COLLECT(exe, a.binaries, a.datas, name="my_app")
```

## onedir vs onefile

| Mode | Startup | Disk |
|---|---|---|
| `--onedir` (default) | Fast — DLLs are already on disk | ~128 MB folder (FFmpeg DLLs are large) |
| `--onefile` | Slow first launch — unpacks ~128 MB of DLLs to a temp directory every time | Single `.exe` |

**Recommendation:** Use `--onedir` unless you have a specific reason to ship a single file. The FFmpeg and TurboJPEG DLLs total roughly 128 MB, so `--onefile` adds noticeable startup latency (several seconds) on every launch while the temporary extraction completes.

## Troubleshooting

### `ImportError: DLL load failed` or `FileNotFoundError` for a DLL

The DLLs were not collected into the bundle, or they ended up in the wrong subdirectory.

- Verify the hook is being picked up: run `pyinstaller --additional-hooks-dir=... --debug=imports your_app.py` and look for `hook-vex` in the output.
- Check that your installed `vex` package directory contains `_vex_core.pyd` **and** the six DLLs. If you installed from source, make sure `cmake --install` (or your copy step) placed them next to the `.pyd`.

### `ModuleNotFoundError: No module named 'vex._vex_core'`

PyInstaller didn't detect the hidden import.

- Confirm the hook directory path is correct and contains `hook-vex.py`.
- As a workaround, add `--hidden-import=vex._vex_core` to the CLI.

### `ImportError: numpy` not found

Add `--hidden-import=numpy` or make sure numpy is installed in the environment PyInstaller is running from.

### `--onefile` is very slow to start

This is expected — see [onedir vs onefile](#onedir-vs-onefile) above. Switch to `--onedir` for fast startup.
