"""PyInstaller hook for the ``vex`` package.

Ensures that:

1. The compiled C++ extension (``_vex_core.pyd``) and the six runtime DLLs
   it depends on (FFmpeg libs + TurboJPEG) are collected into the frozen
   application.  ``collect_dynamic_libs`` walks the installed package
   directory looking for ``.pyd`` and ``.dll`` files.

2. Hidden imports are declared so that PyInstaller's analysis picks up
   ``vex._vex_core`` (a C extension that is imported at runtime via
   ``importlib`` / ``__import__`` rather than a top-level ``import``
   statement) and ``numpy`` (used by the public API but potentially
   missed when the user's own code doesn't import it directly).

Usage — add this directory to PyInstaller's hook search path:

    # CLI
    pyinstaller --additional-hooks-dir=path/to/vex/pyinstaller  app.py

    # .spec file
    a = Analysis(
        ...
        hookspath=["path/to/vex/pyinstaller"],
    )
"""

from PyInstaller.utils.hooks import collect_dynamic_libs

# Collect _vex_core.pyd + avcodec-*.dll, avformat-*.dll, avutil-*.dll,
# swscale-*.dll, swresample-*.dll, and turbojpeg.dll that sit next to
# the .pyd inside the installed vex package directory.
datas = []
binaries = collect_dynamic_libs("vex")

hiddenimports = [
    "vex._vex_core",  # C extension loaded at runtime by _load_native()
    "numpy",          # used by the public API for blob / offset arrays
]
