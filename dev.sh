#!/usr/bin/env bash
# dev.sh — Development helper for the vex project.
# Usage: ./dev.sh <command>
#
# Commands:
#   build          Configure + build (Debug)
#   build-release  Configure + build (Release)
#   test           Run Python tests (builds Release first)
#   install        Create a venv, install deps, build Release, make vex importable
#   clean          Remove build directory
#   fixtures       Generate test video fixtures
#   run-example    Run all examples against fixtures

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$PROJECT_ROOT/build"
VENV_DIR="$PROJECT_ROOT/.venv"
PYTHON_PKG="$PROJECT_ROOT/python"

# ── Helpers ────────────────────────────────────────────────────────────────

log()  { echo "==> $*"; }
err()  { echo "ERROR: $*" >&2; exit 1; }

ensure_cmake() {
    command -v cmake >/dev/null 2>&1 || err "cmake not found. Install CMake >= 3.16."
}

ensure_venv() {
    if [ ! -d "$VENV_DIR" ]; then
        err "venv not found. Run './dev.sh install' first."
    fi
    source "$VENV_DIR/Scripts/activate" 2>/dev/null || source "$VENV_DIR/bin/activate" 2>/dev/null
}

cmake_configure() {
    local build_type="${1:-Release}"
    ensure_cmake
    mkdir -p "$BUILD_DIR"
    log "Configuring ($build_type)..."
    cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
        -G "Visual Studio 17 2022" -A x64
}

cmake_build() {
    local config="${1:-Release}"
    ensure_cmake
    if [ ! -f "$BUILD_DIR/vex.sln" ]; then
        cmake_configure "$config"
    fi
    log "Building ($config)..."
    cmake --build "$BUILD_DIR" --config "$config"
}

# ── Commands ───────────────────────────────────────────────────────────────

cmd_build() {
    cmake_configure Debug
    cmake_build Debug
    log "Debug build complete."
}

cmd_build_release() {
    cmake_configure Release
    cmake_build Release
    log "Release build complete."
}

cmd_test() {
    cmd_build_release
    log "Running tests..."
    if [ -d "$VENV_DIR" ]; then
        ensure_venv
    fi
    PYTHONPATH="$PYTHON_PKG" python -m pytest tests/ -v --tb=short
}

cmd_install() {
    log "Creating virtual environment in $VENV_DIR ..."
    python -m venv "$VENV_DIR"

    # Activate
    source "$VENV_DIR/Scripts/activate" 2>/dev/null || source "$VENV_DIR/bin/activate" 2>/dev/null

    log "Installing Python dependencies..."
    pip install --upgrade pip
    pip install numpy pytest

    log "Building Release..."
    cmd_build_release

    # Create a .pth file so vex is importable from the venv
    local site_pkgs
    site_pkgs=$(python -c "import site; print(site.getsitepackages()[0])")
    echo "$PYTHON_PKG" > "$site_pkgs/vex.pth"

    log "Install complete."
    log "Activate with:  source .venv/Scripts/activate  (or .venv/bin/activate on Linux/Mac)"
    log "Then:           python -c \"import vex; print(vex.__version__)\""
}

cmd_clean() {
    log "Removing build directory..."
    rm -rf "$BUILD_DIR"
    # Remove generated .pyd and DLLs from python/vex/
    rm -f "$PYTHON_PKG/vex/_vex_core"*.pyd
    rm -f "$PYTHON_PKG/vex/"*.dll
    log "Clean complete."
}

cmd_fixtures() {
    log "Generating test fixtures..."
    if [ -d "$VENV_DIR" ]; then
        ensure_venv
    fi
    python "$PROJECT_ROOT/tools/generate_fixtures.py" "$@"
}

cmd_run_example() {
    if [ -d "$VENV_DIR" ]; then
        ensure_venv
    fi
    log "Running examples..."
    for f in "$PROJECT_ROOT"/examples/*.py; do
        if [ -f "$f" ]; then
            log "Running $(basename "$f") ..."
            PYTHONPATH="$PYTHON_PKG" python "$f"
            echo
        fi
    done
    log "All examples complete."
}

# ── Dispatch ───────────────────────────────────────────────────────────────

case "${1:-help}" in
    build)          cmd_build ;;
    build-release)  cmd_build_release ;;
    test)           cmd_test ;;
    install)        cmd_install ;;
    clean)          cmd_clean ;;
    fixtures)       shift; cmd_fixtures "$@" ;;
    run-example)    cmd_run_example ;;
    help|--help|-h)
        echo "Usage: ./dev.sh <command>"
        echo ""
        echo "Commands:"
        echo "  build          Configure + build (Debug)"
        echo "  build-release  Configure + build (Release)"
        echo "  test           Run Python tests (builds Release first)"
        echo "  install        Create venv, install deps, build, make vex importable"
        echo "  clean          Remove build artifacts"
        echo "  fixtures       Generate test video fixtures"
        echo "  run-example    Run all examples"
        ;;
    *)
        err "Unknown command: $1. Run './dev.sh help' for usage."
        ;;
esac
