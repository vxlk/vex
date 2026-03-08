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
#   bench          Run speed comparison vs FFmpeg and generate chart
#   run-example    Run all examples against fixtures
#   lint           Run C++ and Python linters (read-only checks)
#   fmt            Auto-format C++ and Python code
#   check          Run lint + build with warnings-as-errors

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

    log "Building Release..."
    cmd_build_release

    log "Installing vex package (editable)..."
    pip install -e ".[test]"

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

cmd_bench() {
    if [ -d "$VENV_DIR" ]; then
        ensure_venv
    fi
    log "Running benchmark (vex vs FFmpeg)..."
    PYTHONPATH="$PYTHON_PKG" python "$PROJECT_ROOT/tools/benchmark.py" "$@"
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

cmd_lint() {
    local rc=0

    # ── C++: clang-format dry-run ──
    log "Checking C++ formatting (clang-format)..."
    local cpp_files
    cpp_files=$(find "$PROJECT_ROOT/src" -type f \( -name '*.cpp' -o -name '*.h' \))
    if [ -n "$cpp_files" ]; then
        if ! echo "$cpp_files" | xargs clang-format --dry-run --Werror 2>&1; then
            log "C++ formatting issues found. Run './dev.sh fmt' to fix."
            rc=1
        fi
    fi

    # ── Python: ruff lint ──
    log "Checking Python (ruff lint)..."
    if ! ruff check "$PROJECT_ROOT/python" "$PROJECT_ROOT/tools" "$PROJECT_ROOT/examples" "$PROJECT_ROOT/tests" 2>&1; then
        rc=1
    fi

    # ── Python: ruff format check ──
    log "Checking Python formatting (ruff format)..."
    if ! ruff format --check "$PROJECT_ROOT/python" "$PROJECT_ROOT/tools" "$PROJECT_ROOT/examples" "$PROJECT_ROOT/tests" 2>&1; then
        log "Python formatting issues found. Run './dev.sh fmt' to fix."
        rc=1
    fi

    if [ $rc -eq 0 ]; then
        log "All lint checks passed."
    else
        log "Lint checks failed."
        exit 1
    fi
}

cmd_fmt() {
    # ── C++: clang-format ──
    log "Formatting C++ (clang-format)..."
    local cpp_files
    cpp_files=$(find "$PROJECT_ROOT/src" -type f \( -name '*.cpp' -o -name '*.h' \))
    if [ -n "$cpp_files" ]; then
        echo "$cpp_files" | xargs clang-format -i
    fi

    # ── Python: ruff ──
    log "Formatting Python (ruff)..."
    ruff check --fix "$PROJECT_ROOT/python" "$PROJECT_ROOT/tools" "$PROJECT_ROOT/examples" "$PROJECT_ROOT/tests" 2>&1 || true
    ruff format "$PROJECT_ROOT/python" "$PROJECT_ROOT/tools" "$PROJECT_ROOT/examples" "$PROJECT_ROOT/tests"

    log "Formatting complete."
}

cmd_check() {
    # Step 1: lint
    cmd_lint

    # Step 2: build with warnings-as-errors
    log "Building with warnings-as-errors..."
    ensure_cmake
    mkdir -p "$BUILD_DIR"
    cmake -S "$PROJECT_ROOT" -B "$BUILD_DIR" \
        -G "Visual Studio 17 2022" -A x64 \
        -DCMAKE_CXX_FLAGS="/WX"
    cmake --build "$BUILD_DIR" --config Release -- /p:TreatWarningsAsErrors=true

    log "All checks passed."
}

# ── Dispatch ───────────────────────────────────────────────────────────────

case "${1:-help}" in
    build)          cmd_build ;;
    build-release)  cmd_build_release ;;
    test)           cmd_test ;;
    install)        cmd_install ;;
    clean)          cmd_clean ;;
    fixtures)       shift; cmd_fixtures "$@" ;;
    bench)          shift; cmd_bench "$@" ;;
    run-example)    cmd_run_example ;;
    lint)           cmd_lint ;;
    fmt)            cmd_fmt ;;
    check)          cmd_check ;;
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
        echo "  bench          Run speed comparison vs FFmpeg and generate chart"
        echo "  run-example    Run all examples"
        echo "  lint           Run C++ and Python linters (read-only)"
        echo "  fmt            Auto-format C++ and Python code"
        echo "  check          Lint + build with warnings-as-errors"
        ;;
    *)
        err "Unknown command: $1. Run './dev.sh help' for usage."
        ;;
esac
