cmake_minimum_required(VERSION 3.16)
project(vex VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Compile definitions ─────────────────────────────────────────────────────
add_compile_definitions(_CRT_SECURE_NO_WARNINGS NOMINMAX)

# ── MSVC flags ──────────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W3 /MP /GL)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} /LTCG")
endif()

# ── FFmpeg ──────────────────────────────────────────────────────────────────
# Search order: deps/ffmpeg/ then FFMPEG_DIR env variable
set(FFMPEG_SEARCH_PATHS "")
if(EXISTS "${CMAKE_SOURCE_DIR}/deps/ffmpeg")
    list(APPEND FFMPEG_SEARCH_PATHS "${CMAKE_SOURCE_DIR}/deps/ffmpeg")
endif()
if(DEFINED ENV{FFMPEG_DIR})
    list(APPEND FFMPEG_SEARCH_PATHS "$ENV{FFMPEG_DIR}")
endif()

set(FFMPEG_FOUND FALSE)
foreach(_dir ${FFMPEG_SEARCH_PATHS})
    if(EXISTS "${_dir}/include/libavcodec" AND EXISTS "${_dir}/lib")
        set(FFMPEG_INCLUDE_DIR "${_dir}/include")
        set(FFMPEG_LIB_DIR "${_dir}/lib")
        set(FFMPEG_BIN_DIR "${_dir}/bin")
        set(FFMPEG_FOUND TRUE)
        message(STATUS "FFmpeg found at: ${_dir}")
        break()
    endif()
endforeach()

if(NOT FFMPEG_FOUND)
    message(FATAL_ERROR
        "FFmpeg shared/dev build not found.\n"
        "Please place it in deps/ffmpeg/ or set the FFMPEG_DIR environment variable.\n"
        "Expected layout:\n"
        "  <dir>/include/libavcodec/avcodec.h\n"
        "  <dir>/lib/avcodec.lib\n"
        "  <dir>/bin/avcodec-*.dll\n"
        "See deps/README.md for download instructions.")
endif()

# Create imported targets for FFmpeg libraries
set(FFMPEG_LIBS avcodec avformat avutil swscale)
foreach(_lib ${FFMPEG_LIBS})
    add_library(${_lib} SHARED IMPORTED)
    set_target_properties(${_lib} PROPERTIES
        IMPORTED_IMPLIB "${FFMPEG_LIB_DIR}/${_lib}.lib"
        INTERFACE_INCLUDE_DIRECTORIES "${FFMPEG_INCLUDE_DIR}"
    )
    # Find the DLL for runtime (avcodec-61.dll, etc.)
    file(GLOB _dll "${FFMPEG_BIN_DIR}/${_lib}*.dll")
    if(_dll)
        list(GET _dll 0 _dll_first)
        set_target_properties(${_lib} PROPERTIES
            IMPORTED_LOCATION "${_dll_first}"
        )
    endif()
endforeach()

# swresample is needed at runtime even though we don't link against it directly
file(GLOB SWRESAMPLE_DLLS "${FFMPEG_BIN_DIR}/swresample*.dll")

# ── TurboJPEG ──────────────────────────────────────────────────────────────
# Search order: deps/turbojpeg/ then TURBOJPEG_DIR env variable
set(TURBOJPEG_SEARCH_PATHS "")
if(EXISTS "${CMAKE_SOURCE_DIR}/deps/turbojpeg")
    list(APPEND TURBOJPEG_SEARCH_PATHS "${CMAKE_SOURCE_DIR}/deps/turbojpeg")
endif()
if(DEFINED ENV{TURBOJPEG_DIR})
    list(APPEND TURBOJPEG_SEARCH_PATHS "$ENV{TURBOJPEG_DIR}")
endif()

set(TURBOJPEG_FOUND FALSE)
foreach(_dir ${TURBOJPEG_SEARCH_PATHS})
    # Check for include/turbojpeg.h
    set(_inc "")
    if(EXISTS "${_dir}/include/turbojpeg.h")
        set(_inc "${_dir}/include")
    elseif(EXISTS "${_dir}/turbojpeg.h")
        set(_inc "${_dir}")
    endif()

    # Check for lib
    set(_lib_file "")
    if(EXISTS "${_dir}/lib/turbojpeg.lib")
        set(_lib_file "${_dir}/lib/turbojpeg.lib")
    elseif(EXISTS "${_dir}/lib/turbojpeg-static.lib")
        set(_lib_file "${_dir}/lib/turbojpeg-static.lib")
    endif()

    if(_inc AND _lib_file)
        set(TURBOJPEG_INCLUDE_DIR "${_inc}")
        set(TURBOJPEG_LIBRARY "${_lib_file}")
        set(TURBOJPEG_FOUND TRUE)
        message(STATUS "TurboJPEG found at: ${_dir}")
        message(STATUS "  include: ${_inc}")
        message(STATUS "  lib:     ${_lib_file}")
        break()
    endif()
endforeach()

if(NOT TURBOJPEG_FOUND)
    message(FATAL_ERROR
        "TurboJPEG (libjpeg-turbo) not found.\n"
        "Please place it in deps/turbojpeg/ or set the TURBOJPEG_DIR environment variable.\n"
        "Expected layout:\n"
        "  <dir>/include/turbojpeg.h\n"
        "  <dir>/lib/turbojpeg.lib (or turbojpeg-static.lib)\n"
        "See deps/README.md for download instructions.")
endif()

# Use dynamic linking if turbojpeg.lib (import lib) found, static otherwise
if(TURBOJPEG_LIBRARY MATCHES "turbojpeg-static")
    add_library(turbojpeg STATIC IMPORTED)
    set_target_properties(turbojpeg PROPERTIES
        IMPORTED_LOCATION "${TURBOJPEG_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TURBOJPEG_INCLUDE_DIR}"
    )
else()
    add_library(turbojpeg SHARED IMPORTED)
    set_target_properties(turbojpeg PROPERTIES
        IMPORTED_IMPLIB "${TURBOJPEG_LIBRARY}"
        INTERFACE_INCLUDE_DIRECTORIES "${TURBOJPEG_INCLUDE_DIR}"
    )
    # Find turbojpeg DLL for runtime copy
    get_filename_component(_tj_dir "${TURBOJPEG_LIBRARY}" DIRECTORY)
    get_filename_component(_tj_root "${_tj_dir}" DIRECTORY)
    file(GLOB TURBOJPEG_DLLS "${_tj_root}/bin/turbojpeg*.dll")
endif()

# ── pybind11 via FetchContent ──────────────────────────────────────────────
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG        v2.13.6
)
FetchContent_MakeAvailable(pybind11)

# ── Python ──────────────────────────────────────────────────────────────────
find_package(Python REQUIRED COMPONENTS Interpreter Development NumPy)

# ── Source files ────────────────────────────────────────────────────────────
set(VEX_SOURCES
    src/metrics.cpp
    src/hw_accel.cpp
    src/index_scanner.cpp
    src/scaler.cpp
    src/encoder.cpp
    src/decoder.cpp
    src/atlas.cpp
    src/disk_writer.cpp
    src/async_handle.cpp
    src/orchestrator.cpp
)

# ── Build the Python extension module ───────────────────────────────────────
pybind11_add_module(_vex_core src/bindings.cpp ${VEX_SOURCES})

target_include_directories(_vex_core PRIVATE
    ${CMAKE_SOURCE_DIR}/src
    ${Python_NumPy_INCLUDE_DIRS}
)

target_link_libraries(_vex_core PRIVATE
    avcodec
    avformat
    avutil
    swscale
    turbojpeg
)

# ── Post-build: copy .pyd to python/vex/ ────────────────────────────────────
add_custom_command(TARGET _vex_core POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory
        "${CMAKE_SOURCE_DIR}/python/vex"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$<TARGET_FILE:_vex_core>"
        "${CMAKE_SOURCE_DIR}/python/vex/"
    COMMENT "Copying _vex_core.pyd to python/vex/"
)

# ── Post-build: copy FFmpeg DLLs to output directory ────────────────────────
foreach(_lib ${FFMPEG_LIBS})
    get_target_property(_dll_path ${_lib} IMPORTED_LOCATION)
    if(_dll_path)
        add_custom_command(TARGET _vex_core POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll_path}"
                "$<TARGET_FILE_DIR:_vex_core>"
            COMMENT "Copying ${_dll_path}"
        )
        # Also copy to python/vex/ for direct import
        add_custom_command(TARGET _vex_core POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll_path}"
                "${CMAKE_SOURCE_DIR}/python/vex/"
        )
    endif()
endforeach()

# Copy swresample DLL(s) too — needed at runtime
foreach(_dll ${SWRESAMPLE_DLLS})
    add_custom_command(TARGET _vex_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_dll}"
            "$<TARGET_FILE_DIR:_vex_core>"
        COMMENT "Copying ${_dll}"
    )
    add_custom_command(TARGET _vex_core POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${_dll}"
            "${CMAKE_SOURCE_DIR}/python/vex/"
    )
endforeach()

# Copy TurboJPEG DLL(s) if dynamically linked
if(TURBOJPEG_DLLS)
    foreach(_dll ${TURBOJPEG_DLLS})
        add_custom_command(TARGET _vex_core POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}"
                "$<TARGET_FILE_DIR:_vex_core>"
            COMMENT "Copying ${_dll}"
        )
        add_custom_command(TARGET _vex_core POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${_dll}"
                "${CMAKE_SOURCE_DIR}/python/vex/"
        )
    endforeach()
endif()

message(STATUS "vex ${PROJECT_VERSION} — building _vex_core Python extension")
