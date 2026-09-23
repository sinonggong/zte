cmake_minimum_required(VERSION 3.20)
project(pi0_node_array_host LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# With no CMAKE_BUILD_TYPE, cmake passes no optimisation flag at all -- every
# binary this project has built was compiled at -O0. That is the wrong default
# for a runtime whose whole point is meeting a latency budget, and it silently
# understated nothing: it inflated every host-side number ever measured.
#
# RelWithDebInfo rather than Release: board bring-up wants symbols, and -O2
# against -O3 is not the difference that matters here.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
        "Build type; RelWithDebInfo by default so the runtime is optimised" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
        Debug Release RelWithDebInfo MinSizeRel)
endif()
message(STATUS "pi0 host runtime build type: ${CMAKE_BUILD_TYPE}")

# The SDK install is lowercase `sdk` on Linux and uppercase `SDK` in the older
# Windows layout. Probe for both rather than hardcoding one.
if(DEFINED ENV{ACHRONIX_SDK_ROOT} AND NOT "$ENV{ACHRONIX_SDK_ROOT}" STREQUAL "")
    set(_ACHRONIX_SDK_ROOT_DEFAULT "$ENV{ACHRONIX_SDK_ROOT}")
elseif(IS_DIRECTORY "/opt/achronix/sdk")
    set(_ACHRONIX_SDK_ROOT_DEFAULT "/opt/achronix/sdk")
else()
    set(_ACHRONIX_SDK_ROOT_DEFAULT "/opt/achronix/SDK")
endif()

set(ACHRONIX_SDK_ROOT "${_ACHRONIX_SDK_ROOT_DEFAULT}" CACHE PATH "Achronix SDK root")
set(ACHRONIX_SDK_LIB_DIR "${ACHRONIX_SDK_ROOT}/lib" CACHE PATH "Achronix SDK shared library directory")

# libacxdev.so lives in drivers/lib/ in a source SDK and in lib/ in a
# pre-compiled one. Pointing at a directory that does not exist leaves acxdev
# unresolvable at link time.
if(IS_DIRECTORY "${ACHRONIX_SDK_ROOT}/drivers/lib")
    set(_ACHRONIX_DRV_LIB_DIR_DEFAULT "${ACHRONIX_SDK_ROOT}/drivers/lib")
else()
    set(_ACHRONIX_DRV_LIB_DIR_DEFAULT "${ACHRONIX_SDK_ROOT}/lib")
endif()
set(ACHRONIX_DRV_LIB_DIR "${_ACHRONIX_DRV_LIB_DIR_DEFAULT}" CACHE PATH "Achronix driver shared library directory")

find_package(Threads REQUIRED)

# Node-array host tools: the chunk runner, the S0 replay, the ATU inspector and the DBI probe (all link the Achronix SDK).

# Read-only probe of the PCIe controller's DBI routes and DMA registers. Board
# tooling: links the SDK like the bring-up CLI, needs no other source.
add_executable(pi0_dbi_probe pi0_dbi_probe.cpp)
target_include_directories(pi0_dbi_probe PRIVATE
    ${ACHRONIX_SDK_ROOT}/include
    ${ACHRONIX_SDK_ROOT}/drivers/lib
)
target_link_directories(pi0_dbi_probe PRIVATE
    ${ACHRONIX_SDK_LIB_DIR}
    ${ACHRONIX_DRV_LIB_DIR}
)
target_link_libraries(pi0_dbi_probe PRIVATE acxsdk acxdev rt)

# S0 node-array silicon replay (paper/sw/s0_vectors.py vectors through the BAR0 host window + bulk GDDR6 path).
add_executable(pi0_s0_replay pi0_s0_replay.cpp pi0_fpga_sdk_backend.cpp pi0_fpga_mock_backend.cpp)
target_include_directories(pi0_s0_replay PRIVATE
    ${ACHRONIX_SDK_ROOT}/include
    ${ACHRONIX_SDK_ROOT}/drivers/lib
)
target_link_directories(pi0_s0_replay PRIVATE
    ${ACHRONIX_SDK_LIB_DIR}
    ${ACHRONIX_DRV_LIB_DIR}
)
target_link_libraries(pi0_s0_replay PRIVATE acxsdk acxdev Threads::Threads rt)

# ATU (BAR -> NoC translation) inspector / mover for the node-array bitstreams' PIO window.
add_executable(pi0_atu_tool pi0_atu_tool.cpp)
target_include_directories(pi0_atu_tool PRIVATE ${ACHRONIX_SDK_ROOT}/include ${ACHRONIX_SDK_ROOT}/drivers/lib)
target_link_directories(pi0_atu_tool PRIVATE ${ACHRONIX_SDK_LIB_DIR} ${ACHRONIX_DRV_LIB_DIR})
target_link_libraries(pi0_atu_tool PRIVATE acxsdk acxdev Threads::Threads rt)

# Whole-chunk runner for node-array bitstreams: image through a sliding BAR1 ATU window, broadcast start, compare.
add_executable(pi0_chunk_run pi0_chunk_run.cpp)
target_include_directories(pi0_chunk_run PRIVATE ${ACHRONIX_SDK_ROOT}/include ${ACHRONIX_SDK_ROOT}/drivers/lib)
target_link_directories(pi0_chunk_run PRIVATE ${ACHRONIX_SDK_LIB_DIR} ${ACHRONIX_DRV_LIB_DIR})
target_link_libraries(pi0_chunk_run PRIVATE acxsdk acxdev Threads::Threads rt)
