cmake_minimum_required(VERSION 3.20)

# ============================================================================
# CUDA Configuration (must be before project())
# ============================================================================
option(ENABLE_CUDA "Enable CUDA GPU acceleration" ON)

# Set CUDA 13.0 path BEFORE project() to ensure CMake uses the correct CUDA version
if(ENABLE_CUDA AND EXISTS "/usr/local/cuda-13.0")
    set(CMAKE_CUDA_COMPILER "/usr/local/cuda-13.0/bin/nvcc" CACHE FILEPATH "CUDA compiler")
    set(CUDAToolkit_ROOT "/usr/local/cuda-13.0" CACHE PATH "CUDA Toolkit root directory")
    set(CMAKE_PREFIX_PATH "/usr/local/cuda-13.0" ${CMAKE_PREFIX_PATH})
endif()

# Set CUDA architectures BEFORE project() to avoid CMake using default (sm_52) during compiler detection
# Check environment variable first, then default to 90
# Note: Cache variable TRITON_CUDA_ARCHITECTURES is set after project() and will override this if needed
if(ENABLE_CUDA)
    if(DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
        set(CMAKE_CUDA_ARCHITECTURES "$ENV{CMAKE_CUDA_ARCHITECTURES}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "90")
    endif()
endif()

# Enable C++20 standard before project() to avoid compiler feature detection issues
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(ENABLE_CUDA)
    project(triton-vm-cpp VERSION 1.0.0 LANGUAGES CXX CUDA)
else()
    project(triton-vm-cpp VERSION 1.0.0 LANGUAGES CXX)
endif()

find_package(Python3 COMPONENTS Interpreter REQUIRED)

# C++ standard already set before project()
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


# ============================================================================
# Build Options
# ============================================================================
option(BUILD_TESTS "Build unit tests" ON)
option(BUILD_BENCHMARKS "Build benchmarks" OFF)
option(BUILD_GPU_TESTS "Build GPU co-verification tests (requires CUDA)" ON)

# ============================================================================
# CUDA Setup
# ============================================================================
if(ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    
    # CUDA architectures (compile time is roughly proportional to number of archs).
    #
    # CUDA architectures to compile for. Default includes common datacenter GPUs.
    # - sm_90: H100, H200 (Hopper), RTX 5090 (Blackwell compatibility)
    # - sm_89: RTX 4090, L40 (Ada Lovelace)
    #
    # Note: sm_120 (Blackwell native) requires CUDA 13.0+
    # For fastest local iteration, compile only for your actual GPU:
    #   cmake -S . -B build -DTRITON_CUDA_ARCHITECTURES=90
    set(TRITON_CUDA_ARCHITECTURES "90" CACHE STRING "CUDA arch list for nvcc (e.g. 90 or 89;90)")
    # Only override CMAKE_CUDA_ARCHITECTURES if it wasn't already set by environment variable
    if(NOT DEFINED ENV{CMAKE_CUDA_ARCHITECTURES})
        set(CMAKE_CUDA_ARCHITECTURES "${TRITON_CUDA_ARCHITECTURES}")
    endif()

    # Speed up nvcc compilation (CUDA 12+): parallel compilation.
    # Set empty to disable if your nvcc doesn't support it.
    set(TRITON_NVCC_THREADS "0" CACHE STRING "nvcc --threads value (0=auto). Set empty to disable.")
    
    # Minimal CUDA flags to avoid command line length issues
    #
    # NOTE (Ubuntu 24.04 / glibc 2.39+ / NVCC 12.x):
    # NVCC sometimes ends up compiling glibc headers with C23 IEC 60559 extensions enabled,
    # which makes glibc expose `_Float32/_Float64/_Float128/...` in C++ headers. Those types
    # are not valid C++ keywords and NVCC then fails in system headers.
    #
    # We defensively force the IEC 60559 "want" macros off and also undefine __STDC_VERSION__
    # (NVCC may define it even in C++ mode), to keep glibc from emitting those prototypes.
    # IMPORTANT: glibc enables `_Float*` declarations if these macros are *defined at all* (value doesn't matter),
    # so we must explicitly *undefine* them for NVCC.
    set(CMAKE_CUDA_FLAGS "--expt-relaxed-constexpr -diag-suppress=177 -U__STDC_VERSION__ -U__STDC_WANT_IEC_60559_TYPES_EXT__ -U__STDC_WANT_IEC_60559_BFP_EXT__ -U__STDC_WANT_IEC_60559_FUNCS_EXT__ -U__STDC_WANT_IEC_60559_FUNCS_EXT2__")
    if(NOT "${TRITON_NVCC_THREADS}" STREQUAL "")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads=${TRITON_NVCC_THREADS}")
    endif()

    # CUDA 13.0 workaround: limit ptxas optimizations to prevent hangs on complex kernels
    # ptxas (PTX assembler) in CUDA 13.0 can hang indefinitely even with -O2 compiler optimization
    # Using --ptxas-options=-O1 limits ptxas optimizer while keeping nvcc compiler optimizations
    if(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0" AND CUDAToolkit_VERSION VERSION_LESS "14.0")
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-O1")
        message(STATUS "CUDA 13.0 detected: limiting ptxas optimizations to -O1 to prevent hangs")
    endif()

    # Enable libstdc++ runtime assertions in debug/profiling builds.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_GLIBCXX_ASSERTIONS")
    add_compile_definitions(_GLIBCXX_ASSERTIONS)

    # ------------------------------------------------------------------------
    # GCC 13 + NVCC (CUDA 12.x) + glibc note
    #
    # Do NOT define __STDC_WANT_IEC_60559_* for CUDA compilation:
    # forcing these macros can make glibc expose `_Float*` declarations in C++
    # headers that NVCC cannot compile in C++ mode (e.g. `_Float128`), causing
    # build failures in system headers.
    # ------------------------------------------------------------------------
    # Use -O2 for CUDA to prevent ptxas from hanging on complex kernels
    # -O3 can cause ptxas to hang indefinitely on quotient transition kernels
    set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O2")
    
    # Static link CUDA runtime for portability (no libcudart.so needed on target)
    set(CMAKE_CUDA_RUNTIME_LIBRARY Static)
    
    add_compile_definitions(TRITON_CUDA_ENABLED)
    message(STATUS "CUDA enabled: ${CUDAToolkit_VERSION}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    message(STATUS "CUDA disabled")
endif()

# ============================================================================
# Dependencies
# ============================================================================
find_package(Threads REQUIRED)

# OpenMP for CPU parallelization (critical for performance on multi-core systems)
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found - enabling parallel CPU execution")
    message(STATUS "  OpenMP version: ${OpenMP_CXX_VERSION}")
    message(STATUS "  OpenMP CXX flags: ${OpenMP_CXX_FLAGS}")
    message(STATUS "  Set OMP_NUM_THREADS environment variable to control thread count")
    message(STATUS "  Recommended: OMP_NUM_THREADS=96 for Threadripper 9995WX (96 physical cores)")
    # Explicitly enable OpenMP compiler flags
    add_compile_definitions(_OPENMP)
else()
    message(FATAL_ERROR "OpenMP not found - required for CPU parallelization")
endif()

# TBB for C++17 parallel algorithms (std::execution::par) and task-based parallelism
find_package(TBB QUIET)
if(TBB_FOUND)
    message(STATUS "TBB found - enabling parallel STL algorithms and task parallelism")
    add_compile_definitions(TVM_USE_TBB)
else()
    message(STATUS "TBB not found - using sequential fallbacks")
endif()

# Taskflow for dependency graph-based task parallelism
include(FetchContent)
FetchContent_Declare(
    taskflow
    GIT_REPOSITORY https://github.com/taskflow/taskflow.git
    GIT_TAG v3.8.0
)
set(TF_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(TF_BUILD_TESTS OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(taskflow)
if(taskflow_POPULATED)
    message(STATUS "Taskflow found - enabling dependency graph-based task parallelism")
    add_compile_definitions(TVM_USE_TASKFLOW)
    # Taskflow v3.8.0 is header-only, source dir is available via taskflow_SOURCE_DIR
    message(STATUS "  Taskflow source dir: ${taskflow_SOURCE_DIR}")
else()
    message(STATUS "Taskflow not found - dependency graph parallelism disabled")
endif()

# nlohmann/json for JSON parsing
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG v3.11.3
)
FetchContent_MakeAvailable(json)

# Google Test for unit testing
if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    enable_testing()
endif()

# ============================================================================
# Rust FFI Libraries
# ============================================================================
find_program(CARGO cargo REQUIRED)

# Degree lowering FFI
set(RUST_DEGREE_FFI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rust/degree_lowering_ffi)
set(RUST_DEGREE_FFI_LIB ${RUST_DEGREE_FFI_DIR}/target/release/libdegree_lowering_ffi.a)

add_custom_target(degree_lowering_ffi_build
    COMMAND ${CARGO} build --release --manifest-path ${RUST_DEGREE_FFI_DIR}/Cargo.toml
    WORKING_DIRECTORY ${RUST_DEGREE_FFI_DIR}
    COMMENT "Building Rust FFI library for degree lowering"
    BYPRODUCTS ${RUST_DEGREE_FFI_LIB}
)

add_library(degree_lowering_ffi STATIC IMPORTED)
add_dependencies(degree_lowering_ffi degree_lowering_ffi_build)
set_target_properties(degree_lowering_ffi PROPERTIES
    IMPORTED_LOCATION ${RUST_DEGREE_FFI_LIB}
)

# Bincode FFI
set(RUST_BINCODE_FFI_DIR ${CMAKE_CURRENT_SOURCE_DIR}/rust/bincode_ffi)
set(RUST_BINCODE_FFI_LIB ${RUST_BINCODE_FFI_DIR}/target/release/libbincode_ffi.a)

add_custom_target(bincode_ffi_build
    COMMAND ${CARGO} build --release --manifest-path ${RUST_BINCODE_FFI_DIR}/Cargo.toml
    WORKING_DIRECTORY ${RUST_BINCODE_FFI_DIR}
    COMMENT "Building Rust FFI library for bincode serialization"
    BYPRODUCTS ${RUST_BINCODE_FFI_LIB}
)

add_library(bincode_ffi STATIC IMPORTED)
add_dependencies(bincode_ffi bincode_ffi_build)
set_target_properties(bincode_ffi PROPERTIES
    IMPORTED_LOCATION ${RUST_BINCODE_FFI_LIB}
)

# ============================================================================
# Main Library Sources (CPU)
# ============================================================================
set(TRITON_VM_SOURCES
    src/types/b_field_element.cpp
    src/types/x_field_element.cpp
    src/types/digest.cpp
    src/table/master_table.cpp
    src/table/ram_bezout.cpp
    src/table/table_padding.cpp
    src/table/degree_lowering_cpp.cpp
    src/table/degree_lowering_main_cpp.cpp
    src/table/extend_helpers.cpp
    src/table/table_commitment.cpp
    src/hash/tip5.cpp
    src/merkle/merkle_tree.cpp
    src/fri/fri.cpp
    src/proof_stream/proof_stream.cpp
    src/quotient/constraint_evaluations.cpp
    src/quotient/quotient.cpp
    src/quotient/constraint_evaluations_xfe_main.cpp
    src/ntt/ntt.cpp
    src/lde/lde_randomized.cpp
    src/lde/lde_table.cpp
    src/stark/challenges.cpp
    src/stark.cpp
    src/chacha12_rng.cpp
    src/test_data_loader.cpp
    src/backend/backend.cpp
    src/vm/vm.cpp
    src/vm/vm_state.cpp
    src/vm/aet.cpp
    src/vm/program.cpp
    src/vm/op_stack.cpp
)

# ============================================================================
# GPU Sources (CUDA)
# ============================================================================
if(ENABLE_CUDA)
    set(TRITON_VM_GPU_SOURCES
        # Backend implementations
        src/gpu/cuda_memory.cpp
        src/gpu/cuda_backend.cu
        src/gpu/cuda_ntt.cu
        src/gpu/cuda_tip5.cu
        src/gpu/cuda_merkle.cu
        src/gpu/cuda_lde.cu
        src/gpu/cuda_quotient.cu
        src/gpu/cuda_fri.cu
        
        # GPU-resident proof generation (zero-copy design)
        src/gpu/gpu_proof_context.cu
        src/gpu/gpu_stark.cu
        src/gpu/gpu_stark_aux_cpu.cpp
        
        # CUDA kernels (in implementation order)
        src/gpu/kernels/bfield_kernel.cu    # 1. BField arithmetic (foundation)
        src/gpu/kernels/xfield_kernel.cu    # 2. XField arithmetic
        src/gpu/kernels/ntt_kernel.cu       # 3. NTT/INTT
        src/gpu/kernels/tip5_kernel.cu      # 4. Tip5 hash
        src/gpu/kernels/merkle_kernel.cu    # 5. Merkle tree
        src/gpu/kernels/lde_kernel.cu       # 6. LDE
        src/gpu/kernels/fiat_shamir_kernel.cu # 7. Fiat-Shamir
        src/gpu/kernels/extend_kernel.cu    # 8. Table extension
        src/gpu/kernels/aux_extend_full_kernel.cu  # 8b. Full aux table extension
        src/gpu/kernels/quotient_kernel.cu  # 9. Quotient
        src/gpu/kernels/fri_kernel.cu       # 10. FRI
        src/gpu/kernels/gather_kernel.cu    # 11. Gather rows
        src/gpu/kernels/row_hash_kernel.cu  # 12. Row hashing
        src/gpu/kernels/randomized_lde_kernel.cu  # 13. Randomized LDE
        src/gpu/kernels/quotient_compute_kernel.cu  # 13a. Quotient post-processing (interp + segment LDE)
        src/gpu/kernels/challenges_kernel.cu  # 13a2. Compute derived challenges on GPU
        # src/gpu/kernels/quotient_constraints_kernel.cu  # 13b. Disabled - nvcc hangs on 6200 lines.
        src/gpu/kernels/quotient_constraints_split_kernel.cu  # 13b-split. Split kernels: init/cons/term
        src/gpu/kernels/quotient_transition_part0_kernel.cu  # 13b-tran0. Transition constraints part 0/4
        src/gpu/kernels/quotient_transition_part1_kernel.cu  # 13b-tran1. Transition constraints part 1/4
        src/gpu/kernels/quotient_transition_part2_kernel.cu  # 13b-tran2. Transition constraints part 2/4
        src/gpu/kernels/quotient_transition_part3_kernel.cu  # 13b-tran3. Transition constraints part 3/4
        src/gpu/kernels/proof_kernel.cu     # 14. Proof encoding
        src/gpu/kernels/table_fill_kernel.cu  # 15. Table filling and padding
        src/gpu/kernels/phase1_kernel.cu      # 16. GPU Phase 1 (table creation from AET)
        src/gpu/kernels/bezout_kernel.cu      # 17. GPU Bézout coefficient computation
        src/gpu/kernels/poly_mul_kernel.cu    # 18. GPU polynomial multiplication
        src/gpu/kernels/degree_lowering_main_kernel.cu  # 19. GPU degree lowering for main table
        src/gpu/kernels/u32_table_kernel.cu     # 20. GPU U32 table fill
    )
    
    # Create placeholder files if they don't exist (for initial build)
    foreach(GPU_SRC ${TRITON_VM_GPU_SOURCES})
        if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${GPU_SRC})
            message(STATUS "GPU source not yet implemented: ${GPU_SRC}")
        endif()
    endforeach()
    
    # Filter to only existing files for now
    set(TRITON_VM_GPU_SOURCES_EXISTING "")
    foreach(GPU_SRC ${TRITON_VM_GPU_SOURCES})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${GPU_SRC})
            list(APPEND TRITON_VM_GPU_SOURCES_EXISTING ${GPU_SRC})
        endif()
    endforeach()
endif()

# ============================================================================
# Main Library
# ============================================================================
add_library(triton_vm ${TRITON_VM_SOURCES})

if(ENABLE_CUDA AND TRITON_VM_GPU_SOURCES_EXISTING)
    target_sources(triton_vm PRIVATE ${TRITON_VM_GPU_SOURCES_EXISTING})
endif()

# Enable SIMD optimizations for x86_64 (AVX2 and FMA)
# Only apply to C++ files, not CUDA files
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_compile_options(triton_vm PRIVATE 
        $<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mfma>
    )
    add_compile_definitions(XFE_SIMD_AVX2_ENABLED=1)
    message(STATUS "AVX2 and FMA SIMD optimizations enabled for C++ code")
endif()

target_include_directories(triton_vm PUBLIC 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(triton_vm PUBLIC 
    nlohmann_json::nlohmann_json
    Threads::Threads
    degree_lowering_ffi
    OpenMP::OpenMP_CXX
    $<$<BOOL:${TBB_FOUND}>:TBB::tbb>
    bincode_ffi
)

# Taskflow is header-only, so we need to add include directory manually
if(taskflow_POPULATED)
    # Taskflow v3.8.0 is header-only, add include directory
    # Include directory should point to the parent of taskflow/ directory
    target_include_directories(triton_vm PUBLIC 
        $<$<BOOL:${taskflow_POPULATED}>:${taskflow_SOURCE_DIR}>
    )
endif()

if(ENABLE_CUDA)
    target_link_libraries(triton_vm PUBLIC CUDA::cudart_static)
endif()

if(UNIX)
    target_link_libraries(triton_vm PUBLIC dl m)
endif()

# Test data path
set(TEST_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../triton-cli-1.0.0/test_data")
target_compile_definitions(triton_vm PUBLIC 
    TEST_DATA_DIR="${TEST_DATA_DIR}"
)

# ============================================================================
# Unit Tests (CPU)
# ============================================================================
if(BUILD_TESTS)
    set(TEST_SOURCES
        tests/test_b_field_element.cpp
        tests/test_x_field_element.cpp
        tests/test_data_loader.cpp
        tests/test_stark.cpp
        tests/test_tables.cpp
        tests/test_tip5.cpp
        tests/test_merkle.cpp
        tests/test_table_commitment.cpp
        tests/test_fri.cpp
        tests/test_proof_stream.cpp
        tests/test_quotient.cpp
        tests/test_functional_verification.cpp
        tests/test_compute_and_compare.cpp
        tests/test_io_verification.cpp
        tests/test_lde_verification.cpp
        tests/test_ntt.cpp
        tests/test_lde_cases.cpp
        tests/test_lde_fixed_seed.cpp
        tests/test_lde_randomized_implementation.cpp
        tests/test_lde_full_randomized.cpp
        tests/test_lde_table_pipeline.cpp
        tests/test_fiat_shamir.cpp
        tests/test_aux_tables.cpp
        tests/test_extend_step.cpp
        tests/test_debug_arithmetic.cpp
        tests/test_extend_row256.cpp
        tests/test_trace_lookup_accumulation.cpp
        tests/test_lookup_row_count.cpp
        tests/test_hash_table_debug.cpp
        tests/test_hash_table_manual_debug.cpp
        tests/test_all_steps_verification.cpp
        tests/test_step_by_step_comparison.cpp
        tests/test_vm_trace_execution.cpp
        tests/test_all_proof_steps.cpp
        tests/test_table_padding_verification.cpp
        tests/test_rust_ffi_vs_cpp_comparison.cpp
        tests/test_pure_cpp_proof_verification.cpp
    )
    
    add_executable(triton_vm_tests ${TEST_SOURCES})
    target_link_libraries(triton_vm_tests 
        triton_vm
        GTest::gtest_main
        GTest::gmock
    )
    
    target_compile_definitions(triton_vm_tests PRIVATE 
        TEST_DATA_DIR="${TEST_DATA_DIR}"
    )
    
    include(GoogleTest)
    gtest_discover_tests(triton_vm_tests)
endif()

# ============================================================================
# GPU Co-Verification Tests
# ============================================================================
if(BUILD_TESTS AND BUILD_GPU_TESTS AND ENABLE_CUDA)
    set(GPU_TEST_SOURCES
        # Co-verification tests (in kernel implementation order)
        tests/co_verify/co_verify_bfield.cpp    # 1. BField arithmetic
        tests/co_verify/co_verify_xfield.cpp    # 2. XField arithmetic
        tests/co_verify/co_verify_ntt.cpp       # 3. NTT
        tests/co_verify/co_verify_tip5.cpp      # 4. Tip5
        tests/co_verify/co_verify_merkle.cpp    # 5. Merkle
        tests/co_verify/co_verify_lde.cpp       # 6. LDE
        tests/co_verify/co_verify_fiat_shamir.cpp # 7. Fiat-Shamir
        tests/co_verify/co_verify_extend.cpp    # 8. Table extension
        tests/co_verify/co_verify_quotient.cpp  # 9. Quotient
        # tests/co_verify/co_verify_quotient_constraints.cpp # 9b. Quotient constraints (AIR eval) (disabled; using Rust FFI)
        tests/co_verify/co_verify_fri.cpp       # 10. FRI
        tests/co_verify/co_verify_randomized_lde.cpp  # 11. Randomized LDE
        tests/co_verify/co_verify_gather.cpp    # 11. Gather
        tests/co_verify/co_verify_degree_lowering.cpp # 12. Degree lowering
    )
    
    # Filter to existing files
    set(GPU_TEST_SOURCES_EXISTING "")
    foreach(TEST_SRC ${GPU_TEST_SOURCES})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SRC})
            list(APPEND GPU_TEST_SOURCES_EXISTING ${TEST_SRC})
        endif()
    endforeach()
    
    if(GPU_TEST_SOURCES_EXISTING)
        add_executable(triton_vm_gpu_tests ${GPU_TEST_SOURCES_EXISTING})
        target_link_libraries(triton_vm_gpu_tests 
            triton_vm
            GTest::gtest_main
            GTest::gmock
            CUDA::cudart_static
        )
        target_include_directories(triton_vm_gpu_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/co_verify
        )
        
        gtest_discover_tests(triton_vm_gpu_tests)
        message(STATUS "GPU co-verification tests enabled")
    else()
        message(STATUS "GPU co-verification tests: source files not yet created")
    endif()
endif()

## (disabled) Generate quotient constraint header for GPU build
## Keeping this off speeds up builds significantly.

# ============================================================================
# GPU-Only Tests (for testing CUDA kernels in isolation)
# ============================================================================
if(BUILD_TESTS AND ENABLE_CUDA)
    set(CUDA_UNIT_TEST_SOURCES "")
    
    # Add GPU-only tests as they're created
    file(GLOB CUDA_UNIT_TEST_SOURCES "tests/gpu/*.cpp" "tests/gpu/*.cu")
    
    if(CUDA_UNIT_TEST_SOURCES)
        add_executable(triton_vm_cuda_tests ${CUDA_UNIT_TEST_SOURCES})
        target_link_libraries(triton_vm_cuda_tests 
            triton_vm
            GTest::gtest_main
            CUDA::cudart_static
        )
        gtest_discover_tests(triton_vm_cuda_tests)
    endif()
endif()

# ============================================================================
# GPU Integration Tests (end-to-end testing with real trace data)
# ============================================================================
if(BUILD_TESTS AND BUILD_GPU_TESTS AND ENABLE_CUDA)
    set(GPU_INTEGRATION_TEST_SOURCES
        tests/integration/gpu_proof_integration_test.cpp
    )
    
    # Filter to existing files
    set(GPU_INTEGRATION_TEST_SOURCES_EXISTING "")
    foreach(TEST_SRC ${GPU_INTEGRATION_TEST_SOURCES})
        if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/${TEST_SRC})
            list(APPEND GPU_INTEGRATION_TEST_SOURCES_EXISTING ${TEST_SRC})
        endif()
    endforeach()
    
    if(GPU_INTEGRATION_TEST_SOURCES_EXISTING)
        add_executable(triton_vm_gpu_integration_tests ${GPU_INTEGRATION_TEST_SOURCES_EXISTING})
        target_link_libraries(triton_vm_gpu_integration_tests 
            triton_vm
            GTest::gtest_main
            GTest::gmock
            CUDA::cudart_static
        )
        target_include_directories(triton_vm_gpu_integration_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/tests/co_verify
        )
        
        gtest_discover_tests(triton_vm_gpu_integration_tests
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        message(STATUS "GPU integration tests enabled")
    else()
        message(STATUS "GPU integration tests: source files not yet created")
    endif()
endif()

# ============================================================================
# Main Executable
# ============================================================================
add_executable(triton_vm_prove main.cpp)
target_link_libraries(triton_vm_prove triton_vm)
target_include_directories(triton_vm_prove PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# ============================================================================
# GPU Proof Generation Executables
# ============================================================================
if(ENABLE_CUDA)
    # Hybrid GPU prover (GPU for LDE/Merkle, CPU for rest)
    add_executable(triton_vm_prove_gpu main_gpu.cpp)
    target_link_libraries(triton_vm_prove_gpu triton_vm CUDA::cudart_static)
    target_include_directories(triton_vm_prove_gpu PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )
    
    # Full GPU zero-copy prover (no OpenMP - uses std::thread for CPU parallelization)
    add_executable(triton_vm_prove_gpu_full main_gpu_full.cpp)
    target_link_libraries(triton_vm_prove_gpu_full triton_vm CUDA::cudart_static)
    target_include_directories(triton_vm_prove_gpu_full PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Debug tool: compare decoded proof streams (Rust vs GPU)
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/compare_proof_streams.cpp)
        add_executable(compare_proof_streams tools/compare_proof_streams.cpp)
        target_link_libraries(compare_proof_streams triton_vm CUDA::cudart_static)
        target_include_directories(compare_proof_streams PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
    endif()
    
    # Trace dump tool: dump AET and domain data for comparison with Rust
    add_executable(dump_trace src/tools/dump_trace.cpp)
    target_link_libraries(dump_trace triton_vm)
    target_include_directories(dump_trace PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    )

    # Debug tool: print proof item encoding sizes
    if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/print_proof_item_encoding_sizes.cpp)
        add_executable(print_proof_item_encoding_sizes tools/print_proof_item_encoding_sizes.cpp)
        target_link_libraries(print_proof_item_encoding_sizes triton_vm CUDA::cudart_static)
        target_include_directories(print_proof_item_encoding_sizes PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
    endif()
    
    # ============================================================================
    # GPU Prover Server (Neptune integration)
    # ============================================================================
    set(PROVER_SERVER_SOURCES
        src/prover_server/main.cpp
        src/prover_server/server.cpp
        src/prover_server/protocol.cpp
        src/prover_server/prover.cpp
    )
    
    add_executable(gpu_prover_server ${PROVER_SERVER_SOURCES})
    target_include_directories(gpu_prover_server PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/src/prover_server
    )
    target_link_libraries(gpu_prover_server triton_vm CUDA::cudart_static)
    target_compile_definitions(gpu_prover_server PRIVATE TRITON_CUDA_ENABLED)
    
    message(STATUS "  GPU Prover Server: Enabled")
endif()

# ============================================================================
# Configuration Summary
# ============================================================================
message(STATUS "")
message(STATUS "=================================================")
message(STATUS "Triton VM C++ Configuration:")
message(STATUS "=================================================")
message(STATUS "  Version:           ${PROJECT_VERSION}")
message(STATUS "  C++ Standard:      ${CMAKE_CXX_STANDARD}")
message(STATUS "  Build Tests:       ${BUILD_TESTS}")
message(STATUS "  Build GPU Tests:   ${BUILD_GPU_TESTS}")
message(STATUS "  CUDA Enabled:      ${ENABLE_CUDA}")
if(ENABLE_CUDA)
    message(STATUS "  CUDA Version:      ${CUDAToolkit_VERSION}")
    message(STATUS "  CUDA Archs:        ${CMAKE_CUDA_ARCHITECTURES}")
endif()
message(STATUS "  Test Data Dir:     ${TEST_DATA_DIR}")
message(STATUS "=================================================")
message(STATUS "")
