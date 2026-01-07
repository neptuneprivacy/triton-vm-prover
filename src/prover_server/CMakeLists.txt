# GPU Prover Server for Neptune

set(PROVER_SERVER_SOURCES
    main.cpp
    server.cpp
    protocol.cpp
    prover.cpp
)

add_executable(gpu_prover_server ${PROVER_SERVER_SOURCES})

target_include_directories(gpu_prover_server PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/src/prover_server
)

# Link against the main triton-vm library
target_link_libraries(gpu_prover_server PRIVATE
    triton_vm_core
    nlohmann_json::nlohmann_json
)

# Link CUDA if available
if(TRITON_CUDA_ENABLED)
    target_link_libraries(gpu_prover_server PRIVATE
        triton_vm_gpu
    )
    target_compile_definitions(gpu_prover_server PRIVATE TRITON_CUDA_ENABLED)
endif()

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(gpu_prover_server PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install
install(TARGETS gpu_prover_server
    RUNTIME DESTINATION bin
)

