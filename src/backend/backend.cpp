#include "backend/backend.hpp"
#include "backend/cpu_backend.hpp"
#include "backend/cuda_backend.hpp"

namespace triton_vm {

std::unique_ptr<Backend> Backend::create(BackendType type) {
    switch (type) {
        case BackendType::CPU:
            return std::make_unique<CpuBackend>();
        case BackendType::CUDA:
            return std::make_unique<CudaBackend>();
        default:
            throw std::runtime_error("Unknown backend type");
    }
}

} // namespace triton_vm

