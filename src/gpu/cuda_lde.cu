/**
 * CUDA LDE Backend Implementation
 */

#ifdef TRITON_CUDA_ENABLED

#include "backend/cuda_backend.hpp"
#include "gpu/cuda_memory.hpp"
#include "gpu/cuda_common.cuh"

namespace triton_vm {
namespace gpu {
namespace kernels {
    // Forward declarations
    void lde_column_device(const uint64_t* d_trace, size_t trace_len,
                          uint64_t* d_extended, size_t extended_len,
                          uint64_t trace_offset, uint64_t extended_offset,
                          cudaStream_t stream);
    void lde_batch_device(const uint64_t** d_traces, size_t num_columns,
                         size_t trace_len, uint64_t** d_extended,
                         size_t extended_len, uint64_t trace_offset,
                         uint64_t extended_offset, cudaStream_t stream);
}
}

void CudaBackend::lde_column(
    const BFieldElement* trace_column,
    size_t trace_len,
    BFieldElement* extended_column,
    size_t extended_len,
    BFieldElement trace_offset,
    BFieldElement extended_offset
) {
    // Convert to raw u64
    std::vector<uint64_t> h_trace(trace_len);
    for (size_t i = 0; i < trace_len; ++i) {
        h_trace[i] = trace_column[i].value();
    }
    
    // Allocate device memory
    gpu::DeviceBuffer<uint64_t> d_trace(trace_len);
    gpu::DeviceBuffer<uint64_t> d_extended(extended_len);
    
    // Upload
    d_trace.upload(h_trace.data(), trace_len);
    
    // Execute
    gpu::kernels::lde_column_device(
        d_trace.data(), trace_len,
        d_extended.data(), extended_len,
        trace_offset.value(), extended_offset.value(),
        stream_
    );
    cudaStreamSynchronize(stream_);
    
    // Download
    std::vector<uint64_t> h_extended(extended_len);
    d_extended.download(h_extended.data(), extended_len);
    
    for (size_t i = 0; i < extended_len; ++i) {
        extended_column[i] = BFieldElement(h_extended[i]);
    }
}

void CudaBackend::lde_batch(
    const BFieldElement** columns,
    size_t num_columns,
    size_t trace_len,
    BFieldElement** extended,
    size_t extended_len,
    BFieldElement trace_offset,
    BFieldElement extended_offset
) {
    // For now, process columns sequentially
    // TODO: Implement batched version with multi-stream processing
    for (size_t c = 0; c < num_columns; ++c) {
        lde_column(
            columns[c], trace_len,
            extended[c], extended_len,
            trace_offset, extended_offset
        );
    }
}

} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

