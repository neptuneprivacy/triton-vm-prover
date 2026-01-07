/**
 * Fiat-Shamir CUDA Kernel Implementation
 * 
 * GPU-accelerated Fiat-Shamir transform using Tip5 sponge.
 * 
 * This implementation delegates to the Tip5 batch permutation kernel
 * for consistency with the CPU implementation.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/fiat_shamir_kernel.cuh"
#include "gpu/kernels/tip5_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Constants matching Tip5
static constexpr int STATE_SIZE = 16;
static constexpr int RATE = 10;

// ============================================================================
// Kernels
// ============================================================================

/**
 * Initialize sponge state for variable-length input
 * Sets entire state to zeros (variable-length mode)
 */
__global__ void fs_init_varlen_kernel(uint64_t* state) {
    int idx = threadIdx.x;
    if (idx < STATE_SIZE) {
        state[idx] = 0;
    }
}

/**
 * Absorb one chunk into sponge (overwrite rate, then permute)
 * This kernel overwrites the rate portion with data
 */
__global__ void fs_overwrite_rate_kernel(
    uint64_t* state,
    const uint64_t* data
) {
    int idx = threadIdx.x;
    if (idx < RATE) {
        state[idx] = data[idx];
    }
}

/**
 * Overwrite RATE portion from a device array slice with padding:
 * padded = [data..., 1, 0, 0, ...] to multiple of RATE.
 */
__global__ void fs_overwrite_rate_from_device_kernel(
    uint64_t* state,
    const uint64_t* data,
    size_t data_len,
    size_t chunk_base   // base index into padded stream (multiple of RATE)
) {
    int idx = threadIdx.x;
    if (idx >= RATE) return;
    size_t pos = chunk_base + (size_t)idx;
    uint64_t v = 0;
    if (pos < data_len) {
        v = data[pos];
    } else if (pos == data_len) {
        v = 1;
    } else {
        v = 0;
    }
    state[idx] = v;
}

__global__ void fs_copy_rate_to_output_kernel(
    const uint64_t* state,
    uint64_t* output,
    size_t out_offset,
    size_t take
) {
    int idx = threadIdx.x;
    if ((size_t)idx < take) {
        output[out_offset + (size_t)idx] = state[idx];
    }
}

__global__ void fs_consume_pool_for_indices_kernel(
    const uint64_t* pool_rate, // [RATE]
    size_t upper_bound,
    size_t* out_indices,
    uint32_t* out_count,
    size_t target_count
) {
    // single-threaded consumption (deterministic order)
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    static constexpr uint64_t MAX_VALUE = 0xFFFFFFFF00000000ULL;
    uint32_t c = *out_count;
    // Consume in "ProofStream::sample_indices" order: pool is built from squeezed.rbegin()
    // and then pop_back() => effectively consumes squeezed in forward order (0..RATE-1).
    for (int i = 0; i < RATE && c < target_count; ++i) {
        uint64_t elem = pool_rate[i];
        if (elem != MAX_VALUE) {
            out_indices[c++] = (size_t)(elem % upper_bound);
        }
    }
    *out_count = c;
}

/**
 * Extract rate portion for squeeze
 */
__global__ void fs_extract_rate_kernel(
    const uint64_t* state,
    uint64_t* output
) {
    int idx = threadIdx.x;
    if (idx < RATE) {
        output[idx] = state[idx];
    }
}

// ============================================================================
// Host Interface
// ============================================================================

void fs_init_sponge_gpu(uint64_t* d_state, cudaStream_t stream) {
    fs_init_varlen_kernel<<<1, STATE_SIZE, 0, stream>>>(d_state);
}

void fs_absorb_gpu(
    uint64_t* d_state,
    const uint64_t* h_data,
    size_t data_len,
    const uint16_t* d_sbox_table,
    const uint64_t* d_mds_matrix,
    const uint64_t* d_round_constants,
    cudaStream_t stream
) {
    (void)d_sbox_table;
    (void)d_mds_matrix;
    (void)d_round_constants;
    
    // Pad data on host (matching CPU pad_and_absorb_all)
    std::vector<uint64_t> padded(h_data, h_data + data_len);
    padded.push_back(1);  // Padding indicator
    while (padded.size() % RATE != 0) {
        padded.push_back(0);
    }
    
    // Allocate device buffer for one chunk
    uint64_t* d_chunk;
    cudaMalloc(&d_chunk, RATE * sizeof(uint64_t));
    
    // Absorb in chunks of RATE
    for (size_t chunk = 0; chunk < padded.size(); chunk += RATE) {
        // Upload chunk
        cudaMemcpyAsync(d_chunk, &padded[chunk], RATE * sizeof(uint64_t),
                        cudaMemcpyHostToDevice, stream);
        
        // Overwrite rate portion
        fs_overwrite_rate_kernel<<<1, RATE, 0, stream>>>(d_state, d_chunk);
        
        // Permute using existing Tip5 kernel
        tip5_permutation_gpu(d_state, 1, stream);
    }
    
    cudaFree(d_chunk);
}

void fs_absorb_device_gpu(
    uint64_t* d_state,
    const uint64_t* d_data,
    size_t data_len,
    cudaStream_t stream
) {
    // Pad length: (data_len + 1) rounded up to multiple of RATE
    size_t padded_len = data_len + 1;
    padded_len = ((padded_len + RATE - 1) / RATE) * RATE;

    // Absorb in chunks of RATE: overwrite rate with padded data then permute.
    for (size_t chunk = 0; chunk < padded_len; chunk += RATE) {
        fs_overwrite_rate_from_device_kernel<<<1, RATE, 0, stream>>>(
            d_state, d_data, data_len, chunk
        );
        tip5_permutation_gpu(d_state, 1, stream);
    }
}

void fs_squeeze_gpu(
    uint64_t* d_state,
    uint64_t* d_output,
    const uint16_t* d_sbox_table,
    const uint64_t* d_mds_matrix,
    const uint64_t* d_round_constants,
    cudaStream_t stream
) {
    (void)d_sbox_table;
    (void)d_mds_matrix;
    (void)d_round_constants;
    
    // Extract rate portion
    fs_extract_rate_kernel<<<1, RATE, 0, stream>>>(d_state, d_output);
    
    // Permute for next squeeze
    tip5_permutation_gpu(d_state, 1, stream);
}

void fs_sample_scalars_gpu(
    uint64_t* d_state,
    uint64_t* d_output,
    size_t count,
    const uint16_t* d_sbox_table,
    const uint64_t* d_mds_matrix,
    const uint64_t* d_round_constants,
    cudaStream_t stream
) {
    (void)d_sbox_table;
    (void)d_mds_matrix;
    (void)d_round_constants;
    
    // Each XFieldElement needs 3 BFieldElements
    size_t elements_needed = count * 3;
    
    // Allocate temporary buffer for squeezed elements
    uint64_t* d_pool;
    cudaMalloc(&d_pool, RATE * sizeof(uint64_t));
    
    std::vector<uint64_t> h_output(elements_needed);
    size_t out_idx = 0;
    std::vector<uint64_t> pool;
    
    while (out_idx < elements_needed) {
        // Squeeze if pool is empty
        if (pool.empty()) {
            // Extract rate portion to pool
            fs_extract_rate_kernel<<<1, RATE, 0, stream>>>(d_state, d_pool);
            cudaStreamSynchronize(stream);
            
            std::vector<uint64_t> squeezed(RATE);
            cudaMemcpy(squeezed.data(), d_pool, RATE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
            for (int i = 0; i < RATE; ++i) {
                pool.push_back(squeezed[i]);
            }
            
            // Permute for next squeeze
            tip5_permutation_gpu(d_state, 1, stream);
        }
        
        h_output[out_idx++] = pool.front();
        pool.erase(pool.begin());
    }
    
    // Upload result to device
    cudaMemcpy(d_output, h_output.data(), elements_needed * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    cudaFree(d_pool);
}

void fs_sample_scalars_device_gpu(
    uint64_t* d_state,
    uint64_t* d_output,
    size_t count,
    cudaStream_t stream
) {
    // Each XFieldElement needs 3 BFieldElements
    size_t elements_needed = count * 3;

    size_t out_off = 0;
    while (out_off < elements_needed) {
        size_t take = std::min((size_t)RATE, elements_needed - out_off);
        fs_copy_rate_to_output_kernel<<<1, RATE, 0, stream>>>(
            d_state, d_output, out_off, take
        );
        tip5_permutation_gpu(d_state, 1, stream);
        out_off += take;
    }
}

void fs_sample_indices_gpu(
    uint64_t* d_state,
    size_t* d_output,
    size_t upper_bound,
    size_t count,
    const uint16_t* d_sbox_table,
    const uint64_t* d_mds_matrix,
    const uint64_t* d_round_constants,
    cudaStream_t stream
) {
    (void)d_sbox_table;
    (void)d_mds_matrix;
    (void)d_round_constants;
    
    static constexpr uint64_t MAX_VALUE = 0xFFFFFFFF00000000ULL;
    
    uint64_t* d_pool;
    cudaMalloc(&d_pool, RATE * sizeof(uint64_t));
    
    std::vector<size_t> h_output;
    h_output.reserve(count);
    std::vector<uint64_t> pool;
    
    while (h_output.size() < count) {
        // Refill pool if needed (reverse order to match CPU)
        if (pool.empty()) {
            fs_extract_rate_kernel<<<1, RATE, 0, stream>>>(d_state, d_pool);
            cudaStreamSynchronize(stream);
            
            std::vector<uint64_t> squeezed(RATE);
            cudaMemcpy(squeezed.data(), d_pool, RATE * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            
            // Add in reverse order to match CPU behavior
            for (int i = RATE - 1; i >= 0; --i) {
                pool.push_back(squeezed[i]);
            }
            
            tip5_permutation_gpu(d_state, 1, stream);
        }
        
        uint64_t elem = pool.back();
        pool.pop_back();
        
        // Skip max value for uniform distribution
        if (elem != MAX_VALUE) {
            h_output.push_back(elem % upper_bound);
        }
    }
    
    cudaMemcpy(d_output, h_output.data(), count * sizeof(size_t), cudaMemcpyHostToDevice);
    
    cudaFree(d_pool);
}

void fs_sample_indices_device_gpu(
    uint64_t* d_state,
    size_t* d_output,
    size_t upper_bound,
    size_t count,
    cudaStream_t stream
) {
    // We generate enough squeezes to virtually guarantee `count` accepted indices
    // without ever reading counters back to host.
    // Each squeeze provides RATE candidates; rejection only happens on MAX_VALUE (prob ~ 2^-32).
    size_t max_squeezes = (count / RATE) + 4;

    uint64_t* d_pool = nullptr;
    uint32_t* d_count = nullptr;
    cudaMalloc(&d_pool, RATE * sizeof(uint64_t));
    cudaMalloc(&d_count, sizeof(uint32_t));
    cudaMemsetAsync(d_count, 0, sizeof(uint32_t), stream);

    for (size_t s = 0; s < max_squeezes; ++s) {
        fs_extract_rate_kernel<<<1, RATE, 0, stream>>>(d_state, d_pool);
        tip5_permutation_gpu(d_state, 1, stream);
        fs_consume_pool_for_indices_kernel<<<1, 1, 0, stream>>>(
            d_pool, upper_bound, d_output, d_count, count
        );
    }

    cudaFree(d_pool);
    cudaFree(d_count);
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
