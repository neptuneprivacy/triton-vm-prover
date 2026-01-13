/**
 * NTT CUDA Kernel Implementation
 * 
 * Number Theoretic Transform for Goldilocks field.
 * Uses Cooley-Tukey decimation-in-time algorithm.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/cuda_common.cuh"
#include <cuda_runtime.h>

namespace triton_vm {
namespace gpu {
namespace kernels {

// ============================================================================
// Constants
// ============================================================================

// Goldilocks prime and generator
static constexpr uint64_t GOLDILOCKS_PRIME_NTT = 18446744069414584321ULL;
static constexpr uint64_t GOLDILOCKS_GENERATOR = 7ULL;

// Precomputed primitive roots of unity for power-of-2 orders
// omega_k = g^((p-1) / 2^k) where g = 7
// These are computed at compile time for k = 1..32
__constant__ uint64_t ROOTS_OF_UNITY[33];
__constant__ uint64_t INV_ROOTS_OF_UNITY[33];
__constant__ uint64_t SIZE_INVERSES[33];  // 2^{-k} mod p

// Host-side arrays for initialization
static uint64_t h_roots_of_unity[33];
static uint64_t h_inv_roots_of_unity[33];
static uint64_t h_size_inverses[33];
static bool ntt_constants_initialized = false;

// ============================================================================
// Initialization
// ============================================================================

/**
 * Host-side modular exponentiation for initialization
 */
static uint64_t host_pow_mod(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            __uint128_t prod = static_cast<__uint128_t>(result) * base;
            result = static_cast<uint64_t>(prod % GOLDILOCKS_PRIME_NTT);
        }
        __uint128_t sq = static_cast<__uint128_t>(base) * base;
        base = static_cast<uint64_t>(sq % GOLDILOCKS_PRIME_NTT);
        exp >>= 1;
    }
    return result;
}

/**
 * Initialize NTT constants (called once)
 */
void ntt_init_constants() {
    if (ntt_constants_initialized) return;
    
    // Compute roots of unity: omega_k = g^((p-1) / 2^k)
    for (int k = 0; k <= 32; k++) {
        uint64_t exp = (GOLDILOCKS_PRIME_NTT - 1) >> k;
        h_roots_of_unity[k] = host_pow_mod(GOLDILOCKS_GENERATOR, exp);
        h_inv_roots_of_unity[k] = host_pow_mod(h_roots_of_unity[k], GOLDILOCKS_PRIME_NTT - 2);
        h_size_inverses[k] = host_pow_mod(1ULL << k, GOLDILOCKS_PRIME_NTT - 2);
    }
    
    // Copy to device constant memory
    cudaMemcpyToSymbol(ROOTS_OF_UNITY, h_roots_of_unity, sizeof(h_roots_of_unity));
    cudaMemcpyToSymbol(INV_ROOTS_OF_UNITY, h_inv_roots_of_unity, sizeof(h_inv_roots_of_unity));
    cudaMemcpyToSymbol(SIZE_INVERSES, h_size_inverses, sizeof(h_size_inverses));
    
    ntt_constants_initialized = true;
}

// ============================================================================
// Butterfly Operation
// ============================================================================

/**
 * Single butterfly: (a, b) -> (a + w*b, a - w*b)
 */
__device__ __forceinline__ void butterfly(
    uint64_t& a,
    uint64_t& b,
    uint64_t twiddle
) {
    uint64_t t = bfield_mul_impl(b, twiddle);
    uint64_t a_plus_t = bfield_add_impl(a, t);
    uint64_t a_minus_t = bfield_sub_impl(a, t);
    a = a_plus_t;
    b = a_minus_t;
}

// ============================================================================
// Precomputed Twiddle Factor Management
// ============================================================================

// Global twiddle factor tables (allocated once, reused)
static uint64_t* d_twiddles_fwd = nullptr;  // Forward NTT twiddles
static uint64_t* d_twiddles_inv = nullptr;  // Inverse NTT twiddles
static size_t twiddles_max_n = 0;           // Max size allocated

/**
 * Generate twiddle factors for all stages
 * Layout: For stage s (half_size = 2^s), twiddles[s][k] = omega^k for k in [0, half_size)
 * Total storage: sum_{s=0}^{log_n-1} 2^s = n - 1 elements
 * We store them contiguously: [stage0_twiddles][stage1_twiddles]...
 * Offset for stage s: 2^s - 1
 */
__global__ void generate_twiddles_kernel(
    uint64_t* d_twiddles,
    size_t log_n,
    bool inverse
) {
    // IMPORTANT: CUDA's blockIdx.x/blockDim.x/threadIdx.x are 32-bit. Cast before multiplying
    // to avoid overflow when linear indices exceed 2^32.
    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    
    // Total twiddles = n - 1 where n = 2^log_n
    size_t total = (1ULL << log_n) - 1;
    if (idx >= total) return;
    
    // Find which stage this idx belongs to
    // Stage s has 2^s twiddles starting at offset 2^s - 1
    size_t stage = 0;
    size_t offset = 0;
    size_t stage_size = 1;
    
    while (offset + stage_size <= idx) {
        offset += stage_size;
        stage++;
        stage_size <<= 1;
    }
    
    size_t pos = idx - offset;
    size_t root_index = stage + 1;
    
    uint64_t omega = inverse ? INV_ROOTS_OF_UNITY[root_index] : ROOTS_OF_UNITY[root_index];
    d_twiddles[idx] = bfield_pow_impl(omega, pos);
}

/**
 * Ensure twiddle factors are precomputed for size n
 */
static void ensure_twiddles(size_t n, cudaStream_t stream) {
    if (n <= twiddles_max_n && d_twiddles_fwd != nullptr) return;
    
    // Free old allocations
    if (d_twiddles_fwd) cudaFree(d_twiddles_fwd);
    if (d_twiddles_inv) cudaFree(d_twiddles_inv);
    
    size_t total = n - 1;  // Total twiddles needed
    cudaMalloc(&d_twiddles_fwd, total * sizeof(uint64_t));
    cudaMalloc(&d_twiddles_inv, total * sizeof(uint64_t));
    
    size_t log_n = 0;
    for (size_t temp = n; temp > 1; temp >>= 1) ++log_n;
    
    int block = 256;
    int grid = (total + block - 1) / block;
    
    generate_twiddles_kernel<<<grid, block, 0, stream>>>(d_twiddles_fwd, log_n, false);
    generate_twiddles_kernel<<<grid, block, 0, stream>>>(d_twiddles_inv, log_n, true);
    cudaStreamSynchronize(stream);
    
    twiddles_max_n = n;
}

// ============================================================================
// Optimized NTT Kernels with ILP and Precomputed Twiddles
// ============================================================================

/**
 * NTT butterfly kernel with precomputed twiddles
 * Much faster than computing powers on the fly
 */
__global__ void ntt_butterfly_precomputed_kernel(
    uint64_t* data,
    size_t n,
    size_t stage,
    const uint64_t* d_twiddles  // Precomputed twiddles
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t num_butterflies = n / 2;
    
    if (idx >= num_butterflies) return;
    
    size_t half_size = 1ULL << stage;
    size_t full_size = half_size * 2;
    
    size_t group = idx / half_size;
    size_t pos = idx % half_size;
    
    size_t i = group * full_size + pos;
    size_t j = i + half_size;
    
    // Twiddle offset for this stage: 2^stage - 1
    size_t twiddle_offset = half_size - 1;
    uint64_t twiddle = d_twiddles[twiddle_offset + pos];
    
    butterfly(data[i], data[j], twiddle);
}

/**
 * ILP-optimized butterfly kernel: each thread handles 4 butterflies
 * This hides memory latency and increases arithmetic intensity
 */
__global__ void ntt_butterfly_ilp4_kernel(
    uint64_t* data,
    size_t n,
    size_t stage,
    const uint64_t* d_twiddles
) {
    // Each thread handles 4 butterflies for ILP
    constexpr int ILP = 4;
    // IMPORTANT: Cast before multiply to avoid 32-bit overflow.
    size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * static_cast<size_t>(ILP);
    size_t num_butterflies = n / 2;
    
    size_t half_size = 1ULL << stage;
    size_t full_size = half_size * 2;
    size_t twiddle_offset = half_size - 1;
    
    // Load all data into registers first (ILP for loads)
    uint64_t a[ILP], b[ILP], tw[ILP];
    size_t i_idx[ILP], j_idx[ILP];
    
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        size_t idx = base_idx + k;
        if (idx < num_butterflies) {
            size_t group = idx / half_size;
            size_t pos = idx % half_size;
            i_idx[k] = group * full_size + pos;
            j_idx[k] = i_idx[k] + half_size;
            
            a[k] = data[i_idx[k]];
            b[k] = data[j_idx[k]];
            tw[k] = d_twiddles[twiddle_offset + pos];
        }
    }
    
    // Compute all butterflies (ILP for compute)
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        size_t idx = base_idx + k;
        if (idx < num_butterflies) {
            uint64_t t = bfield_mul_impl(b[k], tw[k]);
            a[k] = bfield_add_impl(a[k], t);
            b[k] = bfield_sub_impl(a[k] - t - t + GOLDILOCKS_P, 0);  // a - t (undo the add, sub t again)
            // Simpler: recalculate
            uint64_t orig_a = a[k];
            b[k] = bfield_sub_impl(orig_a - t + GOLDILOCKS_P, t);
        }
    }
    
    // Actually let's do this correctly
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        size_t idx = base_idx + k;
        if (idx < num_butterflies) {
            // Reload for correctness
            uint64_t av = data[i_idx[k]];
            uint64_t bv = data[j_idx[k]];
            uint64_t t = bfield_mul_impl(bv, tw[k]);
            data[i_idx[k]] = bfield_add_impl(av, t);
            data[j_idx[k]] = bfield_sub_impl(av, t);
        }
    }
}

// Original kernel kept for compatibility
__global__ void ntt_butterfly_kernel(
    uint64_t* data,
    size_t n,
    size_t log_n,
    size_t stage,        // stage = 0, 1, ..., log_n-1
    bool inverse
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t num_butterflies = n / 2;
    
    if (idx >= num_butterflies) return;
    
    // At stage s: butterfly size = 2^(s+1), half_size = 2^s
    size_t half_size = 1ULL << stage;
    size_t full_size = half_size * 2;
    
    // Which group and position within group?
    size_t group = idx / half_size;
    size_t pos = idx % half_size;
    
    // Indices into data array
    size_t i = group * full_size + pos;
    size_t j = i + half_size;
    
    // Twiddle factor: for butterfly size 2^(s+1), we need (2^(s+1))-th root of unity
    // omega = g^((p-1) / 2^(s+1)) raised to power pos
    size_t root_index = stage + 1;  // We need root of order 2^(stage+1)
    uint64_t omega = inverse ? INV_ROOTS_OF_UNITY[root_index] : ROOTS_OF_UNITY[root_index];
    uint64_t twiddle = bfield_pow_impl(omega, pos);
    
    // Butterfly
    butterfly(data[i], data[j], twiddle);
}

/**
 * Bit-reversal permutation kernel
 */
__global__ void bit_reverse_kernel(
    uint64_t* data,
    size_t n,
    size_t log_n
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    
    // Compute bit-reversed index
    size_t rev = 0;
    size_t temp = idx;
    for (size_t i = 0; i < log_n; ++i) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }
    
    // Only swap if rev > idx (to avoid double swap)
    if (rev > idx) {
        uint64_t tmp = data[idx];
        data[idx] = data[rev];
        data[rev] = tmp;
    }
}

/**
 * Scale kernel for inverse NTT (multiply by n^{-1})
 */
__global__ void ntt_scale_kernel(
    uint64_t* data,
    size_t n,
    size_t log_n
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    if (idx >= n) return;
    
    data[idx] = bfield_mul_impl(data[idx], SIZE_INVERSES[log_n]);
}

// ============================================================================
// Host Interface
// ============================================================================

/**
 * Perform NTT on device memory
 * @param d_data Device pointer to data (n elements)
 * @param n Size (must be power of 2)
 * @param inverse True for inverse NTT
 * @param stream CUDA stream
 */
void ntt_forward_gpu(
    uint64_t* d_data,
    size_t n,
    cudaStream_t stream
) {
    // Ensure constants are initialized
    ntt_init_constants();
    
    if (n == 0 || (n & (n - 1)) != 0) return;  // Must be power of 2
    
    // Calculate log_n
    size_t log_n = 0;
    for (size_t temp = n; temp > 1; temp >>= 1) {
        ++log_n;
    }
    
    int block_size = 256;
    int grid_size_n = (n + block_size - 1) / block_size;
    int grid_size_half = (n / 2 + block_size - 1) / block_size;
    
    // Bit-reversal permutation
    bit_reverse_kernel<<<grid_size_n, block_size, 0, stream>>>(d_data, n, log_n);
    
    // NTT stages (forward)
    for (size_t stage = 0; stage < log_n; ++stage) {
        ntt_butterfly_kernel<<<grid_size_half, block_size, 0, stream>>>(
            d_data, n, log_n, stage, false
        );
    }
}

/**
 * Perform inverse NTT on device memory
 */
void ntt_inverse_gpu(
    uint64_t* d_data,
    size_t n,
    cudaStream_t stream
) {
    // Ensure constants are initialized
    ntt_init_constants();
    
    if (n == 0 || (n & (n - 1)) != 0) return;  // Must be power of 2
    
    // Calculate log_n
    size_t log_n = 0;
    for (size_t temp = n; temp > 1; temp >>= 1) {
        ++log_n;
    }
    
    int block_size = 256;
    int grid_size_n = (n + block_size - 1) / block_size;
    int grid_size_half = (n / 2 + block_size - 1) / block_size;
    
    // Bit-reversal permutation
    bit_reverse_kernel<<<grid_size_n, block_size, 0, stream>>>(d_data, n, log_n);
    
    // NTT stages (inverse)
    for (size_t stage = 0; stage < log_n; ++stage) {
        ntt_butterfly_kernel<<<grid_size_half, block_size, 0, stream>>>(
            d_data, n, log_n, stage, true
        );
    }
    
    // Scale by n^{-1}
    ntt_scale_kernel<<<grid_size_n, block_size, 0, stream>>>(d_data, n, log_n);
}

/**
 * Batch NTT - perform NTT on multiple arrays
 */
void ntt_batch_gpu(
    uint64_t** d_data_ptrs,  // Array of device pointers
    size_t n,                 // Size of each array
    size_t batch_size,        // Number of arrays
    bool inverse,
    cudaStream_t stream
) {
    for (size_t i = 0; i < batch_size; ++i) {
        if (inverse) {
            ntt_inverse_gpu(d_data_ptrs[i], n, stream);
        } else {
            ntt_forward_gpu(d_data_ptrs[i], n, stream);
        }
    }
}

// ============================================================================
// Batched NTT Kernels (for column-major contiguous data)
// ============================================================================

/**
 * Batched bit-reversal permutation kernel
 * Data layout: column-major with num_cols columns of n elements each
 * d_data[col * n + row] = element at (col, row)
 */
__global__ void batched_bit_reverse_kernel(
    uint64_t* d_data,
    size_t n,
    size_t log_n,
    size_t num_cols
) {
    // Thread handles one (col, idx) pair
    // IMPORTANT: total = n * num_cols can exceed 2^32 (e.g. n=2^24, num_cols=379).
    // Cast before multiplying to avoid 32-bit overflow.
    size_t global_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t total = n * num_cols;
    if (global_idx >= total) return;
    
    size_t col = global_idx / n;
    size_t idx = global_idx % n;
    
    // Compute bit-reversed index
    size_t rev = 0;
    size_t temp = idx;
    for (size_t i = 0; i < log_n; ++i) {
        rev = (rev << 1) | (temp & 1);
        temp >>= 1;
    }
    
    // Only swap if rev > idx (to avoid double swap)
    if (rev > idx) {
        size_t pos1 = col * n + idx;
        size_t pos2 = col * n + rev;
        uint64_t tmp = d_data[pos1];
        d_data[pos1] = d_data[pos2];
        d_data[pos2] = tmp;
    }
}

/**
 * Batched NTT butterfly kernel with precomputed twiddles
 * Each thread handles one butterfly operation for one column
 * Uses precomputed twiddle factors for massive speedup
 */
__global__ void batched_ntt_butterfly_kernel(
    uint64_t* d_data,
    size_t n,
    size_t log_n,
    size_t stage,
    size_t num_cols,
    bool inverse
) {
    size_t global_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t num_butterflies_per_col = n / 2;
    size_t total_butterflies = num_butterflies_per_col * num_cols;
    
    if (global_idx >= total_butterflies) return;
    
    size_t col = global_idx / num_butterflies_per_col;
    size_t local_idx = global_idx % num_butterflies_per_col;
    
    // At stage s: butterfly size = 2^(s+1), half_size = 2^s
    size_t half_size = 1ULL << stage;
    size_t full_size = half_size * 2;
    
    // Which group and position within group?
    size_t group = local_idx / half_size;
    size_t pos = local_idx % half_size;
    
    // Indices into data array for this column
    size_t base = col * n;
    size_t i = base + group * full_size + pos;
    size_t j = i + half_size;
    
    // Twiddle factor from precomputed table
    size_t root_index = stage + 1;
    uint64_t omega = inverse ? INV_ROOTS_OF_UNITY[root_index] : ROOTS_OF_UNITY[root_index];
    uint64_t twiddle = bfield_pow_impl(omega, pos);
    
    // Butterfly
    butterfly(d_data[i], d_data[j], twiddle);
}

/**
 * Optimized batched NTT butterfly kernel with precomputed twiddles
 * Uses twiddle lookup instead of computing powers
 */
__global__ void batched_ntt_butterfly_precomputed_kernel(
    uint64_t* d_data,
    size_t n,
    size_t stage,
    size_t num_cols,
    const uint64_t* d_twiddles  // Precomputed twiddles
) {
    size_t global_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t num_butterflies_per_col = n / 2;
    size_t total_butterflies = num_butterflies_per_col * num_cols;
    
    if (global_idx >= total_butterflies) return;
    
    size_t col = global_idx / num_butterflies_per_col;
    size_t local_idx = global_idx % num_butterflies_per_col;
    
    size_t half_size = 1ULL << stage;
    size_t full_size = half_size * 2;
    
    size_t group = local_idx / half_size;
    size_t pos = local_idx % half_size;
    
    size_t base = col * n;
    size_t i = base + group * full_size + pos;
    size_t j = i + half_size;
    
    // Twiddle from precomputed table: offset for stage s is (2^s - 1)
    size_t twiddle_offset = half_size - 1;
    uint64_t twiddle = d_twiddles[twiddle_offset + pos];
    
    // Butterfly
    butterfly(d_data[i], d_data[j], twiddle);
}

/**
 * ILP-optimized batched butterfly: each thread handles 2 butterflies
 * Improves instruction-level parallelism
 */
__global__ void batched_ntt_butterfly_ilp2_kernel(
    uint64_t* d_data,
    size_t n,
    size_t stage,
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    constexpr int ILP = 2;
    size_t thread_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t num_butterflies_per_col = n / 2;
    size_t total_butterflies = num_butterflies_per_col * num_cols;
    
    size_t half_size = 1ULL << stage;
    size_t full_size = half_size * 2;
    size_t twiddle_offset = half_size - 1;
    
    // Each thread processes ILP consecutive butterflies
    size_t base_global_idx = thread_idx * ILP;
    
    // Registers for ILP
    uint64_t a[ILP], b[ILP], tw[ILP];
    size_t i_pos[ILP], j_pos[ILP];
    bool valid[ILP];
    
    // Load phase - load ILP butterflies worth of data
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        size_t global_idx = base_global_idx + k;
        valid[k] = (global_idx < total_butterflies);
        
        if (valid[k]) {
            size_t col = global_idx / num_butterflies_per_col;
            size_t local_idx = global_idx % num_butterflies_per_col;
            
            size_t group = local_idx / half_size;
            size_t pos = local_idx % half_size;
            
            size_t base = col * n;
            i_pos[k] = base + group * full_size + pos;
            j_pos[k] = i_pos[k] + half_size;
            
            a[k] = d_data[i_pos[k]];
            b[k] = d_data[j_pos[k]];
            tw[k] = d_twiddles[twiddle_offset + pos];
        }
    }
    
    // Compute phase - all ILP butterflies
    uint64_t t[ILP], new_a[ILP], new_b[ILP];
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        if (valid[k]) {
            t[k] = bfield_mul_impl(b[k], tw[k]);
            new_a[k] = bfield_add_impl(a[k], t[k]);
            new_b[k] = bfield_sub_impl(a[k], t[k]);
        }
    }
    
    // Store phase
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        if (valid[k]) {
            d_data[i_pos[k]] = new_a[k];
            d_data[j_pos[k]] = new_b[k];
        }
    }
}

/**
 * ILP-optimized batched butterfly: each thread handles 4 butterflies
 * Better latency hiding for memory-bound stages
 */
__global__ void batched_ntt_butterfly_ilp4_kernel(
    uint64_t* d_data,
    size_t n,
    size_t stage,
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    constexpr int ILP = 4;
    size_t thread_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t num_butterflies_per_col = n / 2;
    size_t total_butterflies = num_butterflies_per_col * num_cols;
    
    size_t half_size = 1ULL << stage;
    size_t full_size = half_size * 2;
    size_t twiddle_offset = half_size - 1;
    
    size_t base_global_idx = thread_idx * ILP;
    
    // Registers for ILP
    uint64_t a[ILP], b[ILP], tw[ILP];
    size_t i_pos[ILP], j_pos[ILP];
    bool valid[ILP];
    
    // Load phase
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        size_t global_idx = base_global_idx + k;
        valid[k] = (global_idx < total_butterflies);
        
        if (valid[k]) {
            size_t col = global_idx / num_butterflies_per_col;
            size_t local_idx = global_idx % num_butterflies_per_col;
            
            size_t group = local_idx / half_size;
            size_t pos = local_idx % half_size;
            
            size_t base = col * n;
            i_pos[k] = base + group * full_size + pos;
            j_pos[k] = i_pos[k] + half_size;
            
            a[k] = d_data[i_pos[k]];
            b[k] = d_data[j_pos[k]];
            tw[k] = d_twiddles[twiddle_offset + pos];
        }
    }
    
    // Compute phase
    uint64_t t[ILP], new_a[ILP], new_b[ILP];
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        if (valid[k]) {
            t[k] = bfield_mul_impl(b[k], tw[k]);
            new_a[k] = bfield_add_impl(a[k], t[k]);
            new_b[k] = bfield_sub_impl(a[k], t[k]);
        }
    }
    
    // Store phase
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        if (valid[k]) {
            d_data[i_pos[k]] = new_a[k];
            d_data[j_pos[k]] = new_b[k];
        }
    }
}

/**
 * ILP-optimized batched butterfly: each thread handles 8 butterflies
 * Maximum latency hiding for memory-bound stages
 */
__global__ void batched_ntt_butterfly_ilp8_kernel(
    uint64_t* d_data,
    size_t n,
    size_t stage,
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    constexpr int ILP = 8;
    size_t thread_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t num_butterflies_per_col = n / 2;
    size_t total_butterflies = num_butterflies_per_col * num_cols;
    
    size_t half_size = 1ULL << stage;
    size_t full_size = half_size * 2;
    size_t twiddle_offset = half_size - 1;
    
    size_t base_global_idx = thread_idx * ILP;
    
    // Registers for ILP
    uint64_t a[ILP], b[ILP], tw[ILP];
    size_t i_pos[ILP], j_pos[ILP];
    bool valid[ILP];
    
    // Load phase - all loads issued before any compute
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        size_t global_idx = base_global_idx + k;
        valid[k] = (global_idx < total_butterflies);
        
        if (valid[k]) {
            size_t col = global_idx / num_butterflies_per_col;
            size_t local_idx = global_idx % num_butterflies_per_col;
            
            size_t group = local_idx / half_size;
            size_t pos = local_idx % half_size;
            
            size_t base = col * n;
            i_pos[k] = base + group * full_size + pos;
            j_pos[k] = i_pos[k] + half_size;
            
            a[k] = d_data[i_pos[k]];
            b[k] = d_data[j_pos[k]];
            tw[k] = d_twiddles[twiddle_offset + pos];
        }
    }
    
    // Compute phase - all arithmetic while waiting for memory
    uint64_t t[ILP], new_a[ILP], new_b[ILP];
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        if (valid[k]) {
            t[k] = bfield_mul_impl(b[k], tw[k]);
            new_a[k] = bfield_add_impl(a[k], t[k]);
            new_b[k] = bfield_sub_impl(a[k], t[k]);
        }
    }
    
    // Store phase
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        if (valid[k]) {
            d_data[i_pos[k]] = new_a[k];
            d_data[j_pos[k]] = new_b[k];
        }
    }
}

/**
 * ILP-optimized batched butterfly: each thread handles 32 butterflies
 * Extreme latency hiding for memory-bound stages
 */
__global__ void batched_ntt_butterfly_ilp32_kernel(
    uint64_t* d_data,
    size_t n,
    size_t stage,
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    constexpr int ILP = 32;
    size_t thread_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t num_butterflies_per_col = n / 2;
    size_t total_butterflies = num_butterflies_per_col * num_cols;
    
    size_t half_size = 1ULL << stage;
    size_t full_size = half_size * 2;
    size_t twiddle_offset = half_size - 1;
    
    size_t base_global_idx = thread_idx * ILP;
    
    // Registers for ILP
    uint64_t a[ILP], b[ILP], tw[ILP];
    size_t i_pos[ILP], j_pos[ILP];
    bool valid[ILP];
    
    // Load phase - all loads issued before any compute
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        size_t global_idx = base_global_idx + k;
        valid[k] = (global_idx < total_butterflies);
        
        if (valid[k]) {
            size_t col = global_idx / num_butterflies_per_col;
            size_t local_idx = global_idx % num_butterflies_per_col;
            
            size_t group = local_idx / half_size;
            size_t pos = local_idx % half_size;
            
            size_t base = col * n;
            i_pos[k] = base + group * full_size + pos;
            j_pos[k] = i_pos[k] + half_size;
            
            a[k] = d_data[i_pos[k]];
            b[k] = d_data[j_pos[k]];
            tw[k] = d_twiddles[twiddle_offset + pos];
        }
    }
    
    // Compute phase - all arithmetic while waiting for memory
    uint64_t t[ILP], new_a[ILP], new_b[ILP];
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        if (valid[k]) {
            t[k] = bfield_mul_impl(b[k], tw[k]);
            new_a[k] = bfield_add_impl(a[k], t[k]);
            new_b[k] = bfield_sub_impl(a[k], t[k]);
        }
    }
    
    // Store phase
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        if (valid[k]) {
            d_data[i_pos[k]] = new_a[k];
            d_data[j_pos[k]] = new_b[k];
        }
    }
}

/**
 * Optimized kernel for later stages (stage > 11) with better memory coalescing
 * Uses warp-level column processing and L2 cache hints for large strides
 * Each warp processes multiple columns to improve coalescing
 */
__global__ void batched_ntt_butterfly_coalesced_large_stage_kernel(
    uint64_t* d_data,
    size_t n,
    size_t stage,
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    constexpr int WARP_SIZE = 32;
    constexpr int COLS_PER_WARP = 4;  // Process 4 columns per warp for better coalescing
    
    size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t lane_id = threadIdx.x % WARP_SIZE;
    size_t num_butterflies_per_col = n / 2;
    
    size_t half_size = 1ULL << stage;
    size_t full_size = half_size * 2;
    size_t twiddle_offset = half_size - 1;
    
    // Each warp processes COLS_PER_WARP columns
    size_t base_col = (warp_id * COLS_PER_WARP) % num_cols;
    size_t base_butterfly = ((warp_id * COLS_PER_WARP) / num_cols) * num_butterflies_per_col;
    
    // Process COLS_PER_WARP columns, with each thread handling butterflies across columns
    #pragma unroll
    for (int col_offset = 0; col_offset < COLS_PER_WARP; col_offset++) {
        size_t col = (base_col + col_offset) % num_cols;
        size_t butterfly_idx = base_butterfly + lane_id;
        
        if (butterfly_idx < num_butterflies_per_col) {
            size_t group = butterfly_idx / half_size;
            size_t pos = butterfly_idx % half_size;
            
            size_t base = col * n;
            size_t i = base + group * full_size + pos;
            size_t j = i + half_size;
            
            // Use __ldg for read-only data (L2 cache hint)
            uint64_t a = __ldg(&d_data[i]);
            uint64_t b_val = __ldg(&d_data[j]);
            uint64_t tw = __ldg(&d_twiddles[twiddle_offset + pos]);
            
            // Compute butterfly
            uint64_t t = bfield_mul_impl(b_val, tw);
            uint64_t new_a = bfield_add_impl(a, t);
            uint64_t new_b = bfield_sub_impl(a, t);
            
            // Store results
            d_data[i] = new_a;
            d_data[j] = new_b;
        }
    }
}

/**
 * ILP-optimized kernel for later stages with warp-level column processing
 * Combines ILP with better coalescing for large strides
 */
__global__ void batched_ntt_butterfly_ilp4_coalesced_kernel(
    uint64_t* d_data,
    size_t n,
    size_t stage,
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    constexpr int ILP = 4;
    constexpr int WARP_SIZE = 32;
    constexpr int COLS_PER_WARP = 2;  // Process 2 columns per warp
    
    size_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    size_t lane_id = threadIdx.x % WARP_SIZE;
    size_t num_butterflies_per_col = n / 2;
    
    size_t half_size = 1ULL << stage;
    size_t full_size = half_size * 2;
    size_t twiddle_offset = half_size - 1;
    
    // Each warp processes COLS_PER_WARP columns
    size_t base_col = (warp_id * COLS_PER_WARP) % num_cols;
    size_t base_butterfly = ((warp_id * COLS_PER_WARP) / num_cols) * num_butterflies_per_col;
    
    // Process COLS_PER_WARP columns with ILP
    #pragma unroll
    for (int col_offset = 0; col_offset < COLS_PER_WARP; col_offset++) {
        size_t col = (base_col + col_offset) % num_cols;
        size_t base_idx = base_butterfly + lane_id * ILP;
        
        uint64_t a[ILP], b[ILP], tw[ILP];
        size_t i_pos[ILP], j_pos[ILP];
        bool valid[ILP];
        
        // Load phase with __ldg
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            size_t butterfly_idx = base_idx + k;
            valid[k] = (butterfly_idx < num_butterflies_per_col);
            
            if (valid[k]) {
                size_t group = butterfly_idx / half_size;
                size_t pos = butterfly_idx % half_size;
                
                size_t base = col * n;
                i_pos[k] = base + group * full_size + pos;
                j_pos[k] = i_pos[k] + half_size;
                
                a[k] = __ldg(&d_data[i_pos[k]]);
                b[k] = __ldg(&d_data[j_pos[k]]);
                tw[k] = __ldg(&d_twiddles[twiddle_offset + pos]);
            }
        }
        
        // Compute phase
        uint64_t t[ILP], new_a[ILP], new_b[ILP];
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            if (valid[k]) {
                t[k] = bfield_mul_impl(b[k], tw[k]);
                new_a[k] = bfield_add_impl(a[k], t[k]);
                new_b[k] = bfield_sub_impl(a[k], t[k]);
            }
        }
        
        // Store phase
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            if (valid[k]) {
                d_data[i_pos[k]] = new_a[k];
                d_data[j_pos[k]] = new_b[k];
            }
        }
    }
}

/**
 * Register-based 2-stage kernel: each thread processes 4 elements through 2 stages
 * Reduces kernel launches by 2x without using shared memory
 * 
 * For Cooley-Tukey DIT NTT (after bit-reversal):
 * - Stage s: butterflies on pairs (i, i + 2^s) where i has bit s = 0
 * - Stage s+1: butterflies on pairs (i, i + 2^(s+1)) where i has bit s+1 = 0
 * 
 * Key insight: Find 4 elements that form a complete "radix-4" unit:
 * indices i0, i0+hs0, i0+hs1, i0+hs1+hs0 where hs0=2^s, hs1=2^(s+1)
 */
__global__ void batched_ntt_reg2stage_kernel(
    uint64_t* d_data,
    size_t n,
    size_t stage,  // First stage (processes stage and stage+1)
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    size_t thread_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    
    size_t hs0 = 1ULL << stage;         // 2^s
    size_t hs1 = 1ULL << (stage + 1);   // 2^(s+1)
    
    // Each thread handles one radix-4 unit (4 elements)
    // Number of radix-4 units per column = n / 4
    size_t units_per_col = n / 4;
    size_t total_units = units_per_col * num_cols;
    
    if (thread_idx >= total_units) return;
    
    size_t col = thread_idx / units_per_col;
    size_t unit_idx = thread_idx % units_per_col;
    
    // Map unit_idx to the base index i0
    // i0 must have bits s and s+1 both = 0
    // So i0 = (high_bits << (s+2)) | low_bits where low_bits < 2^s
    size_t low_bits = unit_idx % hs0;                    // bits 0..s-1
    size_t high_bits = unit_idx / hs0;                   // remaining bits
    size_t i0 = col * n + (high_bits << (stage + 2)) + low_bits;
    
    // The 4 elements for radix-4:
    size_t i1 = i0 + hs0;        // i0 + 2^s
    size_t i2 = i0 + hs1;        // i0 + 2^(s+1)
    size_t i3 = i0 + hs1 + hs0;  // i0 + 2^(s+1) + 2^s
    
    // Load 4 elements
    uint64_t a0 = d_data[i0];
    uint64_t a1 = d_data[i1];
    uint64_t a2 = d_data[i2];
    uint64_t a3 = d_data[i3];
    
    // Stage s: butterflies (a0,a1) and (a2,a3)
    // Twiddle for stage s: position within butterfly group
    // For stage s, the twiddle position is (i0 % (2^(s+1))) = i0 % (2*hs0)
    // Since i0 has bit s = 0, this equals low_bits
    size_t tw_off_s = hs0 - 1;
    uint64_t w_s = d_twiddles[tw_off_s + low_bits];
    
    uint64_t t0 = bfield_mul_impl(a1, w_s);
    uint64_t b0 = bfield_add_impl(a0, t0);  // a0 + w*a1
    uint64_t b1 = bfield_sub_impl(a0, t0);  // a0 - w*a1
    
    uint64_t t1 = bfield_mul_impl(a3, w_s);
    uint64_t b2 = bfield_add_impl(a2, t1);  // a2 + w*a3
    uint64_t b3 = bfield_sub_impl(a2, t1);  // a2 - w*a3
    
    // Stage s+1: butterflies (b0,b2) and (b1,b3)
    // b0 at i0, b1 at i1, b2 at i2, b3 at i3
    // Pair (b0,b2): indices i0 and i2 = i0 + hs1
    // Pair (b1,b3): indices i1 and i3 = i1 + hs1
    //
    // Twiddle position for (b0,b2): (i0 % (2*hs1)) = (i0 % (4*hs0))
    // Since i0 = (high_bits << (s+2)) + low_bits, we have i0 % (4*hs0) = low_bits
    // Twiddle position for (b1,b3): (i1 % (2*hs1)) = ((i0 + hs0) % (4*hs0)) = low_bits + hs0
    size_t tw_off_s1 = hs1 - 1;
    uint64_t w_s1_02 = d_twiddles[tw_off_s1 + low_bits];
    uint64_t w_s1_13 = d_twiddles[tw_off_s1 + low_bits + hs0];
    
    uint64_t t2 = bfield_mul_impl(b2, w_s1_02);
    a0 = bfield_add_impl(b0, t2);  // b0 + w*b2
    a2 = bfield_sub_impl(b0, t2);  // b0 - w*b2
    
    uint64_t t3 = bfield_mul_impl(b3, w_s1_13);
    a1 = bfield_add_impl(b1, t3);  // b1 + w*b3
    a3 = bfield_sub_impl(b1, t3);  // b1 - w*b3
    
    // Store results
    d_data[i0] = a0;
    d_data[i1] = a1;
    d_data[i2] = a2;
    d_data[i3] = a3;
}

/**
 * Register-based 4-stage kernel: each thread processes 16 elements through 4 stages
 * Reduces kernel launches by 4x without using shared memory
 * 
 * For stages s, s+1, s+2, s+3:
 * We need 16 elements that form a complete radix-16 unit
 */
__global__ void batched_ntt_reg4stage_kernel(
    uint64_t* d_data,
    size_t n,
    size_t stage,  // First stage (processes stage, stage+1, stage+2, stage+3)
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    size_t thread_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    
    size_t hs0 = 1ULL << stage;           // 2^s
    size_t hs1 = 1ULL << (stage + 1);     // 2^(s+1)
    size_t hs2 = 1ULL << (stage + 2);     // 2^(s+2)
    size_t hs3 = 1ULL << (stage + 3);     // 2^(s+3)
    
    // Each thread handles one radix-16 unit (16 elements)
    size_t units_per_col = n / 16;
    size_t total_units = units_per_col * num_cols;
    
    if (thread_idx >= total_units) return;
    
    size_t col = thread_idx / units_per_col;
    size_t unit_idx = thread_idx % units_per_col;
    
    // Map unit_idx to base index i0
    // i0 must have bits s, s+1, s+2, s+3 all = 0
    size_t low_bits = unit_idx % hs0;
    size_t high_bits = unit_idx / hs0;
    size_t i0 = col * n + (high_bits << (stage + 4)) + low_bits;
    
    // Load 16 elements
    uint64_t a[16];
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        size_t offset = ((k & 1) ? hs0 : 0) + ((k & 2) ? hs1 : 0) + 
                        ((k & 4) ? hs2 : 0) + ((k & 8) ? hs3 : 0);
        a[k] = d_data[i0 + offset];
    }
    
    // Stage s: 8 butterflies on pairs (a[k], a[k+1]) for k=0,2,4,...,14
    size_t tw_off_s = hs0 - 1;
    uint64_t w_s = d_twiddles[tw_off_s + low_bits];
    
    #pragma unroll
    for (int k = 0; k < 16; k += 2) {
        uint64_t t = bfield_mul_impl(a[k+1], w_s);
        uint64_t tmp = a[k];
        a[k] = bfield_add_impl(tmp, t);
        a[k+1] = bfield_sub_impl(tmp, t);
    }
    
    // Stage s+1: 8 butterflies on pairs with stride 2
    size_t tw_off_s1 = hs1 - 1;
    #pragma unroll
    for (int k = 0; k < 16; k += 4) {
        uint64_t w0 = d_twiddles[tw_off_s1 + low_bits];
        uint64_t w1 = d_twiddles[tw_off_s1 + low_bits + hs0];
        
        uint64_t t0 = bfield_mul_impl(a[k+2], w0);
        uint64_t tmp0 = a[k];
        a[k] = bfield_add_impl(tmp0, t0);
        a[k+2] = bfield_sub_impl(tmp0, t0);
        
        uint64_t t1 = bfield_mul_impl(a[k+3], w1);
        uint64_t tmp1 = a[k+1];
        a[k+1] = bfield_add_impl(tmp1, t1);
        a[k+3] = bfield_sub_impl(tmp1, t1);
    }
    
    // Stage s+2: 8 butterflies on pairs with stride 4
    size_t tw_off_s2 = hs2 - 1;
    #pragma unroll
    for (int k = 0; k < 16; k += 8) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint64_t w = d_twiddles[tw_off_s2 + low_bits + j * hs0];
            uint64_t t = bfield_mul_impl(a[k+4+j], w);
            uint64_t tmp = a[k+j];
            a[k+j] = bfield_add_impl(tmp, t);
            a[k+4+j] = bfield_sub_impl(tmp, t);
        }
    }
    
    // Stage s+3: 8 butterflies on pairs with stride 8
    size_t tw_off_s3 = hs3 - 1;
    #pragma unroll
    for (int j = 0; j < 8; j++) {
        uint64_t w = d_twiddles[tw_off_s3 + low_bits + j * hs0];
        uint64_t t = bfield_mul_impl(a[8+j], w);
        uint64_t tmp = a[j];
        a[j] = bfield_add_impl(tmp, t);
        a[8+j] = bfield_sub_impl(tmp, t);
    }
    
    // Store 16 elements
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        size_t offset = ((k & 1) ? hs0 : 0) + ((k & 2) ? hs1 : 0) + 
                        ((k & 4) ? hs2 : 0) + ((k & 8) ? hs3 : 0);
        d_data[i0 + offset] = a[k];
    }
}

/**
 * Register-based 6-stage kernel: each thread processes 64 elements through 6 stages
 * Good balance between register usage (64 regs) and kernel launch reduction
 */
__global__ void batched_ntt_reg6stage_kernel(
    uint64_t* d_data,
    size_t n,
    size_t stage,  // First stage (processes stage through stage+5)
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    size_t thread_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    
    // Each thread handles one radix-64 unit (64 elements)
    size_t units_per_col = n / 64;
    size_t total_units = units_per_col * num_cols;
    
    if (thread_idx >= total_units) return;
    
    size_t col = thread_idx / units_per_col;
    size_t unit_idx = thread_idx % units_per_col;
    
    size_t hs0 = 1ULL << stage;
    
    // Map unit_idx to base index i0
    size_t low_bits = unit_idx % hs0;
    size_t high_bits = unit_idx / hs0;
    size_t i0 = col * n + (high_bits << (stage + 6)) + low_bits;
    
    // Load 64 elements
    uint64_t a[64];
    #pragma unroll
    for (int k = 0; k < 64; k++) {
        size_t offset = 0;
        if (k & 1) offset += (1ULL << stage);
        if (k & 2) offset += (1ULL << (stage + 1));
        if (k & 4) offset += (1ULL << (stage + 2));
        if (k & 8) offset += (1ULL << (stage + 3));
        if (k & 16) offset += (1ULL << (stage + 4));
        if (k & 32) offset += (1ULL << (stage + 5));
        a[k] = d_data[i0 + offset];
    }
    
    // Stage 0: 32 butterflies, stride 1
    {
        size_t tw_off = hs0 - 1;
        uint64_t w = d_twiddles[tw_off + low_bits];
        #pragma unroll
        for (int k = 0; k < 64; k += 2) {
            uint64_t t = bfield_mul_impl(a[k+1], w);
            uint64_t tmp = a[k];
            a[k] = bfield_add_impl(tmp, t);
            a[k+1] = bfield_sub_impl(tmp, t);
        }
    }
    
    // Stage 1: 32 butterflies, stride 2
    {
        size_t tw_off = (hs0 << 1) - 1;
        uint64_t w0 = d_twiddles[tw_off + low_bits];
        uint64_t w1 = d_twiddles[tw_off + low_bits + hs0];
        #pragma unroll
        for (int k = 0; k < 64; k += 4) {
            uint64_t t0 = bfield_mul_impl(a[k+2], w0);
            uint64_t tmp0 = a[k];
            a[k] = bfield_add_impl(tmp0, t0);
            a[k+2] = bfield_sub_impl(tmp0, t0);
            
            uint64_t t1 = bfield_mul_impl(a[k+3], w1);
            uint64_t tmp1 = a[k+1];
            a[k+1] = bfield_add_impl(tmp1, t1);
            a[k+3] = bfield_sub_impl(tmp1, t1);
        }
    }
    
    // Stage 2: 32 butterflies, stride 4
    {
        size_t tw_off = (hs0 << 2) - 1;
        #pragma unroll
        for (int k = 0; k < 64; k += 8) {
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                uint64_t w = d_twiddles[tw_off + low_bits + j * hs0];
                uint64_t t = bfield_mul_impl(a[k+4+j], w);
                uint64_t tmp = a[k+j];
                a[k+j] = bfield_add_impl(tmp, t);
                a[k+4+j] = bfield_sub_impl(tmp, t);
            }
        }
    }
    
    // Stage 3: 32 butterflies, stride 8
    {
        size_t tw_off = (hs0 << 3) - 1;
        #pragma unroll
        for (int k = 0; k < 64; k += 16) {
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                uint64_t w = d_twiddles[tw_off + low_bits + j * hs0];
                uint64_t t = bfield_mul_impl(a[k+8+j], w);
                uint64_t tmp = a[k+j];
                a[k+j] = bfield_add_impl(tmp, t);
                a[k+8+j] = bfield_sub_impl(tmp, t);
            }
        }
    }
    
    // Stage 4: 32 butterflies, stride 16
    {
        size_t tw_off = (hs0 << 4) - 1;
        #pragma unroll
        for (int k = 0; k < 64; k += 32) {
            #pragma unroll
            for (int j = 0; j < 16; j++) {
                uint64_t w = d_twiddles[tw_off + low_bits + j * hs0];
                uint64_t t = bfield_mul_impl(a[k+16+j], w);
                uint64_t tmp = a[k+j];
                a[k+j] = bfield_add_impl(tmp, t);
                a[k+16+j] = bfield_sub_impl(tmp, t);
            }
        }
    }
    
    // Stage 5: 32 butterflies, stride 32
    {
        size_t tw_off = (hs0 << 5) - 1;
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            uint64_t w = d_twiddles[tw_off + low_bits + j * hs0];
            uint64_t t = bfield_mul_impl(a[32+j], w);
            uint64_t tmp = a[j];
            a[j] = bfield_add_impl(tmp, t);
            a[32+j] = bfield_sub_impl(tmp, t);
        }
    }
    
    // Store 64 elements
    #pragma unroll
    for (int k = 0; k < 64; k++) {
        size_t offset = 0;
        if (k & 1) offset += (1ULL << stage);
        if (k & 2) offset += (1ULL << (stage + 1));
        if (k & 4) offset += (1ULL << (stage + 2));
        if (k & 8) offset += (1ULL << (stage + 3));
        if (k & 16) offset += (1ULL << (stage + 4));
        if (k & 32) offset += (1ULL << (stage + 5));
        d_data[i0 + offset] = a[k];
    }
}

/**
 * Radix-4 butterfly: processes 2 stages at once
 * (a0, a1, a2, a3) -> radix-4 DIF butterfly
 * Reduces kernel launches by 2x and improves arithmetic intensity
 */
__device__ __forceinline__ void radix4_butterfly(
    uint64_t& a0, uint64_t& a1, uint64_t& a2, uint64_t& a3,
    uint64_t w1, uint64_t w2, uint64_t w3
) {
    // First stage (radix-2)
    uint64_t t0 = bfield_mul_impl(a2, w2);
    uint64_t t1 = bfield_mul_impl(a3, w2);
    uint64_t b0 = bfield_add_impl(a0, t0);
    uint64_t b2 = bfield_sub_impl(a0, t0);
    uint64_t b1 = bfield_add_impl(a1, t1);
    uint64_t b3 = bfield_sub_impl(a1, t1);
    
    // Second stage with different twiddles
    uint64_t t2 = bfield_mul_impl(b1, w1);
    uint64_t t3 = bfield_mul_impl(b3, w3);
    a0 = bfield_add_impl(b0, t2);
    a1 = bfield_sub_impl(b0, t2);
    a2 = bfield_add_impl(b2, t3);
    a3 = bfield_sub_impl(b2, t3);
}

/**
 * Radix-4 batched NTT butterfly kernel
 * Processes 2 stages at once, reducing global memory traffic
 */
__global__ void batched_ntt_radix4_kernel(
    uint64_t* d_data,
    size_t n,
    size_t stage,  // Even stage number (0, 2, 4, ...)
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    size_t thread_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t num_radix4_per_col = n / 4;  // Each radix-4 handles 4 elements
    size_t total_radix4 = num_radix4_per_col * num_cols;
    
    if (thread_idx >= total_radix4) return;
    
    size_t col = thread_idx / num_radix4_per_col;
    size_t local_idx = thread_idx % num_radix4_per_col;
    
    // For stage s and s+1:
    // Stage s: half_size = 2^s, full_size = 2^(s+1)
    // Stage s+1: half_size = 2^(s+1), full_size = 2^(s+2)
    size_t half_s = 1ULL << stage;
    size_t full_s = half_s * 2;
    size_t half_s1 = full_s;
    size_t full_s1 = half_s1 * 2;
    
    // Calculate indices for radix-4 butterfly
    // In radix-4, we need 4 elements spaced by half_s
    size_t group_s1 = local_idx / half_s1;
    size_t pos_s1 = local_idx % half_s1;
    
    size_t base_idx = col * n + group_s1 * full_s1 + pos_s1;
    size_t i0 = base_idx;
    size_t i1 = base_idx + half_s;
    size_t i2 = base_idx + half_s1;
    size_t i3 = base_idx + half_s1 + half_s;
    
    // Load 4 elements
    uint64_t a0 = d_data[i0];
    uint64_t a1 = d_data[i1];
    uint64_t a2 = d_data[i2];
    uint64_t a3 = d_data[i3];
    
    // Get twiddle factors
    size_t pos_in_group = pos_s1 % half_s;
    size_t sub_group = pos_s1 / half_s;
    
    size_t tw_off_s = half_s - 1;
    size_t tw_off_s1 = half_s1 - 1;
    
    uint64_t w_s = d_twiddles[tw_off_s + pos_in_group];  // Stage s twiddle
    uint64_t w_s1 = d_twiddles[tw_off_s1 + pos_s1];      // Stage s+1 twiddle
    
    // Apply stage s butterflies first
    uint64_t t0 = bfield_mul_impl(a1, w_s);
    uint64_t b0 = bfield_add_impl(a0, t0);
    uint64_t b1 = bfield_sub_impl(a0, t0);
    
    uint64_t t1 = bfield_mul_impl(a3, w_s);
    uint64_t b2 = bfield_add_impl(a2, t1);
    uint64_t b3 = bfield_sub_impl(a2, t1);
    
    // Apply stage s+1 butterflies
    uint64_t t2 = bfield_mul_impl(b2, w_s1);
    a0 = bfield_add_impl(b0, t2);
    a2 = bfield_sub_impl(b0, t2);
    
    // For b1 and b3, we need w_s1 with adjusted position
    uint64_t w_s1_adj = d_twiddles[tw_off_s1 + pos_s1 + half_s];
    uint64_t t3 = bfield_mul_impl(b3, w_s1_adj);
    a1 = bfield_add_impl(b1, t3);
    a3 = bfield_sub_impl(b1, t3);
    
    // Store results
    d_data[i0] = a0;
    d_data[i1] = a1;
    d_data[i2] = a2;
    d_data[i3] = a3;
}

/**
 * Batched scale kernel for inverse NTT
 */
__global__ void batched_ntt_scale_kernel(
    uint64_t* d_data,
    size_t n,
    size_t log_n,
    size_t num_cols
) {
    size_t idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    size_t total = n * num_cols;
    if (idx >= total) return;
    
    d_data[idx] = bfield_mul_impl(d_data[idx], SIZE_INVERSES[log_n]);
}

// ============================================================================
// Fused Multi-Stage NTT Kernel (Shared Memory Optimization)
// ============================================================================

/**
 * Fused NTT kernel: processes FUSED_STAGES stages in shared memory
 * Each block handles one 1024-element chunk per column
 * Stages 0-9 are fused (2^10 = 1024 elements per block)
 * This dramatically reduces global memory traffic
 */
constexpr int FUSED_STAGES = 10;  // Process 10 stages in shared memory (1024 elements)
constexpr int FUSED_SIZE = 1 << FUSED_STAGES;  // 1024

__global__ void batched_ntt_fused_kernel(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    size_t start_stage,  // Which global stage to start at
    const uint64_t* d_twiddles
) {
    // Shared memory for 1024 elements
    __shared__ uint64_t smem[FUSED_SIZE];
    
    // Block handles one chunk of FUSED_SIZE elements from one column
    size_t num_chunks_per_col = n / FUSED_SIZE;
    size_t total_chunks = num_chunks_per_col * num_cols;
    size_t chunk_idx = blockIdx.x;
    
    if (chunk_idx >= total_chunks) return;
    
    size_t col = chunk_idx / num_chunks_per_col;
    size_t chunk_in_col = chunk_idx % num_chunks_per_col;
    
    // Global offset for this chunk
    size_t global_base = col * n + chunk_in_col * FUSED_SIZE;
    
    // Load into shared memory (coalesced: 4 elements per thread)
    constexpr int ELEMS_PER_THREAD = FUSED_SIZE / 256;  // 4
    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {
        size_t local_idx = threadIdx.x + k * 256;
        smem[local_idx] = d_data[global_base + local_idx];
    }
    __syncthreads();
    
    // Process FUSED_STAGES stages in shared memory
    for (size_t stage = 0; stage < FUSED_STAGES; stage++) {
        size_t half_size = 1ULL << stage;
        size_t full_size = half_size * 2;
        size_t twiddle_base = (1ULL << (start_stage + stage)) - 1;
        
        // Each thread handles multiple butterflies per stage
        size_t num_butterflies_this_stage = FUSED_SIZE / 2;
        size_t butterflies_per_thread = (num_butterflies_this_stage + 255) / 256;
        
        for (size_t b = 0; b < butterflies_per_thread; b++) {
            size_t butterfly_idx = threadIdx.x + b * 256;
            if (butterfly_idx >= num_butterflies_this_stage) break;
            
            size_t group = butterfly_idx / half_size;
            size_t pos = butterfly_idx % half_size;
            
            size_t i = group * full_size + pos;
            size_t j = i + half_size;
            
            // For early stages within chunk, pos < half_size
            // Need to compute global twiddle position accounting for chunk offset
            size_t global_pos = (chunk_in_col * (FUSED_SIZE >> (stage + 1)) + group) * half_size + pos;
            if (global_pos < half_size) {
                // Use precomputed twiddle
                uint64_t twiddle = d_twiddles[twiddle_base + global_pos];
                butterfly(smem[i], smem[j], twiddle);
            } else {
                // Fallback: compute on the fly
                uint64_t omega = ROOTS_OF_UNITY[start_stage + stage + 1];
                uint64_t twiddle = bfield_pow_impl(omega, global_pos);
                butterfly(smem[i], smem[j], twiddle);
            }
        }
        __syncthreads();
    }
    
    // Write back to global memory (coalesced)
    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {
        size_t local_idx = threadIdx.x + k * 256;
        d_data[global_base + local_idx] = smem[local_idx];
    }
}

/**
 * Simplified fused kernel for first 10 stages only (stage 0-9)
 * This is simpler because twiddle factors are always within bounds
 */
__global__ void batched_ntt_fused_first10_kernel(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    __shared__ uint64_t smem[FUSED_SIZE];
    
    size_t num_chunks_per_col = n / FUSED_SIZE;
    size_t chunk_idx = blockIdx.x;
    size_t total_chunks = num_chunks_per_col * num_cols;
    
    if (chunk_idx >= total_chunks) return;
    
    size_t col = chunk_idx / num_chunks_per_col;
    size_t chunk_in_col = chunk_idx % num_chunks_per_col;
    size_t global_base = col * n + chunk_in_col * FUSED_SIZE;
    
    // Coalesced load
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        smem[threadIdx.x + k * 256] = d_data[global_base + threadIdx.x + k * 256];
    }
    __syncthreads();
    
    // Process stages 0-9 in shared memory
    #pragma unroll
    for (int stage = 0; stage < FUSED_STAGES; stage++) {
        size_t half_size = 1ULL << stage;
        size_t full_size = half_size * 2;
        size_t num_butterflies = FUSED_SIZE / 2;  // 512
        size_t twiddle_offset = half_size - 1;
        
        // 2 butterflies per thread (512 / 256 = 2)
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            size_t butterfly_idx = threadIdx.x + b * 256;
            
            size_t group = butterfly_idx / half_size;
            size_t pos = butterfly_idx % half_size;
            
            size_t i = group * full_size + pos;
            size_t j = i + half_size;
            
            uint64_t twiddle = d_twiddles[twiddle_offset + pos];
            butterfly(smem[i], smem[j], twiddle);
        }
        __syncthreads();
    }
    
    // Coalesced store
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        d_data[global_base + threadIdx.x + k * 256] = smem[threadIdx.x + k * 256];
    }
}

// ============================================================================
// Extended Fused Kernel (11 stages, 2048 elements) for better performance
// ============================================================================

constexpr int FUSED_STAGES_EXT = 11;  // Process 11 stages in shared memory (2048 elements)
constexpr int FUSED_SIZE_EXT = 1 << FUSED_STAGES_EXT;  // 2048

constexpr int FUSED_STAGES_12 = 12;  // Process 12 stages in shared memory (4096 elements)
constexpr int FUSED_SIZE_12 = 1 << FUSED_STAGES_12;  // 4096

/**
 * Extended fused kernel for first 11 stages (stage 0-10)
 * Uses 2048 elements per block, requiring 16KB shared memory
 * 512 threads per block, 4 elements per thread
 */
__global__ void batched_ntt_fused_first11_kernel(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    __shared__ uint64_t smem[FUSED_SIZE_EXT];
    
    size_t num_chunks_per_col = n / FUSED_SIZE_EXT;
    size_t chunk_idx = blockIdx.x;
    size_t total_chunks = num_chunks_per_col * num_cols;
    
    if (chunk_idx >= total_chunks) return;
    
    size_t col = chunk_idx / num_chunks_per_col;
    size_t chunk_in_col = chunk_idx % num_chunks_per_col;
    size_t global_base = col * n + chunk_in_col * FUSED_SIZE_EXT;
    
    // Coalesced load: 4 elements per thread with 512 threads
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        smem[threadIdx.x + k * 512] = d_data[global_base + threadIdx.x + k * 512];
    }
    __syncthreads();
    
    // Process stages 0-10 in shared memory
    #pragma unroll
    for (int stage = 0; stage < FUSED_STAGES_EXT; stage++) {
        size_t half_size = 1ULL << stage;
        size_t full_size = half_size * 2;
        size_t num_butterflies = FUSED_SIZE_EXT / 2;  // 1024
        size_t twiddle_offset = half_size - 1;
        
        // 2 butterflies per thread (1024 / 512 = 2)
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            size_t butterfly_idx = threadIdx.x + b * 512;
            
            size_t group = butterfly_idx / half_size;
            size_t pos = butterfly_idx % half_size;
            
            size_t i = group * full_size + pos;
            size_t j = i + half_size;
            
            uint64_t twiddle = d_twiddles[twiddle_offset + pos];
            butterfly(smem[i], smem[j], twiddle);
        }
        __syncthreads();
    }
    
    // Coalesced store
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        d_data[global_base + threadIdx.x + k * 512] = smem[threadIdx.x + k * 512];
    }
}

/**
 * Extended fused inverse NTT kernel for first 11 stages
 */
__global__ void batched_intt_fused_first11_kernel(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    const uint64_t* d_twiddles_inv
) {
    __shared__ uint64_t smem[FUSED_SIZE_EXT];
    
    size_t num_chunks_per_col = n / FUSED_SIZE_EXT;
    size_t chunk_idx = blockIdx.x;
    size_t total_chunks = num_chunks_per_col * num_cols;
    
    if (chunk_idx >= total_chunks) return;
    
    size_t col = chunk_idx / num_chunks_per_col;
    size_t chunk_in_col = chunk_idx % num_chunks_per_col;
    size_t global_base = col * n + chunk_in_col * FUSED_SIZE_EXT;
    
    // Coalesced load
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        smem[threadIdx.x + k * 512] = d_data[global_base + threadIdx.x + k * 512];
    }
    __syncthreads();
    
    // Process stages 0-10 in shared memory (inverse)
    #pragma unroll
    for (int stage = 0; stage < FUSED_STAGES_EXT; stage++) {
        size_t half_size = 1ULL << stage;
        size_t full_size = half_size * 2;
        size_t twiddle_offset = half_size - 1;
        
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            size_t butterfly_idx = threadIdx.x + b * 512;
            
            size_t group = butterfly_idx / half_size;
            size_t pos = butterfly_idx % half_size;
            
            size_t i = group * full_size + pos;
            size_t j = i + half_size;
            
            uint64_t twiddle = d_twiddles_inv[twiddle_offset + pos];
            butterfly(smem[i], smem[j], twiddle);
        }
        __syncthreads();
    }
    
    // Coalesced store
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        d_data[global_base + threadIdx.x + k * 512] = smem[threadIdx.x + k * 512];
    }
}

/**
 * Extended fused kernel for first 12 stages (stage 0-11)
 * Uses 4096 elements per block, requiring 32KB shared memory
 * 1024 threads per block, 4 elements per thread
 */
__global__ void batched_ntt_fused_first12_kernel(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    __shared__ uint64_t smem[FUSED_SIZE_12];
    
    size_t num_chunks_per_col = n / FUSED_SIZE_12;
    size_t chunk_idx = blockIdx.x;
    size_t total_chunks = num_chunks_per_col * num_cols;
    
    if (chunk_idx >= total_chunks) return;
    
    size_t col = chunk_idx / num_chunks_per_col;
    size_t chunk_in_col = chunk_idx % num_chunks_per_col;
    size_t global_base = col * n + chunk_in_col * FUSED_SIZE_12;
    
    // Coalesced load: 4 elements per thread with 1024 threads
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        smem[threadIdx.x + k * 1024] = d_data[global_base + threadIdx.x + k * 1024];
    }
    __syncthreads();
    
    // Process stages 0-11 in shared memory
    #pragma unroll
    for (int stage = 0; stage < FUSED_STAGES_12; stage++) {
        size_t half_size = 1ULL << stage;
        size_t full_size = half_size * 2;
        size_t num_butterflies = FUSED_SIZE_12 / 2;  // 2048
        size_t twiddle_offset = half_size - 1;
        
        // 2 butterflies per thread (2048 / 1024 = 2)
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            size_t butterfly_idx = threadIdx.x + b * 1024;
            
            size_t group = butterfly_idx / half_size;
            size_t pos = butterfly_idx % half_size;
            
            size_t i = group * full_size + pos;
            size_t j = i + half_size;
            
            uint64_t twiddle = d_twiddles[twiddle_offset + pos];
            butterfly(smem[i], smem[j], twiddle);
        }
        __syncthreads();
    }
    
    // Coalesced store
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        d_data[global_base + threadIdx.x + k * 1024] = smem[threadIdx.x + k * 1024];
    }
}

/**
 * Extended fused inverse NTT kernel for first 12 stages
 */
__global__ void batched_intt_fused_first12_kernel(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    const uint64_t* d_twiddles_inv
) {
    __shared__ uint64_t smem[FUSED_SIZE_12];
    
    size_t num_chunks_per_col = n / FUSED_SIZE_12;
    size_t chunk_idx = blockIdx.x;
    size_t total_chunks = num_chunks_per_col * num_cols;
    
    if (chunk_idx >= total_chunks) return;
    
    size_t col = chunk_idx / num_chunks_per_col;
    size_t chunk_in_col = chunk_idx % num_chunks_per_col;
    size_t global_base = col * n + chunk_in_col * FUSED_SIZE_12;
    
    // Coalesced load
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        smem[threadIdx.x + k * 1024] = d_data[global_base + threadIdx.x + k * 1024];
    }
    __syncthreads();
    
    // Process stages 0-11 in shared memory (inverse)
    #pragma unroll
    for (int stage = 0; stage < FUSED_STAGES_12; stage++) {
        size_t half_size = 1ULL << stage;
        size_t full_size = half_size * 2;
        size_t twiddle_offset = half_size - 1;
        
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            size_t butterfly_idx = threadIdx.x + b * 1024;
            
            size_t group = butterfly_idx / half_size;
            size_t pos = butterfly_idx % half_size;
            
            size_t i = group * full_size + pos;
            size_t j = i + half_size;
            
            uint64_t twiddle = d_twiddles_inv[twiddle_offset + pos];
            butterfly(smem[i], smem[j], twiddle);
        }
        __syncthreads();
    }
    
    // Coalesced store
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        d_data[global_base + threadIdx.x + k * 1024] = smem[threadIdx.x + k * 1024];
    }
}

/**
 * Warp-shuffle optimized fused kernel for first 12 stages (forward NTT)
 * Currently identical to regular 12-stage kernel
 * TODO: Implement warp shuffle optimizations for early stages (0-4)
 */
__global__ void batched_ntt_fused_warp_first12_kernel(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    const uint64_t* d_twiddles
) {
    __shared__ uint64_t smem[FUSED_SIZE_12];
    
    size_t num_chunks_per_col = n / FUSED_SIZE_12;
    size_t chunk_idx = blockIdx.x;
    size_t total_chunks = num_chunks_per_col * num_cols;
    
    if (chunk_idx >= total_chunks) return;
    
    size_t col = chunk_idx / num_chunks_per_col;
    size_t chunk_in_col = chunk_idx % num_chunks_per_col;
    size_t global_base = col * n + chunk_in_col * FUSED_SIZE_12;
    
    // Coalesced load: 4 elements per thread with 1024 threads
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        smem[threadIdx.x + k * 1024] = d_data[global_base + threadIdx.x + k * 1024];
    }
    __syncthreads();
    
    // Process stages 0-11 in shared memory
    // TODO: Use warp shuffle for stages 0-4 for better performance
    #pragma unroll
    for (int stage = 0; stage < FUSED_STAGES_12; stage++) {
        size_t half_size = 1ULL << stage;
        size_t full_size = half_size * 2;
        size_t num_butterflies = FUSED_SIZE_12 / 2;  // 2048
        size_t twiddle_offset = half_size - 1;
        
        // 2 butterflies per thread (2048 / 1024 = 2)
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            size_t butterfly_idx = threadIdx.x + b * 1024;
            
            size_t group = butterfly_idx / half_size;
            size_t pos = butterfly_idx % half_size;
            
            size_t i = group * full_size + pos;
            size_t j = i + half_size;
            
            uint64_t twiddle = d_twiddles[twiddle_offset + pos];
            butterfly(smem[i], smem[j], twiddle);
        }
        __syncthreads();
    }
    
    // Coalesced store
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        d_data[global_base + threadIdx.x + k * 1024] = smem[threadIdx.x + k * 1024];
    }
}

/**
 * Warp-shuffle optimized fused kernel for first 12 stages (inverse NTT)
 * Currently identical to regular 12-stage kernel
 * TODO: Implement warp shuffle optimizations for early stages (0-4)
 */
__global__ void batched_intt_fused_warp_first12_kernel(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    const uint64_t* d_twiddles_inv
) {
    __shared__ uint64_t smem[FUSED_SIZE_12];
    
    size_t num_chunks_per_col = n / FUSED_SIZE_12;
    size_t chunk_idx = blockIdx.x;
    size_t total_chunks = num_chunks_per_col * num_cols;
    
    if (chunk_idx >= total_chunks) return;
    
    size_t col = chunk_idx / num_chunks_per_col;
    size_t chunk_in_col = chunk_idx % num_chunks_per_col;
    size_t global_base = col * n + chunk_in_col * FUSED_SIZE_12;
    
    // Coalesced load
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        smem[threadIdx.x + k * 1024] = d_data[global_base + threadIdx.x + k * 1024];
    }
    __syncthreads();
    
    // Process stages 0-11 in shared memory (inverse)
    // TODO: Use warp shuffle for stages 0-4 for better performance
    #pragma unroll
    for (int stage = 0; stage < FUSED_STAGES_12; stage++) {
        size_t half_size = 1ULL << stage;
        size_t full_size = half_size * 2;
        size_t twiddle_offset = half_size - 1;
        
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            size_t butterfly_idx = threadIdx.x + b * 1024;
            
            size_t group = butterfly_idx / half_size;
            size_t pos = butterfly_idx % half_size;
            
            size_t i = group * full_size + pos;
            size_t j = i + half_size;
            
            uint64_t twiddle = d_twiddles_inv[twiddle_offset + pos];
            butterfly(smem[i], smem[j], twiddle);
        }
        __syncthreads();
    }
    
    // Coalesced store
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        d_data[global_base + threadIdx.x + k * 1024] = smem[threadIdx.x + k * 1024];
    }
}

/**
 * Batched forward NTT for contiguous column-major data
 * Uses fused shared-memory kernel for first 10-12 stages + ILP4 kernel for remaining
 * @param d_data Device pointer to data (num_cols * n elements, column-major)
 * @param n Size of each column (must be power of 2)
 * @param num_cols Number of columns
 * @param stream CUDA stream
 */
void ntt_forward_batched_gpu(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    cudaStream_t stream
) {
    ntt_init_constants();
    
    if (n == 0 || (n & (n - 1)) != 0 || num_cols == 0) return;
    
    size_t log_n = 0;
    for (size_t temp = n; temp > 1; temp >>= 1) {
        ++log_n;
    }
    
    // Ensure twiddles are precomputed
    ensure_twiddles(n, stream);
    
    size_t total_elements = n * num_cols;
    size_t total_butterflies = (n / 2) * num_cols;
    
    int block_size = 256;
    int grid_elements = (total_elements + block_size - 1) / block_size;
    
    // For ILP=4 kernel, we need 1/4 as many threads
    constexpr int ILP = 4;
    int grid_butterflies_ilp = ((total_butterflies + ILP - 1) / ILP + block_size - 1) / block_size;
    
    // Batched bit-reversal
    batched_bit_reverse_kernel<<<grid_elements, block_size, 0, stream>>>(
        d_data, n, log_n, num_cols
    );
    
    // Choose fused kernel based on n size and environment
    // Default: 12-stage fused + reg6stage for best performance on large NTTs
    size_t start_stage = 0;
    const bool disable_fused = (std::getenv("TRITON_DISABLE_FUSED_NTT") != nullptr);
    const bool disable_fused12 = (std::getenv("TRITON_NTT_NO_FUSED12") != nullptr);
    const bool use_ext_fused = (std::getenv("TRITON_NTT_FUSED11") != nullptr);
    const bool use_reg2stage = (std::getenv("TRITON_NTT_REG2STAGE") != nullptr);
    const bool use_reg4stage = (std::getenv("TRITON_NTT_REG4STAGE") != nullptr);
    const bool disable_reg6stage = (std::getenv("TRITON_NTT_NO_REG6STAGE") != nullptr);
    const bool use_ilp8 = (std::getenv("TRITON_NTT_ILP8") != nullptr);
    const bool use_ilp32 = (std::getenv("TRITON_NTT_ILP32") != nullptr);
    
    // Fused12 + reg6stage are now DEFAULT for large NTTs (best performance)
    const bool use_fused12 = !disable_fused12 && (n >= FUSED_SIZE_12);
    const bool use_reg6stage = !disable_reg6stage;
    
    if (!disable_fused) {
        if (use_fused12) {
            // Use 12-stage fused kernel (32KB shared memory, 1024 threads) - DEFAULT for large NTTs
            size_t num_chunks = (n / FUSED_SIZE_12) * num_cols;
            batched_ntt_fused_first12_kernel<<<num_chunks, 1024, 0, stream>>>(
                d_data, n, num_cols, d_twiddles_fwd
            );
            start_stage = FUSED_STAGES_12;  // Continue from stage 12
        } else if ((use_ext_fused || n >= 4194304) && n >= FUSED_SIZE_EXT) {
            // Use extended 11-stage fused kernel for large NTTs (4M+ elements)
            size_t num_chunks = (n / FUSED_SIZE_EXT) * num_cols;
            batched_ntt_fused_first11_kernel<<<num_chunks, 512, 0, stream>>>(
                d_data, n, num_cols, d_twiddles_fwd
            );
            start_stage = FUSED_STAGES_EXT;  // Continue from stage 11
        } else if (n >= FUSED_SIZE) {
            // Use standard 10-stage fused kernel
            size_t num_chunks = (n / FUSED_SIZE) * num_cols;
            batched_ntt_fused_first10_kernel<<<num_chunks, 256, 0, stream>>>(
                d_data, n, num_cols, d_twiddles_fwd
            );
            start_stage = FUSED_STAGES;  // Continue from stage 10
        }
    }
    
    // Choose kernel for remaining stages
    if (use_reg6stage && !disable_fused) {
        // Register-based 6-stage kernel: process 6 stages per kernel launch
        size_t total_units6 = (n / 64) * num_cols;
        int grid_units6 = (total_units6 + block_size - 1) / block_size;
        size_t total_units4 = (n / 16) * num_cols;
        int grid_units4 = (total_units4 + block_size - 1) / block_size;
        size_t total_groups = (n / 4) * num_cols;
        int grid_groups = (total_groups + block_size - 1) / block_size;
        
        size_t stage = start_stage;
        // Process groups of 6 stages
        for (; stage + 5 < log_n; stage += 6) {
            batched_ntt_reg6stage_kernel<<<grid_units6, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
        // Handle remaining stages with reg4stage, reg2stage, or ILP4
        for (; stage + 3 < log_n; stage += 4) {
            batched_ntt_reg4stage_kernel<<<grid_units4, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
        for (; stage + 1 < log_n; stage += 2) {
            batched_ntt_reg2stage_kernel<<<grid_groups, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
        if (stage < log_n) {
            batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
    } else if (use_reg4stage && !disable_fused) {
        // Register-based 4-stage kernel: process 4 stages per kernel launch
        size_t total_units = (n / 16) * num_cols;
        int grid_units = (total_units + block_size - 1) / block_size;
        size_t total_groups = (n / 4) * num_cols;
        int grid_groups = (total_groups + block_size - 1) / block_size;
        
        size_t stage = start_stage;
        // Process groups of 4 stages
        for (; stage + 3 < log_n; stage += 4) {
            batched_ntt_reg4stage_kernel<<<grid_units, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
        // Handle remaining 1-3 stages with reg2stage or ILP4
        for (; stage + 1 < log_n; stage += 2) {
            batched_ntt_reg2stage_kernel<<<grid_groups, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
        if (stage < log_n) {
            batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
    } else if (use_reg2stage && !disable_fused) {
        // Register-based 2-stage kernel: process 2 stages per kernel launch
        size_t total_groups = (n / 4) * num_cols;
        int grid_groups = (total_groups + block_size - 1) / block_size;
        
        for (size_t stage = start_stage; stage + 1 < log_n; stage += 2) {
            batched_ntt_reg2stage_kernel<<<grid_groups, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
        // Handle odd remaining stage with ILP4
        if ((log_n - start_stage) % 2 == 1) {
            batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                d_data, n, log_n - 1, num_cols, d_twiddles_fwd
            );
        }
    } else if (disable_fused && use_reg4stage) {
        // Full register-based 4-stage mode (no shared memory)
        size_t total_units = (n / 16) * num_cols;
        int grid_units = (total_units + block_size - 1) / block_size;
        size_t total_groups = (n / 4) * num_cols;
        int grid_groups = (total_groups + block_size - 1) / block_size;
        
        size_t stage = 0;
        for (; stage + 3 < log_n; stage += 4) {
            batched_ntt_reg4stage_kernel<<<grid_units, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
        for (; stage + 1 < log_n; stage += 2) {
            batched_ntt_reg2stage_kernel<<<grid_groups, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
        if (stage < log_n) {
            batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
    } else if (disable_fused && use_reg2stage) {
        // Full register-based 2-stage mode (no shared memory at all)
        size_t total_groups = (n / 4) * num_cols;
        int grid_groups = (total_groups + block_size - 1) / block_size;
        
        for (size_t stage = 0; stage + 1 < log_n; stage += 2) {
            batched_ntt_reg2stage_kernel<<<grid_groups, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_fwd
            );
        }
        // Handle odd remaining stage
        if (log_n % 2 == 1) {
            batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                d_data, n, log_n - 1, num_cols, d_twiddles_fwd
            );
        }
    } else if (use_ilp32) {
        // ILP32 kernel for extreme latency hiding
        constexpr int ILP32 = 32;
        constexpr size_t LARGE_STAGE_THRESHOLD = 11;
        const bool use_coalesced = (std::getenv("TRITON_NTT_COALESCED") != nullptr);
        int grid_ilp32 = ((total_butterflies + ILP32 - 1) / ILP32 + block_size - 1) / block_size;
        
        for (size_t stage = start_stage; stage < log_n; ++stage) {
            if ((stage > LARGE_STAGE_THRESHOLD || use_coalesced) && num_cols >= 2) {
                // Use coalesced ILP4 kernel for large stages
                constexpr int WARP_SIZE = 32;
                constexpr int COLS_PER_WARP = 2;
                size_t num_butterflies_per_col = n / 2;
                size_t num_warps = ((num_butterflies_per_col + WARP_SIZE - 1) / WARP_SIZE) * 
                                   ((num_cols + COLS_PER_WARP - 1) / COLS_PER_WARP);
                int grid_warps = (num_warps + block_size - 1) / block_size;
                batched_ntt_butterfly_ilp4_coalesced_kernel<<<grid_warps, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_fwd
                );
            } else {
                batched_ntt_butterfly_ilp32_kernel<<<grid_ilp32, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_fwd
                );
            }
        }
    } else if (use_ilp8) {
        // ILP8 kernel for maximum latency hiding
        constexpr int ILP8 = 8;
        constexpr size_t LARGE_STAGE_THRESHOLD = 11;
        const bool use_coalesced = (std::getenv("TRITON_NTT_COALESCED") != nullptr);
        int grid_ilp8 = ((total_butterflies + ILP8 - 1) / ILP8 + block_size - 1) / block_size;
        
        for (size_t stage = start_stage; stage < log_n; ++stage) {
            if ((stage > LARGE_STAGE_THRESHOLD || use_coalesced) && num_cols >= 2) {
                // Use coalesced ILP4 kernel for large stages
                constexpr int WARP_SIZE = 32;
                constexpr int COLS_PER_WARP = 2;
                size_t num_butterflies_per_col = n / 2;
                size_t num_warps = ((num_butterflies_per_col + WARP_SIZE - 1) / WARP_SIZE) * 
                                   ((num_cols + COLS_PER_WARP - 1) / COLS_PER_WARP);
                int grid_warps = (num_warps + block_size - 1) / block_size;
                batched_ntt_butterfly_ilp4_coalesced_kernel<<<grid_warps, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_fwd
                );
            } else {
                batched_ntt_butterfly_ilp8_kernel<<<grid_ilp8, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_fwd
                );
            }
        }
    } else {
        // Default: Use coalesced kernels for large stages (stage > 11), ILP4 for smaller stages
        constexpr size_t LARGE_STAGE_THRESHOLD = 11;
        const bool use_coalesced = (std::getenv("TRITON_NTT_COALESCED") != nullptr);
        
        for (size_t stage = start_stage; stage < log_n; ++stage) {
            if ((stage > LARGE_STAGE_THRESHOLD || use_coalesced) && num_cols >= 2) {
                // Use coalesced kernel for large stages with better memory access patterns
                // Grid size: number of warps needed
                constexpr int WARP_SIZE = 32;
                constexpr int COLS_PER_WARP = 4;
                size_t num_butterflies_per_col = n / 2;
                size_t num_warps = ((num_butterflies_per_col + WARP_SIZE - 1) / WARP_SIZE) * 
                                   ((num_cols + COLS_PER_WARP - 1) / COLS_PER_WARP);
                int grid_warps = (num_warps + block_size - 1) / block_size;
                batched_ntt_butterfly_coalesced_large_stage_kernel<<<grid_warps, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_fwd
                );
            } else {
                // Use standard ILP4 kernel for smaller stages
                batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_fwd
                );
            }
        }
    }
}

/**
 * Fused inverse NTT kernel for first 10 stages
 */
__global__ void batched_intt_fused_first10_kernel(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    const uint64_t* d_twiddles_inv
) {
    __shared__ uint64_t smem[FUSED_SIZE];
    
    size_t num_chunks_per_col = n / FUSED_SIZE;
    size_t chunk_idx = blockIdx.x;
    size_t total_chunks = num_chunks_per_col * num_cols;
    
    if (chunk_idx >= total_chunks) return;
    
    size_t col = chunk_idx / num_chunks_per_col;
    size_t chunk_in_col = chunk_idx % num_chunks_per_col;
    size_t global_base = col * n + chunk_in_col * FUSED_SIZE;
    
    // Coalesced load
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        smem[threadIdx.x + k * 256] = d_data[global_base + threadIdx.x + k * 256];
    }
    __syncthreads();
    
    // Process stages 0-9 in shared memory (inverse)
    #pragma unroll
    for (int stage = 0; stage < FUSED_STAGES; stage++) {
        size_t half_size = 1ULL << stage;
        size_t full_size = half_size * 2;
        size_t twiddle_offset = half_size - 1;
        
        #pragma unroll
        for (int b = 0; b < 2; b++) {
            size_t butterfly_idx = threadIdx.x + b * 256;
            
            size_t group = butterfly_idx / half_size;
            size_t pos = butterfly_idx % half_size;
            
            size_t i = group * full_size + pos;
            size_t j = i + half_size;
            
            uint64_t twiddle = d_twiddles_inv[twiddle_offset + pos];
            butterfly(smem[i], smem[j], twiddle);
        }
        __syncthreads();
    }
    
    // Coalesced store
    #pragma unroll
    for (int k = 0; k < 4; k++) {
        d_data[global_base + threadIdx.x + k * 256] = smem[threadIdx.x + k * 256];
    }
}

/**
 * Batched inverse NTT for contiguous column-major data
 * Uses fused shared-memory kernel for first 10-11 stages + ILP4 kernel for remaining
 */
void ntt_inverse_batched_gpu(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    cudaStream_t stream
) {
    ntt_init_constants();
    
    if (n == 0 || (n & (n - 1)) != 0 || num_cols == 0) return;
    
    size_t log_n = 0;
    for (size_t temp = n; temp > 1; temp >>= 1) {
        ++log_n;
    }
    
    // Ensure twiddles are precomputed
    ensure_twiddles(n, stream);
    
    size_t total_elements = n * num_cols;
    size_t total_butterflies = (n / 2) * num_cols;
    
    int block_size = 256;
    int grid_elements = (total_elements + block_size - 1) / block_size;
    
    constexpr int ILP = 4;
    int grid_butterflies_ilp = ((total_butterflies + ILP - 1) / ILP + block_size - 1) / block_size;
    
    // Batched bit-reversal
    batched_bit_reverse_kernel<<<grid_elements, block_size, 0, stream>>>(
        d_data, n, log_n, num_cols
    );
    
    // Choose fused kernel based on n size and environment
    // Default: 12-stage fused + reg6stage for best performance on large NTTs
    size_t start_stage = 0;
    const bool disable_fused = (std::getenv("TRITON_DISABLE_FUSED_NTT") != nullptr);
    const bool disable_fused12 = (std::getenv("TRITON_NTT_NO_FUSED12") != nullptr);
    const bool use_ext_fused = (std::getenv("TRITON_NTT_FUSED11") != nullptr);
    const bool use_reg2stage = (std::getenv("TRITON_NTT_REG2STAGE") != nullptr);
    const bool use_reg4stage = (std::getenv("TRITON_NTT_REG4STAGE") != nullptr);
    const bool disable_reg6stage = (std::getenv("TRITON_NTT_NO_REG6STAGE") != nullptr);
    const bool use_ilp8 = (std::getenv("TRITON_NTT_ILP8") != nullptr);
    const bool use_ilp32 = (std::getenv("TRITON_NTT_ILP32") != nullptr);
    
    // Fused12 + reg6stage are now DEFAULT for large NTTs (best performance)
    const bool use_fused12 = !disable_fused12 && (n >= FUSED_SIZE_12);
    const bool use_reg6stage = !disable_reg6stage;
    
    if (!disable_fused) {
        if (use_fused12) {
            // Use 12-stage fused kernel (32KB shared memory, 1024 threads) - DEFAULT for large NTTs
            size_t num_chunks = (n / FUSED_SIZE_12) * num_cols;
            batched_intt_fused_first12_kernel<<<num_chunks, 1024, 0, stream>>>(
                d_data, n, num_cols, d_twiddles_inv
            );
            start_stage = FUSED_STAGES_12;  // Continue from stage 12
        } else if ((use_ext_fused || n >= 4194304) && n >= FUSED_SIZE_EXT) {
            // Use extended 11-stage fused kernel for large NTTs (4M+ elements)
            size_t num_chunks = (n / FUSED_SIZE_EXT) * num_cols;
            batched_intt_fused_first11_kernel<<<num_chunks, 512, 0, stream>>>(
                d_data, n, num_cols, d_twiddles_inv
            );
            start_stage = FUSED_STAGES_EXT;  // Continue from stage 11
        } else if (n >= FUSED_SIZE) {
            // Use standard 10-stage fused kernel
            size_t num_chunks = (n / FUSED_SIZE) * num_cols;
            batched_intt_fused_first10_kernel<<<num_chunks, 256, 0, stream>>>(
                d_data, n, num_cols, d_twiddles_inv
            );
            start_stage = FUSED_STAGES;  // Continue from stage 10
        }
    }
    
    // Choose kernel for remaining stages
    if (use_reg6stage && !disable_fused) {
        // Register-based 6-stage kernel
        size_t total_units6 = (n / 64) * num_cols;
        int grid_units6 = (total_units6 + block_size - 1) / block_size;
        size_t total_units4 = (n / 16) * num_cols;
        int grid_units4 = (total_units4 + block_size - 1) / block_size;
        size_t total_groups = (n / 4) * num_cols;
        int grid_groups = (total_groups + block_size - 1) / block_size;
        
        size_t stage = start_stage;
        for (; stage + 5 < log_n; stage += 6) {
            batched_ntt_reg6stage_kernel<<<grid_units6, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
        for (; stage + 3 < log_n; stage += 4) {
            batched_ntt_reg4stage_kernel<<<grid_units4, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
        for (; stage + 1 < log_n; stage += 2) {
            batched_ntt_reg2stage_kernel<<<grid_groups, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
        if (stage < log_n) {
            batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
    } else if (use_reg4stage && !disable_fused) {
        // Register-based 4-stage kernel
        size_t total_units = (n / 16) * num_cols;
        int grid_units = (total_units + block_size - 1) / block_size;
        size_t total_groups = (n / 4) * num_cols;
        int grid_groups = (total_groups + block_size - 1) / block_size;
        
        size_t stage = start_stage;
        for (; stage + 3 < log_n; stage += 4) {
            batched_ntt_reg4stage_kernel<<<grid_units, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
        for (; stage + 1 < log_n; stage += 2) {
            batched_ntt_reg2stage_kernel<<<grid_groups, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
        if (stage < log_n) {
            batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
    } else if (use_reg2stage && !disable_fused) {
        // Register-based 2-stage kernel
        size_t total_groups = (n / 4) * num_cols;
        int grid_groups = (total_groups + block_size - 1) / block_size;
        
        for (size_t stage = start_stage; stage + 1 < log_n; stage += 2) {
            batched_ntt_reg2stage_kernel<<<grid_groups, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
        if ((log_n - start_stage) % 2 == 1) {
            batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                d_data, n, log_n - 1, num_cols, d_twiddles_inv
            );
        }
    } else if (disable_fused && use_reg4stage) {
        // Full register-based 4-stage mode
        size_t total_units = (n / 16) * num_cols;
        int grid_units = (total_units + block_size - 1) / block_size;
        size_t total_groups = (n / 4) * num_cols;
        int grid_groups = (total_groups + block_size - 1) / block_size;
        
        size_t stage = 0;
        for (; stage + 3 < log_n; stage += 4) {
            batched_ntt_reg4stage_kernel<<<grid_units, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
        for (; stage + 1 < log_n; stage += 2) {
            batched_ntt_reg2stage_kernel<<<grid_groups, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
        if (stage < log_n) {
            batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
    } else if (disable_fused && use_reg2stage) {
        // Full register-based 2-stage mode
        size_t total_groups = (n / 4) * num_cols;
        int grid_groups = (total_groups + block_size - 1) / block_size;
        
        for (size_t stage = 0; stage + 1 < log_n; stage += 2) {
            batched_ntt_reg2stage_kernel<<<grid_groups, block_size, 0, stream>>>(
                d_data, n, stage, num_cols, d_twiddles_inv
            );
        }
        if (log_n % 2 == 1) {
            batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                d_data, n, log_n - 1, num_cols, d_twiddles_inv
            );
        }
    } else if (use_ilp32) {
        // ILP32 kernel for extreme latency hiding
        constexpr int ILP32 = 32;
        constexpr size_t LARGE_STAGE_THRESHOLD = 11;
        const bool use_coalesced = (std::getenv("TRITON_NTT_COALESCED") != nullptr);
        size_t num_butterflies_per_col = n / 2;
        int grid_ilp32 = ((total_butterflies + ILP32 - 1) / ILP32 + block_size - 1) / block_size;
        
        for (size_t stage = start_stage; stage < log_n; ++stage) {
            if ((stage > LARGE_STAGE_THRESHOLD || use_coalesced) && num_cols >= 2) {
                // Use coalesced ILP4 kernel for large stages
                constexpr int WARP_SIZE = 32;
                constexpr int COLS_PER_WARP = 2;
                size_t num_warps = ((num_butterflies_per_col + WARP_SIZE - 1) / WARP_SIZE) * 
                                   ((num_cols + COLS_PER_WARP - 1) / COLS_PER_WARP);
                int grid_warps = (num_warps + block_size - 1) / block_size;
                batched_ntt_butterfly_ilp4_coalesced_kernel<<<grid_warps, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_inv
                );
            } else {
                batched_ntt_butterfly_ilp32_kernel<<<grid_ilp32, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_inv
                );
            }
        }
    } else if (use_ilp8) {
        // ILP8 kernel
        constexpr int ILP8 = 8;
        constexpr size_t LARGE_STAGE_THRESHOLD = 11;
        const bool use_coalesced = (std::getenv("TRITON_NTT_COALESCED") != nullptr);
        size_t num_butterflies_per_col = n / 2;
        int grid_ilp8 = ((total_butterflies + ILP8 - 1) / ILP8 + block_size - 1) / block_size;
        
        for (size_t stage = start_stage; stage < log_n; ++stage) {
            if ((stage > LARGE_STAGE_THRESHOLD || use_coalesced) && num_cols >= 2) {
                // Use coalesced ILP4 kernel for large stages
                constexpr int WARP_SIZE = 32;
                constexpr int COLS_PER_WARP = 2;
                size_t num_warps = ((num_butterflies_per_col + WARP_SIZE - 1) / WARP_SIZE) * 
                                   ((num_cols + COLS_PER_WARP - 1) / COLS_PER_WARP);
                int grid_warps = (num_warps + block_size - 1) / block_size;
                batched_ntt_butterfly_ilp4_coalesced_kernel<<<grid_warps, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_inv
                );
            } else {
                batched_ntt_butterfly_ilp8_kernel<<<grid_ilp8, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_inv
                );
            }
        }
    } else {
        // Default: Use coalesced kernels for large stages (stage > 11), ILP4 for smaller stages
        constexpr size_t LARGE_STAGE_THRESHOLD = 11;
        const bool use_coalesced = (std::getenv("TRITON_NTT_COALESCED") != nullptr);
        
        for (size_t stage = start_stage; stage < log_n; ++stage) {
            if ((stage > LARGE_STAGE_THRESHOLD || use_coalesced) && num_cols >= 2) {
                // Use coalesced kernel for large stages with better memory access patterns
                constexpr int WARP_SIZE = 32;
                constexpr int COLS_PER_WARP = 4;
                size_t num_butterflies_per_col = n / 2;
                size_t num_warps = ((num_butterflies_per_col + WARP_SIZE - 1) / WARP_SIZE) * 
                                   ((num_cols + COLS_PER_WARP - 1) / COLS_PER_WARP);
                int grid_warps = (num_warps + block_size - 1) / block_size;
                batched_ntt_butterfly_coalesced_large_stage_kernel<<<grid_warps, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_inv
                );
            } else {
                // Use standard ILP4 kernel for smaller stages
                batched_ntt_butterfly_ilp4_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
                    d_data, n, stage, num_cols, d_twiddles_inv
                );
            }
        }
    }
    
    // Batched scale by n^{-1}
    batched_ntt_scale_kernel<<<grid_elements, block_size, 0, stream>>>(
        d_data, n, log_n, num_cols
    );
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
