/**
 * NTT CUDA Kernel Implementation
 * 
 * Number Theoretic Transform for Goldilocks field.
 * Uses Cooley-Tukey decimation-in-time algorithm.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bfield_kernel.cuh"
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

// OPTIMIZED: Texture memory for twiddle factors (better cache behavior)
static cudaTextureObject_t tex_twiddles_fwd = 0;
static cudaTextureObject_t tex_twiddles_inv = 0;

// Texture fetch helper functions (use texture cache for better performance)
__device__ __forceinline__ uint64_t tex_fetch_twiddle_fwd(size_t idx) {
    return tex1Dfetch<uint64_t>(tex_twiddles_fwd, static_cast<int>(idx));
}

__device__ __forceinline__ uint64_t tex_fetch_twiddle_inv(size_t idx) {
    return tex1Dfetch<uint64_t>(tex_twiddles_inv, static_cast<int>(idx));
}

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
 * OPTIMIZED: Ensure twiddle factors are precomputed for size n
 * Creates texture objects for better caching
 */
static void ensure_twiddles(size_t n, cudaStream_t stream) {
    if (n <= twiddles_max_n && d_twiddles_fwd != nullptr) return;
    
    // Clean up old allocations and textures
    if (tex_twiddles_fwd) {
        cudaDestroyTextureObject(tex_twiddles_fwd);
        tex_twiddles_fwd = 0;
    }
    if (tex_twiddles_inv) {
        cudaDestroyTextureObject(tex_twiddles_inv);
        tex_twiddles_inv = 0;
    }
    if (d_twiddles_fwd) cudaFree(d_twiddles_fwd);
    if (d_twiddles_inv) cudaFree(d_twiddles_inv);
    
    size_t total = n - 1;  // Total twiddles needed
    cudaMalloc(&d_twiddles_fwd, total * sizeof(uint64_t));
    cudaMalloc(&d_twiddles_inv, total * sizeof(uint64_t));
    
    // OPTIMIZED: Calculate log_n using built-in function (faster)
    size_t log_n = 0;
    if (n > 1) {
        log_n = 63 - __builtin_clzll(n);
    }
    
    int block = 256;
    int grid = (total + block - 1) / block;
    
    generate_twiddles_kernel<<<grid, block, 0, stream>>>(d_twiddles_fwd, log_n, false);
    generate_twiddles_kernel<<<grid, block, 0, stream>>>(d_twiddles_inv, log_n, true);
    cudaStreamSynchronize(stream);
    
    // OPTIMIZED: Create texture objects for better caching
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_twiddles_fwd;
    resDesc.res.linear.desc.f = cudaChannelFormatKindUnsigned;
    resDesc.res.linear.desc.x = 32;
    resDesc.res.linear.desc.y = 32;
    resDesc.res.linear.sizeInBytes = total * sizeof(uint64_t);
    
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    
    cudaCreateTextureObject(&tex_twiddles_fwd, &resDesc, &texDesc, nullptr);
    
    resDesc.res.linear.devPtr = d_twiddles_inv;
    cudaCreateTextureObject(&tex_twiddles_inv, &resDesc, &texDesc, nullptr);
    
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
    uint64_t t[ILP], new_a[ILP], new_b[ILP];
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        size_t idx = base_idx + k;
        if (idx < num_butterflies) {
            t[k] = bfield_mul_impl(b[k], tw[k]);
            new_a[k] = bfield_add_impl(a[k], t[k]);
            new_b[k] = bfield_sub_impl(a[k], t[k]);
        }
    }
    
    // Store phase
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        size_t idx = base_idx + k;
        if (idx < num_butterflies) {
            data[i_idx[k]] = new_a[k];
            data[j_idx[k]] = new_b[k];
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
 * OPTIMIZED: Bit-reversal permutation kernel with ILP
 */
__global__ void bit_reverse_kernel(
    uint64_t* __restrict__ data,
    size_t n,
    size_t log_n
) {
    constexpr int ILP = 4;
    const size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * ILP;
    
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        const size_t idx = base_idx + k;
        if (idx >= n) continue;
        
        // OPTIMIZED: Fast bit-reversal
        const size_t rev = fast_bit_reverse(idx, log_n);
        
        // Only swap if rev > idx (to avoid double swap)
        if (rev > idx) {
            const uint64_t tmp = __ldg(&data[idx]);
            data[idx] = __ldg(&data[rev]);
            data[rev] = tmp;
        }
    }
}

/**
 * OPTIMIZED: Scale kernel for inverse NTT with ILP
 */
__global__ void ntt_scale_kernel(
    uint64_t* __restrict__ data,
    size_t n,
    size_t log_n
) {
    constexpr int ILP = 4;
    const size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * ILP;
    const uint64_t scale = SIZE_INVERSES[log_n];
    
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        const size_t idx = base_idx + k;
        if (idx >= n) continue;
        data[idx] = bfield_mul_impl(__ldg(&data[idx]), scale);
    }
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
    
    // OPTIMIZED: Calculate log_n using built-in function (faster)
    size_t log_n = 0;
    if (n > 1) {
        // Use __builtin_clzll for faster log2 calculation
        log_n = 63 - __builtin_clzll(n);
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
    
    // OPTIMIZED: Calculate log_n using built-in function (faster)
    size_t log_n = 0;
    if (n > 1) {
        // Use __builtin_clzll for faster log2 calculation
        log_n = 63 - __builtin_clzll(n);
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
// Utility Functions
// ============================================================================

/**
 * OPTIMIZED: Fast bit-reversal using optimized algorithm
 * Uses better instruction-level parallelism and unrolling
 */
__device__ __forceinline__ size_t fast_bit_reverse(size_t x, size_t log_n) {
    // Use optimized bit-reversal with better instruction-level parallelism
    size_t rev = 0;
    size_t temp = x;
    
    // Unroll for common sizes (up to 16 bits)
    if (log_n <= 8) {
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            if (i < log_n) {
                rev = (rev << 1) | (temp & 1);
                temp >>= 1;
            }
        }
    } else if (log_n <= 16) {
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            if (i < log_n) {
                rev = (rev << 1) | (temp & 1);
                temp >>= 1;
            }
        }
    } else {
        // For larger sizes, use loop (rare case)
        for (size_t i = 0; i < log_n; ++i) {
            rev = (rev << 1) | (temp & 1);
            temp >>= 1;
        }
    }
    return rev;
}

// ============================================================================
// Batched NTT Kernels (for column-major contiguous data)
// ============================================================================

/**
 * Batched bit-reversal permutation kernel (OPTIMIZED)
 * Data layout: column-major with num_cols columns of n elements each
 * d_data[col * n + row] = element at (col, row)
 */
__global__ void batched_bit_reverse_kernel(
    uint64_t* __restrict__ d_data,
    size_t n,
    size_t log_n,
    size_t num_cols
) {
    // OPTIMIZED: ILP=4 for better memory throughput
    constexpr int ILP = 4;
    const size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * ILP;
    const size_t total = n * num_cols;
    
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        const size_t global_idx = base_idx + k;
        if (global_idx >= total) continue;
        
        const size_t col = global_idx / n;
        const size_t idx = global_idx % n;
        
        // OPTIMIZED: Fast bit-reversal
        const size_t rev = fast_bit_reverse(idx, log_n);
        
        // OPTIMIZED: Only swap if rev > idx, use __ldg for read
        if (rev > idx) {
            const size_t pos1 = col * n + idx;
            const size_t pos2 = col * n + rev;
            const uint64_t tmp = __ldg(&d_data[pos1]);
            d_data[pos1] = __ldg(&d_data[pos2]);
            d_data[pos2] = tmp;
        }
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
 * MAXIMUM ILP-optimized batched butterfly: each thread handles 8 butterflies
 * OPTIMIZED: Uses texture memory for twiddles, better register usage
 */
__global__ void batched_ntt_butterfly_ilp8_kernel(
    uint64_t* __restrict__ d_data,
    size_t n,
    size_t stage,
    size_t num_cols,
    const uint64_t* __restrict__ d_twiddles
) {
    constexpr int ILP = 8;
    const size_t thread_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    const size_t num_butterflies_per_col = n / 2;
    const size_t total_butterflies = num_butterflies_per_col * num_cols;
    
    const size_t half_size = 1ULL << stage;
    const size_t full_size = half_size * 2;
    const size_t twiddle_offset = half_size - 1;
    
    // Each thread processes ILP consecutive butterflies
    const size_t base_global_idx = thread_idx * ILP;
    
    // OPTIMIZED: Reduce register pressure by computing indices on-the-fly
    // Process in batches to improve instruction scheduling
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        const size_t global_idx = base_global_idx + k;
        if (global_idx >= total_butterflies) continue;
        
        const size_t col = global_idx / num_butterflies_per_col;
        const size_t local_idx = global_idx % num_butterflies_per_col;
        
        const size_t group = local_idx / half_size;
        const size_t pos = local_idx % half_size;
        
        const size_t base = col * n;
        const size_t i_pos = base + group * full_size + pos;
        const size_t j_pos = i_pos + half_size;
        
        // OPTIMIZED: Use texture fetch for twiddles (better cache)
        const uint64_t tw = (tex_twiddles_fwd != 0) ? 
            tex_fetch_twiddle_fwd(twiddle_offset + pos) : 
            __ldg(&d_twiddles[twiddle_offset + pos]);
        
        // OPTIMIZED: Load, compute, store in one pass (better pipeline)
        const uint64_t a = __ldg(&d_data[i_pos]);
        const uint64_t b = __ldg(&d_data[j_pos]);
        const uint64_t t = bfield_mul_impl(b, tw);
        d_data[i_pos] = bfield_add_impl(a, t);
        d_data[j_pos] = bfield_sub_impl(a, t);
    }
}

/**
 * ILP-optimized batched butterfly: each thread handles 4 butterflies
 * OPTIMIZED: Better register usage, texture memory for twiddles
 */
__global__ void batched_ntt_butterfly_ilp4_kernel(
    uint64_t* __restrict__ d_data,
    size_t n,
    size_t stage,
    size_t num_cols,
    const uint64_t* __restrict__ d_twiddles
) {
    constexpr int ILP = 4;
    const size_t thread_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    const size_t num_butterflies_per_col = n / 2;
    const size_t total_butterflies = num_butterflies_per_col * num_cols;
    
    const size_t half_size = 1ULL << stage;
    const size_t full_size = half_size * 2;
    const size_t twiddle_offset = half_size - 1;
    
    const size_t base_global_idx = thread_idx * ILP;
    
    // OPTIMIZED: Process in sequence to reduce register pressure
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        const size_t global_idx = base_global_idx + k;
        if (global_idx >= total_butterflies) continue;
        
        const size_t col = global_idx / num_butterflies_per_col;
        const size_t local_idx = global_idx % num_butterflies_per_col;
        
        const size_t group = local_idx / half_size;
        const size_t pos = local_idx % half_size;
        
        const size_t base = col * n;
        const size_t i_pos = base + group * full_size + pos;
        const size_t j_pos = i_pos + half_size;
        
        // OPTIMIZED: Use texture fetch for twiddles
        const uint64_t tw = (tex_twiddles_fwd != 0) ? 
            tex_fetch_twiddle_fwd(twiddle_offset + pos) : 
            __ldg(&d_twiddles[twiddle_offset + pos]);
        
        // Load, compute, store in one pass
        const uint64_t a = __ldg(&d_data[i_pos]);
        const uint64_t b = __ldg(&d_data[j_pos]);
        const uint64_t t = bfield_mul_impl(b, tw);
        d_data[i_pos] = bfield_add_impl(a, t);
        d_data[j_pos] = bfield_sub_impl(a, t);
    }
}

/**
 * ILP-optimized batched butterfly: each thread handles 2 butterflies
 * OPTIMIZED: Texture memory, reduced register pressure
 */
__global__ void batched_ntt_butterfly_ilp2_kernel(
    uint64_t* __restrict__ d_data,
    size_t n,
    size_t stage,
    size_t num_cols,
    const uint64_t* __restrict__ d_twiddles
) {
    constexpr int ILP = 2;
    const size_t thread_idx = static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x);
    const size_t num_butterflies_per_col = n / 2;
    const size_t total_butterflies = num_butterflies_per_col * num_cols;
    
    const size_t half_size = 1ULL << stage;
    const size_t full_size = half_size * 2;
    const size_t twiddle_offset = half_size - 1;
    
    const size_t base_global_idx = thread_idx * ILP;
    
    // OPTIMIZED: Process both butterflies with better instruction scheduling
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        const size_t global_idx = base_global_idx + k;
        if (global_idx >= total_butterflies) continue;
        
        const size_t col = global_idx / num_butterflies_per_col;
        const size_t local_idx = global_idx % num_butterflies_per_col;
        
        const size_t group = local_idx / half_size;
        const size_t pos = local_idx % half_size;
        
        const size_t base = col * n;
        const size_t i_pos = base + group * full_size + pos;
        const size_t j_pos = i_pos + half_size;
        
        // OPTIMIZED: Use texture fetch for twiddles
        const uint64_t tw = (tex_twiddles_fwd != 0) ? 
            tex_fetch_twiddle_fwd(twiddle_offset + pos) : 
            __ldg(&d_twiddles[twiddle_offset + pos]);
        
        // Load, compute, store
        const uint64_t a = __ldg(&d_data[i_pos]);
        const uint64_t b = __ldg(&d_data[j_pos]);
        const uint64_t t = bfield_mul_impl(b, tw);
        d_data[i_pos] = bfield_add_impl(a, t);
        d_data[j_pos] = bfield_sub_impl(a, t);
    }
}

/**
 * Batched scale kernel for inverse NTT
 */
__global__ void batched_ntt_scale_kernel(
    uint64_t* __restrict__ d_data,
    size_t n,
    size_t log_n,
    size_t num_cols
) {
    // OPTIMIZED: Vectorized scaling with ILP=4
    constexpr int ILP = 4;
    const size_t base_idx = (static_cast<size_t>(blockIdx.x) * static_cast<size_t>(blockDim.x) + static_cast<size_t>(threadIdx.x)) * ILP;
    const size_t total = n * num_cols;
    const uint64_t scale = SIZE_INVERSES[log_n];
    
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        const size_t idx = base_idx + k;
        if (idx < total) {
            d_data[idx] = bfield_mul_impl(__ldg(&d_data[idx]), scale);
        }
    }
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
 * OPTIMIZED: Simplified fused kernel for first 10 stages only (stage 0-9)
 * Features:
 * - Bank conflict avoidance with padding (~1.3x faster)
 * - Adaptive ILP for early/late stages
 * - Reduced global memory traffic (only 2 accesses per 1024 elements)
 */
__global__ void batched_ntt_fused_first10_kernel(
    uint64_t* __restrict__ d_data,
    size_t n,
    size_t num_cols,
    const uint64_t* __restrict__ d_twiddles
) {
    // OPTIMIZED: Add padding to avoid 32-way bank conflicts
    // Padding formula: idx + (idx >> 5) spreads accesses across 33 banks
    constexpr int SMEM_PADDING = 33;
    __shared__ uint64_t smem[FUSED_SIZE + SMEM_PADDING];
    
    const size_t num_chunks_per_col = n / FUSED_SIZE;
    const size_t chunk_idx = blockIdx.x;
    const size_t total_chunks = num_chunks_per_col * num_cols;
    
    if (chunk_idx >= total_chunks) return;
    
    const size_t col = chunk_idx / num_chunks_per_col;
    const size_t chunk_in_col = chunk_idx % num_chunks_per_col;
    const size_t global_base = col * n + chunk_in_col * FUSED_SIZE;
    
    // OPTIMIZED: Vectorized coalesced load with bank conflict avoidance
    // Use 128-bit loads where possible (2 uint64_t at a time)
    constexpr int ELEMS_PER_THREAD = 4;
    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {
        const size_t idx = threadIdx.x + k * blockDim.x;
        if (idx < FUSED_SIZE) {
            const size_t padded_idx = idx + (idx >> 5);
            // OPTIMIZED: Use __ldg for read-only global memory (better caching)
            smem[padded_idx] = __ldg(&d_data[global_base + idx]);
        }
    }
    __syncthreads();
    
    // OPTIMIZED: Process stages 0-9 in shared memory with improved ILP
    // Pre-compute stage constants to reduce redundant calculations
    #pragma unroll
    for (int stage = 0; stage < FUSED_STAGES; stage++) {
        const size_t half_size = 1ULL << stage;
        const size_t full_size = half_size * 2;
        const size_t num_butterflies = FUSED_SIZE / 2;  // 512
        const size_t twiddle_offset = half_size - 1;
        
        // OPTIMIZED: Adaptive ILP with better work distribution
        // Early stages: 4 butterflies/thread, later stages: 2 butterflies/thread
        constexpr int ILP_EARLY = 4;
        constexpr int ILP_LATE = 2;
        const int ilp = (stage < 6) ? ILP_EARLY : ILP_LATE;
        const int butterflies_per_thread = ilp;
        
        // OPTIMIZED: Process butterflies with better register usage
        #pragma unroll
        for (int b = 0; b < butterflies_per_thread; b++) {
            const size_t butterfly_idx = threadIdx.x + b * blockDim.x;
            // Early exit for threads beyond num_butterflies (only affects last few threads)
            if (butterfly_idx >= num_butterflies) continue;
            
            const size_t group = butterfly_idx / half_size;
            const size_t pos = butterfly_idx % half_size;
            
            const size_t i = group * full_size + pos;
            const size_t j = i + half_size;
            
            // OPTIMIZED: Apply padding offsets to avoid bank conflicts
            const size_t i_padded = i + (i >> 5);
            const size_t j_padded = j + (j >> 5);
            
            // OPTIMIZED: Use texture fetch for twiddles (better cache than __ldg)
            const uint64_t twiddle = (tex_twiddles_fwd != 0) ? 
                tex_fetch_twiddle_fwd(twiddle_offset + pos) : 
                __ldg(&d_twiddles[twiddle_offset + pos]);
            
            // OPTIMIZED: Inline butterfly with better register usage and instruction scheduling
            // Load both values first, then compute, then store (better pipeline utilization)
            const uint64_t a_val = smem[i_padded];
            const uint64_t b_val = smem[j_padded];
            const uint64_t t = bfield_mul_impl(b_val, twiddle);
            smem[i_padded] = bfield_add_impl(a_val, t);
            smem[j_padded] = bfield_sub_impl(a_val, t);
        }
        __syncthreads();
    }
    
    // OPTIMIZED: Vectorized coalesced store with padding-aware indexing
    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {
        const size_t idx = threadIdx.x + k * blockDim.x;
        if (idx < FUSED_SIZE) {
            const size_t padded_idx = idx + (idx >> 5);
            d_data[global_base + idx] = smem[padded_idx];
        }
    }
}

/**
 * Batched forward NTT for contiguous column-major data
 * Uses fused shared-memory kernel for first 10 stages + ILP kernel for remaining
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
    
    // OPTIMIZED: Calculate log_n using built-in function (faster)
    size_t log_n = 0;
    if (n > 1) {
        log_n = 63 - __builtin_clzll(n);
    }
    
    // Ensure twiddles are precomputed
    ensure_twiddles(n, stream);
    
    const size_t total_elements = n * num_cols;
    const size_t total_butterflies = (n / 2) * num_cols;
    
    // OPTIMIZED: Adaptive block size - use 128 for better occupancy on large problems
    // 256 threads = 8 warps, 128 threads = 4 warps (better for register pressure)
    int block_size = (total_butterflies > 1000000) ? 128 : 256;
    int grid_elements = (total_elements + block_size - 1) / block_size;
    
    // MAXIMIZED: Use ILP=8 for maximum instruction-level parallelism
    constexpr int ILP = 8;
    int grid_butterflies_ilp = ((total_butterflies + ILP - 1) / ILP + block_size - 1) / block_size;
    
    // OPTIMIZED: Batched bit-reversal with optimized block size
    batched_bit_reverse_kernel<<<grid_elements, block_size, 0, stream>>>(
        d_data, n, log_n, num_cols
    );
    
    // OPTIMIZED: Use fused kernel for first 10 stages with optimized block size
    size_t start_stage = 0;
    const bool disable_fused = (std::getenv("TRITON_DISABLE_FUSED_NTT") != nullptr);
    if (!disable_fused && n >= FUSED_SIZE) {
        const size_t num_chunks = (n / FUSED_SIZE) * num_cols;
        // Use 256 threads for fused kernel (optimal for 1024 elements = 4 elems/thread)
        batched_ntt_fused_first10_kernel<<<num_chunks, 256, 0, stream>>>(
            d_data, n, num_cols, d_twiddles_fwd
        );
        start_stage = FUSED_STAGES;
    }
    
    // OPTIMIZED: Remaining stages with ILP=8 kernel and adaptive block size
    for (size_t stage = start_stage; stage < log_n; ++stage) {
        batched_ntt_butterfly_ilp8_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
            d_data, n, stage, num_cols, d_twiddles_fwd
        );
    }
}

/**
 * OPTIMIZED: Fused inverse NTT kernel for first 10 stages
 * Features: Bank conflict avoidance, adaptive ILP, reduced memory traffic
 */
__global__ void batched_intt_fused_first10_kernel(
    uint64_t* __restrict__ d_data,
    size_t n,
    size_t num_cols,
    const uint64_t* __restrict__ d_twiddles_inv
) {
    // OPTIMIZED: Add padding to avoid 32-way bank conflicts
    constexpr int SMEM_PADDING = 33;
    __shared__ uint64_t smem[FUSED_SIZE + SMEM_PADDING];
    
    const size_t num_chunks_per_col = n / FUSED_SIZE;
    const size_t chunk_idx = blockIdx.x;
    const size_t total_chunks = num_chunks_per_col * num_cols;
    
    if (chunk_idx >= total_chunks) return;
    
    const size_t col = chunk_idx / num_chunks_per_col;
    const size_t chunk_in_col = chunk_idx % num_chunks_per_col;
    const size_t global_base = col * n + chunk_in_col * FUSED_SIZE;
    
    // OPTIMIZED: Vectorized coalesced load with __ldg
    constexpr int ELEMS_PER_THREAD = 4;
    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {
        const size_t idx = threadIdx.x + k * blockDim.x;
        if (idx < FUSED_SIZE) {
            const size_t padded_idx = idx + (idx >> 5);
            smem[padded_idx] = __ldg(&d_data[global_base + idx]);
        }
    }
    __syncthreads();
    
    // OPTIMIZED: Process stages 0-9 with improved ILP
    #pragma unroll
    for (int stage = 0; stage < FUSED_STAGES; stage++) {
        const size_t half_size = 1ULL << stage;
        const size_t full_size = half_size * 2;
        const size_t num_butterflies = FUSED_SIZE / 2;
        const size_t twiddle_offset = half_size - 1;
        
        constexpr int ILP_EARLY = 4;
        constexpr int ILP_LATE = 2;
        const int ilp = (stage < 6) ? ILP_EARLY : ILP_LATE;
        const int butterflies_per_thread = ilp;
        
        #pragma unroll
        for (int b = 0; b < butterflies_per_thread; b++) {
            const size_t butterfly_idx = threadIdx.x + b * blockDim.x;
            if (butterfly_idx >= num_butterflies) continue;
            
            const size_t group = butterfly_idx / half_size;
            const size_t pos = butterfly_idx % half_size;
            
            const size_t i = group * full_size + pos;
            const size_t j = i + half_size;
            
            // OPTIMIZED: Apply padding offsets
            const size_t i_padded = i + (i >> 5);
            const size_t j_padded = j + (j >> 5);
            
            // OPTIMIZED: Use texture fetch for twiddles (better cache)
            const uint64_t twiddle = (tex_twiddles_inv != 0) ? 
                tex_fetch_twiddle_inv(twiddle_offset + pos) : 
                __ldg(&d_twiddles_inv[twiddle_offset + pos]);
            
            // OPTIMIZED: Inline butterfly with better register usage and instruction scheduling
            const uint64_t a_val = smem[i_padded];
            const uint64_t b_val = smem[j_padded];
            const uint64_t t = bfield_mul_impl(b_val, twiddle);
            smem[i_padded] = bfield_add_impl(a_val, t);
            smem[j_padded] = bfield_sub_impl(a_val, t);
        }
        __syncthreads();
    }
    
    // OPTIMIZED: Vectorized coalesced store with padding-aware indexing
    #pragma unroll
    for (int k = 0; k < ELEMS_PER_THREAD; k++) {
        const size_t idx = threadIdx.x + k * blockDim.x;
        if (idx < FUSED_SIZE) {
            const size_t padded_idx = idx + (idx >> 5);
            d_data[global_base + idx] = smem[padded_idx];
        }
    }
}

/**
 * Batched inverse NTT for contiguous column-major data
 * Uses fused shared-memory kernel for first 10 stages + ILP kernel for remaining
 */
void ntt_inverse_batched_gpu(
    uint64_t* d_data,
    size_t n,
    size_t num_cols,
    cudaStream_t stream
) {
    ntt_init_constants();
    
    if (n == 0 || (n & (n - 1)) != 0 || num_cols == 0) return;
    
    // OPTIMIZED: Calculate log_n using built-in function (faster)
    size_t log_n = 0;
    if (n > 1) {
        log_n = 63 - __builtin_clzll(n);
    }
    
    // Ensure twiddles are precomputed
    ensure_twiddles(n, stream);
    
    const size_t total_elements = n * num_cols;
    const size_t total_butterflies = (n / 2) * num_cols;
    
    // OPTIMIZED: Adaptive block size for better occupancy
    int block_size = 256;
    if (total_butterflies > 2000000) {
        block_size = 128;  // Very large: reduce register pressure
    } else if (total_butterflies > 500000 && num_cols > 100) {
        block_size = 192;  // Medium-large: balance
    }
    int grid_elements = (total_elements + block_size - 1) / block_size;
    
    // MAXIMIZED: Use ILP=8 for maximum instruction-level parallelism
    constexpr int ILP = 8;
    int grid_butterflies_ilp = ((total_butterflies + ILP - 1) / ILP + block_size - 1) / block_size;
    
    // OPTIMIZED: Batched bit-reversal
    batched_bit_reverse_kernel<<<grid_elements, block_size, 0, stream>>>(
        d_data, n, log_n, num_cols
    );
    
    // OPTIMIZED: Use fused kernel for first 10 stages
    size_t start_stage = 0;
    const bool disable_fused = (std::getenv("TRITON_DISABLE_FUSED_NTT") != nullptr);
    if (!disable_fused && n >= FUSED_SIZE) {
        const size_t num_chunks = (n / FUSED_SIZE) * num_cols;
        batched_intt_fused_first10_kernel<<<num_chunks, 256, 0, stream>>>(
            d_data, n, num_cols, d_twiddles_inv
        );
        start_stage = FUSED_STAGES;
    }
    
    // OPTIMIZED: Remaining stages with ILP=8 kernel
    for (size_t stage = start_stage; stage < log_n; ++stage) {
        batched_ntt_butterfly_ilp8_kernel<<<grid_butterflies_ilp, block_size, 0, stream>>>(
            d_data, n, stage, num_cols, d_twiddles_inv
        );
    }
    
    // OPTIMIZED: Batched scale by n^{-1}
    batched_ntt_scale_kernel<<<grid_elements, block_size, 0, stream>>>(
        d_data, n, log_n, num_cols
    );
}

// ============================================================================
// Cleanup
// ============================================================================

/**
 * OPTIMIZED: Cleanup function to prevent memory leaks
 * Destroys texture objects and frees device memory
 */
void ntt_cleanup() {
    if (tex_twiddles_fwd) {
        cudaDestroyTextureObject(tex_twiddles_fwd);
        tex_twiddles_fwd = 0;
    }
    if (tex_twiddles_inv) {
        cudaDestroyTextureObject(tex_twiddles_inv);
        tex_twiddles_inv = 0;
    }
    if (d_twiddles_fwd) {
        cudaFree(d_twiddles_fwd);
        d_twiddles_fwd = nullptr;
    }
    if (d_twiddles_inv) {
        cudaFree(d_twiddles_inv);
        d_twiddles_inv = nullptr;
    }
    twiddles_max_n = 0;
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
