/**
 * FRI Protocol CUDA Kernel Implementation
 * 
 * GPU-accelerated FRI folding operations.
 * Matches the CPU implementation in fri.cpp
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/kernels/xfield_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace triton_vm {
namespace gpu {
namespace kernels {

// ============================================================================
// FRI Folding Kernel - Matching CPU split_and_fold()
// ============================================================================

/**
 * FRI folding step - matches CPU FriRound::split_and_fold()
 * 
 * For each i in 0..n/2:
 *   scaled_offset_inv = folding_challenge * domain_point_inverses[i]
 *   left_summand = (1 + scaled_offset_inv) * codeword[i]
 *   right_summand = (1 - scaled_offset_inv) * codeword[n/2 + i]
 *   folded[i] = (left_summand + right_summand) * two_inv
 * 
 * @param d_codeword Input codeword (XFE, 3 values each)
 * @param codeword_len Number of XFieldElements in codeword
 * @param d_challenge Folding challenge (XFE, 3 values)
 * @param d_domain_inv Inverses of domain points (BFE, half_len values)
 * @param d_two_inv Inverse of 2 (BFE, 1 value)
 * @param d_folded Output folded codeword (half_len XFEs)
 */
__global__ void fri_fold_kernel(
    const uint64_t* d_codeword,
    size_t codeword_len,
    const uint64_t* d_challenge,
    const uint64_t* d_domain_inv,
    uint64_t two_inv,
    uint64_t* d_folded
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t half_len = codeword_len / 2;
    
    if (idx >= half_len) return;
    
    // Load challenge
    uint64_t ch0 = d_challenge[0];
    uint64_t ch1 = d_challenge[1];
    uint64_t ch2 = d_challenge[2];
    
    // scaled_offset_inv = folding_challenge * domain_point_inverses[i]
    // This is XFE * BFE (scalar multiplication)
    uint64_t dom_inv = d_domain_inv[idx];
    uint64_t soi0 = bfield_mul_impl(ch0, dom_inv);
    uint64_t soi1 = bfield_mul_impl(ch1, dom_inv);
    uint64_t soi2 = bfield_mul_impl(ch2, dom_inv);
    
    // one = (1, 0, 0)
    // one_plus_soi = (1 + soi0, soi1, soi2)
    uint64_t one_plus_soi0 = bfield_add_impl(1, soi0);
    uint64_t one_plus_soi1 = soi1;
    uint64_t one_plus_soi2 = soi2;
    
    // one_minus_soi = (1 - soi0, -soi1, -soi2)
    uint64_t one_minus_soi0 = bfield_sub_impl(1, soi0);
    uint64_t one_minus_soi1 = bfield_neg_impl(soi1);
    uint64_t one_minus_soi2 = bfield_neg_impl(soi2);
    
    // Load codeword[i] and codeword[half_len + i]
    size_t left_offset = idx * 3;
    size_t right_offset = (idx + half_len) * 3;
    
    uint64_t left0 = d_codeword[left_offset + 0];
    uint64_t left1 = d_codeword[left_offset + 1];
    uint64_t left2 = d_codeword[left_offset + 2];
    
    uint64_t right0 = d_codeword[right_offset + 0];
    uint64_t right1 = d_codeword[right_offset + 1];
    uint64_t right2 = d_codeword[right_offset + 2];
    
    // left_summand = (1 + scaled_offset_inv) * codeword[i]
    uint64_t ls0, ls1, ls2;
    xfield_mul_impl(one_plus_soi0, one_plus_soi1, one_plus_soi2,
                    left0, left1, left2,
                    ls0, ls1, ls2);
    
    // right_summand = (1 - scaled_offset_inv) * codeword[half_len + i]
    uint64_t rs0, rs1, rs2;
    xfield_mul_impl(one_minus_soi0, one_minus_soi1, one_minus_soi2,
                    right0, right1, right2,
                    rs0, rs1, rs2);
    
    // sum = left_summand + right_summand
    uint64_t sum0 = bfield_add_impl(ls0, rs0);
    uint64_t sum1 = bfield_add_impl(ls1, rs1);
    uint64_t sum2 = bfield_add_impl(ls2, rs2);
    
    // folded[i] = sum * two_inv (scalar multiplication)
    size_t out_offset = idx * 3;
    d_folded[out_offset + 0] = bfield_mul_impl(sum0, two_inv);
    d_folded[out_offset + 1] = bfield_mul_impl(sum1, two_inv);
    d_folded[out_offset + 2] = bfield_mul_impl(sum2, two_inv);
}

// Optimized: compute domain inverses on the fly as inv_offset * inv_generator^i.
// Process a small contiguous chunk per thread to amortize pow() cost.
template<int ITEMS_PER_THREAD>
__global__ void fri_fold_fast_kernel(
    const uint64_t* d_codeword,
    size_t codeword_len,
    const uint64_t* d_challenge,
    uint64_t inv_offset,
    uint64_t inv_generator,
    uint64_t two_inv,
    uint64_t* d_folded
) {
    const size_t half_len = codeword_len / 2;
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t start = tid * static_cast<size_t>(ITEMS_PER_THREAD);
    if (start >= half_len) return;

    // Load challenge once
    const uint64_t ch0 = d_challenge[0];
    const uint64_t ch1 = d_challenge[1];
    const uint64_t ch2 = d_challenge[2];

    // inv(x_start) = inv_offset * inv_generator^start
    uint64_t inv_gpow = bfield_pow_impl(inv_generator, start);
    uint64_t dom_inv = bfield_mul_impl(inv_offset, inv_gpow);

#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
        const size_t idx = start + static_cast<size_t>(j);
        if (idx >= half_len) break;

        // scaled_offset_inv = folding_challenge * dom_inv (XFE * BFE scalar)
        const uint64_t soi0 = bfield_mul_impl(ch0, dom_inv);
        const uint64_t soi1 = bfield_mul_impl(ch1, dom_inv);
        const uint64_t soi2 = bfield_mul_impl(ch2, dom_inv);

        const uint64_t one_plus_soi0 = bfield_add_impl(1, soi0);
        const uint64_t one_plus_soi1 = soi1;
        const uint64_t one_plus_soi2 = soi2;

        const uint64_t one_minus_soi0 = bfield_sub_impl(1, soi0);
        const uint64_t one_minus_soi1 = bfield_neg_impl(soi1);
        const uint64_t one_minus_soi2 = bfield_neg_impl(soi2);

        const size_t left_offset = idx * 3;
        const size_t right_offset = (idx + half_len) * 3;

        const uint64_t left0 = d_codeword[left_offset + 0];
        const uint64_t left1 = d_codeword[left_offset + 1];
        const uint64_t left2 = d_codeword[left_offset + 2];

        const uint64_t right0 = d_codeword[right_offset + 0];
        const uint64_t right1 = d_codeword[right_offset + 1];
        const uint64_t right2 = d_codeword[right_offset + 2];

        uint64_t ls0, ls1, ls2;
        xfield_mul_impl(one_plus_soi0, one_plus_soi1, one_plus_soi2,
                        left0, left1, left2,
                        ls0, ls1, ls2);

        uint64_t rs0, rs1, rs2;
        xfield_mul_impl(one_minus_soi0, one_minus_soi1, one_minus_soi2,
                        right0, right1, right2,
                        rs0, rs1, rs2);

        const uint64_t sum0 = bfield_add_impl(ls0, rs0);
        const uint64_t sum1 = bfield_add_impl(ls1, rs1);
        const uint64_t sum2 = bfield_add_impl(ls2, rs2);

        const size_t out_offset = idx * 3;
        d_folded[out_offset + 0] = bfield_mul_impl(sum0, two_inv);
        d_folded[out_offset + 1] = bfield_mul_impl(sum1, two_inv);
        d_folded[out_offset + 2] = bfield_mul_impl(sum2, two_inv);

        // dom_inv_{i+1} = dom_inv_i * inv_generator
        dom_inv = bfield_mul_impl(dom_inv, inv_generator);
    }
}

/**
 * Compute domain point inverses for FRI folding
 * domain_points[i] = offset * generator^i
 * domain_inv[i] = 1 / domain_points[i]
 */
__global__ void compute_domain_inverses_kernel(
    uint64_t offset,
    uint64_t generator,
    uint64_t* d_domain_inv,
    size_t half_len
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= half_len) return;
    
    // Compute offset * generator^idx
    uint64_t gen_power = bfield_pow_impl(generator, idx);
    uint64_t domain_point = bfield_mul_impl(offset, gen_power);
    
    // Compute inverse
    d_domain_inv[idx] = bfield_inv_impl(domain_point);
}

// Optimized domain inverses: inv(x_i) = inv_offset * inv_generator^i.
template<int ITEMS_PER_THREAD>
__global__ void compute_domain_inverses_fast_kernel(
    uint64_t inv_offset,
    uint64_t inv_generator,
    uint64_t* d_domain_inv,
    size_t half_len
) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t start = tid * static_cast<size_t>(ITEMS_PER_THREAD);
    if (start >= half_len) return;

    // inv_generator^start
    uint64_t gpow = bfield_pow_impl(inv_generator, start);
    uint64_t cur = bfield_mul_impl(inv_offset, gpow);

#pragma unroll
    for (int j = 0; j < ITEMS_PER_THREAD; ++j) {
        const size_t idx = start + static_cast<size_t>(j);
        if (idx >= half_len) break;
        d_domain_inv[idx] = cur;
        cur = bfield_mul_impl(cur, inv_generator);
    }
}

// ============================================================================
// Host Interface
// ============================================================================

/**
 * Perform FRI folding on GPU
 */
void fri_fold_gpu(
    const uint64_t* d_codeword,
    size_t codeword_len,
    const uint64_t* d_challenge,
    const uint64_t* d_domain_inv,
    uint64_t two_inv,
    uint64_t* d_folded,
    cudaStream_t stream
) {
    size_t half_len = codeword_len / 2;
    int block_size = 256;
    int grid_size = (half_len + block_size - 1) / block_size;
    
    fri_fold_kernel<<<grid_size, block_size, 0, stream>>>(
        d_codeword,
        codeword_len,
        d_challenge,
        d_domain_inv,
        two_inv,
        d_folded
    );
}

void fri_fold_gpu_fast(
    const uint64_t* d_codeword,
    size_t codeword_len,
    const uint64_t* d_challenge,
    uint64_t inv_offset,
    uint64_t inv_generator,
    uint64_t two_inv,
    uint64_t* d_folded,
    cudaStream_t stream
) {
    const size_t half_len = codeword_len / 2;
    constexpr int block_size = 256;
    // 4 items per thread amortizes pow(inv_generator, start) while keeping register use reasonable.
    constexpr int ITEMS = 4;
    const int grid_size = (int)((half_len + (block_size * ITEMS) - 1) / (block_size * ITEMS));
    fri_fold_fast_kernel<ITEMS><<<grid_size, block_size, 0, stream>>>(
        d_codeword,
        codeword_len,
        d_challenge,
        inv_offset,
        inv_generator,
        two_inv,
        d_folded
    );
}

/**
 * Compute domain inverses on GPU
 */
void compute_domain_inverses_gpu(
    uint64_t offset,
    uint64_t generator,
    uint64_t* d_domain_inv,
    size_t half_len,
    cudaStream_t stream
) {
    int block_size = 256;
    int grid_size = (half_len + block_size - 1) / block_size;
    
    compute_domain_inverses_kernel<<<grid_size, block_size, 0, stream>>>(
        offset, generator, d_domain_inv, half_len
    );
}

void compute_domain_inverses_fast_gpu(
    uint64_t inv_offset,
    uint64_t inv_generator,
    uint64_t* d_domain_inv,
    size_t half_len,
    cudaStream_t stream
) {
    constexpr int block_size = 256;
    constexpr int ITEMS = 8;
    const int grid_size = (int)((half_len + (size_t)block_size * ITEMS - 1) / ((size_t)block_size * ITEMS));
    compute_domain_inverses_fast_kernel<ITEMS><<<grid_size, block_size, 0, stream>>>(
        inv_offset, inv_generator, d_domain_inv, half_len
    );
}

// Host-side modular power for two_inv computation
static uint64_t host_pow_mod(uint64_t base, uint64_t exp) {
    uint64_t result = 1;
    while (exp > 0) {
        if (exp & 1) {
            __uint128_t prod = static_cast<__uint128_t>(result) * base;
            result = static_cast<uint64_t>(prod % GOLDILOCKS_P);
        }
        __uint128_t sq = static_cast<__uint128_t>(base) * base;
        base = static_cast<uint64_t>(sq % GOLDILOCKS_P);
        exp >>= 1;
    }
    return result;
}

// Backward compatibility wrapper
void fri_fold_device(
    const uint64_t* d_codeword,
    size_t codeword_len,
    const uint64_t* d_challenge,
    uint64_t* d_folded,
    cudaStream_t stream
) {
    // This simplified version doesn't use proper domain inverses
    // Use fri_fold_gpu with proper domain_inv for correctness
    size_t half_len = codeword_len / 2;
    
    // Allocate temporary domain inverses (set to 1 for simplified testing)
    uint64_t* d_domain_inv;
    cudaMalloc(&d_domain_inv, half_len * sizeof(uint64_t));
    
    // Fill with 1s (identity)
    std::vector<uint64_t> ones(half_len, 1);
    cudaMemcpy(d_domain_inv, ones.data(), half_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    // Use two_inv = inverse of 2 in Goldilocks
    // 2^(-1) mod P where P = 2^64 - 2^32 + 1
    uint64_t two_inv = host_pow_mod(2, GOLDILOCKS_P - 2);
    
    fri_fold_gpu(d_codeword, codeword_len, d_challenge, d_domain_inv, two_inv, d_folded, stream);
    
    cudaFree(d_domain_inv);
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
