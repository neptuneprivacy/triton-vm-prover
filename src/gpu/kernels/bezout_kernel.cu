/**
 * GPU-Accelerated Bézout Coefficient Computation with Memory Pool
 * 
 * Uses a pre-allocated GPU memory pool to avoid per-operation allocation overhead.
 * Falls back to CPU for operations that don't benefit from GPU acceleration.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bezout_kernel.cuh"
#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/kernels/ntt_kernel.cuh"
#include "gpu/cuda_common.cuh"
#include "ntt/ntt.hpp"
#include <cuda_runtime.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <chrono>
#include <cstdlib>
#include <mutex>

namespace triton_vm {
namespace gpu {
namespace kernels {

static constexpr uint64_t P = 18446744069414584321ULL;

// ============================================================================
// GPU Memory Pool - Avoids per-operation allocation overhead
// ============================================================================

class GpuMemoryPool {
public:
    static GpuMemoryPool& instance() {
        static GpuMemoryPool pool;
        return pool;
    }
    
    // Get scratch buffers for NTT multiplication
    // Returns three buffers of size `padded_size` each
    bool get_ntt_buffers(size_t padded_size, uint64_t** d_a, uint64_t** d_b, uint64_t** d_res) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        size_t required = 3 * padded_size * sizeof(uint64_t);
        if (required > capacity_) {
            // Need to resize
            if (buffer_) cudaFree(buffer_);
            capacity_ = std::max(required, size_t(64 * 1024 * 1024));  // Min 64MB
            if (cudaMalloc(&buffer_, capacity_) != cudaSuccess) {
                buffer_ = nullptr;
                capacity_ = 0;
                return false;
            }
        }
        
        *d_a = reinterpret_cast<uint64_t*>(buffer_);
        *d_b = *d_a + padded_size;
        *d_res = *d_b + padded_size;
        return true;
    }
    
    ~GpuMemoryPool() {
        if (buffer_) cudaFree(buffer_);
    }
    
private:
    GpuMemoryPool() : buffer_(nullptr), capacity_(0) {}
    
    std::mutex mutex_;
    void* buffer_;
    size_t capacity_;
};

// ============================================================================
// GPU Kernels
// ============================================================================

__global__ void pointwise_mul_kernel(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    uint64_t* __restrict__ result,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    result[idx] = bfield_mul_impl(a[idx], b[idx]);
}

__global__ void poly_derivative_kernel(
    const uint64_t* __restrict__ coeffs,
    uint64_t* __restrict__ deriv,
    size_t degree
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= degree) return;
    deriv[idx] = bfield_mul_impl(coeffs[idx + 1], static_cast<uint64_t>(idx + 1));
}

__global__ void poly_eval_kernel(
    const uint64_t* __restrict__ coeffs,
    size_t num_coeffs,
    const uint64_t* __restrict__ points,
    size_t num_points,
    uint64_t* __restrict__ results
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_points) return;
    
    uint64_t x = points[idx];
    uint64_t acc = 0;
    for (int i = static_cast<int>(num_coeffs) - 1; i >= 0; --i) {
        acc = bfield_mul_impl(acc, x);
        acc = bfield_add_impl(acc, coeffs[i]);
    }
    results[idx] = acc;
}

// ============================================================================
// GPU NTT Polynomial Multiplication with Memory Pool
// ============================================================================

bool gpu_poly_mul_pooled(
    const std::vector<uint64_t>& a,
    const std::vector<uint64_t>& b,
    std::vector<uint64_t>& result,
    cudaStream_t stream
) {
    size_t result_size = a.size() + b.size() - 1;
    size_t padded = 1;
    while (padded < result_size) padded *= 2;
    
    uint64_t *d_a, *d_b, *d_res;
    if (!GpuMemoryPool::instance().get_ntt_buffers(padded, &d_a, &d_b, &d_res)) {
        return false;  // Pool allocation failed
    }
    
    // Zero and copy
    cudaMemsetAsync(d_a, 0, padded * sizeof(uint64_t), stream);
    cudaMemsetAsync(d_b, 0, padded * sizeof(uint64_t), stream);
    cudaMemcpyAsync(d_a, a.data(), a.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), b.size() * sizeof(uint64_t), cudaMemcpyHostToDevice, stream);
    
    ntt_init_constants();
    ntt_forward_gpu(d_a, padded, stream);
    ntt_forward_gpu(d_b, padded, stream);
    
    size_t block = 256;
    size_t grid = (padded + block - 1) / block;
    pointwise_mul_kernel<<<grid, block, 0, stream>>>(d_a, d_b, d_res, padded);
    
    ntt_inverse_gpu(d_res, padded, stream);
    
    result.resize(result_size);
    cudaMemcpyAsync(result.data(), d_res, result_size * sizeof(uint64_t), cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    
    return true;
}

// ============================================================================
// Polynomial Operations (CPU with optional GPU for large)
// ============================================================================

std::vector<BFieldElement> poly_mul_hybrid(
    const std::vector<BFieldElement>& a,
    const std::vector<BFieldElement>& b,
    cudaStream_t stream
) {
    if (a.empty() || b.empty()) return {BFieldElement::zero()};
    if (a.size() == 1 && a[0].is_zero()) return {BFieldElement::zero()};
    if (b.size() == 1 && b[0].is_zero()) return {BFieldElement::zero()};
    
    const size_t result_size = a.size() + b.size() - 1;
    
    // For large polynomials (>= 4096), try GPU with memory pool
    if (result_size >= 4096 && stream != nullptr) {
        std::vector<uint64_t> a_raw(a.size()), b_raw(b.size());
        for (size_t i = 0; i < a.size(); ++i) a_raw[i] = a[i].value();
        for (size_t i = 0; i < b.size(); ++i) b_raw[i] = b[i].value();
        
        std::vector<uint64_t> result_raw;
        if (gpu_poly_mul_pooled(a_raw, b_raw, result_raw, stream)) {
            std::vector<BFieldElement> result(result_size);
            for (size_t i = 0; i < result_size; ++i) {
                result[i] = BFieldElement(result_raw[i]);
            }
            return result;
        }
        // Fall through to CPU if GPU fails
    }
    
    // CPU NTT for medium polynomials
    if (result_size >= 64) {
        size_t n = 1;
        while (n < result_size) n *= 2;
        
        std::vector<BFieldElement> a_padded(n, BFieldElement::zero());
        std::vector<BFieldElement> b_padded(n, BFieldElement::zero());
        std::copy(a.begin(), a.end(), a_padded.begin());
        std::copy(b.begin(), b.end(), b_padded.begin());
        
        NTT::forward(a_padded);
        NTT::forward(b_padded);
        
        for (size_t i = 0; i < n; ++i) {
            a_padded[i] *= b_padded[i];
        }
        
        NTT::inverse(a_padded);
        a_padded.resize(result_size);
        return a_padded;
    }
    
    // Naive for small
    std::vector<BFieldElement> r(result_size, BFieldElement::zero());
    for (size_t i = 0; i < a.size(); ++i) {
        for (size_t j = 0; j < b.size(); ++j) {
            r[i + j] += a[i] * b[j];
        }
    }
    return r;
}

// ============================================================================
// Main Bézout Computation
// ============================================================================

std::pair<std::vector<BFieldElement>, std::vector<BFieldElement>>
gpu_compute_bezout_coefficients(
    const std::vector<BFieldElement>& unique_ramps,
    cudaStream_t stream
) {
    const size_t n = unique_ramps.size();
    if (n == 0) return {{}, {}};
    
    const bool profile = (std::getenv("TVM_PROFILE_GPU_BEZOUT") != nullptr);
    auto t_start = std::chrono::high_resolution_clock::now();
    auto log_time = [&](const char* msg) {
        if (profile) {
            auto now = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration<double, std::milli>(now - t_start).count();
            std::cout << "    GPU Bézout: " << msg << ": " << ms << " ms" << std::endl;
            t_start = now;
        }
    };
    
    // Helper lambdas
    auto trim = [](std::vector<BFieldElement> v) {
        while (!v.empty() && v.back().is_zero()) v.pop_back();
        if (v.empty()) v.push_back(BFieldElement::zero());
        return v;
    };
    
    auto poly_sub = [](const std::vector<BFieldElement>& a, const std::vector<BFieldElement>& b) {
        std::vector<BFieldElement> r(std::max(a.size(), b.size()), BFieldElement::zero());
        for (size_t i = 0; i < a.size(); ++i) r[i] += a[i];
        for (size_t i = 0; i < b.size(); ++i) r[i] -= b[i];
        return r;
    };
    
    auto poly_mul = [stream](const std::vector<BFieldElement>& a, const std::vector<BFieldElement>& b) {
        return poly_mul_hybrid(a, b, stream);
    };
    
    auto poly_eval = [](const std::vector<BFieldElement>& a, BFieldElement x) {
        BFieldElement acc = BFieldElement::zero();
        for (int i = static_cast<int>(a.size()) - 1; i >= 0; --i) {
            acc = acc * x + a[static_cast<size_t>(i)];
        }
        return acc;
    };
    
    auto poly_derivative = [](const std::vector<BFieldElement>& a) {
        if (a.size() <= 1) return std::vector<BFieldElement>{BFieldElement::zero()};
        std::vector<BFieldElement> d(a.size() - 1, BFieldElement::zero());
        for (size_t i = 1; i < a.size(); ++i) {
            d[i - 1] = a[i] * BFieldElement(static_cast<uint64_t>(i));
        }
        return d;
    };
    
    auto poly_div_exact = [&trim](const std::vector<BFieldElement>& dividend,
                                   const std::vector<BFieldElement>& divisor) {
        auto a = trim(dividend);
        auto d = trim(divisor);
        size_t deg_a = a.size() - 1;
        size_t deg_d = d.size() - 1;
        if (deg_a < deg_d) return std::vector<BFieldElement>{BFieldElement::zero()};

        std::vector<BFieldElement> a_desc(a.rbegin(), a.rend());
        std::vector<BFieldElement> d_desc(d.rbegin(), d.rend());

        size_t q_deg = deg_a - deg_d;
        std::vector<BFieldElement> q_desc(q_deg + 1, BFieldElement::zero());
        std::vector<BFieldElement> rem = a_desc;
        BFieldElement d_lead = d_desc[0];
        for (size_t k = 0; k <= q_deg; ++k) {
            BFieldElement lead = rem[k];
            if (lead.is_zero()) continue;
            BFieldElement qk = lead / d_lead;
            q_desc[k] += qk;
            for (size_t j = 0; j <= deg_d; ++j) {
                rem[k + j] -= qk * d_desc[j];
            }
        }
        std::vector<BFieldElement> q(q_desc.rbegin(), q_desc.rend());
        return trim(q);
    };
    
    // Build linear factors
    std::vector<std::vector<BFieldElement>> factors(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        factors[i] = {BFieldElement::zero() - unique_ramps[i], BFieldElement::one()};
    }
    
    // Product tree for rp(x)
    std::vector<std::vector<BFieldElement>> tree_factors = factors;
    while (tree_factors.size() > 1) {
        size_t new_sz = (tree_factors.size() + 1) / 2;
        std::vector<std::vector<BFieldElement>> new_factors(new_sz);
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < new_sz; ++i) {
            size_t left = 2 * i;
            size_t right = 2 * i + 1;
            if (right < tree_factors.size()) {
                new_factors[i] = poly_mul(tree_factors[left], tree_factors[right]);
            } else {
                new_factors[i] = tree_factors[left];
            }
        }
        tree_factors = std::move(new_factors);
    }
    std::vector<BFieldElement> rp = tree_factors.empty() ? 
        std::vector<BFieldElement>{BFieldElement::one()} : tree_factors[0];
    log_time("Product tree");
    
    // fd = rp'
    std::vector<BFieldElement> fd = poly_derivative(rp);
    
    // Evaluate fd at all roots (parallel)
    std::vector<BFieldElement> fd_evals(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        fd_evals[i] = poly_eval(fd, unique_ramps[i]);
    }
    
    // Batch inversion
    std::vector<BFieldElement> b_in_roots = BFieldElement::batch_inversion(fd_evals);
    log_time("Derivative + Eval + Inversion");
    
    // Lagrange interpolation with block-based prefix/suffix
    const size_t BLOCK_SIZE = 64;
    const size_t num_blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Block products
    std::vector<std::vector<BFieldElement>> block_prods(num_blocks);
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < num_blocks; ++b) {
        size_t start = b * BLOCK_SIZE;
        size_t end = std::min(start + BLOCK_SIZE, n);
        std::vector<BFieldElement> prod{BFieldElement::one()};
        for (size_t i = start; i < end; ++i) {
            prod = poly_mul(prod, factors[i]);
        }
        block_prods[b] = std::move(prod);
    }
    log_time("Block products");
    
    // Block prefix/suffix
    std::vector<std::vector<BFieldElement>> block_prefix(num_blocks + 1);
    block_prefix[0] = {BFieldElement::one()};
    for (size_t b = 0; b < num_blocks; ++b) {
        block_prefix[b + 1] = poly_mul(block_prefix[b], block_prods[b]);
    }
    
    std::vector<std::vector<BFieldElement>> block_suffix(num_blocks + 1);
    block_suffix[num_blocks] = {BFieldElement::one()};
    for (size_t b = num_blocks; b > 0; --b) {
        block_suffix[b - 1] = poly_mul(block_suffix[b], block_prods[b - 1]);
    }
    log_time("Block prefix/suffix");
    
    // Per-element prefix/suffix
    std::vector<std::vector<BFieldElement>> prefix(n + 1);
    std::vector<std::vector<BFieldElement>> suffix(n + 1);
    prefix[0] = block_prefix[0];
    suffix[n] = block_suffix[num_blocks];
    
    #pragma omp parallel for schedule(static)
    for (size_t b = 0; b < num_blocks; ++b) {
        size_t start = b * BLOCK_SIZE;
        size_t end = std::min(start + BLOCK_SIZE, n);
        
        std::vector<BFieldElement> local_prefix = block_prefix[b];
        for (size_t i = start; i < end; ++i) {
            prefix[i] = local_prefix;
            local_prefix = poly_mul(local_prefix, factors[i]);
        }
        prefix[end] = local_prefix;
        
        std::vector<BFieldElement> local_suffix = block_suffix[b + 1];
        for (size_t i = end; i > start; --i) {
            suffix[i] = local_suffix;
            local_suffix = poly_mul(local_suffix, factors[i - 1]);
        }
        suffix[start] = local_suffix;
    }
    log_time("Per-element prefix/suffix");
    
    // Basis polynomials
    std::vector<std::vector<BFieldElement>> basis_polys(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        std::vector<BFieldElement> basis = poly_mul(prefix[i], suffix[i + 1]);
        for (auto& c : basis) c *= b_in_roots[i];
        basis_polys[i] = std::move(basis);
    }
    log_time("Basis polynomials");
    
    // Sum basis polynomials
    std::vector<BFieldElement> b_coeffs(n, BFieldElement::zero());
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < basis_polys[i].size() && j < n; ++j) {
            b_coeffs[j] += basis_polys[i][j];
        }
    }
    b_coeffs = trim(b_coeffs);
    
    // a = (1 - fd*b) / rp
    std::vector<BFieldElement> fd_b = poly_mul(fd, b_coeffs);
    std::vector<BFieldElement> one_minus_fd_b = poly_sub({BFieldElement::one()}, fd_b);
    one_minus_fd_b = trim(one_minus_fd_b);
    std::vector<BFieldElement> a_coeffs = poly_div_exact(one_minus_fd_b, rp);
    log_time("Compute a coefficients");
    
    // Resize
    a_coeffs.resize(n, BFieldElement::zero());
    b_coeffs.resize(n, BFieldElement::zero());
    
    return {a_coeffs, b_coeffs};
}

// ============================================================================
// Export functions
// ============================================================================

void gpu_poly_mul(const uint64_t* d_a, size_t a_size, const uint64_t* d_b, size_t b_size,
                  uint64_t* d_result, cudaStream_t stream) {
    // Not used in new implementation
}

void gpu_product_tree(const uint64_t* d_roots, size_t n, uint64_t* d_result, cudaStream_t stream) {
    // Not used in new implementation  
}

void gpu_poly_eval_batch(const uint64_t* d_coeffs, size_t degree, const uint64_t* d_points,
                          size_t n, uint64_t* d_results, cudaStream_t stream) {
    if (n == 0) return;
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    poly_eval_kernel<<<grid, block, 0, stream>>>(d_coeffs, degree + 1, d_points, n, d_results);
}

void gpu_batch_inversion(const uint64_t* d_values, size_t n, uint64_t* d_results, cudaStream_t stream) {
    std::vector<uint64_t> values(n);
    cudaMemcpy(values.data(), d_values, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    std::vector<uint64_t> prefix(n + 1);
    prefix[0] = 1;
    for (size_t i = 0; i < n; ++i) {
        __uint128_t prod = static_cast<__uint128_t>(prefix[i]) * values[i];
        prefix[i + 1] = static_cast<uint64_t>(prod % P);
    }
    
    uint64_t total_inv = BFieldElement(prefix[n]).inverse().value();
    
    std::vector<uint64_t> inverses(n);
    uint64_t suffix_inv = total_inv;
    for (size_t i = n; i > 0; --i) {
        __uint128_t prod = static_cast<__uint128_t>(prefix[i - 1]) * suffix_inv;
        inverses[i - 1] = static_cast<uint64_t>(prod % P);
        prod = static_cast<__uint128_t>(suffix_inv) * values[i - 1];
        suffix_inv = static_cast<uint64_t>(prod % P);
    }
    
    cudaMemcpy(d_results, inverses.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

