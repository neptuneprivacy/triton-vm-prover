/**
 * GPU-accelerated polynomial multiplication using NTT
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/poly_mul_kernel.cuh"
#include "gpu/kernels/ntt_kernel.cuh"
#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/cuda_common.cuh"
#include <cuda_runtime.h>
#include <cstdio>

namespace {

using triton_vm::gpu::kernels::bfield_mul_impl;

// Pointwise multiplication kernel
__global__ void pointwise_mul_kernel(
    uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    size_t n
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    a[idx] = bfield_mul_impl(a[idx], b[idx]);
}

} // anonymous namespace

extern "C" {

int gpu_poly_mul_available(void) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

int gpu_poly_mul_ntt(
    const uint64_t* a_coeffs, size_t a_size,
    const uint64_t* b_coeffs, size_t b_size,
    uint64_t* result_coeffs, size_t* result_size
) {
    if (a_size == 0 || b_size == 0) {
        *result_size = 1;
        result_coeffs[0] = 0;
        return 0;
    }
    
    size_t actual_result_size = a_size + b_size - 1;
    *result_size = actual_result_size;
    
    // Pad to next power of 2
    size_t n = 1;
    while (n < actual_result_size) n *= 2;
    
    // Allocate GPU memory
    uint64_t *d_a = nullptr, *d_b = nullptr;
    cudaError_t err;
    
    err = cudaMalloc(&d_a, n * sizeof(uint64_t));
    if (err != cudaSuccess) return -1;
    
    err = cudaMalloc(&d_b, n * sizeof(uint64_t));
    if (err != cudaSuccess) {
        cudaFree(d_a);
        return -2;
    }
    
    // Zero-initialize and copy
    err = cudaMemset(d_a, 0, n * sizeof(uint64_t));
    if (err != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return -3; }
    
    err = cudaMemset(d_b, 0, n * sizeof(uint64_t));
    if (err != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return -4; }
    
    err = cudaMemcpy(d_a, a_coeffs, a_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return -5; }
    
    err = cudaMemcpy(d_b, b_coeffs, b_size * sizeof(uint64_t), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return -6; }
    
    // Initialize NTT constants
    triton_vm::gpu::kernels::ntt_init_constants();
    
    // Forward NTT on both
    triton_vm::gpu::kernels::ntt_forward_gpu(d_a, n, nullptr);
    triton_vm::gpu::kernels::ntt_forward_gpu(d_b, n, nullptr);
    
    // Pointwise multiplication
    size_t block = 256;
    size_t grid = (n + block - 1) / block;
    pointwise_mul_kernel<<<grid, block>>>(d_a, d_b, n);
    
    // Inverse NTT
    triton_vm::gpu::kernels::ntt_inverse_gpu(d_a, n, nullptr);
    
    // Synchronize and check for errors
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return -7; }
    
    // Copy result back
    err = cudaMemcpy(result_coeffs, d_a, actual_result_size * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { cudaFree(d_a); cudaFree(d_b); return -8; }
    
    cudaFree(d_a);
    cudaFree(d_b);
    
    return 0;
}

} // extern "C"

#else // !TRITON_CUDA_ENABLED

extern "C" {

int gpu_poly_mul_available(void) {
    return 0;
}

int gpu_poly_mul_ntt(
    const uint64_t* a_coeffs, size_t a_size,
    const uint64_t* b_coeffs, size_t b_size,
    uint64_t* result_coeffs, size_t* result_size
) {
    return -1;  // Not available
}

} // extern "C"

#endif // TRITON_CUDA_ENABLED

