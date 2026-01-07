/**
 * GPU Transition Constraint Evaluation Kernel - Part 0/4
 * Handles transition constraints 0-99 (of 398 total)
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/kernels/xfield_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm::gpu {

static constexpr uint64_t INV_R = 18446744065119617025ULL;

struct Bfe {
    uint64_t v;
    __device__ __forceinline__ Bfe() : v(0) {}
    __device__ __forceinline__ explicit Bfe(uint64_t x) : v(x) {}
    __device__ __forceinline__ static Bfe zero() { return Bfe(0); }
    __device__ __forceinline__ static Bfe one() { return Bfe(1); }
    __device__ __forceinline__ static Bfe from_raw_u64(uint64_t raw) {
        return Bfe(kernels::bfield_mul_impl(raw, INV_R));
    }
};

__device__ __forceinline__ Bfe operator+(Bfe a, Bfe b) { return Bfe(kernels::bfield_add_impl(a.v, b.v)); }
__device__ __forceinline__ Bfe operator-(Bfe a, Bfe b) { return Bfe(kernels::bfield_sub_impl(a.v, b.v)); }
__device__ __forceinline__ Bfe operator*(Bfe a, Bfe b) { return Bfe(kernels::bfield_mul_impl(a.v, b.v)); }

struct Xfe {
    uint64_t c0, c1, c2;
    __device__ __forceinline__ Xfe() : c0(0), c1(0), c2(0) {}
    __device__ __forceinline__ explicit Xfe(Bfe b) : c0(b.v), c1(0), c2(0) {}
    __device__ __forceinline__ Xfe(uint64_t a0, uint64_t a1, uint64_t a2) : c0(a0), c1(a1), c2(a2) {}
    __device__ __forceinline__ static Xfe zero() { return Xfe(0, 0, 0); }
    __device__ __forceinline__ static Xfe one() { return Xfe(1, 0, 0); }
};

__device__ __forceinline__ Xfe operator+(Xfe a, Xfe b) {
    uint64_t r0, r1, r2;
    kernels::xfield_add_impl(a.c0, a.c1, a.c2, b.c0, b.c1, b.c2, r0, r1, r2);
    return Xfe(r0, r1, r2);
}
__device__ __forceinline__ Xfe operator-(Xfe a, Xfe b) {
    uint64_t r0, r1, r2;
    kernels::xfield_sub_impl(a.c0, a.c1, a.c2, b.c0, b.c1, b.c2, r0, r1, r2);
    return Xfe(r0, r1, r2);
}
__device__ __forceinline__ Xfe operator*(Xfe a, Xfe b) {
    uint64_t r0, r1, r2;
    kernels::xfield_mul_impl(a.c0, a.c1, a.c2, b.c0, b.c1, b.c2, r0, r1, r2);
    return Xfe(r0, r1, r2);
}
__device__ __forceinline__ Xfe operator*(Xfe a, Bfe s) {
    uint64_t r0, r1, r2;
    kernels::xfield_scalar_mul_impl(a.c0, a.c1, a.c2, s.v, r0, r1, r2);
    return Xfe(r0, r1, r2);
}
__device__ __forceinline__ Xfe operator*(Bfe s, Xfe a) { return a * s; }
__device__ __forceinline__ Xfe operator+(Xfe a, Bfe b) { return a + Xfe(b); }
__device__ __forceinline__ Xfe operator+(Bfe a, Xfe b) { return Xfe(a) + b; }
__device__ __forceinline__ Xfe operator-(Xfe a, Bfe b) { return a - Xfe(b); }
__device__ __forceinline__ Xfe operator-(Bfe a, Xfe b) { return Xfe(a) - b; }

} // namespace triton_vm::gpu

#include "gpu/kernels/quotient_transition_part0_generated.cuh"

namespace triton_vm::gpu::kernels {

__global__ void quotient_transition_part0_kernel(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    size_t unit_distance,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    uint64_t* d_out  // accumulates into existing values
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= quotient_len) return;

    size_t next_row = (row + unit_distance) % quotient_len;

    const triton_vm::gpu::Bfe* current_main_row =
        reinterpret_cast<const triton_vm::gpu::Bfe*>(d_main_lde + row * main_width);
    const triton_vm::gpu::Bfe* next_main_row =
        reinterpret_cast<const triton_vm::gpu::Bfe*>(d_main_lde + next_row * main_width);
    const triton_vm::gpu::Xfe* current_aux_row =
        reinterpret_cast<const triton_vm::gpu::Xfe*>(d_aux_lde + (row * aux_width * 3));
    const triton_vm::gpu::Xfe* next_aux_row =
        reinterpret_cast<const triton_vm::gpu::Xfe*>(d_aux_lde + (next_row * aux_width * 3));
    const triton_vm::gpu::Xfe* challenges = reinterpret_cast<const triton_vm::gpu::Xfe*>(d_challenges);
    const triton_vm::gpu::Xfe* weights = reinterpret_cast<const triton_vm::gpu::Xfe*>(d_weights);

    triton_vm::gpu::Xfe acc = triton_vm::gpu::quotient_gen::eval_transition_part0_weighted(
        current_main_row, current_aux_row, next_main_row, next_aux_row, challenges, weights
    );

    // Write to output
    size_t out_off = row * 3;
    d_out[out_off + 0] = acc.c0;
    d_out[out_off + 1] = acc.c1;
    d_out[out_off + 2] = acc.c2;
}

void launch_quotient_transition_part0(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    size_t unit_distance,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    uint64_t* d_out,
    cudaStream_t stream
) {
    if (quotient_len == 0) return;
    int block = 256;  // Increased from 128 for better occupancy
    int grid = static_cast<int>((quotient_len + block - 1) / block);
    quotient_transition_part0_kernel<<<grid, block, 0, stream>>>(
        d_main_lde, main_width, d_aux_lde, aux_width, quotient_len, unit_distance,
        d_challenges, d_weights, d_out
    );
}

} // namespace triton_vm::gpu::kernels

#endif // TRITON_CUDA_ENABLED

