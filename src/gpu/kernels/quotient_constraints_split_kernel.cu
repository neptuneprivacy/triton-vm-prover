/**
 * Split GPU Quotient Constraint Evaluation Kernels
 *
 * Uses 4 separate kernels (one per constraint type) to avoid nvcc compilation
 * issues with the single 6200-line kernel.
 *
 * Each kernel evaluates one constraint type and accumulates into a per-row output.
 * The final combine kernel sums: q[row] = init*init_inv + cons*cons_inv + tran*tran_inv + term*term_inv
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/quotient_constraints_kernel.cuh"
#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/kernels/xfield_kernel.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm::gpu {

// Rust BFieldElement::from_raw_u64() decoding: raw * INV_R mod p.
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

// Include the split constraint evaluators
#include "gpu/kernels/quotient_initial_generated.cuh"
#include "gpu/kernels/quotient_consistency_generated.cuh"
// Note: transition is still too big, included separately
#include "gpu/kernels/quotient_terminal_generated.cuh"

namespace triton_vm::gpu::kernels {

//-----------------------------------------------------------------------------
// Kernel 1: Initial constraints
//-----------------------------------------------------------------------------
__global__ void quotient_initial_kernel(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    const uint64_t* d_init_inv_bfe,
    uint64_t* d_out  // quotient_len*3
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= quotient_len) return;

    const triton_vm::gpu::Bfe* main_row =
        reinterpret_cast<const triton_vm::gpu::Bfe*>(d_main_lde + row * main_width);
    const triton_vm::gpu::Xfe* aux_row =
        reinterpret_cast<const triton_vm::gpu::Xfe*>(d_aux_lde + (row * aux_width * 3));
    const triton_vm::gpu::Xfe* challenges = reinterpret_cast<const triton_vm::gpu::Xfe*>(d_challenges);
    const triton_vm::gpu::Xfe* weights = reinterpret_cast<const triton_vm::gpu::Xfe*>(d_weights);

    triton_vm::gpu::Xfe acc = triton_vm::gpu::quotient_gen::eval_initial_weighted(
        main_row, aux_row, challenges, weights
    );
    
    triton_vm::gpu::Bfe init_inv(d_init_inv_bfe[row]);
    triton_vm::gpu::Xfe result = acc * init_inv;

    size_t out_off = row * 3;
    d_out[out_off + 0] = result.c0;
    d_out[out_off + 1] = result.c1;
    d_out[out_off + 2] = result.c2;
}

//-----------------------------------------------------------------------------
// Kernel 2: Consistency constraints
//-----------------------------------------------------------------------------
__global__ void quotient_consistency_kernel(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    const uint64_t* d_cons_inv_bfe,
    uint64_t* d_out
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= quotient_len) return;

    const triton_vm::gpu::Bfe* main_row =
        reinterpret_cast<const triton_vm::gpu::Bfe*>(d_main_lde + row * main_width);
    const triton_vm::gpu::Xfe* aux_row =
        reinterpret_cast<const triton_vm::gpu::Xfe*>(d_aux_lde + (row * aux_width * 3));
    const triton_vm::gpu::Xfe* challenges = reinterpret_cast<const triton_vm::gpu::Xfe*>(d_challenges);
    const triton_vm::gpu::Xfe* weights = reinterpret_cast<const triton_vm::gpu::Xfe*>(d_weights);

    triton_vm::gpu::Xfe acc = triton_vm::gpu::quotient_gen::eval_consistency_weighted(
        main_row, aux_row, challenges, weights
    );
    
    triton_vm::gpu::Bfe cons_inv(d_cons_inv_bfe[row]);
    triton_vm::gpu::Xfe result = acc * cons_inv;

    size_t out_off = row * 3;
    d_out[out_off + 0] = result.c0;
    d_out[out_off + 1] = result.c1;
    d_out[out_off + 2] = result.c2;
}

//-----------------------------------------------------------------------------
// Kernel 3: Terminal constraints
//-----------------------------------------------------------------------------
__global__ void quotient_terminal_kernel(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    const uint64_t* d_term_inv_bfe,
    uint64_t* d_out
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= quotient_len) return;

    const triton_vm::gpu::Bfe* main_row =
        reinterpret_cast<const triton_vm::gpu::Bfe*>(d_main_lde + row * main_width);
    const triton_vm::gpu::Xfe* aux_row =
        reinterpret_cast<const triton_vm::gpu::Xfe*>(d_aux_lde + (row * aux_width * 3));
    const triton_vm::gpu::Xfe* challenges = reinterpret_cast<const triton_vm::gpu::Xfe*>(d_challenges);
    const triton_vm::gpu::Xfe* weights = reinterpret_cast<const triton_vm::gpu::Xfe*>(d_weights);

    triton_vm::gpu::Xfe acc = triton_vm::gpu::quotient_gen::eval_terminal_weighted(
        main_row, aux_row, challenges, weights
    );
    
    triton_vm::gpu::Bfe term_inv(d_term_inv_bfe[row]);
    triton_vm::gpu::Xfe result = acc * term_inv;

    size_t out_off = row * 3;
    d_out[out_off + 0] = result.c0;
    d_out[out_off + 1] = result.c1;
    d_out[out_off + 2] = result.c2;
}

//-----------------------------------------------------------------------------
// Kernel 4: Combine results: out = init + cons + tran + term
//-----------------------------------------------------------------------------
__global__ void quotient_combine_kernel(
    const uint64_t* d_init,
    const uint64_t* d_cons,
    const uint64_t* d_tran,
    const uint64_t* d_term,
    size_t quotient_len,
    uint64_t* d_out
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= quotient_len) return;

    size_t off = row * 3;
    
    triton_vm::gpu::Xfe init(d_init[off], d_init[off+1], d_init[off+2]);
    triton_vm::gpu::Xfe cons(d_cons[off], d_cons[off+1], d_cons[off+2]);
    triton_vm::gpu::Xfe tran(d_tran[off], d_tran[off+1], d_tran[off+2]);
    triton_vm::gpu::Xfe term(d_term[off], d_term[off+1], d_term[off+2]);

    triton_vm::gpu::Xfe result = init + cons + tran + term;

    d_out[off + 0] = result.c0;
    d_out[off + 1] = result.c1;
    d_out[off + 2] = result.c2;
}

//-----------------------------------------------------------------------------
// Host launcher for split kernels (without transition - that's in a separate file)
//-----------------------------------------------------------------------------
void compute_quotient_split_partial(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    const uint64_t* d_init_inv_bfe,
    const uint64_t* d_cons_inv_bfe,
    const uint64_t* d_term_inv_bfe,
    uint64_t* d_out_init,
    uint64_t* d_out_cons,
    uint64_t* d_out_term,
    cudaStream_t stream
) {
    if (quotient_len == 0) return;
    int block = 256;  // Increased from 128 for better occupancy
    int grid = static_cast<int>((quotient_len + block - 1) / block);
    
    // Launch all 3 kernels - they can overlap on the GPU
    quotient_initial_kernel<<<grid, block, 0, stream>>>(
        d_main_lde, main_width, d_aux_lde, aux_width, quotient_len,
        d_challenges, d_weights, d_init_inv_bfe, d_out_init
    );
    
    quotient_consistency_kernel<<<grid, block, 0, stream>>>(
        d_main_lde, main_width, d_aux_lde, aux_width, quotient_len,
        d_challenges, d_weights, d_cons_inv_bfe, d_out_cons
    );
    
    quotient_terminal_kernel<<<grid, block, 0, stream>>>(
        d_main_lde, main_width, d_aux_lde, aux_width, quotient_len,
        d_challenges, d_weights, d_term_inv_bfe, d_out_term
    );
}

void combine_quotient_results(
    const uint64_t* d_init,
    const uint64_t* d_cons,
    const uint64_t* d_tran,
    const uint64_t* d_term,
    size_t quotient_len,
    uint64_t* d_out,
    cudaStream_t stream
) {
    if (quotient_len == 0) return;
    int block = 256;  // Increased from 128 for better occupancy
    int grid = static_cast<int>((quotient_len + block - 1) / block);
    quotient_combine_kernel<<<grid, block, 0, stream>>>(
        d_init, d_cons, d_tran, d_term, quotient_len, d_out
    );
}

//-----------------------------------------------------------------------------
// Helper: Add XFieldElement arrays in-place
//-----------------------------------------------------------------------------
__global__ void add_xfield_kernel(
    uint64_t* d_out,
    const uint64_t* d_add,
    size_t len
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= len) return;

    size_t off = row * 3;
    triton_vm::gpu::Xfe a(d_out[off], d_out[off+1], d_out[off+2]);
    triton_vm::gpu::Xfe b(d_add[off], d_add[off+1], d_add[off+2]);
    triton_vm::gpu::Xfe result = a + b;
    d_out[off + 0] = result.c0;
    d_out[off + 1] = result.c1;
    d_out[off + 2] = result.c2;
}

void add_xfield_arrays(
    uint64_t* d_out,
    const uint64_t* d_add,
    size_t len,
    cudaStream_t stream
) {
    if (len == 0) return;
    int block = 128;
    int grid = static_cast<int>((len + block - 1) / block);
    add_xfield_kernel<<<grid, block, 0, stream>>>(d_out, d_add, len);
}

//-----------------------------------------------------------------------------
// Helper: Scale XFieldElements by BFieldElement scalars
//-----------------------------------------------------------------------------
__global__ void scale_xfield_kernel(
    const uint64_t* d_in,
    const uint64_t* d_scalars_bfe,
    size_t len,
    uint64_t* d_out
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= len) return;

    size_t off = row * 3;
    triton_vm::gpu::Xfe x(d_in[off], d_in[off+1], d_in[off+2]);
    triton_vm::gpu::Bfe s(d_scalars_bfe[row]);
    triton_vm::gpu::Xfe result = x * s;
    d_out[off + 0] = result.c0;
    d_out[off + 1] = result.c1;
    d_out[off + 2] = result.c2;
}

void scale_xfield_by_bfield(
    const uint64_t* d_in,
    const uint64_t* d_scalars_bfe,
    size_t len,
    uint64_t* d_out,
    cudaStream_t stream
) {
    if (len == 0) return;
    int block = 128;
    int grid = static_cast<int>((len + block - 1) / block);
    scale_xfield_kernel<<<grid, block, 0, stream>>>(d_in, d_scalars_bfe, len, d_out);
}

//-----------------------------------------------------------------------------
// FUSED kernel: Sum 4 transition parts + scale by zerofier in ONE pass
// Eliminates: 1 memcpy + 3 add kernels + 1 scale kernel + 1 memcpy
//-----------------------------------------------------------------------------
__global__ void fused_sum4_scale_kernel(
    const uint64_t* d_part0,  // [len*3]
    const uint64_t* d_part1,  // [len*3]
    const uint64_t* d_part2,  // [len*3]
    const uint64_t* d_part3,  // [len*3]
    const uint64_t* d_scale,  // [len] - BFieldElement scalars
    size_t len,
    uint64_t* d_out           // [len*3]
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= len) return;
    
    size_t off = row * 3;
    
    // Load all 4 parts
    triton_vm::gpu::Xfe p0(d_part0[off], d_part0[off + 1], d_part0[off + 2]);
    triton_vm::gpu::Xfe p1(d_part1[off], d_part1[off + 1], d_part1[off + 2]);
    triton_vm::gpu::Xfe p2(d_part2[off], d_part2[off + 1], d_part2[off + 2]);
    triton_vm::gpu::Xfe p3(d_part3[off], d_part3[off + 1], d_part3[off + 2]);
    
    // Sum all 4
    triton_vm::gpu::Xfe sum = p0 + p1 + p2 + p3;
    
    // Scale by zerofier inverse
    triton_vm::gpu::Bfe scale(d_scale[row]);
    triton_vm::gpu::Xfe result = sum * scale;
    
    // Write output
    d_out[off + 0] = result.c0;
    d_out[off + 1] = result.c1;
    d_out[off + 2] = result.c2;
}

void fused_sum4_scale_transition(
    const uint64_t* d_part0,
    const uint64_t* d_part1,
    const uint64_t* d_part2,
    const uint64_t* d_part3,
    const uint64_t* d_scale,
    size_t len,
    uint64_t* d_out,
    cudaStream_t stream
) {
    if (len == 0) return;
    int block = 256;
    int grid = static_cast<int>((len + block - 1) / block);
    fused_sum4_scale_kernel<<<grid, block, 0, stream>>>(
        d_part0, d_part1, d_part2, d_part3, d_scale, len, d_out
    );
}

} // namespace triton_vm::gpu::kernels

#endif // TRITON_CUDA_ENABLED

