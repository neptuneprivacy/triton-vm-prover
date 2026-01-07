/**
 * GPU Quotient Constraint Evaluation Kernel
 *
 * Evaluates all AIR constraints on the quotient domain on GPU and produces
 * the per-row combined quotient value:
 *   q[row] = init(row)*init_inv[row] + cons(row)*cons_inv[row] + tran(row)*tran_inv[row] + term(row)*term_inv[row]
 *
 * Uses generated weighted evaluators from `include/gpu/kernels/quotient_constraints_generated.cuh`.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/quotient_constraints_kernel.cuh"
#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/kernels/xfield_kernel.cuh"

#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm::gpu {

// Rust BFieldElement::from_raw_u64() decoding: raw * INV_R mod p.
// Observed: from_raw(1).value() == INV_R.
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

static __device__ __forceinline__ Xfe load_xfe_u64(const uint64_t* base, size_t idx_xfe) {
    size_t off = idx_xfe * 3;
    return Xfe(base[off + 0], base[off + 1], base[off + 2]);
}

} // namespace triton_vm::gpu

// Generated weighted AIR constraint evaluators (requires Bfe/Xfe above)
#include "gpu/kernels/quotient_constraints_generated.cuh"

namespace triton_vm::gpu::kernels {

__global__ void quotient_constraints_eval_kernel(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    size_t unit_distance,
    const uint64_t* d_challenges, // 63*3
    const uint64_t* d_weights,    // 596*3
    const uint64_t* d_init_inv_bfe, // quotient_len
    const uint64_t* d_cons_inv_bfe,
    const uint64_t* d_tran_inv_bfe,
    const uint64_t* d_term_inv_bfe,
    uint64_t* d_out              // quotient_len*3
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= quotient_len) return;

    size_t next_row = (row + unit_distance) % quotient_len;

    // pointers to main rows (Bfe array)
    // Main rows are stored as canonical BFEs (u64). Wrap as Bfe on the fly.
    // We reinterpret to Bfe since Bfe is a trivial wrapper around one u64.
    const triton_vm::gpu::Bfe* current_main_row =
        reinterpret_cast<const triton_vm::gpu::Bfe*>(d_main_lde + row * main_width);
    const triton_vm::gpu::Bfe* next_main_row =
        reinterpret_cast<const triton_vm::gpu::Bfe*>(d_main_lde + next_row * main_width);

    // pointers to aux rows (Xfe array)
    const triton_vm::gpu::Xfe* current_aux_row =
        reinterpret_cast<const triton_vm::gpu::Xfe*>(d_aux_lde + (row * aux_width * 3));
    const triton_vm::gpu::Xfe* next_aux_row =
        reinterpret_cast<const triton_vm::gpu::Xfe*>(d_aux_lde + (next_row * aux_width * 3));

    const triton_vm::gpu::Xfe* challenges = reinterpret_cast<const triton_vm::gpu::Xfe*>(d_challenges);
    const triton_vm::gpu::Xfe* weights = reinterpret_cast<const triton_vm::gpu::Xfe*>(d_weights);

    triton_vm::gpu::Xfe init_acc = triton_vm::gpu::quotient_gen::eval_initial_weighted(
        current_main_row, current_aux_row, challenges, weights
    );
    triton_vm::gpu::Xfe cons_acc = triton_vm::gpu::quotient_gen::eval_consistency_weighted(
        current_main_row, current_aux_row, challenges, weights
    );
    triton_vm::gpu::Xfe tran_acc = triton_vm::gpu::quotient_gen::eval_transition_weighted(
        current_main_row, current_aux_row, next_main_row, next_aux_row, challenges, weights
    );
    triton_vm::gpu::Xfe term_acc = triton_vm::gpu::quotient_gen::eval_terminal_weighted(
        current_main_row, current_aux_row, challenges, weights
    );

    triton_vm::gpu::Bfe init_inv(d_init_inv_bfe[row]);
    triton_vm::gpu::Bfe cons_inv(d_cons_inv_bfe[row]);
    triton_vm::gpu::Bfe tran_inv(d_tran_inv_bfe[row]);
    triton_vm::gpu::Bfe term_inv(d_term_inv_bfe[row]);

    triton_vm::gpu::Xfe q = init_acc * init_inv;
    q = q + (cons_acc * cons_inv);
    q = q + (tran_acc * tran_inv);
    q = q + (term_acc * term_inv);

    size_t out_off = row * 3;
    d_out[out_off + 0] = q.c0;
    d_out[out_off + 1] = q.c1;
    d_out[out_off + 2] = q.c2;
}

void compute_quotient_values_gpu(
    const uint64_t* d_main_lde,
    size_t main_width,
    const uint64_t* d_aux_lde,
    size_t aux_width,
    size_t quotient_len,
    size_t unit_distance,
    const uint64_t* d_challenges,
    const uint64_t* d_weights,
    const uint64_t* d_init_inv_bfe,
    const uint64_t* d_cons_inv_bfe,
    const uint64_t* d_tran_inv_bfe,
    const uint64_t* d_term_inv_bfe,
    uint64_t* d_out_quotient_values,
    cudaStream_t stream
) {
    if (quotient_len == 0) return;
    int block = 128;
    int grid = static_cast<int>((quotient_len + block - 1) / block);
    quotient_constraints_eval_kernel<<<grid, block, 0, stream>>>(
        d_main_lde, main_width,
        d_aux_lde, aux_width,
        quotient_len, unit_distance,
        d_challenges, d_weights,
        d_init_inv_bfe, d_cons_inv_bfe, d_tran_inv_bfe, d_term_inv_bfe,
        d_out_quotient_values
    );
}

} // namespace triton_vm::gpu::kernels

#endif // TRITON_CUDA_ENABLED


