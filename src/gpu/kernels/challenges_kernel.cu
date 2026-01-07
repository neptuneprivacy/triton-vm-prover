/**
 * Challenges CUDA Kernel Implementation
 *
 * Computes derived challenges on GPU:
 *   - StandardInputTerminal
 *   - StandardOutputTerminal
 *   - LookupTablePublicTerminal
 *   - CompressedProgramDigest
 *
 * Derived formula uses EvalArg terminal:
 *   acc = 1
 *   for symbol in symbols: acc = challenge * acc + symbol
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/challenges_kernel.cuh"
#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/kernels/xfield_kernel.cuh"
#include <cuda_runtime.h>

namespace triton_vm::gpu::kernels {

namespace {
static constexpr size_t CH_COUNT = 63;
static constexpr size_t SAMPLE_COUNT = 59;
static constexpr size_t IDX_COMPRESS_PROG_DIGEST_IND = 0;
static constexpr size_t IDX_STD_INPUT_IND = 1;
static constexpr size_t IDX_STD_OUTPUT_IND = 2;
static constexpr size_t IDX_LOOKUP_PUBLIC_IND = 54;

static constexpr size_t IDX_STD_INPUT_TERM = 59;
static constexpr size_t IDX_STD_OUTPUT_TERM = 60;
static constexpr size_t IDX_LOOKUP_PUBLIC_TERM = 61;
static constexpr size_t IDX_COMPRESSED_PROG_DIGEST = 62;

struct Xfe {
    uint64_t c0, c1, c2;
    __device__ __forceinline__ Xfe() : c0(0), c1(0), c2(0) {}
    __device__ __forceinline__ Xfe(uint64_t a0, uint64_t a1, uint64_t a2) : c0(a0), c1(a1), c2(a2) {}
};

__device__ __forceinline__ Xfe xfe_from_bfe(uint64_t b) { return Xfe(b, 0, 0); }

__device__ __forceinline__ Xfe xfe_add(Xfe a, Xfe b) {
    uint64_t r0, r1, r2;
    triton_vm::gpu::kernels::xfield_add_impl(a.c0, a.c1, a.c2, b.c0, b.c1, b.c2, r0, r1, r2);
    return Xfe(r0, r1, r2);
}

__device__ __forceinline__ Xfe xfe_mul(Xfe a, Xfe b) {
    uint64_t r0, r1, r2;
    triton_vm::gpu::kernels::xfield_mul_impl(a.c0, a.c1, a.c2, b.c0, b.c1, b.c2, r0, r1, r2);
    return Xfe(r0, r1, r2);
}

__device__ __forceinline__ Xfe evalarg_terminal_bfe(
    const uint64_t* symbols_bfe,
    size_t len,
    Xfe challenge
) {
    Xfe acc = xfe_from_bfe(1);
    for (size_t i = 0; i < len; ++i) {
        acc = xfe_add(xfe_mul(challenge, acc), xfe_from_bfe(symbols_bfe[i]));
    }
    return acc;
}

} // namespace

__global__ void compute_derived_challenges_kernel(
    uint64_t* d_challenges_xfe,           // [63*3]
    const uint64_t* d_program_digest_bfe, // [5]
    const uint64_t* d_input_bfe,          // [input_len]
    size_t input_len,
    const uint64_t* d_output_bfe,         // [output_len]
    size_t output_len,
    const uint64_t* d_lookup_table_bfe    // [256]
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;

    // Read challenge indeterminates
    auto read_ch = [&](size_t idx) -> Xfe {
        size_t off = idx * 3;
        return Xfe(d_challenges_xfe[off + 0], d_challenges_xfe[off + 1], d_challenges_xfe[off + 2]);
    };
    Xfe ind_prog = read_ch(IDX_COMPRESS_PROG_DIGEST_IND);
    Xfe ind_in   = read_ch(IDX_STD_INPUT_IND);
    Xfe ind_out  = read_ch(IDX_STD_OUTPUT_IND);
    Xfe ind_lut  = read_ch(IDX_LOOKUP_PUBLIC_IND);

    // Compute terminals
    Xfe compressed_digest = evalarg_terminal_bfe(d_program_digest_bfe, 5, ind_prog);
    Xfe input_term = evalarg_terminal_bfe(d_input_bfe, input_len, ind_in);
    Xfe output_term = evalarg_terminal_bfe(d_output_bfe, output_len, ind_out);
    Xfe lookup_term = evalarg_terminal_bfe(d_lookup_table_bfe, 256, ind_lut);

    auto write_ch = [&](size_t idx, Xfe v) {
        size_t off = idx * 3;
        d_challenges_xfe[off + 0] = v.c0;
        d_challenges_xfe[off + 1] = v.c1;
        d_challenges_xfe[off + 2] = v.c2;
    };
    write_ch(IDX_STD_INPUT_TERM, input_term);
    write_ch(IDX_STD_OUTPUT_TERM, output_term);
    write_ch(IDX_LOOKUP_PUBLIC_TERM, lookup_term);
    write_ch(IDX_COMPRESSED_PROG_DIGEST, compressed_digest);
}

void compute_derived_challenges_gpu(
    uint64_t* d_challenges_xfe,
    const uint64_t* d_program_digest_bfe,
    const uint64_t* d_input_bfe,
    size_t input_len,
    const uint64_t* d_output_bfe,
    size_t output_len,
    const uint64_t* d_lookup_table_bfe,
    cudaStream_t stream
) {
    compute_derived_challenges_kernel<<<1, 1, 0, stream>>>(
        d_challenges_xfe,
        d_program_digest_bfe,
        d_input_bfe,
        input_len,
        d_output_bfe,
        output_len,
        d_lookup_table_bfe
    );
}

} // namespace triton_vm::gpu::kernels

#endif // TRITON_CUDA_ENABLED


