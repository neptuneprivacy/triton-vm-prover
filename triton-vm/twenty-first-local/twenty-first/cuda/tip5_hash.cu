// CUDA kernel for Tip5 batch hashing (Merkle tree leaf construction)
// This kernel hashes millions of variable-length rows in parallel for ZK proof generation
//
// Algorithm: Tip5 (https://eprint.iacr.org/2023/107.pdf)
// - Poseidon-based sponge with RATE=10, CAPACITY=6, STATE_SIZE=16
// - 5 rounds with S-box (lookup + power-7) + MDS matrix + round constants
// - Field: BFieldElement (P = 2^64 - 2^32 + 1) in Montgomery form

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "field_arithmetic.cuh"


// Tip5 constants
#define STATE_SIZE 16
#define RATE 10
#define CAPACITY 6
#define DIGEST_LEN 5
#define NUM_ROUNDS 5
#define NUM_SPLIT_AND_LOOKUP 4

//============================================================================
// Tip5 Lookup Table (256 bytes)
//============================================================================

__constant__ u8 LOOKUP_TABLE[256] = {
    0, 7, 26, 63, 124, 215, 85, 254, 214, 228, 45, 185, 140, 173, 33, 240, 29, 177, 176, 32, 8,
    110, 87, 202, 204, 99, 150, 106, 230, 14, 235, 128, 213, 239, 212, 138, 23, 130, 208, 6, 44,
    71, 93, 116, 146, 189, 251, 81, 199, 97, 38, 28, 73, 179, 95, 84, 152, 48, 35, 119, 49, 88,
    242, 3, 148, 169, 72, 120, 62, 161, 166, 83, 175, 191, 137, 19, 100, 129, 112, 55, 221, 102,
    218, 61, 151, 237, 68, 164, 17, 147, 46, 234, 203, 216, 22, 141, 65, 57, 123, 12, 244, 54, 219,
    231, 96, 77, 180, 154, 5, 253, 133, 165, 98, 195, 205, 134, 245, 30, 9, 188, 59, 142, 186, 197,
    181, 144, 92, 31, 224, 163, 111, 74, 58, 69, 113, 196, 67, 246, 225, 10, 121, 50, 60, 157, 90,
    122, 2, 250, 101, 75, 178, 159, 24, 36, 201, 11, 243, 132, 198, 190, 114, 233, 39, 52, 21, 209,
    108, 238, 91, 187, 18, 104, 194, 37, 153, 34, 200, 143, 126, 155, 236, 118, 64, 80, 172, 89,
    94, 193, 135, 183, 86, 107, 252, 13, 167, 206, 136, 220, 207, 103, 171, 160, 76, 182, 227, 217,
    158, 56, 174, 4, 66, 109, 139, 162, 184, 211, 249, 47, 125, 232, 117, 43, 16, 42, 127, 20, 241,
    25, 149, 105, 156, 51, 53, 168, 145, 247, 223, 79, 78, 226, 15, 222, 82, 115, 70, 210, 27, 41,
    1, 170, 40, 131, 192, 229, 248, 255
};

//============================================================================
// Tip5 Round Constants (80 elements = 5 rounds * 16 state elements)
//============================================================================

__constant__ u64 ROUND_CONSTANTS[NUM_ROUNDS * STATE_SIZE] = {
    13630775303355457758ULL, 16896927574093233874ULL, 10379449653650130495ULL, 1965408364413093495ULL,
    15232538947090185111ULL, 15892634398091747074ULL, 3989134140024871768ULL, 2851411912127730865ULL,
    8709136439293758776ULL, 3694858669662939734ULL, 12692440244315327141ULL, 10722316166358076749ULL,
    12745429320441639448ULL, 17932424223723990421ULL, 7558102534867937463ULL, 15551047435855531404ULL,

    17532528648579384106ULL, 5216785850422679555ULL, 15418071332095031847ULL, 11921929762955146258ULL,
    9738718993677019874ULL, 3464580399432997147ULL, 13408434769117164050ULL, 264428218649616431ULL,
    4436247869008081381ULL, 4063129435850804221ULL, 2865073155741120117ULL, 5749834437609765994ULL,
    6804196764189408435ULL, 17060469201292988508ULL, 9475383556737206708ULL, 12876344085611465020ULL,

    13835756199368269249ULL, 1648753455944344172ULL, 9836124473569258483ULL, 12867641597107932229ULL,
    11254152636692960595ULL, 16550832737139861108ULL, 11861573970480733262ULL, 1256660473588673495ULL,
    13879506000676455136ULL, 10564103842682358721ULL, 16142842524796397521ULL, 3287098591948630584ULL,
    685911471061284805ULL, 5285298776918878023ULL, 18310953571768047354ULL, 3142266350630002035ULL,

    549990724933663297ULL, 4901984846118077401ULL, 11458643033696775769ULL, 8706785264119212710ULL,
    12521758138015724072ULL, 11877914062416978196ULL, 11333318251134523752ULL, 3933899631278608623ULL,
    16635128972021157924ULL, 10291337173108950450ULL, 4142107155024199350ULL, 16973934533787743537ULL,
    11068111539125175221ULL, 17546769694830203606ULL, 5315217744825068993ULL, 4609594252909613081ULL,

    3350107164315270407ULL, 17715942834299349177ULL, 9600609149219873996ULL, 12894357635820003949ULL,
    4597649658040514631ULL, 7735563950920491847ULL, 1663379455870887181ULL, 13889298103638829706ULL,
    7375530351220884434ULL, 3502022433285269151ULL, 9231805330431056952ULL, 9252272755288523725ULL,
    10014268662326746219ULL, 15565031632950843234ULL, 1209725273521819323ULL, 6024642864597845108ULL
};

//============================================================================
// Tip5 S-box Layer
//============================================================================

/// Split-and-lookup S-box for first 4 state elements
/// Applies lookup table to each byte of the element
__device__ void split_and_lookup(u64* element) {
    // Get raw bytes (already in Montgomery form)
    u8 bytes[8];
    for (int i = 0; i < 8; i++) {
        bytes[i] = (u8)((*element >> (i * 8)) & 0xFF);
    }

    // Apply lookup table to each byte
    for (int i = 0; i < 8; i++) {
        bytes[i] = LOOKUP_TABLE[bytes[i]];
    }

    // Reconstruct u64 from bytes
    u64 result = 0;
    for (int i = 0; i < 8; i++) {
        result |= ((u64)bytes[i]) << (i * 8);
    }
    *element = result;
}

/// Power-7 S-box for remaining 12 state elements: x^7 = x * x^2 * x^4
__device__ __forceinline__ u64 power_7(u64 x) {
    u64 x2 = bfe_square(x);          // x^2
    u64 x4 = bfe_square(x2);         // x^4
    u64 x3 = bfe_mul(x, x2);         // x^3
    u64 x7 = bfe_mul(x3, x4);        // x^7
    return x7;
}

/// S-box layer: split-and-lookup for first 4, power-7 for rest
__device__ void sbox_layer(u64* state) {
    // First 4 elements: split-and-lookup
    for (int i = 0; i < NUM_SPLIT_AND_LOOKUP; i++) {
        split_and_lookup(&state[i]);
    }

    // Remaining 12 elements: x^7
    for (int i = NUM_SPLIT_AND_LOOKUP; i < STATE_SIZE; i++) {
        state[i] = power_7(state[i]);
    }
}

//============================================================================
// Tip5 MDS Layer (generated_function)
// This is a highly optimized circulant MDS matrix multiplication
// Based on the original Rust implementation's generated_function
//============================================================================

// Include auto-generated MDS function (300 lines of optimized matrix operations)
#include "tip5_generated.cu"

__device__ void mds_generated(u64* state) {
    u64 lo[STATE_SIZE], hi[STATE_SIZE];

    // Split into low and high 32-bit parts
    for (int i = 0; i < STATE_SIZE; i++) {
        hi[i] = state[i] >> 32;
        lo[i] = state[i] & 0xffffffff;
    }

    u64 lo_out[STATE_SIZE], hi_out[STATE_SIZE];
    generated_function(lo, lo_out);
    generated_function(hi, hi_out);

    // Recombine with reduction (from reference implementation)
    for (int r = 0; r < STATE_SIZE; r++) {
        u64 res;
        u64 lo_i = lo_out[r] >> 4;
        asm("{\n\t"
            "shl.b64            %0, %2, 28;\n\t"
            "add.cc.u64         %0, %1, %0;\n\t"
            "shr.b64            %2, %2, 36;\n\t"
            "addc.u64           %1, %2, 0;\n\t"
            "shl.b64            %2, %1, 32;\n\t"
            "sub.u64            %1, %2, %1;\n\t"
            "add.cc.u64         %0, %1, %0;\n\t"
            "addc.u64           %1, 0, 0;\n\t"
            "neg.s64            %1, %1;\n\t"
            "and.b64            %1, %1, 0xFFFFFFFF;\n\t"
            "add.u64            %0, %0, %1;\n\t"
            "}"
            : "=l"(res), "+l"(lo_i)
            : "l"(hi_out[r]));

        state[r] = res;
    }
}

//============================================================================
// Tip5 Permutation
//============================================================================

/// One round of Tip5 permutation
__device__ void tip5_round(u64* state, int round_index) {
    // 1. S-box layer
    sbox_layer(state);

    // 2. MDS layer
    mds_generated(state);

    // 3. Add round constants
    for (int i = 0; i < STATE_SIZE; i++) {
        u64 rc = ROUND_CONSTANTS[round_index * STATE_SIZE + i];
        state[i] = bfe_add(state[i], to_montgomery(rc));
    }
}

/// Full Tip5 permutation (5 rounds)
__device__ void tip5_permutation(u64* state) {
    for (int i = 0; i < NUM_ROUNDS; i++) {
        tip5_round(state, i);
    }
}

//============================================================================
// Variable-Length Hashing (Sponge Construction)
//============================================================================

/// Pad input and absorb all elements
/// Padding: append [1, 0, 0, ...] to fill last chunk
__device__ void pad_and_absorb_all(u64* state, const u64* input, u32 input_len) {
    u32 num_full_chunks = input_len / RATE;
    u32 remainder = input_len % RATE;

    u64 bfe_zero = to_montgomery(0);
    u64 bfe_one = to_montgomery(1);

    // Absorb full chunks
    for (u32 chunk_idx = 0; chunk_idx < num_full_chunks; chunk_idx++) {
        // Copy RATE elements to state[0..RATE]
        for (int i = 0; i < RATE; i++) {
            state[i] = input[chunk_idx * RATE + i];
        }

        // Permute
        tip5_permutation(state);
    }

    // Absorb last chunk with padding
    u64 last_chunk[RATE];
    for (int i = 0; i < RATE; i++) {
        last_chunk[i] = bfe_zero;
    }

    // Copy remainder
    for (u32 i = 0; i < remainder; i++) {
        last_chunk[i] = input[num_full_chunks * RATE + i];
    }

    // Padding: append 1
    last_chunk[remainder] = bfe_one;

    // Absorb last chunk
    for (int i = 0; i < RATE; i++) {
        state[i] = last_chunk[i];
    }
    tip5_permutation(state);
}

//============================================================================
// Batch Hashing Kernel
//============================================================================

/// Hash multiple variable-length rows in parallel
/// Each block processes one row
///
/// Grid configuration:
///   - gridDim.x = num_rows (number of rows to hash)
///   - blockDim.x = 256 (threads per block, used for cooperation if needed)
///
/// Parameters:
///   input_rows: Flattened row data [row0_elem0, row0_elem1, ..., row1_elem0, ...]
///               Elements are in MONTGOMERY FORM (already converted by Rust)
///   output_digests: Output digests [5 elements per row], in MONTGOMERY FORM
///   row_length: Number of elements per row (e.g., 379 for main table, 88 for aux)
///   num_rows: Total number of rows to hash
///   row_start: Starting row index (for dual-GPU parallel hashing)
extern "C" __global__ void tip5_hash_batch(
    const u64* input_rows,
    u64* output_digests,
    u32 row_length,
    u32 num_rows,
    u32 row_start
) {
    // Multiple threads per block, each thread processes a different row
    // This improves SM occupancy and reduces scheduling overhead
    u32 local_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row_idx >= num_rows) return;

    // Compute global row index (for reading from input table)
    u32 row_idx = row_start + local_row_idx;

    // Get pointers to this row's data
    // CRITICAL: Cast to u64 to avoid integer overflow for large tables
    // (row_idx * row_length can exceed 2^32 for 16M+ rows)
    const u64* row_data = input_rows + ((u64)row_idx * (u64)row_length);
    // Output digests are written contiguously (local indexing)
    u64* digest_out = output_digests + ((u64)local_row_idx * DIGEST_LEN);

    // Initialize Tip5 state for VariableLength domain
    // VariableLength: state = [0, 0, ..., 0] (capacity is all zeros)
    u64 state[STATE_SIZE];
    u64 bfe_zero = to_montgomery(0);
    for (int i = 0; i < STATE_SIZE; i++) {
        state[i] = bfe_zero;
    }

    // Pad and absorb all input
    pad_and_absorb_all(state, row_data, row_length);

    // Squeeze: extract first DIGEST_LEN elements
    for (int i = 0; i < DIGEST_LEN; i++) {
        digest_out[i] = state[i];
    }
}
