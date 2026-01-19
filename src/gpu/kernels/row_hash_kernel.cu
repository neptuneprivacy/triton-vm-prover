/**
 * Row Hashing CUDA Kernel Implementation
 * 
 * Hashes table rows to digests using Tip5 sponge.
 * Each row is padded and absorbed into a separate sponge, then squeezed.
 * 
 * Uses the inline tip5_permutation device function from tip5_kernel.cu
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bfield_kernel.cuh"
#include <cuda_runtime.h>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

// ============================================================================
// Tip5 Constants (must match tip5_kernel.cu)
// ============================================================================

static constexpr int TIP5_STATE_SIZE = 16;
static constexpr int TIP5_RATE = 10;
static constexpr int TIP5_CAPACITY = 6;
static constexpr int TIP5_NUM_ROUNDS = 5;
static constexpr int TIP5_NUM_SPLIT_AND_LOOKUP = 4;
static constexpr int DIGEST_LEN = 5;

// R^2 mod P for Montgomery conversion
static constexpr uint64_t R2 = 0xFFFFFFFE00000001ULL;

// Lookup table for S-box (same as in tip5_kernel.cu)
__constant__ uint8_t ROW_HASH_LOOKUP_TABLE[256] = {
      0,   7,  26,  63, 124, 215,  85, 254, 214, 228,  45, 185, 140, 173,  33, 240,
     29, 177, 176,  32,   8, 110,  87, 202, 204,  99, 150, 106, 230,  14, 235, 128,
    213, 239, 212, 138,  23, 130, 208,   6,  44,  71,  93, 116, 146, 189, 251,  81,
    199,  97,  38,  28,  73, 179,  95,  84, 152,  48,  35, 119,  49,  88, 242,   3,
    148, 169,  72, 120,  62, 161, 166,  83, 175, 191, 137,  19, 100, 129, 112,  55,
    221, 102, 218,  61, 151, 237,  68, 164,  17, 147,  46, 234, 203, 216,  22, 141,
     65,  57, 123,  12, 244,  54, 219, 231,  96,  77, 180, 154,   5, 253, 133, 165,
     98, 195, 205, 134, 245,  30,   9, 188,  59, 142, 186, 197, 181, 144,  92,  31,
    224, 163, 111,  74,  58,  69, 113, 196,  67, 246, 225,  10, 121,  50,  60, 157,
     90, 122,   2, 250, 101,  75, 178, 159,  24,  36, 201,  11, 243, 132, 198, 190,
    114, 233,  39,  52,  21, 209, 108, 238,  91, 187,  18, 104, 194,  37, 153,  34,
    200, 143, 126, 155, 236, 118,  64,  80, 172,  89,  94, 193, 135, 183,  86, 107,
    252,  13, 167, 206, 136, 220, 207, 103, 171, 160,  76, 182, 227, 217, 158,  56,
    174,   4,  66, 109, 139, 162, 184, 211, 249,  47, 125, 232, 117,  43,  16,  42,
    127,  20, 241,  25, 149, 105, 156,  51,  53, 168, 145, 247, 223,  79,  78, 226,
     15, 222,  82, 115,  70, 210,  27,  41,   1, 170,  40, 131, 192, 229, 248, 255
};

// MDS matrix first column
__constant__ uint64_t ROW_HASH_MDS[16] = {
    61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034,
    56951, 27521, 41351, 40901, 12021, 59689, 26798, 17845
};

// Round constants
__constant__ uint64_t ROW_HASH_RC[80] = {
    0xBD2A3DEB61AB60DEULL, 0xEA7DF21AD9547ED2ULL, 0x900B3677A1DE063FULL, 0x1B46887E876C8677ULL,
    0xD364D977889CFB97ULL, 0xDC8DFAC843699F02ULL, 0x375C405D7190DB58ULL, 0x27924006D2B0D4B1ULL,
    0x78DD1172D483CD38ULL, 0x3346C66244882A56ULL, 0xB0249B279F498AA5ULL, 0x94CD51BE79338D4DULL,
    0xB0E0DC7052C5B218ULL, 0xF8DCC4D248ADAD95ULL, 0x68E3C635FEC868B7ULL, 0xD7D06B3FFB6B0D8CULL,
    0xF3500DEA20EF032AULL, 0x4865BF175BBA5803ULL, 0xD5F7FE3027287A27ULL, 0xA57333F44E193412ULL,
    0x8726E153A977EAE2ULL, 0x3014A98463FC191BULL, 0xBA145461AF39B212ULL, 0x03AB70105933202FULL,
    0x3D90B7EEBFCF71E5ULL, 0x386322B1CC520BFDULL, 0x27C2C8DAF774F675ULL, 0x4FCB83F50309BC6AULL,
    0x5E6D5CE8275F3CB3ULL, 0xECC2F6592C8F905CULL, 0x837F532461E609B4ULL, 0xB2B1F6B95C92C93CULL,
    0xC0027AF556411DC1ULL, 0x16E18C885FC2A26CULL, 0x8880EF183D9F2BF3ULL, 0xB2930BDB5CA88C45ULL,
    0x9C2EC8322E1C1553ULL, 0xE5B05EAF3220A674ULL, 0xA49CC6AE4B861C4EULL, 0x11708E0AEB86EBD7ULL,
    0xC09DE92BBC3902E0ULL, 0x929B3C79516BCBC1ULL, 0xE006E5BF738F27D1ULL, 0x2D9E1EC0EAC8EA38ULL,
    0x0984D8D94BF937C5ULL, 0x4959273C220E6747ULL, 0xFE1D934207E796FAULL, 0x2B9B9298F2F6DD73ULL,
    0x07A1F5A67D6E3A41ULL, 0x4407593EE73743D9ULL, 0x9F054720EF802E59ULL, 0x78D4B711336E6AA6ULL,
    0xADC638AEF3C8B228ULL, 0xA4D6D3E86AFB2114ULL, 0x9D4808E725531968ULL, 0x369804DF3866D0EFULL,
    0xE6DBD9A9D2215024ULL, 0x8ED22CA212EE85B2ULL, 0x397BB882FCD23EB6ULL, 0xEB8F8786D7277531ULL,
    0x9999D4CDAFF543B5ULL, 0xF382A61217F192D6ULL, 0x49C37260B026ADC1ULL, 0x3FF8918CE35C1019ULL,
    0x2E7DF8B76080BD07ULL, 0xF5DBAC250B8A28B9ULL, 0x853C3727AE9DA4CCULL, 0xB2F1F5F3D9E5A26DULL,
    0x3FCE22012D337847ULL, 0x6B5A3E6DB7EEE347ULL, 0x171582CD59DDE50DULL, 0xC0C0B3095EE62A8AULL,
    0x665B25C6F6A203D2ULL, 0x3099AED93B6AE69FULL, 0x801DF6092BE69C38ULL, 0x8066AD0CDFFF43CDULL,
    0x8AF9D44A5F4FDC6BULL, 0xD80219CD97C0D762ULL, 0x10C9CEBA14148EBBULL, 0x539BD4C3F2F24474ULL
};

// ============================================================================
// Inline Tip5 Functions (for use within row hashing kernel)
// ============================================================================

__device__ __forceinline__ uint64_t rh_monty_reduce(uint64_t hi, uint64_t lo) {
    uint64_t shifted = lo << 32;
    uint64_t a = lo + shifted;
    bool overflow_a = (a < lo) || (a < shifted);
    
    uint64_t b = a - (a >> 32);
    if (overflow_a) b -= 1;
    
    uint64_t r = hi - b;
    bool underflow = (hi < b);
    
    if (underflow) {
        r -= (1ULL + ~GOLDILOCKS_P);
    }
    
    return r;
}

__device__ __forceinline__ uint64_t rh_split_and_lookup(uint64_t val) {
    // Convert to Montgomery form
    uint64_t lo, hi;
    asm("mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;"
        : "=l"(lo), "=l"(hi)
        : "l"(val), "l"(R2));
    uint64_t montgomery_val = rh_monty_reduce(hi, lo);
    
    // Apply lookup on each byte
    uint64_t sbox_out = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t byte = static_cast<uint8_t>((montgomery_val >> (i * 8)) & 0xFF);
        uint8_t looked_up = ROW_HASH_LOOKUP_TABLE[byte];
        sbox_out |= static_cast<uint64_t>(looked_up) << (i * 8);
    }
    
    return rh_monty_reduce(0, sbox_out);
}

__device__ __forceinline__ uint64_t rh_power7_sbox(uint64_t x) {
    uint64_t x2 = bfield_mul_impl(x, x);
    uint64_t x4 = bfield_mul_impl(x2, x2);
    return bfield_mul_impl(bfield_mul_impl(x, x2), x4);
}

// Optimized MDS layer - preload state into registers to avoid array access overhead
// The MDS matrix is circulant, enabling efficient computation with register reuse
__device__ __forceinline__ void rh_mds_layer(uint64_t state[TIP5_STATE_SIZE]) {
    // Load state into registers (avoid repeated array indexing in inner loop)
    const uint64_t s0 = state[0], s1 = state[1], s2 = state[2], s3 = state[3];
    const uint64_t s4 = state[4], s5 = state[5], s6 = state[6], s7 = state[7];
    const uint64_t s8 = state[8], s9 = state[9], s10 = state[10], s11 = state[11];
    const uint64_t s12 = state[12], s13 = state[13], s14 = state[14], s15 = state[15];
    
    // Load MDS coefficients (from constant memory, will be cached)
    const uint64_t m0 = ROW_HASH_MDS[0], m1 = ROW_HASH_MDS[1], m2 = ROW_HASH_MDS[2], m3 = ROW_HASH_MDS[3];
    const uint64_t m4 = ROW_HASH_MDS[4], m5 = ROW_HASH_MDS[5], m6 = ROW_HASH_MDS[6], m7 = ROW_HASH_MDS[7];
    const uint64_t m8 = ROW_HASH_MDS[8], m9 = ROW_HASH_MDS[9], m10 = ROW_HASH_MDS[10], m11 = ROW_HASH_MDS[11];
    const uint64_t m12 = ROW_HASH_MDS[12], m13 = ROW_HASH_MDS[13], m14 = ROW_HASH_MDS[14], m15 = ROW_HASH_MDS[15];
    
    // Compute MDS matrix-vector product: new_state[r] = sum(MDS[(r-c)&15] * state[c])
    // Each row uses a rotated version of the MDS coefficients
    
    // Row 0: m0*s0 + m15*s1 + m14*s2 + ... + m1*s15
    state[0] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(
                   bfield_mul_impl(m0, s0), bfield_mul_impl(m15, s1)), bfield_mul_impl(m14, s2)), bfield_mul_impl(m13, s3)),
                   bfield_mul_impl(m12, s4)), bfield_mul_impl(m11, s5)), bfield_mul_impl(m10, s6)), bfield_mul_impl(m9, s7)),
                   bfield_mul_impl(m8, s8)), bfield_mul_impl(m7, s9)), bfield_mul_impl(m6, s10)), bfield_mul_impl(m5, s11)),
                   bfield_mul_impl(m4, s12)), bfield_mul_impl(m3, s13)), bfield_mul_impl(m2, s14)), bfield_mul_impl(m1, s15));
    
    // Row 1: m1*s0 + m0*s1 + m15*s2 + ...
    state[1] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(
                   bfield_mul_impl(m1, s0), bfield_mul_impl(m0, s1)), bfield_mul_impl(m15, s2)), bfield_mul_impl(m14, s3)),
                   bfield_mul_impl(m13, s4)), bfield_mul_impl(m12, s5)), bfield_mul_impl(m11, s6)), bfield_mul_impl(m10, s7)),
                   bfield_mul_impl(m9, s8)), bfield_mul_impl(m8, s9)), bfield_mul_impl(m7, s10)), bfield_mul_impl(m6, s11)),
                   bfield_mul_impl(m5, s12)), bfield_mul_impl(m4, s13)), bfield_mul_impl(m3, s14)), bfield_mul_impl(m2, s15));
    
    // Row 2
    state[2] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(
                   bfield_mul_impl(m2, s0), bfield_mul_impl(m1, s1)), bfield_mul_impl(m0, s2)), bfield_mul_impl(m15, s3)),
                   bfield_mul_impl(m14, s4)), bfield_mul_impl(m13, s5)), bfield_mul_impl(m12, s6)), bfield_mul_impl(m11, s7)),
                   bfield_mul_impl(m10, s8)), bfield_mul_impl(m9, s9)), bfield_mul_impl(m8, s10)), bfield_mul_impl(m7, s11)),
                   bfield_mul_impl(m6, s12)), bfield_mul_impl(m5, s13)), bfield_mul_impl(m4, s14)), bfield_mul_impl(m3, s15));
    
    // Row 3
    state[3] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(
                   bfield_mul_impl(m3, s0), bfield_mul_impl(m2, s1)), bfield_mul_impl(m1, s2)), bfield_mul_impl(m0, s3)),
                   bfield_mul_impl(m15, s4)), bfield_mul_impl(m14, s5)), bfield_mul_impl(m13, s6)), bfield_mul_impl(m12, s7)),
                   bfield_mul_impl(m11, s8)), bfield_mul_impl(m10, s9)), bfield_mul_impl(m9, s10)), bfield_mul_impl(m8, s11)),
                   bfield_mul_impl(m7, s12)), bfield_mul_impl(m6, s13)), bfield_mul_impl(m5, s14)), bfield_mul_impl(m4, s15));
    
    // Row 4
    state[4] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(
                   bfield_mul_impl(m4, s0), bfield_mul_impl(m3, s1)), bfield_mul_impl(m2, s2)), bfield_mul_impl(m1, s3)),
                   bfield_mul_impl(m0, s4)), bfield_mul_impl(m15, s5)), bfield_mul_impl(m14, s6)), bfield_mul_impl(m13, s7)),
                   bfield_mul_impl(m12, s8)), bfield_mul_impl(m11, s9)), bfield_mul_impl(m10, s10)), bfield_mul_impl(m9, s11)),
                   bfield_mul_impl(m8, s12)), bfield_mul_impl(m7, s13)), bfield_mul_impl(m6, s14)), bfield_mul_impl(m5, s15));
    
    // Row 5
    state[5] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(
                   bfield_mul_impl(m5, s0), bfield_mul_impl(m4, s1)), bfield_mul_impl(m3, s2)), bfield_mul_impl(m2, s3)),
                   bfield_mul_impl(m1, s4)), bfield_mul_impl(m0, s5)), bfield_mul_impl(m15, s6)), bfield_mul_impl(m14, s7)),
                   bfield_mul_impl(m13, s8)), bfield_mul_impl(m12, s9)), bfield_mul_impl(m11, s10)), bfield_mul_impl(m10, s11)),
                   bfield_mul_impl(m9, s12)), bfield_mul_impl(m8, s13)), bfield_mul_impl(m7, s14)), bfield_mul_impl(m6, s15));
    
    // Row 6
    state[6] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(
                   bfield_mul_impl(m6, s0), bfield_mul_impl(m5, s1)), bfield_mul_impl(m4, s2)), bfield_mul_impl(m3, s3)),
                   bfield_mul_impl(m2, s4)), bfield_mul_impl(m1, s5)), bfield_mul_impl(m0, s6)), bfield_mul_impl(m15, s7)),
                   bfield_mul_impl(m14, s8)), bfield_mul_impl(m13, s9)), bfield_mul_impl(m12, s10)), bfield_mul_impl(m11, s11)),
                   bfield_mul_impl(m10, s12)), bfield_mul_impl(m9, s13)), bfield_mul_impl(m8, s14)), bfield_mul_impl(m7, s15));
    
    // Row 7
    state[7] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(
                   bfield_mul_impl(m7, s0), bfield_mul_impl(m6, s1)), bfield_mul_impl(m5, s2)), bfield_mul_impl(m4, s3)),
                   bfield_mul_impl(m3, s4)), bfield_mul_impl(m2, s5)), bfield_mul_impl(m1, s6)), bfield_mul_impl(m0, s7)),
                   bfield_mul_impl(m15, s8)), bfield_mul_impl(m14, s9)), bfield_mul_impl(m13, s10)), bfield_mul_impl(m12, s11)),
                   bfield_mul_impl(m11, s12)), bfield_mul_impl(m10, s13)), bfield_mul_impl(m9, s14)), bfield_mul_impl(m8, s15));
    
    // Row 8
    state[8] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(
                   bfield_mul_impl(m8, s0), bfield_mul_impl(m7, s1)), bfield_mul_impl(m6, s2)), bfield_mul_impl(m5, s3)),
                   bfield_mul_impl(m4, s4)), bfield_mul_impl(m3, s5)), bfield_mul_impl(m2, s6)), bfield_mul_impl(m1, s7)),
                   bfield_mul_impl(m0, s8)), bfield_mul_impl(m15, s9)), bfield_mul_impl(m14, s10)), bfield_mul_impl(m13, s11)),
                   bfield_mul_impl(m12, s12)), bfield_mul_impl(m11, s13)), bfield_mul_impl(m10, s14)), bfield_mul_impl(m9, s15));
    
    // Row 9
    state[9] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
               bfield_add_impl(bfield_add_impl(bfield_add_impl(
                   bfield_mul_impl(m9, s0), bfield_mul_impl(m8, s1)), bfield_mul_impl(m7, s2)), bfield_mul_impl(m6, s3)),
                   bfield_mul_impl(m5, s4)), bfield_mul_impl(m4, s5)), bfield_mul_impl(m3, s6)), bfield_mul_impl(m2, s7)),
                   bfield_mul_impl(m1, s8)), bfield_mul_impl(m0, s9)), bfield_mul_impl(m15, s10)), bfield_mul_impl(m14, s11)),
                   bfield_mul_impl(m13, s12)), bfield_mul_impl(m12, s13)), bfield_mul_impl(m11, s14)), bfield_mul_impl(m10, s15));
    
    // Row 10
    state[10] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(
                    bfield_mul_impl(m10, s0), bfield_mul_impl(m9, s1)), bfield_mul_impl(m8, s2)), bfield_mul_impl(m7, s3)),
                    bfield_mul_impl(m6, s4)), bfield_mul_impl(m5, s5)), bfield_mul_impl(m4, s6)), bfield_mul_impl(m3, s7)),
                    bfield_mul_impl(m2, s8)), bfield_mul_impl(m1, s9)), bfield_mul_impl(m0, s10)), bfield_mul_impl(m15, s11)),
                    bfield_mul_impl(m14, s12)), bfield_mul_impl(m13, s13)), bfield_mul_impl(m12, s14)), bfield_mul_impl(m11, s15));
    
    // Row 11
    state[11] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(
                    bfield_mul_impl(m11, s0), bfield_mul_impl(m10, s1)), bfield_mul_impl(m9, s2)), bfield_mul_impl(m8, s3)),
                    bfield_mul_impl(m7, s4)), bfield_mul_impl(m6, s5)), bfield_mul_impl(m5, s6)), bfield_mul_impl(m4, s7)),
                    bfield_mul_impl(m3, s8)), bfield_mul_impl(m2, s9)), bfield_mul_impl(m1, s10)), bfield_mul_impl(m0, s11)),
                    bfield_mul_impl(m15, s12)), bfield_mul_impl(m14, s13)), bfield_mul_impl(m13, s14)), bfield_mul_impl(m12, s15));
    
    // Row 12
    state[12] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(
                    bfield_mul_impl(m12, s0), bfield_mul_impl(m11, s1)), bfield_mul_impl(m10, s2)), bfield_mul_impl(m9, s3)),
                    bfield_mul_impl(m8, s4)), bfield_mul_impl(m7, s5)), bfield_mul_impl(m6, s6)), bfield_mul_impl(m5, s7)),
                    bfield_mul_impl(m4, s8)), bfield_mul_impl(m3, s9)), bfield_mul_impl(m2, s10)), bfield_mul_impl(m1, s11)),
                    bfield_mul_impl(m0, s12)), bfield_mul_impl(m15, s13)), bfield_mul_impl(m14, s14)), bfield_mul_impl(m13, s15));
    
    // Row 13
    state[13] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(
                    bfield_mul_impl(m13, s0), bfield_mul_impl(m12, s1)), bfield_mul_impl(m11, s2)), bfield_mul_impl(m10, s3)),
                    bfield_mul_impl(m9, s4)), bfield_mul_impl(m8, s5)), bfield_mul_impl(m7, s6)), bfield_mul_impl(m6, s7)),
                    bfield_mul_impl(m5, s8)), bfield_mul_impl(m4, s9)), bfield_mul_impl(m3, s10)), bfield_mul_impl(m2, s11)),
                    bfield_mul_impl(m1, s12)), bfield_mul_impl(m0, s13)), bfield_mul_impl(m15, s14)), bfield_mul_impl(m14, s15));
    
    // Row 14
    state[14] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(
                    bfield_mul_impl(m14, s0), bfield_mul_impl(m13, s1)), bfield_mul_impl(m12, s2)), bfield_mul_impl(m11, s3)),
                    bfield_mul_impl(m10, s4)), bfield_mul_impl(m9, s5)), bfield_mul_impl(m8, s6)), bfield_mul_impl(m7, s7)),
                    bfield_mul_impl(m6, s8)), bfield_mul_impl(m5, s9)), bfield_mul_impl(m4, s10)), bfield_mul_impl(m3, s11)),
                    bfield_mul_impl(m2, s12)), bfield_mul_impl(m1, s13)), bfield_mul_impl(m0, s14)), bfield_mul_impl(m15, s15));
    
    // Row 15
    state[15] = bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(bfield_add_impl(
                bfield_add_impl(bfield_add_impl(bfield_add_impl(
                    bfield_mul_impl(m15, s0), bfield_mul_impl(m14, s1)), bfield_mul_impl(m13, s2)), bfield_mul_impl(m12, s3)),
                    bfield_mul_impl(m11, s4)), bfield_mul_impl(m10, s5)), bfield_mul_impl(m9, s6)), bfield_mul_impl(m8, s7)),
                    bfield_mul_impl(m7, s8)), bfield_mul_impl(m6, s9)), bfield_mul_impl(m5, s10)), bfield_mul_impl(m4, s11)),
                    bfield_mul_impl(m3, s12)), bfield_mul_impl(m2, s13)), bfield_mul_impl(m1, s14)), bfield_mul_impl(m0, s15));
}

__device__ __forceinline__ void rh_tip5_permutation(uint64_t state[TIP5_STATE_SIZE]) {
    #pragma unroll
    for (int round = 0; round < TIP5_NUM_ROUNDS; ++round) {
        // S-box layer
        #pragma unroll
        for (int i = 0; i < TIP5_NUM_SPLIT_AND_LOOKUP; ++i) {
            state[i] = rh_split_and_lookup(state[i]);
        }
        #pragma unroll
        for (int i = TIP5_NUM_SPLIT_AND_LOOKUP; i < TIP5_STATE_SIZE; ++i) {
            state[i] = rh_power7_sbox(state[i]);
        }
        
        // MDS layer
        rh_mds_layer(state);
        
        // Add round constants
        int offset = round * TIP5_STATE_SIZE;
        #pragma unroll
        for (int i = 0; i < TIP5_STATE_SIZE; ++i) {
            state[i] = bfield_add_impl(state[i], ROW_HASH_RC[offset + i]);
        }
    }
}

// ============================================================================
// Row Hashing Kernels
// ============================================================================

/**
 * Hash BFieldElement rows to digests using Tip5 sponge
 * 
 * Uses OVERWRITE mode (matching CPU hash_varlen):
 * - Overwrite rate portion with input chunk
 * - Apply permutation
 * - Last chunk includes padding (1 after data, then zeros)
 * 
 * OPTIMIZED: Uses vectorized loads and prefetching to improve memory
 * performance with unified memory. Reduces page fault overhead by
 * loading data in larger chunks and using prefetch hints.
 * 
 * @param d_table Table data, column-major (num_cols * num_rows)
 * @param num_rows Number of rows
 * @param num_cols Number of columns per row
 * @param d_digests Output digests (num_rows * 5 elements)
 */
__global__ void hash_bfield_rows_kernel(
    const uint64_t* __restrict__ d_table,
    size_t num_rows,
    size_t num_cols,
    uint64_t* __restrict__ d_digests
) {
    size_t row_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= num_rows) return;
    
    // Variable-length domain: state starts as all zeros
    uint64_t state[TIP5_STATE_SIZE] = {0};
    
    size_t consumed = 0;
    
    // Process full chunks using OVERWRITE mode (matching CPU)
    // Optimized: Use cache-allocate hints for better memory performance
    // __ldca hints the driver to keep data in L2 cache
    while (consumed + TIP5_RATE <= num_cols) {
        // Load current chunk with cache-allocate hints
        #pragma unroll
        for (int i = 0; i < TIP5_RATE; ++i) {
            size_t elem_idx = consumed + i;
            size_t table_idx = elem_idx * num_rows + row_idx;
            // Use __ldca (cache allocate) to hint keeping data in L2 cache
            state[i] = __ldca(&d_table[table_idx]);
        }
        
        rh_tip5_permutation(state);
        consumed += TIP5_RATE;
    }
    
    // Handle last chunk with padding (matching CPU)
    #pragma unroll
    for (int i = 0; i < TIP5_RATE; ++i) {
        size_t elem_idx = consumed + i;
        if (elem_idx < num_cols) {
            size_t table_idx = elem_idx * num_rows + row_idx;
            state[i] = __ldca(&d_table[table_idx]);
        } else if (elem_idx == num_cols) {
            state[i] = 1;  // Padding indicator
        } else {
            state[i] = 0;  // Zero padding
        }
    }
    rh_tip5_permutation(state);
    
    // Extract digest from first 5 elements of state
    // Use streaming write for output
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        d_digests[row_idx * DIGEST_LEN + i] = state[i];
    }
}

/**
 * Hash XFieldElement rows to digests
 * Each XFieldElement is 3 BFieldElements.
 * 
 * Uses OVERWRITE mode (matching CPU hash_varlen):
 * - Overwrite rate portion with input chunk
 * - Apply permutation
 * - Last chunk includes padding (1 after data, then zeros)
 * 
 * OPTIMIZED: Uses vectorized loads and prefetching to improve memory
 * performance with unified memory. Reduces page fault overhead.
 */
__global__ void hash_xfield_rows_kernel(
    const uint64_t* __restrict__ d_table,
    size_t num_rows,
    size_t num_xfe_cols,
    uint64_t* __restrict__ d_digests
) {
    size_t row_idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row_idx >= num_rows) return;
    
    size_t num_bfe = num_xfe_cols * 3;
    
    // Variable-length domain: state starts as all zeros
    uint64_t state[TIP5_STATE_SIZE] = {0};
    
    size_t consumed = 0;
    
    // Process full chunks using OVERWRITE mode (matching CPU)
    // Optimized: Use cache-allocate hints for better memory performance
    while (consumed + TIP5_RATE <= num_bfe) {
        // Load current chunk with cache-allocate hints
        #pragma unroll
        for (int i = 0; i < TIP5_RATE; ++i) {
            size_t elem_idx = consumed + i;
            size_t xfe_col = elem_idx / 3;
            size_t component = elem_idx % 3;
            // Explicit size_t cast to avoid 32-bit overflow when (xfe_col*3+component)*num_rows > 2^32
            size_t table_idx = (xfe_col * 3 + component) * num_rows + row_idx;
            // Use __ldca (cache allocate) to hint keeping data in L2 cache
            state[i] = __ldca(&d_table[table_idx]);
        }
        rh_tip5_permutation(state);
        consumed += TIP5_RATE;
    }
    
    // Handle last chunk with padding (matching CPU)
    // Remaining elements, then 1, then zeros to fill RATE
    #pragma unroll
    for (int i = 0; i < TIP5_RATE; ++i) {
        size_t elem_idx = consumed + i;
        if (elem_idx < num_bfe) {
            size_t xfe_col = elem_idx / 3;
            size_t component = elem_idx % 3;
            // Explicit size_t cast to avoid 32-bit overflow when (xfe_col*3+component)*num_rows > 2^32
            size_t table_idx = (xfe_col * 3 + component) * num_rows + row_idx;
            state[i] = __ldca(&d_table[table_idx]);
        } else if (elem_idx == num_bfe) {
            state[i] = 1;  // Padding indicator
        } else {
            state[i] = 0;  // Zero padding
        }
    }
    rh_tip5_permutation(state);
    
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        d_digests[row_idx * DIGEST_LEN + i] = state[i];
    }
}

// ============================================================================
// Streaming Row Hashing (Frugal Mode)
// ============================================================================

__global__ void row_sponge_init_kernel(
    uint64_t* __restrict__ d_states,
    size_t num_rows
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    #pragma unroll
    for (int i = 0; i < TIP5_STATE_SIZE; ++i) {
        d_states[row * TIP5_STATE_SIZE + i] = 0;
    }
}

__global__ void row_sponge_absorb_rate_kernel(
    const uint64_t* __restrict__ d_table,   // [batch_cols * num_rows] col-major
    size_t num_rows,
    size_t num_cols_total,
    size_t col_start,
    size_t batch_cols,
    uint64_t* __restrict__ d_states         // [num_rows * 16]
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    uint64_t state[TIP5_STATE_SIZE];
    #pragma unroll
    for (int i = 0; i < TIP5_STATE_SIZE; ++i) {
        state[i] = d_states[row * TIP5_STATE_SIZE + i];
    }

    // Overwrite rate with this chunk (with padding only for last chunk)
    #pragma unroll
    for (int i = 0; i < TIP5_RATE; ++i) {
        size_t col = col_start + (size_t)i;
        uint64_t v = 0;
        if (col < num_cols_total) {
            // Column-major: d_table[local_col * num_rows + row]
            size_t local_col = (col - col_start);
            if (local_col < batch_cols) {
                v = __ldca(&d_table[local_col * num_rows + row]);
            }
        } else if (col == num_cols_total) {
            v = 1; // padding marker
        } else {
            v = 0; // zero padding
        }
        state[i] = v;
    }

    rh_tip5_permutation(state);

    #pragma unroll
    for (int i = 0; i < TIP5_STATE_SIZE; ++i) {
        d_states[row * TIP5_STATE_SIZE + i] = state[i];
    }
}

__global__ void row_sponge_finalize_kernel(
    const uint64_t* __restrict__ d_states,  // [num_rows * 16]
    size_t num_rows,
    uint64_t* __restrict__ d_digests        // [num_rows * 5]
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        d_digests[row * DIGEST_LEN + i] = d_states[row * TIP5_STATE_SIZE + i];
    }
}

// ============================================================================
// Host Interface
// ============================================================================

void hash_bfield_rows_gpu(
    const uint64_t* d_table,
    size_t num_rows,
    size_t num_cols,
    uint64_t* d_digests,
    cudaStream_t stream
) {
    if (num_rows == 0) return;
    
    // Block size 512 for better occupancy on modern GPUs
    int block_size = 512;
    int grid_size = (num_rows + block_size - 1) / block_size;
    
    hash_bfield_rows_kernel<<<grid_size, block_size, 0, stream>>>(
        d_table, num_rows, num_cols, d_digests
    );
}

void hash_xfield_rows_gpu(
    const uint64_t* d_table,
    size_t num_rows,
    size_t num_xfe_cols,
    uint64_t* d_digests,
    cudaStream_t stream
) {
    if (num_rows == 0) return;
    
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    
    hash_xfield_rows_kernel<<<grid_size, block_size, 0, stream>>>(
        d_table, num_rows, num_xfe_cols, d_digests
    );
}

void row_sponge_init_gpu(
    uint64_t* d_states,
    size_t num_rows,
    cudaStream_t stream
) {
    if (num_rows == 0) return;
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    row_sponge_init_kernel<<<grid_size, block_size, 0, stream>>>(d_states, num_rows);
}

void row_sponge_absorb_rate_gpu(
    const uint64_t* d_table,
    size_t num_rows,
    size_t num_cols_total,
    size_t col_start,
    size_t batch_cols,
    uint64_t* d_states,
    cudaStream_t stream
) {
    if (num_rows == 0 || batch_cols == 0) return;
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    row_sponge_absorb_rate_kernel<<<grid_size, block_size, 0, stream>>>(
        d_table, num_rows, num_cols_total, col_start, batch_cols, d_states
    );
}

void row_sponge_finalize_gpu(
    const uint64_t* d_states,
    size_t num_rows,
    uint64_t* d_digests,
    cudaStream_t stream
) {
    if (num_rows == 0) return;
    int block_size = 256;
    int grid_size = (num_rows + block_size - 1) / block_size;
    row_sponge_finalize_kernel<<<grid_size, block_size, 0, stream>>>(
        d_states, num_rows, d_digests
    );
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
