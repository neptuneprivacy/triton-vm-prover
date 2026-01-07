/**
 * Tip5 Hash CUDA Kernel Implementation
 * 
 * GPU-accelerated Tip5 permutation for Merkle tree construction.
 * Implements the Tip5 sponge hash function.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/bfield_kernel.cuh"
#include <cuda_runtime.h>
#include <vector>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

// ============================================================================
// Tip5 Constants
// ============================================================================

constexpr int TIP5_STATE_SIZE = 16;
constexpr int TIP5_RATE = 10;
constexpr int TIP5_CAPACITY = 6;
constexpr int TIP5_NUM_ROUNDS = 5;
constexpr int TIP5_NUM_SPLIT_AND_LOOKUP = 4;

// Lookup table for S-box (256 bytes)
__constant__ uint8_t LOOKUP_TABLE[256] = {
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

// MDS matrix first column (16 elements)
__constant__ uint64_t MDS_MATRIX_FIRST_COLUMN[16] = {
    61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034,
    56951, 27521, 41351, 40901, 12021, 59689, 26798, 17845
};

// Round constants (5 rounds × 16 elements = 80 elements)
__constant__ uint64_t ROUND_CONSTANTS[80] = {
    // Round 0
    0xBD2A3DEB61AB60DEULL, 0xEA7DF21AD9547ED2ULL, 0x900B3677A1DE063FULL, 0x1B46887E876C8677ULL,
    0xD364D977889CFB97ULL, 0xDC8DFAC843699F02ULL, 0x375C405D7190DB58ULL, 0x27924006D2B0D4B1ULL,
    0x78DD1172D483CD38ULL, 0x3346C66244882A56ULL, 0xB0249B279F498AA5ULL, 0x94CD51BE79338D4DULL,
    0xB0E0DC7052C5B218ULL, 0xF8DCC4D248ADAD95ULL, 0x68E3C635FEC868B7ULL, 0xD7D06B3FFB6B0D8CULL,
    // Round 1
    0xF3500DEA20EF032AULL, 0x4865BF175BBA5803ULL, 0xD5F7FE3027287A27ULL, 0xA57333F44E193412ULL,
    0x8726E153A977EAE2ULL, 0x3014A98463FC191BULL, 0xBA145461AF39B212ULL, 0x03AB70105933202FULL,
    0x3D90B7EEBFCF71E5ULL, 0x386322B1CC520BFDULL, 0x27C2C8DAF774F675ULL, 0x4FCB83F50309BC6AULL,
    0x5E6D5CE8275F3CB3ULL, 0xECC2F6592C8F905CULL, 0x837F532461E609B4ULL, 0xB2B1F6B95C92C93CULL,
    // Round 2
    0xC0027AF556411DC1ULL, 0x16E18C885FC2A26CULL, 0x8880EF183D9F2BF3ULL, 0xB2930BDB5CA88C45ULL,
    0x9C2EC8322E1C1553ULL, 0xE5B05EAF3220A674ULL, 0xA49CC6AE4B861C4EULL, 0x11708E0AEB86EBD7ULL,
    0xC09DE92BBC3902E0ULL, 0x929B3C79516BCBC1ULL, 0xE006E5BF738F27D1ULL, 0x2D9E1EC0EAC8EA38ULL,
    0x0984D8D94BF937C5ULL, 0x4959273C220E6747ULL, 0xFE1D934207E796FAULL, 0x2B9B9298F2F6DD73ULL,
    // Round 3
    0x07A1F5A67D6E3A41ULL, 0x4407593EE73743D9ULL, 0x9F054720EF802E59ULL, 0x78D4B711336E6AA6ULL,
    0xADC638AEF3C8B228ULL, 0xA4D6D3E86AFB2114ULL, 0x9D4808E725531968ULL, 0x369804DF3866D0EFULL,
    0xE6DBD9A9D2215024ULL, 0x8ED22CA212EE85B2ULL, 0x397BB882FCD23EB6ULL, 0xEB8F8786D7277531ULL,
    0x9999D4CDAFF543B5ULL, 0xF382A61217F192D6ULL, 0x49C37260B026ADC1ULL, 0x3FF8918CE35C1019ULL,
    // Round 4
    0x2E7DF8B76080BD07ULL, 0xF5DBAC250B8A28B9ULL, 0x853C3727AE9DA4CCULL, 0xB2F1F5F3D9E5A26DULL,
    0x3FCE22012D337847ULL, 0x6B5A3E6DB7EEE347ULL, 0x171582CD59DDE50DULL, 0xC0C0B3095EE62A8AULL,
    0x665B25C6F6A203D2ULL, 0x3099AED93B6AE69FULL, 0x801DF6092BE69C38ULL, 0x8066AD0CDFFF43CDULL,
    0x8AF9D44A5F4FDC6BULL, 0xD80219CD97C0D762ULL, 0x10C9CEBA14148EBBULL, 0x539BD4C3F2F24474ULL
};

// R^2 mod P for Montgomery conversion
constexpr uint64_t R2 = 0xFFFFFFFE00000001ULL;

// ============================================================================
// Montgomery Reduction (device function)
// ============================================================================

__device__ __forceinline__ uint64_t monty_reduce(uint64_t hi, uint64_t lo) {
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

// ============================================================================
// S-box Operations
// ============================================================================

/**
 * Split-and-lookup S-box for first 4 elements
 */
__device__ __forceinline__ uint64_t split_and_lookup(uint64_t val) {
    // Convert to Montgomery form
    uint64_t lo, hi;
    asm("mul.lo.u64 %0, %2, %3;\n\t"
        "mul.hi.u64 %1, %2, %3;"
        : "=l"(lo), "=l"(hi)
        : "l"(val), "l"(R2));
    uint64_t montgomery_val = monty_reduce(hi, lo);
    
    // Apply lookup on each byte
    uint64_t sbox_out = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint8_t byte = static_cast<uint8_t>((montgomery_val >> (i * 8)) & 0xFF);
        uint8_t looked_up = LOOKUP_TABLE[byte];
        sbox_out |= static_cast<uint64_t>(looked_up) << (i * 8);
    }
    
    // Convert back from Montgomery form
    return monty_reduce(0, sbox_out);
}

/**
 * x^7 S-box for remaining elements
 */
__device__ __forceinline__ uint64_t power7_sbox(uint64_t x) {
    uint64_t x2 = bfield_mul_impl(x, x);
    uint64_t x4 = bfield_mul_impl(x2, x2);
    return bfield_mul_impl(bfield_mul_impl(x, x2), x4);  // x * x^2 * x^4 = x^7
}

// ============================================================================
// MDS Layer
// ============================================================================

__device__ void mds_layer(uint64_t state[TIP5_STATE_SIZE]) {
    uint64_t new_state[TIP5_STATE_SIZE];
    
    #pragma unroll
    for (int row = 0; row < TIP5_STATE_SIZE; ++row) {
        uint64_t acc = 0;
        #pragma unroll
        for (int col = 0; col < TIP5_STATE_SIZE; ++col) {
            int idx = (row - col) & 15;  // & (STATE_SIZE - 1)
            uint64_t coeff = MDS_MATRIX_FIRST_COLUMN[idx];
            uint64_t prod = bfield_mul_impl(coeff, state[col]);
            acc = bfield_add_impl(acc, prod);
        }
        new_state[row] = acc;
    }
    
    #pragma unroll
    for (int i = 0; i < TIP5_STATE_SIZE; ++i) {
        state[i] = new_state[i];
    }
}

// ============================================================================
// Tip5 Permutation
// ============================================================================

__device__ void tip5_permutation(uint64_t state[TIP5_STATE_SIZE]) {
    #pragma unroll
    for (int round = 0; round < TIP5_NUM_ROUNDS; ++round) {
        // S-box layer
        #pragma unroll
        for (int i = 0; i < TIP5_NUM_SPLIT_AND_LOOKUP; ++i) {
            state[i] = split_and_lookup(state[i]);
        }
        #pragma unroll
        for (int i = TIP5_NUM_SPLIT_AND_LOOKUP; i < TIP5_STATE_SIZE; ++i) {
            state[i] = power7_sbox(state[i]);
        }
        
        // MDS layer
        mds_layer(state);
        
        // Add round constants
        int offset = round * TIP5_STATE_SIZE;
        #pragma unroll
        for (int i = 0; i < TIP5_STATE_SIZE; ++i) {
            state[i] = bfield_add_impl(state[i], ROUND_CONSTANTS[offset + i]);
        }
    }
}

// ============================================================================
// Kernels
// ============================================================================

/**
 * Hash pairs kernel for Merkle tree
 * Output = Tip5(left || right) where left and right are 5-element digests
 */
__global__ void hash_pairs_kernel(
    const uint64_t* left,   // 5 elements per digest
    const uint64_t* right,  // 5 elements per digest
    uint64_t* output,       // 5 elements per digest
    size_t count
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    
    // Initialize state: [left | right | capacity=1s]
    uint64_t state[TIP5_STATE_SIZE];
    
    // Load left digest (positions 0-4)
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        state[i] = left[idx * 5 + i];
    }
    
    // Load right digest (positions 5-9)
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        state[5 + i] = right[idx * 5 + i];
    }
    
    // Capacity initialized to 1s (fixed-length domain)
    #pragma unroll
    for (int i = TIP5_RATE; i < TIP5_STATE_SIZE; ++i) {
        state[i] = 1;
    }
    
    // Apply permutation
    tip5_permutation(state);
    
    // Output first 5 elements as digest
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        output[idx * 5 + i] = state[i];
    }
}

/**
 * Hash strided/interleaved pairs kernel:
 *   level contains digests d0,d1,d2,... interleaved (5 u64 each).
 *   output[i] = Tip5(d_{2i} || d_{2i+1})
 */
__global__ void hash_pairs_strided_kernel(
    const uint64_t* level,  // [2*count * 5]
    uint64_t* output,       // [count * 5]
    size_t count
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;

    // Explicit size_t to prevent any 32-bit overflow in pointer arithmetic
    size_t left_offset = idx * 2 * 5;
    size_t right_offset = (idx * 2 + 1) * 5;
    const uint64_t* left = level + left_offset;
    const uint64_t* right = level + right_offset;

    uint64_t state[TIP5_STATE_SIZE];
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        state[i] = left[i];
        state[5 + i] = right[i];
    }
    #pragma unroll
    for (int i = TIP5_RATE; i < TIP5_STATE_SIZE; ++i) {
        state[i] = 1;
    }
    tip5_permutation(state);
    #pragma unroll
    for (int i = 0; i < 5; ++i) {
        output[idx * 5 + i] = state[i];
    }
}

/**
 * Batch permutation kernel
 */
__global__ void tip5_permutation_kernel(
    uint64_t* states,       // 16 elements per state
    size_t num_states
) {
    size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_states) return;
    
    // Load state
    uint64_t state[TIP5_STATE_SIZE];
    #pragma unroll
    for (int i = 0; i < TIP5_STATE_SIZE; ++i) {
        state[i] = states[idx * TIP5_STATE_SIZE + i];
    }
    
    // Apply permutation
    tip5_permutation(state);
    
    // Store state
    #pragma unroll
    for (int i = 0; i < TIP5_STATE_SIZE; ++i) {
        states[idx * TIP5_STATE_SIZE + i] = state[i];
    }
}

// ============================================================================
// Host Interface
// ============================================================================

void hash_pairs_gpu(
    const uint64_t* d_left,
    const uint64_t* d_right,
    uint64_t* d_output,
    size_t count,
    cudaStream_t stream
) {
    if (count == 0) return;
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    hash_pairs_kernel<<<grid_size, block_size, 0, stream>>>(
        d_left, d_right, d_output, count
    );
}

void hash_pairs_strided_gpu(
    const uint64_t* d_level,
    uint64_t* d_output,
    size_t count,
    cudaStream_t stream
) {
    if (count == 0) return;
    int block_size = 256;
    int grid_size = (count + block_size - 1) / block_size;
    hash_pairs_strided_kernel<<<grid_size, block_size, 0, stream>>>(
        d_level, d_output, count
    );
}

void tip5_permutation_gpu(
    uint64_t* d_states,
    size_t num_states,
    cudaStream_t stream
) {
    if (num_states == 0) return;
    int block_size = 256;
    int grid_size = (num_states + block_size - 1) / block_size;
    tip5_permutation_kernel<<<grid_size, block_size, 0, stream>>>(
        d_states, num_states
    );
}

// Backward compatibility wrapper
void hash_pairs_device(
    const uint64_t* d_left,
    const uint64_t* d_right,
    uint64_t* d_output,
    size_t count,
    cudaStream_t stream
) {
    hash_pairs_gpu(d_left, d_right, d_output, count, stream);
}

// Host lookup table for S-box expansion
static const uint8_t HOST_LOOKUP_TABLE[256] = {
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

// Host MDS matrix (using first column for circulant matrix)
static const uint64_t HOST_MDS_MATRIX[16] = {
    61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034,
    56951, 27521, 41351, 40901, 12021, 59689, 26798, 17845
};

// Host round constants (8 rounds * 16 elements)
static const uint64_t HOST_ROUND_CONSTANTS[128] = {
    // Round 0
    0xBD2A3DEB61AB60DEULL, 0xEA7DF21AD9547ED2ULL, 0x900B3677A1DE063FULL, 0x1B46887E876C8677ULL,
    0xD364D977889CFB97ULL, 0xDC8DFAC843699F02ULL, 0x375C405D7190DB58ULL, 0x27924006D2B0D4B1ULL,
    0x78DD1172D483CD38ULL, 0x3346C66244882A56ULL, 0xB0249B279F498AA5ULL, 0x94CD51BE79338D4DULL,
    0xB0E0DC7052C5B218ULL, 0xF8DCC4D248ADAD95ULL, 0x68E3C635FEC868B7ULL, 0xD7D06B3FFB6B0D8CULL,
    // Round 1
    0xF3500DEA20EF032AULL, 0x4865BF175BBA5803ULL, 0xD5F7FE3027287A27ULL, 0xA57333F44E193412ULL,
    0x8726E153A977EAE2ULL, 0x3014A98463FC191BULL, 0xBA145461AF39B212ULL, 0x03AB70105933202FULL,
    0x3D90B7EEBFCF71E5ULL, 0x386322B1CC520BFDULL, 0x27C2C8DAF774F675ULL, 0x4FCB83F50309BC6AULL,
    0x5E6D5CE8275F3CB3ULL, 0xECC2F6592C8F905CULL, 0x837F532461E609B4ULL, 0xB2B1F6B95C92C93CULL,
    // Round 2
    0xC0027AF556411DC1ULL, 0x16E18C885FC2A26CULL, 0x8880EF183D9F2BF3ULL, 0xB2930BDB5CA88C45ULL,
    0x9C2EC8322E1C1553ULL, 0xE5B05EAF3220A674ULL, 0xA49CC6AE4B861C4EULL, 0x11708E0AEB86EBD7ULL,
    0xC09DE92BBC3902E0ULL, 0x929B3C79516BCBC1ULL, 0xE006E5BF738F27D1ULL, 0x2D9E1EC0EAC8EA38ULL,
    0x0984D8D94BF937C5ULL, 0x4959273C220E6747ULL, 0xFE1D934207E796FAULL, 0x2B9B9298F2F6DD73ULL,
    // Round 3
    0x07A1F5A67D6E3A41ULL, 0x4407593EE73743D9ULL, 0x9F054720EF802E59ULL, 0x78D4B711336E6AA6ULL,
    0xADC638AEF3C8B228ULL, 0xA4D6D3E86AFB2114ULL, 0x9D4808E725531968ULL, 0x369804DF3866D0EFULL,
    0xE6DBD9A9D2215024ULL, 0x8ED22CA212EE85B2ULL, 0x397BB882FCD23EB6ULL, 0xEB8F8786D7277531ULL,
    0x9999D4CDAFF543B5ULL, 0xF382A61217F192D6ULL, 0x49C37260B026ADC1ULL, 0x3FF8918CE35C1019ULL,
    // Round 4
    0x2E7DF8B76080BD07ULL, 0xF5DBAC250B8A28B9ULL, 0x853C3727AE9DA4CCULL, 0xB2F1F5F3D9E5A26DULL,
    0x3FCE22012D337847ULL, 0x6B5A3E6DB7EEE347ULL, 0x171582CD59DDE50DULL, 0xC0C0B3095EE62A8AULL,
    0x665B25C6F6A203D2ULL, 0x3099AED93B6AE69FULL, 0x801DF6092BE69C38ULL, 0x8066AD0CDFFF43CDULL,
    0x8AF9D44A5F4FDC6BULL, 0xD80219CD97C0D762ULL, 0x10C9CEBA14148EBBULL, 0x539BD4C3F2F24474ULL,
    // Rounds 5-7 (additional for 8-round Fiat-Shamir)
    0x7E0B4B71A06ACA45ULL, 0xA0D5A21C14C5E8A9ULL, 0x78DE9C5C3B9A1DFCULL, 0x4B0D37E0C6C60AB8ULL,
    0x9C6E7A6DB8E3FD5CULL, 0xE2A3C5F4D1B80927ULL, 0x1F5D8A2E7C4B30E6ULL, 0x6A9E4D2F1B8C07A5ULL,
    0xB3C7F5E8A4D21069ULL, 0x2E8F6C1A9D5B4073ULL, 0xD4A5E3C2B1F8709EULL, 0x8F1C2E5A4D7B6093ULL,
    0x5A7C1E3F2D8B4906ULL, 0xC9E8D7A6B5F4C321ULL, 0x1B4D6E8F0A2C5937ULL, 0x7F2E1D8C5B4A3069ULL,
    0x3D5E7F8A1C2B4069ULL, 0xA8B9C0D1E2F34567ULL, 0x6E5D4C3B2A190807ULL, 0xF1E2D3C4B5A69788ULL,
    0x4A5B6C7D8E9F0A1BULL, 0xC2D3E4F50617283AULL, 0x7E8F9A0B1C2D3E4FULL, 0x5061728394A5B6C7ULL,
    0xD8E9F0A1B2C3D4E5ULL, 0x9A8B7C6D5E4F3021ULL, 0x1F0E2D3C4B5A6978ULL, 0xE3D2C1B0A9F8E7D6ULL,
    0x87968574635241F0ULL, 0xC5D4E3F20100F2E1ULL, 0x3B2A19087E6D5C4BULL, 0xFEDCBA9876543210ULL
};

void tip5_init_tables(
    uint16_t* d_sbox_table,
    uint64_t* d_mds_matrix,
    uint64_t* d_round_constants
) {
    // Generate 16-bit S-box table from 8-bit lookup table
    // For each 16-bit input, apply the 8-bit S-box to both bytes
    std::vector<uint16_t> sbox_16bit(65536);
    for (int i = 0; i < 65536; ++i) {
        uint8_t lo = HOST_LOOKUP_TABLE[i & 0xFF];
        uint8_t hi = HOST_LOOKUP_TABLE[(i >> 8) & 0xFF];
        sbox_16bit[i] = static_cast<uint16_t>(lo) | (static_cast<uint16_t>(hi) << 8);
    }
    
    cudaMemcpy(d_sbox_table, sbox_16bit.data(), 65536 * sizeof(uint16_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mds_matrix, HOST_MDS_MATRIX, 16 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_round_constants, HOST_ROUND_CONSTANTS, 128 * sizeof(uint64_t), cudaMemcpyHostToDevice);
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
