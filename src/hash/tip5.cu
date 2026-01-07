#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace {

constexpr int STATE_SIZE = 16;
constexpr int NUM_ROUNDS = 5;
constexpr int DIGEST_LEN = 5;
constexpr int RATE = 10;

constexpr uint64_t GOLDILOCKS_MODULUS = 0xFFFFFFFF00000001ULL;
constexpr uint64_t R2 = 0xFFFFFFFE00000001ULL;
constexpr uint64_t BFE_ONE = 0x0000000000000001ULL;

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

__constant__ uint32_t MDS_COEFF[STATE_SIZE] = {
    61402, 1108, 28750, 33823, 7454, 43244, 53865, 12034,
    56951, 27521, 41351, 40901, 12021, 59689, 26798, 17845
};

__constant__ uint64_t ROUND_CONSTANTS[NUM_ROUNDS][STATE_SIZE] = {
    {
        0xBD2A3DEB61AB60DEULL, 0xEA7DF21AD9547ED2ULL, 0x900B3677A1DE063FULL, 0x1B46887E876C8677ULL,
        0xD364D977889CFB97ULL, 0xDC8DFAC843699F02ULL, 0x375C405D7190DB58ULL, 0x27924006D2B0D4B1ULL,
        0x78DD1172D483CD38ULL, 0x3346C66244882A56ULL, 0xB0249B279F498AA5ULL, 0x94CD51BE79338D4DULL,
        0xB0E0DC7052C5B218ULL, 0xF8DCC4D248ADAD95ULL, 0x68E3C635FEC868B7ULL, 0xD7D06B3FFB6B0D8CULL
    },
    {
        0xF3500DEA20EF032AULL, 0x4865BF175BBA5803ULL, 0xD5F7FE3027287A27ULL, 0xA57333F44E193412ULL,
        0x8726E153A977EAE2ULL, 0x3014A98463FC191BULL, 0xBA145461AF39B212ULL, 0x03AB70105933202FULL,
        0x3D90B7EEBFCF71E5ULL, 0x386322B1CC520BFDULL, 0x27C2C8DAF774F675ULL, 0x4FCB83F50309BC6AULL,
        0x5E6D5CE8275F3CB3ULL, 0xECC2F6592C8F905CULL, 0x837F532461E609B4ULL, 0xB2B1F6B95C92C93CULL
    },
    {
        0xC0027AF556411DC1ULL, 0x16E18C885FC2A26CULL, 0x8880EF183D9F2BF3ULL, 0xB2930BDB5CA88C45ULL,
        0x9C2EC8322E1C1553ULL, 0xE5B05EAF3220A674ULL, 0xA49CC6AE4B861C4EULL, 0x11708E0AEB86EBD7ULL,
        0xC09DE92BBC3902E0ULL, 0x929B3C79516BCBC1ULL, 0xE006E5BF738F27D1ULL, 0x2D9E1EC0EAC8EA38ULL,
        0x0984D8D94BF937C5ULL, 0x4959273C220E6747ULL, 0xFE1D934207E796FAULL, 0x2B9B9298F2F6DD73ULL
    },
    {
        0x07A1F5A67D6E3A41ULL, 0x4407593EE73743D9ULL, 0x9F054720EF802E59ULL, 0x78D4B711336E6AA6ULL,
        0xADC638AEF3C8B228ULL, 0xA4D6D3E86AFB2114ULL, 0x9D4808E725531968ULL, 0x369804DF3866D0EFULL,
        0xE6DBD9A9D2215024ULL, 0x8ED22CA212EE85B2ULL, 0x397BB882FCD23EB6ULL, 0xEB8F8786D7277531ULL,
        0x9999D4CDAFF543B5ULL, 0xF382A61217F192D6ULL, 0x49C37260B026ADC1ULL, 0x3FF8918CE35C1019ULL
    },
    {
        0x2E7DF8B76080BD07ULL, 0xF5DBAC250B8A28B9ULL, 0x853C3727AE9DA4CCULL, 0xB2F1F5F3D9E5A26DULL,
        0x3FCE22012D337847ULL, 0x6B5A3E6DB7EEE347ULL, 0x171582CD59DDE50DULL, 0xC0C0B3095EE62A8AULL,
        0x665B25C6F6A203D2ULL, 0x3099AED93B6AE69FULL, 0x801DF6092BE69C38ULL, 0x8066AD0CDFFF43CDULL,
        0x8AF9D44A5F4FDC6BULL, 0xD80219CD97C0D762ULL, 0x10C9CEBA14148EBBULL, 0x539BD4C3F2F24474ULL
    }
};

__device__ __forceinline__ uint64_t montyred_from_parts(uint64_t xh, uint64_t xl) {
    uint64_t shifted = xl << 32;
    uint64_t a = xl + shifted;
    bool overflow_a = (a < xl) | (a < shifted);

    uint64_t b = a - (a >> 32);
    if (overflow_a) b -= 1;

    uint64_t r = xh - b;
    bool underflow_c = (xh < b);

    if (underflow_c) {
        r -= (1ULL + ~GOLDILOCKS_MODULUS);
    }

    return r;
}

__device__ __forceinline__ uint64_t field_mul(uint64_t a, uint64_t b) {
    uint64_t lo, hi;
    asm("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
    asm("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
    return montyred_from_parts(hi, lo);
}

__device__ __forceinline__ uint64_t x7_computer(uint64_t x) {
    uint64_t x2 = field_mul(x, x);
    uint64_t x4 = field_mul(x2, x2);
    uint64_t x6 = field_mul(x4, x2);
    return field_mul(x6, x);
}

__device__ __forceinline__ uint64_t split_lookup_shared(uint64_t element_in, const uint8_t* s_lookup) {
    uint64_t lo1 = element_in * R2;
    uint64_t hi1 = __umul64hi(element_in, R2);
    uint64_t reduced_in = montyred_from_parts(hi1, lo1);

    uint32_t lo32 = static_cast<uint32_t>(reduced_in);
    uint32_t hi32 = static_cast<uint32_t>(reduced_in >> 32);

    uint64_t b0 = s_lookup[(lo32 >> 0) & 0xFF];
    uint64_t b1 = s_lookup[(lo32 >> 8) & 0xFF];
    uint64_t b2 = s_lookup[(lo32 >> 16) & 0xFF];
    uint64_t b3 = s_lookup[(lo32 >> 24) & 0xFF];
    uint64_t b4 = s_lookup[(hi32 >> 0) & 0xFF];
    uint64_t b5 = s_lookup[(hi32 >> 8) & 0xFF];
    uint64_t b6 = s_lookup[(hi32 >> 16) & 0xFF];
    uint64_t b7 = s_lookup[(hi32 >> 24) & 0xFF];

    uint64_t sbox_out = b0 | (b1 << 8) | (b2 << 16) | (b3 << 24) |
                        (b4 << 32) | (b5 << 40) | (b6 << 48) | (b7 << 56);

    return montyred_from_parts(0, sbox_out);
}

__device__ __forceinline__ uint32_t get_mds_coeff(int i, int j) {
    int idx = (i - j) & (STATE_SIZE - 1);
    return MDS_COEFF[idx];
}

__device__ void mds_layer_original(const uint64_t* state_in, uint64_t* state_out) {
    #pragma unroll
    for (int i = 0; i < STATE_SIZE; ++i) {
        __uint128_t acc = 0;
        #pragma unroll
        for (int j = 0; j < STATE_SIZE; ++j) {
            acc += static_cast<__uint128_t>(state_in[j]) * get_mds_coeff(i, j);
        }

        uint64_t lower = static_cast<uint64_t>(acc);
        uint64_t upper = static_cast<uint64_t>(acc >> 64);
        uint64_t temp = lower + upper * 0xFFFFFFFFULL;
        if (upper > 0xFFFFFFFFULL) {
            temp += (upper >> 32) * 0xFFFFFFFFULL;
        }
        if (temp >= GOLDILOCKS_MODULUS) temp -= GOLDILOCKS_MODULUS;
        if (temp >= GOLDILOCKS_MODULUS) temp -= GOLDILOCKS_MODULUS;
        state_out[i] = temp;
    }
}

__device__ void sbox_layer_shared(const uint64_t* state_in, uint64_t* state_out, const uint8_t* s_lookup) {
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        state_out[i] = split_lookup_shared(state_in[i], s_lookup);
    }
    #pragma unroll
    for (int i = 4; i < STATE_SIZE; ++i) {
        state_out[i] = x7_computer(state_in[i]);
    }
}

__device__ void tip5_permutation_shared(uint64_t* state, const uint8_t* s_lookup) {
    uint64_t temp_state[STATE_SIZE];

    for (int round = 0; round < NUM_ROUNDS; ++round) {
        sbox_layer_shared(state, temp_state, s_lookup);
        mds_layer_original(temp_state, state);
        #pragma unroll
        for (int i = 0; i < STATE_SIZE; ++i) {
            state[i] += ROUND_CONSTANTS[round][i];
            if (state[i] >= GOLDILOCKS_MODULUS || state[i] < ROUND_CONSTANTS[round][i]) {
                state[i] -= GOLDILOCKS_MODULUS;
            }
        }
    }
}

__global__ void tip5_hash_rows_kernel(
    const uint64_t* rows,
    size_t row_len,
    size_t row_stride,
    size_t col_stride,
    uint64_t* out_digests,
    size_t row_count
) {
    __shared__ uint8_t s_lookup[256];
    if (threadIdx.x < 256) {
        s_lookup[threadIdx.x] = LOOKUP_TABLE[threadIdx.x];
    }
    __syncthreads();

    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= row_count) return;

    const uint64_t* row_ptr = rows + idx * row_stride;
    uint64_t state[STATE_SIZE] = {0};

    size_t consumed = 0;
    while (consumed + RATE <= row_len) {
        #pragma unroll
        for (int i = 0; i < RATE; ++i) {
            state[i] = row_ptr[(consumed + i) * col_stride];
        }
        tip5_permutation_shared(state, s_lookup);
        consumed += RATE;
    }

    uint64_t last_chunk[RATE] = {0};
    for (size_t i = 0; i < row_len - consumed; ++i) {
        last_chunk[i] = row_ptr[(consumed + i) * col_stride];
    }
    last_chunk[row_len - consumed] = BFE_ONE;
    #pragma unroll
    for (int i = 0; i < RATE; ++i) {
        state[i] = last_chunk[i];
    }
    tip5_permutation_shared(state, s_lookup);

    uint64_t* out_ptr = out_digests + idx * DIGEST_LEN;
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        out_ptr[i] = state[i];
    }
}

} // namespace

namespace {

int hash_rows_device(
    const uint64_t* d_rows,
    size_t row_count,
    size_t row_len,
    size_t row_stride,
    size_t col_stride,
    uint64_t* d_digests
) {
    const int threads = 256;
    const int blocks = static_cast<int>((row_count + threads - 1) / threads);
    tip5_hash_rows_kernel<<<blocks, threads>>>(
        d_rows,
        row_len,
        row_stride,
        col_stride,
        d_digests,
        row_count);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return 4;
    }
    return 0;
}

} // namespace

extern "C" int tv_gpu_tip5_hash_rows(
    const uint64_t* host_rows,
    size_t row_count,
    size_t row_len,
    size_t row_stride,
    size_t col_stride,
    int device_id,
    uint64_t* out_digests
) {
    if (!host_rows || !out_digests) {
        return 1;
    }
    if (row_len == 0 || row_count == 0) {
        return 0;
    }

    if (device_id >= 0) {
        if (cudaSetDevice(device_id) != cudaSuccess) {
            return -1;
        }
    }

    size_t row_words = row_count * row_stride;
    size_t row_bytes = row_words * sizeof(uint64_t);
    size_t digest_bytes = row_count * DIGEST_LEN * sizeof(uint64_t);

    uint64_t* d_rows = nullptr;
    uint64_t* d_digests = nullptr;

    cudaError_t err = cudaMalloc(&d_rows, row_bytes);
    if (err != cudaSuccess) {
        return 2;
    }
    err = cudaMalloc(&d_digests, digest_bytes);
    if (err != cudaSuccess) {
        cudaFree(d_rows);
        return 2;
    }

    err = cudaMemcpy(d_rows, host_rows, row_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_rows);
        cudaFree(d_digests);
        return 3;
    }

    int status = hash_rows_device(d_rows, row_count, row_len, row_stride, col_stride, d_digests);
    if (status != 0) {
        cudaFree(d_rows);
        cudaFree(d_digests);
        return status;
    }

    err = cudaMemcpy(out_digests, d_digests, digest_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_rows);
    cudaFree(d_digests);
    if (err != cudaSuccess) {
        return 5;
    }

    return 0;
}

extern "C" int tv_gpu_tip5_hash_rows_device(
    const uint64_t* device_rows,
    size_t row_count,
    size_t row_len,
    size_t row_stride,
    size_t col_stride,
    uint64_t* out_digests
) {
    if (!device_rows || !out_digests) {
        return 1;
    }
    if (row_len == 0 || row_count == 0) {
        return 0;
    }
    return hash_rows_device(
        device_rows,
        row_count,
        row_len,
        row_stride,
        col_stride,
        out_digests);
}

__global__ void tip5_hash_pairs_kernel(
    const uint64_t* children,
    size_t child_count,
    size_t child_stride,
    uint64_t* parents
) {
    __shared__ uint8_t s_lookup[256];
    if (threadIdx.x < 256) {
        s_lookup[threadIdx.x] = LOOKUP_TABLE[threadIdx.x];
    }
    __syncthreads();

    size_t parent_idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t parent_count = (child_count + 1) / 2;
    if (parent_idx >= parent_count) {
        return;
    }

    size_t left_idx = parent_idx * 2;
    size_t right_idx = left_idx + 1;
    if (right_idx >= child_count) {
        right_idx = child_count - 1;
    }

    const uint64_t* left = children + left_idx * child_stride;
    const uint64_t* right = children + right_idx * child_stride;

    uint64_t state[STATE_SIZE] = {0};
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        state[i] = left[i];
        state[i + DIGEST_LEN] = right[i];
    }
    #pragma unroll
    for (int i = RATE; i < STATE_SIZE; ++i) {
        state[i] = BFE_ONE;
    }

    tip5_permutation_shared(state, s_lookup);

    uint64_t* out = parents + parent_idx * DIGEST_LEN;
    #pragma unroll
    for (int i = 0; i < DIGEST_LEN; ++i) {
        out[i] = state[i];
    }
}

static int build_merkle_root_device(
    uint64_t* leaves,
    size_t leaf_count,
    size_t leaf_stride,
    uint64_t* workspace,
    uint64_t* out_root
) {
    (void)leaf_stride;
    if (leaf_count == 0) {
        cudaError_t err = cudaMemset(out_root, 0, DIGEST_LEN * sizeof(uint64_t));
        return err == cudaSuccess ? 0 : 7;
    }

    uint64_t* level_a = workspace;
    uint64_t* level_b = workspace + leaf_count * DIGEST_LEN;

    cudaError_t err = cudaMemcpy(
        level_a,
        leaves,
        leaf_count * DIGEST_LEN * sizeof(uint64_t),
        cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        return 8;
    }

    size_t current_count = leaf_count;
    uint64_t* current = level_a;
    uint64_t* next = level_b;

    while (current_count > 1) {
        size_t parent_count = (current_count + 1) / 2;
        const int threads = 256;
        const int blocks = static_cast<int>((parent_count + threads - 1) / threads);
        tip5_hash_pairs_kernel<<<blocks, threads>>>(
            current,
            current_count,
            DIGEST_LEN,
            next);
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            return 9;
        }

        current = next;
        next = (next == level_b) ? level_a : level_b;
        current_count = parent_count;
    }

    err = cudaMemcpy(out_root, current, DIGEST_LEN * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
    if (err != cudaSuccess) {
        return 10;
    }
    return 0;
}

extern "C" int tv_gpu_tip5_build_merkle_root_device(
    uint64_t* device_leaves,
    size_t leaf_count,
    size_t leaf_stride,
    uint64_t* workspace,
    uint64_t* out_root
) {
    if (!device_leaves || !workspace || !out_root) {
        return 1;
    }
    return build_merkle_root_device(
        device_leaves,
        leaf_count,
        leaf_stride,
        workspace,
        out_root);
}

