/**
 * Co-Verification Test for Degree Lowering (Aux cols 49..86)
 *
 * This test isolates degree lowering:
 * - Random main table (BFE) + random aux base part (cols 0..48) + random challenges
 * - Rust FFI `degree_lowering_fill_aux_columns` computes reference degree-lowering cols
 * - CUDA `degree_lowering_only_gpu` computes degree-lowering cols on GPU
 * - Assert bit-for-bit equality for cols 49..86 (and that other cols are unchanged)
 */

#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "types/b_field_element.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/extend_kernel.cuh"
#include <cuda_runtime.h>
#endif

using namespace triton_vm;

extern "C" void degree_lowering_fill_aux_columns(
    const uint64_t* main_ptr,
    size_t num_rows,
    size_t main_cols,
    uint64_t* aux_ptr,
    size_t aux_cols,
    const uint64_t* challenges_ptr,
    size_t challenges_len
);

class DegreeLoweringCoVerifyTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef TRITON_CUDA_ENABLED
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
#else
        GTEST_SKIP() << "CUDA not enabled";
#endif
    }
};

#ifdef TRITON_CUDA_ENABLED

TEST_F(DegreeLoweringCoVerifyTest, DegreeLoweringOnly_BitForBit) {
    constexpr size_t kRows = 64;       // power of 2 not strictly required for degree lowering, but keep small
    constexpr size_t kMainCols = 379;  // expected by Rust degree lowering table
    constexpr size_t kAuxCols = 88;
    constexpr size_t kChallenges = 63;

    std::mt19937_64 rng(12345);
    auto rand_bfe = [&]() -> uint64_t { return rng() % BFieldElement::MODULUS; };

    // Main table (BFE)
    std::vector<uint64_t> main_flat(kRows * kMainCols);
    for (auto& v : main_flat) v = rand_bfe();

    // Aux table (XFE packed as 3*u64 per cell)
    // Initialize cols 0..48 random, cols 49..86 zero, col 87 random.
    std::vector<uint64_t> aux_init(kRows * kAuxCols * 3);
    for (size_t r = 0; r < kRows; ++r) {
        for (size_t c = 0; c < kAuxCols; ++c) {
            size_t idx = (r * kAuxCols + c) * 3;
            if (c < 49 || c == 87) {
                aux_init[idx + 0] = rand_bfe();
                aux_init[idx + 1] = rand_bfe();
                aux_init[idx + 2] = rand_bfe();
            } else {
                aux_init[idx + 0] = 0;
                aux_init[idx + 1] = 0;
                aux_init[idx + 2] = 0;
            }
        }
    }

    // Challenges (XFE packed)
    std::vector<uint64_t> challenges_flat(kChallenges * 3);
    for (auto& v : challenges_flat) v = rand_bfe();

    // CPU reference via Rust FFI (mutates aux buffer in-place, filling cols 49..86)
    std::vector<uint64_t> aux_ref = aux_init;
    degree_lowering_fill_aux_columns(
        main_flat.data(),
        kRows,
        kMainCols,
        aux_ref.data(),
        kAuxCols,
        challenges_flat.data(),
        kChallenges
    );

    // GPU compute degree lowering only
    uint64_t *d_main = nullptr, *d_aux = nullptr, *d_ch = nullptr;
    cudaMalloc(&d_main, main_flat.size() * sizeof(uint64_t));
    cudaMalloc(&d_aux, aux_init.size() * sizeof(uint64_t));
    cudaMalloc(&d_ch, challenges_flat.size() * sizeof(uint64_t));

    cudaMemcpy(d_main, main_flat.data(), main_flat.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aux, aux_init.data(), aux_init.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ch, challenges_flat.data(), challenges_flat.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    gpu::kernels::degree_lowering_only_gpu(d_main, kMainCols, kRows, d_ch, d_aux, 0);
    cudaDeviceSynchronize();

    std::vector<uint64_t> aux_gpu(aux_init.size());
    cudaMemcpy(aux_gpu.data(), d_aux, aux_gpu.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_main);
    cudaFree(d_aux);
    cudaFree(d_ch);

    // Check: cols 0..48 unchanged, col 87 unchanged
    for (size_t i = 0; i < aux_init.size(); ++i) {
        size_t cell = i / 3;
        size_t col = cell % kAuxCols;
        if (col < 49 || col == 87) {
            ASSERT_EQ(aux_gpu[i], aux_init[i]) << "Unexpected modification outside degree-lowering region at i=" << i;
        }
    }

    // Check: cols 49..86 match Rust FFI bit-for-bit
    size_t mismatches = 0;
    for (size_t r = 0; r < kRows; ++r) {
        for (size_t c = 49; c < 87; ++c) {
            size_t idx = (r * kAuxCols + c) * 3;
            for (size_t k = 0; k < 3; ++k) {
                if (aux_gpu[idx + k] != aux_ref[idx + k]) {
                    if (mismatches < 5) {
                        std::cerr << "Mismatch r=" << r << " c=" << c << " k=" << k
                                  << " GPU=" << aux_gpu[idx + k] << " REF=" << aux_ref[idx + k] << "\n";
                    }
                    ++mismatches;
                }
            }
        }
    }

    EXPECT_EQ(mismatches, 0u);
}

#endif // TRITON_CUDA_ENABLED


