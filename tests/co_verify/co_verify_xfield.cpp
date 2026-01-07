/**
 * XFieldElement Co-Verification Tests
 * 
 * Compares CPU XFieldElement implementation with GPU CUDA kernels
 * to ensure they produce identical results.
 */

#include "co_verify_framework.hpp"
#include "types/x_field_element.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/xfield_kernel.cuh"
#include <cuda_runtime.h>
#endif

#include <gtest/gtest.h>
#include <vector>
#include <random>
#include <chrono>

namespace triton_vm {
namespace co_verify {

class XFieldCoVerifyTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef TRITON_CUDA_ENABLED
        cudaError_t err = cudaSetDevice(0);
        if (err != cudaSuccess) {
            GTEST_SKIP() << "No CUDA device available";
        }
#else
        GTEST_SKIP() << "CUDA not enabled";
#endif
    }
    
    std::mt19937_64 rng_{42};
    
    // Generate random XFieldElement
    XFieldElement random_xfield() {
        std::uniform_int_distribution<uint64_t> dist(0, BFieldElement::MODULUS - 1);
        return XFieldElement(
            BFieldElement(dist(rng_)),
            BFieldElement(dist(rng_)),
            BFieldElement(dist(rng_))
        );
    }
    
    // Generate non-zero XFieldElement (for inversion tests)
    XFieldElement random_nonzero_xfield() {
        XFieldElement x;
        do {
            x = random_xfield();
        } while (x.is_zero());
        return x;
    }
    
    // Convert XFieldElement vector to flat uint64_t array
    std::vector<uint64_t> to_flat(const std::vector<XFieldElement>& v) {
        std::vector<uint64_t> result;
        result.reserve(v.size() * 3);
        for (const auto& x : v) {
            result.push_back(x.coeff(0).value());
            result.push_back(x.coeff(1).value());
            result.push_back(x.coeff(2).value());
        }
        return result;
    }
    
    // Convert flat uint64_t array to XFieldElement vector
    std::vector<XFieldElement> from_flat(const std::vector<uint64_t>& v) {
        std::vector<XFieldElement> result;
        result.reserve(v.size() / 3);
        for (size_t i = 0; i < v.size(); i += 3) {
            result.push_back(XFieldElement(
                BFieldElement(v[i]),
                BFieldElement(v[i + 1]),
                BFieldElement(v[i + 2])
            ));
        }
        return result;
    }
    
    // Compare two XFieldElements
    static bool compare_xfield(const XFieldElement& a, const XFieldElement& b) {
        return a == b;
    }
};

// ============================================================================
// Addition Tests
// ============================================================================

TEST_F(XFieldCoVerifyTest, Addition_Random) {
#ifdef TRITON_CUDA_ENABLED
    const size_t N = 10000;
    
    // Generate random inputs
    std::vector<XFieldElement> a_cpu(N), b_cpu(N), c_cpu(N);
    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = random_xfield();
        b_cpu[i] = random_xfield();
    }
    
    // CPU computation
    for (size_t i = 0; i < N; ++i) {
        c_cpu[i] = a_cpu[i] + b_cpu[i];
    }
    
    // GPU computation
    auto a_flat = to_flat(a_cpu);
    auto b_flat = to_flat(b_cpu);
    std::vector<uint64_t> c_flat(N * 3);
    
    uint64_t *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_b, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_c, N * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_a, a_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfield_add_gpu(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(c_flat.data(), d_c, N * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto c_gpu = from_flat(c_flat);
    
    // Compare
    size_t mismatch;
    bool match = compare_vectors(c_cpu, c_gpu, mismatch, compare_xfield);
    EXPECT_TRUE(match) << "Mismatch at index " << mismatch 
                       << ": CPU=" << c_cpu[mismatch] << ", GPU=" << c_gpu[mismatch];
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
#endif
}

TEST_F(XFieldCoVerifyTest, Addition_EdgeCases) {
#ifdef TRITON_CUDA_ENABLED
    std::vector<XFieldElement> a_cpu = {
        XFieldElement::zero(),
        XFieldElement::one(),
        XFieldElement(BFieldElement(BFieldElement::MODULUS - 1), BFieldElement::zero(), BFieldElement::zero()),
        XFieldElement(BFieldElement::zero(), BFieldElement(BFieldElement::MODULUS - 1), BFieldElement::zero()),
        XFieldElement(BFieldElement::zero(), BFieldElement::zero(), BFieldElement(BFieldElement::MODULUS - 1)),
    };
    
    std::vector<XFieldElement> b_cpu = {
        XFieldElement::one(),
        XFieldElement::one(),
        XFieldElement::one(),
        XFieldElement(BFieldElement::zero(), BFieldElement::one(), BFieldElement::zero()),
        XFieldElement(BFieldElement::zero(), BFieldElement::zero(), BFieldElement::one()),
    };
    
    const size_t N = a_cpu.size();
    std::vector<XFieldElement> c_cpu(N);
    for (size_t i = 0; i < N; ++i) {
        c_cpu[i] = a_cpu[i] + b_cpu[i];
    }
    
    auto a_flat = to_flat(a_cpu);
    auto b_flat = to_flat(b_cpu);
    std::vector<uint64_t> c_flat(N * 3);
    
    uint64_t *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_b, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_c, N * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_a, a_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfield_add_gpu(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(c_flat.data(), d_c, N * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto c_gpu = from_flat(c_flat);
    
    size_t mismatch;
    bool match = compare_vectors(c_cpu, c_gpu, mismatch, compare_xfield);
    EXPECT_TRUE(match) << "Mismatch at edge case " << mismatch;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
#endif
}

// ============================================================================
// Subtraction Tests
// ============================================================================

TEST_F(XFieldCoVerifyTest, Subtraction_Random) {
#ifdef TRITON_CUDA_ENABLED
    const size_t N = 10000;
    
    std::vector<XFieldElement> a_cpu(N), b_cpu(N), c_cpu(N);
    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = random_xfield();
        b_cpu[i] = random_xfield();
        c_cpu[i] = a_cpu[i] - b_cpu[i];
    }
    
    auto a_flat = to_flat(a_cpu);
    auto b_flat = to_flat(b_cpu);
    std::vector<uint64_t> c_flat(N * 3);
    
    uint64_t *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_b, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_c, N * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_a, a_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfield_sub_gpu(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(c_flat.data(), d_c, N * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto c_gpu = from_flat(c_flat);
    
    size_t mismatch;
    bool match = compare_vectors(c_cpu, c_gpu, mismatch, compare_xfield);
    EXPECT_TRUE(match) << "Mismatch at index " << mismatch;
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
#endif
}

// ============================================================================
// Multiplication Tests
// ============================================================================

TEST_F(XFieldCoVerifyTest, Multiplication_Random) {
#ifdef TRITON_CUDA_ENABLED
    const size_t N = 10000;
    
    std::vector<XFieldElement> a_cpu(N), b_cpu(N), c_cpu(N);
    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = random_xfield();
        b_cpu[i] = random_xfield();
        c_cpu[i] = a_cpu[i] * b_cpu[i];
    }
    
    auto a_flat = to_flat(a_cpu);
    auto b_flat = to_flat(b_cpu);
    std::vector<uint64_t> c_flat(N * 3);
    
    uint64_t *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_b, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_c, N * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_a, a_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfield_mul_gpu(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(c_flat.data(), d_c, N * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto c_gpu = from_flat(c_flat);
    
    size_t mismatch;
    bool match = compare_vectors(c_cpu, c_gpu, mismatch, compare_xfield);
    EXPECT_TRUE(match) << "Mismatch at index " << mismatch
                       << ": CPU=" << c_cpu[mismatch] << ", GPU=" << c_gpu[mismatch];
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
#endif
}

TEST_F(XFieldCoVerifyTest, Multiplication_EdgeCases) {
#ifdef TRITON_CUDA_ENABLED
    // Test X * X = X² (identity in extension)
    XFieldElement x = XFieldElement(BFieldElement::zero(), BFieldElement::one(), BFieldElement::zero());
    XFieldElement x_squared = x * x;
    XFieldElement expected_x2 = XFieldElement(BFieldElement::zero(), BFieldElement::zero(), BFieldElement::one());
    EXPECT_EQ(x_squared, expected_x2) << "X * X should equal X²";
    
    // Test X³ = X - 1 (the defining relation)
    XFieldElement x_cubed = x * x * x;
    XFieldElement expected_x3 = XFieldElement(
        BFieldElement(BFieldElement::MODULUS - 1),  // -1
        BFieldElement::one(),                        // X
        BFieldElement::zero()                        // 0*X²
    );
    EXPECT_EQ(x_cubed, expected_x3) << "X³ should equal X - 1";
    
    // Test multiplicative identity
    XFieldElement a = random_xfield();
    EXPECT_EQ(a * XFieldElement::one(), a) << "a * 1 = a";
    
    // Test zero
    EXPECT_EQ(a * XFieldElement::zero(), XFieldElement::zero()) << "a * 0 = 0";
#endif
}

// ============================================================================
// Negation Tests
// ============================================================================

TEST_F(XFieldCoVerifyTest, Negation_Random) {
#ifdef TRITON_CUDA_ENABLED
    const size_t N = 10000;
    
    std::vector<XFieldElement> a_cpu(N), c_cpu(N);
    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = random_xfield();
        c_cpu[i] = -a_cpu[i];
    }
    
    auto a_flat = to_flat(a_cpu);
    std::vector<uint64_t> c_flat(N * 3);
    
    uint64_t *d_a, *d_c;
    cudaMalloc(&d_a, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_c, N * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_a, a_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfield_neg_gpu(d_a, d_c, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(c_flat.data(), d_c, N * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto c_gpu = from_flat(c_flat);
    
    size_t mismatch;
    bool match = compare_vectors(c_cpu, c_gpu, mismatch, compare_xfield);
    EXPECT_TRUE(match) << "Mismatch at index " << mismatch;
    
    cudaFree(d_a);
    cudaFree(d_c);
#endif
}

// ============================================================================
// Inversion Tests
// ============================================================================

TEST_F(XFieldCoVerifyTest, Inverse_Random) {
#ifdef TRITON_CUDA_ENABLED
    const size_t N = 1000;  // Fewer elements as inversion is expensive
    
    std::vector<XFieldElement> a_cpu(N), c_cpu(N);
    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = random_nonzero_xfield();
        c_cpu[i] = a_cpu[i].inverse();
    }
    
    auto a_flat = to_flat(a_cpu);
    std::vector<uint64_t> c_flat(N * 3);
    
    uint64_t *d_a, *d_c;
    cudaMalloc(&d_a, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_c, N * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_a, a_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfield_inv_gpu(d_a, d_c, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(c_flat.data(), d_c, N * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto c_gpu = from_flat(c_flat);
    
    size_t mismatch;
    bool match = compare_vectors(c_cpu, c_gpu, mismatch, compare_xfield);
    EXPECT_TRUE(match) << "Mismatch at index " << mismatch
                       << ": CPU=" << c_cpu[mismatch] << ", GPU=" << c_gpu[mismatch];
    
    cudaFree(d_a);
    cudaFree(d_c);
#endif
}

TEST_F(XFieldCoVerifyTest, Inverse_Verify) {
#ifdef TRITON_CUDA_ENABLED
    const size_t N = 100;
    
    std::vector<XFieldElement> a_cpu(N);
    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = random_nonzero_xfield();
    }
    
    auto a_flat = to_flat(a_cpu);
    std::vector<uint64_t> inv_flat(N * 3);
    
    uint64_t *d_a, *d_inv;
    cudaMalloc(&d_a, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_inv, N * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_a, a_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfield_inv_gpu(d_a, d_inv, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(inv_flat.data(), d_inv, N * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto inv_gpu = from_flat(inv_flat);
    
    // Verify a * a^{-1} = 1
    for (size_t i = 0; i < N; ++i) {
        XFieldElement product = a_cpu[i] * inv_gpu[i];
        EXPECT_EQ(product, XFieldElement::one()) 
            << "a * a^{-1} != 1 at index " << i
            << ": a=" << a_cpu[i] << ", a^{-1}=" << inv_gpu[i]
            << ", product=" << product;
    }
    
    cudaFree(d_a);
    cudaFree(d_inv);
#endif
}

// ============================================================================
// Scalar Multiplication Tests
// ============================================================================

TEST_F(XFieldCoVerifyTest, ScalarMul_Random) {
#ifdef TRITON_CUDA_ENABLED
    const size_t N = 10000;
    
    std::vector<XFieldElement> a_cpu(N), c_cpu(N);
    std::vector<BFieldElement> scalars_cpu(N);
    
    std::uniform_int_distribution<uint64_t> dist(0, BFieldElement::MODULUS - 1);
    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = random_xfield();
        scalars_cpu[i] = BFieldElement(dist(rng_));
        c_cpu[i] = a_cpu[i] * scalars_cpu[i];
    }
    
    auto a_flat = to_flat(a_cpu);
    std::vector<uint64_t> scalars_flat(N);
    for (size_t i = 0; i < N; ++i) {
        scalars_flat[i] = scalars_cpu[i].value();
    }
    std::vector<uint64_t> c_flat(N * 3);
    
    uint64_t *d_a, *d_scalars, *d_c;
    cudaMalloc(&d_a, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_scalars, N * sizeof(uint64_t));
    cudaMalloc(&d_c, N * 3 * sizeof(uint64_t));
    
    cudaMemcpy(d_a, a_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scalars, scalars_flat.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice);
    
    gpu::kernels::xfield_scalar_mul_gpu(d_a, d_scalars, d_c, N);
    cudaDeviceSynchronize();
    
    cudaMemcpy(c_flat.data(), d_c, N * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    
    auto c_gpu = from_flat(c_flat);
    
    size_t mismatch;
    bool match = compare_vectors(c_cpu, c_gpu, mismatch, compare_xfield);
    EXPECT_TRUE(match) << "Mismatch at index " << mismatch;
    
    cudaFree(d_a);
    cudaFree(d_scalars);
    cudaFree(d_c);
#endif
}

// ============================================================================
// Benchmark
// ============================================================================

TEST_F(XFieldCoVerifyTest, Benchmark_Multiplication) {
#ifdef TRITON_CUDA_ENABLED
    const size_t N = 1 << 20;  // 1M elements
    
    std::vector<XFieldElement> a_cpu(N), b_cpu(N), c_cpu(N);
    for (size_t i = 0; i < N; ++i) {
        a_cpu[i] = random_xfield();
        b_cpu[i] = random_xfield();
    }
    
    // CPU timing
    auto cpu_start = std::chrono::high_resolution_clock::now();
    for (size_t i = 0; i < N; ++i) {
        c_cpu[i] = a_cpu[i] * b_cpu[i];
    }
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
    
    // GPU timing
    auto a_flat = to_flat(a_cpu);
    auto b_flat = to_flat(b_cpu);
    std::vector<uint64_t> c_flat(N * 3);
    
    uint64_t *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_b, N * 3 * sizeof(uint64_t));
    cudaMalloc(&d_c, N * 3 * sizeof(uint64_t));
    
    auto gpu_start = std::chrono::high_resolution_clock::now();
    cudaMemcpy(d_a, a_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b_flat.data(), N * 3 * sizeof(uint64_t), cudaMemcpyHostToDevice);
    gpu::kernels::xfield_mul_gpu(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    cudaMemcpy(c_flat.data(), d_c, N * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_ms = std::chrono::duration<double, std::milli>(gpu_end - gpu_start).count();
    
    std::cout << "\n[Benchmark] XField Multiplication (" << N << " elements):\n";
    std::cout << "  CPU: " << std::fixed << std::setprecision(2) << cpu_ms << " ms\n";
    std::cout << "  GPU: " << gpu_ms << " ms (including H2D/D2H transfer)\n";
    std::cout << "  Speedup: " << (cpu_ms / gpu_ms) << "x\n";
    
    // Verify correctness
    auto c_gpu = from_flat(c_flat);
    size_t mismatch;
    bool match = compare_vectors(c_cpu, c_gpu, mismatch, compare_xfield);
    EXPECT_TRUE(match) << "Benchmark results mismatch";
    
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
#endif
}

} // namespace co_verify
} // namespace triton_vm

