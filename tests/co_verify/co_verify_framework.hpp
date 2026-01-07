#pragma once

#include <gtest/gtest.h>
#include <functional>
#include <vector>
#include <string>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"
#include "backend/backend.hpp"
#include "backend/cpu_backend.hpp"
#include "backend/cuda_backend.hpp"

namespace triton_vm {
namespace co_verify {

// ============================================================================
// Test Result Reporting
// ============================================================================

struct CoVerifyResult {
    bool passed;
    std::string component_name;
    std::string test_name;
    size_t input_size;
    double cpu_time_ms;
    double gpu_time_ms;
    double speedup;
    size_t first_mismatch_index;
    std::string mismatch_details;
    
    void print() const {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "[" << (passed ? "PASS" : "FAIL") << "] "
                  << component_name << "::" << test_name
                  << " (n=" << input_size << ")";
        
        if (passed) {
            std::cout << " - CPU: " << cpu_time_ms << "ms"
                      << ", GPU: " << gpu_time_ms << "ms"
                      << ", Speedup: " << speedup << "x";
        } else {
            std::cout << " - First mismatch at index " << first_mismatch_index
                      << ": " << mismatch_details;
        }
        std::cout << std::endl;
    }
};

// ============================================================================
// Random Data Generators
// ============================================================================

class RandomGenerator {
public:
    explicit RandomGenerator(uint64_t seed = 42) : rng_(seed) {}
    
    BFieldElement random_bfield() {
        return BFieldElement(dist_(rng_) % BFieldElement::MODULUS);
    }
    
    XFieldElement random_xfield() {
        return XFieldElement(random_bfield(), random_bfield(), random_bfield());
    }
    
    Digest random_digest() {
        Digest d;
        for (size_t i = 0; i < Digest::LEN; ++i) {
            d[i] = random_bfield();
        }
        return d;
    }
    
    std::vector<BFieldElement> random_bfield_vector(size_t n) {
        std::vector<BFieldElement> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = random_bfield();
        }
        return result;
    }
    
    std::vector<XFieldElement> random_xfield_vector(size_t n) {
        std::vector<XFieldElement> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = random_xfield();
        }
        return result;
    }
    
    std::vector<Digest> random_digest_vector(size_t n) {
        std::vector<Digest> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = random_digest();
        }
        return result;
    }
    
    std::vector<uint64_t> random_u64_vector(size_t n) {
        std::vector<uint64_t> result(n);
        for (size_t i = 0; i < n; ++i) {
            result[i] = dist_(rng_);
        }
        return result;
    }
    
    void reset(uint64_t seed) {
        rng_.seed(seed);
    }
    
private:
    std::mt19937_64 rng_;
    std::uniform_int_distribution<uint64_t> dist_;
};

// ============================================================================
// Comparison Utilities
// ============================================================================

inline bool compare_bfield(const BFieldElement& a, const BFieldElement& b) {
    return a.value() == b.value();
}

inline bool compare_xfield(const XFieldElement& a, const XFieldElement& b) {
    return compare_bfield(a.coeff(0), b.coeff(0)) &&
           compare_bfield(a.coeff(1), b.coeff(1)) &&
           compare_bfield(a.coeff(2), b.coeff(2));
}

inline bool compare_digest(const Digest& a, const Digest& b) {
    for (size_t i = 0; i < Digest::LEN; ++i) {
        if (!compare_bfield(a[i], b[i])) return false;
    }
    return true;
}

template<typename T, typename CompareFunc>
bool compare_vectors(
    const std::vector<T>& cpu_result,
    const std::vector<T>& gpu_result,
    size_t& first_mismatch,
    CompareFunc compare_fn
) {
    if (cpu_result.size() != gpu_result.size()) {
        first_mismatch = 0;
        return false;
    }
    
    for (size_t i = 0; i < cpu_result.size(); ++i) {
        if (!compare_fn(cpu_result[i], gpu_result[i])) {
            first_mismatch = i;
            return false;
        }
    }
    return true;
}

// ============================================================================
// Timing Utilities
// ============================================================================

class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }
    
    double stop_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start_);
        return duration.count() / 1000.0;
    }
    
private:
    std::chrono::time_point<std::chrono::high_resolution_clock> start_;
};

// ============================================================================
// Co-Verification Test Base Class
// ============================================================================

/**
 * Base class for CPU vs GPU co-verification tests.
 * 
 * Usage:
 *   class NttCoVerify : public CoVerifyTestBase {
 *       void SetUp() override {
 *           cpu_ = std::make_unique<CpuBackend>();
 *           gpu_ = std::make_unique<CudaBackend>();
 *       }
 *   };
 * 
 *   TEST_F(NttCoVerify, Forward) {
 *       auto input = gen_.random_bfield_vector(1024);
 *       auto result = verify_operation("NTT", "Forward", input,
 *           [&](auto& data) { cpu_->ntt_forward(data.data(), data.size()); },
 *           [&](auto& data) { gpu_->ntt_forward(data.data(), data.size()); }
 *       );
 *       EXPECT_TRUE(result.passed) << result.mismatch_details;
 *   }
 */
class CoVerifyTestBase : public ::testing::Test {
protected:
    void SetUp() override {
        cpu_ = std::make_unique<CpuBackend>();
        
        // Try to create CUDA backend, skip test if unavailable
        try {
            gpu_ = std::make_unique<CudaBackend>();
        } catch (const std::exception& e) {
            GTEST_SKIP() << "CUDA backend unavailable: " << e.what();
        }
    }
    
    /**
     * Verify that CPU and GPU produce the same result for an operation
     * that modifies a vector in-place.
     */
    template<typename T>
    CoVerifyResult verify_inplace_operation(
        const std::string& component,
        const std::string& test_name,
        std::vector<T> input,
        std::function<void(std::vector<T>&)> cpu_op,
        std::function<void(std::vector<T>&)> gpu_op,
        std::function<bool(const T&, const T&)> compare_fn
    ) {
        CoVerifyResult result;
        result.component_name = component;
        result.test_name = test_name;
        result.input_size = input.size();
        
        // Run CPU version
        auto cpu_data = input;
        Timer timer;
        timer.start();
        cpu_op(cpu_data);
        result.cpu_time_ms = timer.stop_ms();
        
        // Run GPU version
        auto gpu_data = input;
        timer.start();
        gpu_op(gpu_data);
        result.gpu_time_ms = timer.stop_ms();
        
        // Compare results
        result.passed = compare_vectors(cpu_data, gpu_data, 
                                        result.first_mismatch_index, compare_fn);
        
        if (result.passed) {
            result.speedup = result.cpu_time_ms / result.gpu_time_ms;
        } else {
            std::ostringstream oss;
            oss << "CPU[" << result.first_mismatch_index << "] != "
                << "GPU[" << result.first_mismatch_index << "]";
            result.mismatch_details = oss.str();
        }
        
        result.print();
        return result;
    }
    
    /**
     * Verify that CPU and GPU produce the same output for given input.
     */
    template<typename InputT, typename OutputT>
    CoVerifyResult verify_function(
        const std::string& component,
        const std::string& test_name,
        const InputT& input,
        std::function<OutputT(const InputT&)> cpu_fn,
        std::function<OutputT(const InputT&)> gpu_fn,
        std::function<bool(const OutputT&, const OutputT&)> compare_fn
    ) {
        CoVerifyResult result;
        result.component_name = component;
        result.test_name = test_name;
        result.input_size = 1;  // Or derive from input
        
        Timer timer;
        
        timer.start();
        auto cpu_result = cpu_fn(input);
        result.cpu_time_ms = timer.stop_ms();
        
        timer.start();
        auto gpu_result = gpu_fn(input);
        result.gpu_time_ms = timer.stop_ms();
        
        result.passed = compare_fn(cpu_result, gpu_result);
        
        if (result.passed) {
            result.speedup = result.cpu_time_ms / result.gpu_time_ms;
        } else {
            result.mismatch_details = "Output mismatch";
            result.first_mismatch_index = 0;
        }
        
        result.print();
        return result;
    }
    
    /**
     * Run verification across multiple sizes
     */
    template<typename T>
    void verify_across_sizes(
        const std::string& component,
        const std::string& test_name,
        const std::vector<size_t>& sizes,
        std::function<std::vector<T>(size_t)> input_generator,
        std::function<void(std::vector<T>&)> cpu_op,
        std::function<void(std::vector<T>&)> gpu_op,
        std::function<bool(const T&, const T&)> compare_fn
    ) {
        for (size_t n : sizes) {
            auto input = input_generator(n);
            auto result = verify_inplace_operation(
                component, test_name + "_n" + std::to_string(n),
                input, cpu_op, gpu_op, compare_fn
            );
            EXPECT_TRUE(result.passed) << result.mismatch_details;
        }
    }
    
protected:
    std::unique_ptr<CpuBackend> cpu_;
    std::unique_ptr<CudaBackend> gpu_;
    RandomGenerator gen_{42};
};

// ============================================================================
// Convenience Macros
// ============================================================================

#define CO_VERIFY_BFIELD_VECTOR(component, test, input, cpu_op, gpu_op) \
    verify_inplace_operation<BFieldElement>( \
        component, test, input, cpu_op, gpu_op, compare_bfield)

#define CO_VERIFY_XFIELD_VECTOR(component, test, input, cpu_op, gpu_op) \
    verify_inplace_operation<XFieldElement>( \
        component, test, input, cpu_op, gpu_op, compare_xfield)

#define CO_VERIFY_DIGEST_VECTOR(component, test, input, cpu_op, gpu_op) \
    verify_inplace_operation<Digest>( \
        component, test, input, cpu_op, gpu_op, compare_digest)

} // namespace co_verify
} // namespace triton_vm

