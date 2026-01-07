#pragma once

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>
#include <cstdint>
#include "cuda_common.cuh"

namespace triton_vm {
namespace gpu {

// ============================================================================
// DeviceBFieldElement - GPU-compatible Goldilocks field element
// ============================================================================

struct DeviceBFieldElement {
    uint64_t value;
    
    __device__ __host__ DeviceBFieldElement() : value(0) {}
    __device__ __host__ explicit DeviceBFieldElement(uint64_t v) : value(v) {}
    
    // Arithmetic operators
    __device__ __forceinline__ DeviceBFieldElement operator+(const DeviceBFieldElement& other) const {
        return DeviceBFieldElement(add_mod(value, other.value));
    }
    
    __device__ __forceinline__ DeviceBFieldElement operator-(const DeviceBFieldElement& other) const {
        return DeviceBFieldElement(sub_mod(value, other.value));
    }
    
    __device__ __forceinline__ DeviceBFieldElement operator*(const DeviceBFieldElement& other) const {
        return DeviceBFieldElement(mul_mod(value, other.value));
    }
    
    __device__ __forceinline__ DeviceBFieldElement& operator+=(const DeviceBFieldElement& other) {
        value = add_mod(value, other.value);
        return *this;
    }
    
    __device__ __forceinline__ DeviceBFieldElement& operator-=(const DeviceBFieldElement& other) {
        value = sub_mod(value, other.value);
        return *this;
    }
    
    __device__ __forceinline__ DeviceBFieldElement& operator*=(const DeviceBFieldElement& other) {
        value = mul_mod(value, other.value);
        return *this;
    }
    
    __device__ __forceinline__ DeviceBFieldElement operator-() const {
        return DeviceBFieldElement(value == 0 ? 0 : GOLDILOCKS_PRIME - value);
    }
    
    __device__ __forceinline__ DeviceBFieldElement inverse() const {
        return DeviceBFieldElement(inv_mod(value));
    }
    
    __device__ __forceinline__ DeviceBFieldElement pow(uint64_t exp) const {
        return DeviceBFieldElement(pow_mod(value, exp));
    }
    
    __device__ __forceinline__ bool operator==(const DeviceBFieldElement& other) const {
        return value == other.value;
    }
    
    __device__ __forceinline__ bool operator!=(const DeviceBFieldElement& other) const {
        return value != other.value;
    }
    
    // Static constants
    __device__ __host__ static DeviceBFieldElement zero() {
        return DeviceBFieldElement(0);
    }
    
    __device__ __host__ static DeviceBFieldElement one() {
        return DeviceBFieldElement(1);
    }
};

// ============================================================================
// DeviceXFieldElement - GPU-compatible cubic extension field element
// ============================================================================

struct DeviceXFieldElement {
    DeviceBFieldElement coeffs[3];  // c0 + c1*x + c2*x^2 where x^3 = x + 1
    
    __device__ __host__ DeviceXFieldElement() {
        coeffs[0] = DeviceBFieldElement::zero();
        coeffs[1] = DeviceBFieldElement::zero();
        coeffs[2] = DeviceBFieldElement::zero();
    }
    
    __device__ __host__ explicit DeviceXFieldElement(DeviceBFieldElement c0) {
        coeffs[0] = c0;
        coeffs[1] = DeviceBFieldElement::zero();
        coeffs[2] = DeviceBFieldElement::zero();
    }
    
    __device__ __host__ DeviceXFieldElement(
        DeviceBFieldElement c0,
        DeviceBFieldElement c1,
        DeviceBFieldElement c2
    ) {
        coeffs[0] = c0;
        coeffs[1] = c1;
        coeffs[2] = c2;
    }
    
    __device__ __forceinline__ DeviceXFieldElement operator+(const DeviceXFieldElement& other) const {
        return DeviceXFieldElement(
            coeffs[0] + other.coeffs[0],
            coeffs[1] + other.coeffs[1],
            coeffs[2] + other.coeffs[2]
        );
    }
    
    __device__ __forceinline__ DeviceXFieldElement operator-(const DeviceXFieldElement& other) const {
        return DeviceXFieldElement(
            coeffs[0] - other.coeffs[0],
            coeffs[1] - other.coeffs[1],
            coeffs[2] - other.coeffs[2]
        );
    }
    
    /**
     * Extension field multiplication
     * (a0 + a1*x + a2*x^2) * (b0 + b1*x + b2*x^2)
     * where x^3 = x + 1
     */
    __device__ __forceinline__ DeviceXFieldElement operator*(const DeviceXFieldElement& other) const {
        // Standard schoolbook multiplication with reduction
        DeviceBFieldElement c0, c1, c2, c3, c4;
        
        c0 = coeffs[0] * other.coeffs[0];
        c1 = coeffs[0] * other.coeffs[1] + coeffs[1] * other.coeffs[0];
        c2 = coeffs[0] * other.coeffs[2] + coeffs[1] * other.coeffs[1] + coeffs[2] * other.coeffs[0];
        c3 = coeffs[1] * other.coeffs[2] + coeffs[2] * other.coeffs[1];
        c4 = coeffs[2] * other.coeffs[2];
        
        // Reduce: x^3 = x + 1
        // c3*x^3 = c3*(x + 1) = c3*x + c3
        // c4*x^4 = c4*x*(x + 1) = c4*x^2 + c4*x
        
        return DeviceXFieldElement(
            c0 + c3,           // constant term + c3 from x^3 reduction
            c1 + c4 + c3,      // x term + c4*x from x^4 + c3*x from x^3
            c2 + c4            // x^2 term + c4 from x^4
        );
    }
    
    __device__ __forceinline__ DeviceXFieldElement& operator+=(const DeviceXFieldElement& other) {
        coeffs[0] += other.coeffs[0];
        coeffs[1] += other.coeffs[1];
        coeffs[2] += other.coeffs[2];
        return *this;
    }
    
    __device__ __forceinline__ DeviceXFieldElement& operator*=(const DeviceXFieldElement& other) {
        *this = *this * other;
        return *this;
    }
    
    __device__ __forceinline__ DeviceXFieldElement operator-() const {
        return DeviceXFieldElement(-coeffs[0], -coeffs[1], -coeffs[2]);
    }
    
    __device__ __host__ static DeviceXFieldElement zero() {
        return DeviceXFieldElement();
    }
    
    __device__ __host__ static DeviceXFieldElement one() {
        return DeviceXFieldElement(DeviceBFieldElement::one());
    }
};

// ============================================================================
// DeviceDigest - GPU-compatible 5-element digest
// ============================================================================

struct DeviceDigest {
    uint64_t elements[5];
    
    __device__ __host__ DeviceDigest() {
        for (int i = 0; i < 5; ++i) {
            elements[i] = 0;
        }
    }
    
    __device__ __host__ bool operator==(const DeviceDigest& other) const {
        for (int i = 0; i < 5; ++i) {
            if (elements[i] != other.elements[i]) return false;
        }
        return true;
    }
};

} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

