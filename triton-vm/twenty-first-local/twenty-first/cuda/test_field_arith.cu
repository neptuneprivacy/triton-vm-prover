#include "field_arithmetic.cuh"
#include <stdio.h>

// Test kernel to verify field arithmetic operations
extern "C" __global__ void test_field_operations(
    const u64* test_values,  // Input test values (pairs of u64)
    u64* add_results,
    u64* mul_results,
    u64* sub_results,
    int num_tests
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_tests) return;

    BFieldElement a = bfield_from_raw(test_values[idx * 2]);
    BFieldElement b = bfield_from_raw(test_values[idx * 2 + 1]);

    BFieldElement sum = bfield_add(a, b);
    BFieldElement product = bfield_mul(a, b);
    BFieldElement difference = bfield_sub(a, b);

    add_results[idx] = sum.value;
    mul_results[idx] = product.value;
    sub_results[idx] = difference.value;

    // Print first test for debugging
    if (idx == 0) {
        printf("GPU: a=%llu, b=%llu\\n", a.value, b.value);
        printf("GPU: a+b=%llu, a*b=%llu, a-b=%llu\\n",
               sum.value, product.value, difference.value);
    }
}
