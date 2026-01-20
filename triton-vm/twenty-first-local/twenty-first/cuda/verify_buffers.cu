#include <cstdint>
#include "field_arithmetic.cuh"

// Kernel to verify two BField buffers are identical
// Returns number of mismatches found (0 = identical)
extern "C" __global__ void verify_buffers_bfield(
    const uint64_t* buffer_a,
    const uint64_t* buffer_b,
    uint64_t n,
    uint64_t* mismatch_count
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        if (buffer_a[idx] != buffer_b[idx]) {
            atomicAdd((unsigned long long*)mismatch_count, 1ULL);
        }
    }
}

// Kernel to verify two XField buffers are identical
// Each XField element consists of 3 BField elements (extension field)
// Returns number of mismatches found (0 = identical)
extern "C" __global__ void verify_buffers_xfield(
    const uint64_t* buffer_a,
    const uint64_t* buffer_b,
    uint64_t n,  // number of XField elements
    uint64_t* mismatch_count
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // Each XField element has 3 components
        uint64_t base_idx = idx * 3;

        bool mismatch = false;
        for (int i = 0; i < 3; i++) {
            if (buffer_a[base_idx + i] != buffer_b[base_idx + i]) {
                mismatch = true;
                break;
            }
        }

        if (mismatch) {
            atomicAdd((unsigned long long*)mismatch_count, 1ULL);
        }
    }
}

// Kernel to copy first few elements for debugging
extern "C" __global__ void copy_sample_bfield(
    const uint64_t* src,
    uint64_t* dst,
    uint64_t n
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        dst[idx] = src[idx];
    }
}

// Kernel to copy specific columns from src table to dst table
// Tables are row-major: data[row * num_columns + col]
extern "C" __global__ void copy_columns_bfield(
    const uint64_t* src,
    uint64_t* dst,
    uint64_t num_rows,
    uint64_t num_columns,
    uint64_t col_start,
    uint64_t col_end
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total_elements = num_rows * (col_end - col_start);

    if (idx < total_elements) {
        // Map linear idx to (row, col_offset)
        uint64_t col_offset = idx % (col_end - col_start);
        uint64_t row = idx / (col_end - col_start);

        // Calculate actual column index
        uint64_t col = col_start + col_offset;

        // Copy from src[row, col] to dst[row, col]
        uint64_t table_idx = row * num_columns + col;
        dst[table_idx] = src[table_idx];
    }
}

// Kernel to copy specific columns from src table to dst table (XField version)
// Each XField element has 3 components
extern "C" __global__ void copy_columns_xfield(
    const uint64_t* src,
    uint64_t* dst,
    uint64_t num_rows,
    uint64_t num_columns,
    uint64_t col_start,
    uint64_t col_end
) {
    uint64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint64_t total_elements = num_rows * (col_end - col_start);

    if (idx < total_elements) {
        // Map linear idx to (row, col_offset)
        uint64_t col_offset = idx % (col_end - col_start);
        uint64_t row = idx / (col_end - col_start);

        // Calculate actual column index
        uint64_t col = col_start + col_offset;

        // Copy all 3 components of XField element
        uint64_t table_idx = (row * num_columns + col) * 3;
        dst[table_idx] = src[table_idx];
        dst[table_idx + 1] = src[table_idx + 1];
        dst[table_idx + 2] = src[table_idx + 2];
    }
}
