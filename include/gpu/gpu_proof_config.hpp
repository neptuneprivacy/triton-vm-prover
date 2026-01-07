#pragma once

#include <cstddef>
#include <cstdint>
#include "quotient/quotient.hpp"

namespace triton_vm {
namespace gpu {

/**
 * GPU Proof Generation Configuration
 * 
 * Supports different modes based on input size:
 * - Input 8:  Full zero-copy (all data on GPU)
 * - Input 16: Full zero-copy (requires 8+ GB GPU)
 * - Input 21: Streaming mode (column batching, requires 24+ GB GPU)
 */
struct GpuProofConfig {
    // =========================================================================
    // Input Size Constants
    // =========================================================================
    
    // Test levels
    static constexpr uint32_t INPUT_FAST_TEST = 8;
    static constexpr uint32_t INPUT_DEVELOPMENT = 16;
    static constexpr uint32_t INPUT_PRODUCTION = 21;
    
    // Table dimensions (Triton VM constants)
    static constexpr size_t MAIN_WIDTH = 379;
    static constexpr size_t AUX_WIDTH = 88;
    // Derive from Rust: NUM_QUOTIENT_SEGMENTS = air::TARGET_DEGREE = 4
    static constexpr size_t QUOTIENT_SEGMENTS = Quotient::NUM_QUOTIENT_SEGMENTS;
    static constexpr size_t DIGEST_LEN = 5;
    static constexpr size_t XFE_LEN = 3;  // XFieldElement = 3 BFieldElements
    
    // FRI parameters
    static constexpr size_t FRI_EXPANSION_FACTOR = 8;
    static constexpr size_t FRI_NUM_COLLINEARITY_CHECKS = 80;
    static constexpr size_t FRI_LAST_CODEWORD_LEN = 512;
    
    // =========================================================================
    // Configuration
    // =========================================================================
    
    uint32_t log2_padded_height;  // The input value (8, 16, or 21)
    bool streaming_mode;          // True for input 21, false for 8/16
    size_t column_batch_size;     // Number of columns to process at once in streaming mode
    size_t max_gpu_memory_bytes;  // Maximum GPU memory to use
    
    // =========================================================================
    // Computed Values
    // =========================================================================
    
    size_t padded_height() const {
        return 1ULL << log2_padded_height;
    }
    
    size_t fri_domain_length() const {
        return padded_height() * FRI_EXPANSION_FACTOR;
    }
    
    size_t num_fri_rounds() const {
        size_t len = fri_domain_length();
        size_t rounds = 0;
        while (len > FRI_LAST_CODEWORD_LEN) {
            len /= 2;
            ++rounds;
        }
        return rounds;
    }
    
    // =========================================================================
    // Memory Estimates
    // =========================================================================
    
    /**
     * Estimate total GPU memory needed for full zero-copy mode
     */
    size_t estimate_full_memory_bytes() const {
        size_t h = padded_height();
        size_t f = fri_domain_length();
        
        size_t total = 0;
        
        // Main table (trace + LDE)
        total += h * MAIN_WIDTH * sizeof(uint64_t);
        total += f * MAIN_WIDTH * sizeof(uint64_t);
        
        // Aux table (trace + LDE, XFE)
        total += h * AUX_WIDTH * XFE_LEN * sizeof(uint64_t);
        total += f * AUX_WIDTH * XFE_LEN * sizeof(uint64_t);
        
        // Quotient segments
        total += f * QUOTIENT_SEGMENTS * XFE_LEN * sizeof(uint64_t);
        
        // Merkle trees (2 * leaves * digest)
        total += 3 * 2 * f * DIGEST_LEN * sizeof(uint64_t);
        
        // FRI data (roughly equal to one fri_domain)
        total += f * XFE_LEN * sizeof(uint64_t);
        total += f * DIGEST_LEN * sizeof(uint64_t);
        
        // Scratch space (2 buffers)
        total += 2 * f * MAIN_WIDTH * sizeof(uint64_t);
        
        // Proof buffer
        total += 10 * 1024 * 1024;  // 10 MB
        
        return total;
    }
    
    /**
     * Estimate GPU memory needed for streaming mode
     */
    size_t estimate_streaming_memory_bytes() const {
        size_t f = fri_domain_length();
        
        size_t total = 0;
        
        // Column batch for LDE (column_batch_size columns)
        total += f * column_batch_size * sizeof(uint64_t);
        
        // Merkle accumulator
        total += 2 * f * DIGEST_LEN * sizeof(uint64_t);
        
        // FRI working set
        total += 2 * f * XFE_LEN * sizeof(uint64_t);
        
        // Quotient segments
        total += f * QUOTIENT_SEGMENTS * XFE_LEN * sizeof(uint64_t);
        
        // Scratch space
        total += 2 * f * column_batch_size * sizeof(uint64_t);
        
        // Proof buffer
        total += 10 * 1024 * 1024;
        
        return total;
    }
    
    // =========================================================================
    // Factory Methods
    // =========================================================================
    
    /**
     * Create config for fast testing (input 8)
     */
    static GpuProofConfig fast_test() {
        GpuProofConfig cfg;
        cfg.log2_padded_height = INPUT_FAST_TEST;
        cfg.streaming_mode = false;
        cfg.column_batch_size = MAIN_WIDTH;  // All columns at once
        cfg.max_gpu_memory_bytes = 4ULL * 1024 * 1024 * 1024;  // 4 GB
        return cfg;
    }
    
    /**
     * Create config for development (input 16)
     */
    static GpuProofConfig development() {
        GpuProofConfig cfg;
        cfg.log2_padded_height = INPUT_DEVELOPMENT;
        cfg.streaming_mode = false;
        cfg.column_batch_size = MAIN_WIDTH;  // All columns at once
        cfg.max_gpu_memory_bytes = 12ULL * 1024 * 1024 * 1024;  // 12 GB
        return cfg;
    }
    
    /**
     * Create config for production (input 21)
     */
    static GpuProofConfig production() {
        GpuProofConfig cfg;
        cfg.log2_padded_height = INPUT_PRODUCTION;
        cfg.streaming_mode = true;
        cfg.column_batch_size = 64;  // Process 64 columns at a time
        cfg.max_gpu_memory_bytes = 24ULL * 1024 * 1024 * 1024;  // 24 GB
        return cfg;
    }
    
    /**
     * Create config from input value, auto-detecting mode
     */
    static GpuProofConfig from_input(uint32_t input) {
        GpuProofConfig cfg;
        cfg.log2_padded_height = input;
        
        // Auto-detect streaming mode based on memory requirements
        cfg.streaming_mode = false;
        cfg.column_batch_size = MAIN_WIDTH;
        cfg.max_gpu_memory_bytes = 24ULL * 1024 * 1024 * 1024;
        
        size_t full_memory = cfg.estimate_full_memory_bytes();
        
        // If full memory exceeds 16 GB, switch to streaming mode
        if (full_memory > 16ULL * 1024 * 1024 * 1024) {
            cfg.streaming_mode = true;
            cfg.column_batch_size = 64;
        }
        
        return cfg;
    }
    
    /**
     * Print configuration summary
     */
    void print() const;
};

// ============================================================================
// Memory Layout Constants
// ============================================================================

/**
 * Input 8 memory layout (all fits in GPU)
 */
struct Input8Layout {
    static constexpr size_t PADDED_HEIGHT = 256;
    static constexpr size_t FRI_LENGTH = 2048;
    static constexpr size_t MAIN_TRACE_SIZE = PADDED_HEIGHT * 379 * 8;     // 776 KB
    static constexpr size_t MAIN_LDE_SIZE = FRI_LENGTH * 379 * 8;          // 6.2 MB
    static constexpr size_t AUX_TRACE_SIZE = PADDED_HEIGHT * 88 * 3 * 8;   // 528 KB
    static constexpr size_t AUX_LDE_SIZE = FRI_LENGTH * 88 * 3 * 8;        // 4.2 MB
    static constexpr size_t QUOTIENT_SIZE = FRI_LENGTH * QUOTIENT_SEGMENTS * 3 * 8;        // 196 KB
    static constexpr size_t MERKLE_SIZE = 2 * FRI_LENGTH * 5 * 8;          // 164 KB
    static constexpr size_t TOTAL_APPROXIMATE = 35 * 1024 * 1024;          // ~35 MB
};

/**
 * Input 16 memory layout (fits in 12 GB GPU)
 */
struct Input16Layout {
    static constexpr size_t PADDED_HEIGHT = 65536;
    static constexpr size_t FRI_LENGTH = 524288;
    static constexpr size_t MAIN_TRACE_SIZE = PADDED_HEIGHT * 379 * 8;     // 199 MB
    static constexpr size_t MAIN_LDE_SIZE = FRI_LENGTH * 379 * 8;          // 1.6 GB
    static constexpr size_t AUX_TRACE_SIZE = PADDED_HEIGHT * 88 * 3 * 8;   // 138 MB
    static constexpr size_t AUX_LDE_SIZE = FRI_LENGTH * 88 * 3 * 8;        // 1.1 GB
    static constexpr size_t QUOTIENT_SIZE = FRI_LENGTH * QUOTIENT_SEGMENTS * 3 * 8;        // 50 MB
    static constexpr size_t MERKLE_SIZE = 2 * FRI_LENGTH * 5 * 8;          // 42 MB
    static constexpr size_t TOTAL_APPROXIMATE = 6400ULL * 1024 * 1024;     // ~6.4 GB
};

/**
 * Input 21 memory layout (streaming mode required)
 */
struct Input21Layout {
    static constexpr size_t PADDED_HEIGHT = 2097152;
    static constexpr size_t FRI_LENGTH = 16777216;
    static constexpr size_t MAIN_TRACE_SIZE = PADDED_HEIGHT * 379 * 8;     // 6.4 GB
    static constexpr size_t MAIN_LDE_SIZE = FRI_LENGTH * 379 * 8;          // 51 GB
    
    // Streaming mode: process in batches
    static constexpr size_t COLUMN_BATCH = 64;
    static constexpr size_t BATCH_LDE_SIZE = FRI_LENGTH * COLUMN_BATCH * 8; // 8.6 GB per batch
    static constexpr size_t STREAMING_TOTAL = 24ULL * 1024 * 1024 * 1024;  // Target 24 GB
};

} // namespace gpu
} // namespace triton_vm

