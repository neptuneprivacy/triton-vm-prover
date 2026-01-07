#pragma once

#include "backend/backend.hpp"

#ifdef TRITON_CUDA_ENABLED

#include <cuda_runtime.h>

namespace triton_vm {

/**
 * CUDA GPU backend implementation.
 * Provides GPU-accelerated implementations of all compute operations.
 */
class CudaBackend : public Backend {
public:
    CudaBackend();
    ~CudaBackend() override;
    
    BackendType type() const override { return BackendType::CUDA; }
    std::string name() const override { return "CUDA"; }
    
    // =========================================================================
    // NTT Operations
    // =========================================================================
    
    void ntt_forward(BFieldElement* data, size_t n) override;
    void ntt_inverse(BFieldElement* data, size_t n) override;
    void ntt_batch(BFieldElement** data, size_t n, size_t batch_size) override;
    
    // =========================================================================
    // Hash Operations
    // =========================================================================
    
    void tip5_permutation_batch(uint64_t* states, size_t num_states) override;
    void hash_pairs(
        const Digest* left,
        const Digest* right,
        Digest* output,
        size_t count
    ) override;
    
    // =========================================================================
    // Merkle Tree Operations
    // =========================================================================
    
    Digest merkle_root(const Digest* leaves, size_t num_leaves) override;
    void merkle_tree_full(
        const Digest* leaves,
        size_t num_leaves,
        Digest* tree
    ) override;
    
    // =========================================================================
    // LDE Operations
    // =========================================================================
    
    void lde_column(
        const BFieldElement* trace_column,
        size_t trace_len,
        BFieldElement* extended_column,
        size_t extended_len,
        BFieldElement trace_offset,
        BFieldElement extended_offset
    ) override;
    
    void lde_batch(
        const BFieldElement** columns,
        size_t num_columns,
        size_t trace_len,
        BFieldElement** extended,
        size_t extended_len,
        BFieldElement trace_offset,
        BFieldElement extended_offset
    ) override;
    
    // =========================================================================
    // Quotient Operations
    // =========================================================================
    
    void evaluate_constraints_batch(
        const BFieldElement* main_rows,
        const XFieldElement* aux_rows,
        size_t num_rows,
        const XFieldElement* challenges,
        XFieldElement* outputs
    ) override;
    
    // =========================================================================
    // FRI Operations
    // =========================================================================
    
    void fri_fold(
        const XFieldElement* codeword,
        size_t codeword_len,
        const XFieldElement& challenge,
        XFieldElement* folded
    ) override;
    
    // =========================================================================
    // CUDA-specific utilities
    // =========================================================================
    
    /**
     * Get CUDA device properties
     */
    static void print_device_info();
    
    /**
     * Synchronize all CUDA operations
     */
    void synchronize();
    
    /**
     * Check last CUDA error
     */
    static void check_error(const char* context);
    
private:
    // CUDA stream for async operations
    cudaStream_t stream_;
    
    // Device memory pools (implementation detail)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace triton_vm

#else // !TRITON_CUDA_ENABLED

namespace triton_vm {

/**
 * Stub CUDA backend when CUDA is not available.
 * All operations throw runtime errors.
 */
class CudaBackend : public Backend {
public:
    CudaBackend() {
        throw std::runtime_error("CUDA backend not available - compile with ENABLE_CUDA=ON");
    }
    
    BackendType type() const override { return BackendType::CUDA; }
    std::string name() const override { return "CUDA (unavailable)"; }
    
    void ntt_forward(BFieldElement*, size_t) override { throw_unavailable(); }
    void ntt_inverse(BFieldElement*, size_t) override { throw_unavailable(); }
    void ntt_batch(BFieldElement**, size_t, size_t) override { throw_unavailable(); }
    void tip5_permutation_batch(uint64_t*, size_t) override { throw_unavailable(); }
    void hash_pairs(const Digest*, const Digest*, Digest*, size_t) override { throw_unavailable(); }
    Digest merkle_root(const Digest*, size_t) override { throw_unavailable(); return Digest(); }
    void merkle_tree_full(const Digest*, size_t, Digest*) override { throw_unavailable(); }
    void lde_column(const BFieldElement*, size_t, BFieldElement*, size_t, BFieldElement, BFieldElement) override { throw_unavailable(); }
    void lde_batch(const BFieldElement**, size_t, size_t, BFieldElement**, size_t, BFieldElement, BFieldElement) override { throw_unavailable(); }
    void evaluate_constraints_batch(const BFieldElement*, const XFieldElement*, size_t, const XFieldElement*, XFieldElement*) override { throw_unavailable(); }
    void fri_fold(const XFieldElement*, size_t, const XFieldElement&, XFieldElement*) override { throw_unavailable(); }
    
private:
    [[noreturn]] void throw_unavailable() {
        throw std::runtime_error("CUDA backend not available");
    }
};

} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

