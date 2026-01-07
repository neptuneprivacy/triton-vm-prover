#pragma once

#include <memory>
#include <vector>
#include <string>
#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"

namespace triton_vm {

// Forward declarations
class MasterMainTable;
class MasterAuxTable;
struct Challenges;
struct ArithmeticDomain;

/**
 * Backend type enumeration
 */
enum class BackendType {
    CPU,    // Pure C++ implementation (reference)
    CUDA    // NVIDIA CUDA GPU implementation
};

/**
 * Convert string to BackendType
 */
inline BackendType backend_from_string(const std::string& s) {
    if (s == "cuda" || s == "CUDA" || s == "gpu" || s == "GPU") {
        return BackendType::CUDA;
    }
    return BackendType::CPU;
}

/**
 * Get backend type from environment variable TRITON_BACKEND
 */
inline BackendType backend_from_env() {
    const char* env = std::getenv("TRITON_BACKEND");
    if (env) {
        return backend_from_string(env);
    }
    return BackendType::CPU;
}

/**
 * Abstract backend interface for compute operations.
 * 
 * This allows switching between CPU and GPU implementations
 * transparently, enabling co-verification testing.
 */
class Backend {
public:
    virtual ~Backend() = default;
    
    /**
     * Get the backend type
     */
    virtual BackendType type() const = 0;
    
    /**
     * Get backend name for logging
     */
    virtual std::string name() const = 0;
    
    // =========================================================================
    // NTT Operations
    // =========================================================================
    
    /**
     * Forward NTT (Number Theoretic Transform) in-place
     * @param data Array of BFieldElements (must be power of 2)
     * @param n Number of elements
     */
    virtual void ntt_forward(BFieldElement* data, size_t n) = 0;
    
    /**
     * Inverse NTT in-place
     * @param data Array of BFieldElements (must be power of 2)
     * @param n Number of elements
     */
    virtual void ntt_inverse(BFieldElement* data, size_t n) = 0;
    
    /**
     * Batch NTT on multiple independent arrays
     * @param data Array of pointers to BFieldElement arrays
     * @param n Size of each array
     * @param batch_size Number of arrays
     */
    virtual void ntt_batch(BFieldElement** data, size_t n, size_t batch_size) = 0;
    
    // =========================================================================
    // Hash Operations (Tip5)
    // =========================================================================
    
    /**
     * Tip5 permutation on a batch of states
     * @param states Array of 16-element states (flattened)
     * @param num_states Number of states
     */
    virtual void tip5_permutation_batch(uint64_t* states, size_t num_states) = 0;
    
    /**
     * Hash pairs of digests (for Merkle tree construction)
     * @param left Left child digests
     * @param right Right child digests  
     * @param output Parent digests
     * @param count Number of pairs
     */
    virtual void hash_pairs(
        const Digest* left,
        const Digest* right,
        Digest* output,
        size_t count
    ) = 0;
    
    // =========================================================================
    // Merkle Tree Operations
    // =========================================================================
    
    /**
     * Build complete Merkle tree from leaves
     * @param leaves Leaf digests
     * @param num_leaves Number of leaves (must be power of 2)
     * @return Root digest
     */
    virtual Digest merkle_root(const Digest* leaves, size_t num_leaves) = 0;
    
    /**
     * Build Merkle tree and return all nodes
     * @param leaves Leaf digests
     * @param num_leaves Number of leaves
     * @param tree Output: all tree nodes (size = 2*num_leaves - 1)
     */
    virtual void merkle_tree_full(
        const Digest* leaves,
        size_t num_leaves,
        Digest* tree
    ) = 0;
    
    // =========================================================================
    // LDE (Low Degree Extension) Operations
    // =========================================================================
    
    /**
     * Low degree extension for a single column
     * @param trace_column Input trace values
     * @param trace_len Length of trace
     * @param extended_column Output extended values
     * @param extended_len Length of extension (must be >= trace_len)
     * @param trace_offset Coset offset for trace domain
     * @param extended_offset Coset offset for extended domain
     */
    virtual void lde_column(
        const BFieldElement* trace_column,
        size_t trace_len,
        BFieldElement* extended_column,
        size_t extended_len,
        BFieldElement trace_offset,
        BFieldElement extended_offset
    ) = 0;
    
    /**
     * Batch LDE for multiple columns
     * @param columns Array of column pointers
     * @param num_columns Number of columns
     * @param trace_len Length of each trace column
     * @param extended Array of output column pointers
     * @param extended_len Length of extended columns
     * @param trace_offset Coset offset for trace domain
     * @param extended_offset Coset offset for extended domain
     */
    virtual void lde_batch(
        const BFieldElement** columns,
        size_t num_columns,
        size_t trace_len,
        BFieldElement** extended,
        size_t extended_len,
        BFieldElement trace_offset,
        BFieldElement extended_offset
    ) = 0;
    
    // =========================================================================
    // Quotient Operations
    // =========================================================================
    
    /**
     * Evaluate AIR constraints at multiple points
     * @param main_rows Main table rows (flattened, row-major)
     * @param aux_rows Auxiliary table rows (flattened, row-major)
     * @param num_rows Number of evaluation points
     * @param challenges Challenge values
     * @param outputs Constraint evaluation outputs
     */
    virtual void evaluate_constraints_batch(
        const BFieldElement* main_rows,
        const XFieldElement* aux_rows,
        size_t num_rows,
        const XFieldElement* challenges,
        XFieldElement* outputs
    ) = 0;
    
    // =========================================================================
    // FRI Operations
    // =========================================================================
    
    /**
     * FRI folding step
     * @param codeword Input codeword
     * @param codeword_len Length of input
     * @param challenge Folding challenge
     * @param folded Output folded codeword (half length)
     */
    virtual void fri_fold(
        const XFieldElement* codeword,
        size_t codeword_len,
        const XFieldElement& challenge,
        XFieldElement* folded
    ) = 0;
    
    // =========================================================================
    // Factory
    // =========================================================================
    
    /**
     * Create a backend instance
     * @param type Backend type to create
     * @return Unique pointer to backend
     */
    static std::unique_ptr<Backend> create(BackendType type);
    
    /**
     * Create backend from environment variable
     */
    static std::unique_ptr<Backend> create_from_env() {
        return create(backend_from_env());
    }
};

} // namespace triton_vm

