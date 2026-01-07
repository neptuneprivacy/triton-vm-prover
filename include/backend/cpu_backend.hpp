#pragma once

#include "backend/backend.hpp"
#include "ntt/ntt.hpp"
#include "hash/tip5.hpp"
#include "merkle/merkle_tree.hpp"
#include <cstring>

namespace triton_vm {

/**
 * CPU backend implementation using existing C++ code.
 * This serves as the reference implementation for co-verification.
 */
class CpuBackend : public Backend {
public:
    CpuBackend() = default;
    ~CpuBackend() override = default;
    
    BackendType type() const override { return BackendType::CPU; }
    std::string name() const override { return "CPU"; }
    
    // =========================================================================
    // NTT Operations
    // =========================================================================
    
    void ntt_forward(BFieldElement* data, size_t n) override {
        std::vector<BFieldElement> vec(data, data + n);
        NTT::forward(vec);
        std::memcpy(data, vec.data(), n * sizeof(BFieldElement));
    }
    
    void ntt_inverse(BFieldElement* data, size_t n) override {
        std::vector<BFieldElement> vec(data, data + n);
        NTT::inverse(vec);
        std::memcpy(data, vec.data(), n * sizeof(BFieldElement));
    }
    
    void ntt_batch(BFieldElement** data, size_t n, size_t batch_size) override {
        for (size_t i = 0; i < batch_size; ++i) {
            std::vector<BFieldElement> vec(data[i], data[i] + n);
            NTT::forward(vec);
            std::memcpy(data[i], vec.data(), n * sizeof(BFieldElement));
        }
    }
    
    // =========================================================================
    // Hash Operations
    // =========================================================================
    
    void tip5_permutation_batch(uint64_t* states, size_t num_states) override {
        for (size_t i = 0; i < num_states; ++i) {
            Tip5 tip5;
            // Copy state elements
            for (size_t j = 0; j < Tip5::STATE_SIZE; ++j) {
                tip5.state[j] = BFieldElement(states[i * Tip5::STATE_SIZE + j]);
            }
            tip5.permutation();
            // Copy back
            for (size_t j = 0; j < Tip5::STATE_SIZE; ++j) {
                states[i * Tip5::STATE_SIZE + j] = tip5.state[j].value();
            }
        }
    }
    
    void hash_pairs(
        const Digest* left,
        const Digest* right,
        Digest* output,
        size_t count
    ) override {
        for (size_t i = 0; i < count; ++i) {
            output[i] = Tip5::hash_pair(left[i], right[i]);
        }
    }
    
    // =========================================================================
    // Merkle Tree Operations
    // =========================================================================
    
    Digest merkle_root(const Digest* leaves, size_t num_leaves) override {
        std::vector<Digest> leaves_vec(leaves, leaves + num_leaves);
        MerkleTree tree(leaves_vec);
        return tree.root();
    }
    
    void merkle_tree_full(
        const Digest* leaves,
        size_t num_leaves,
        Digest* tree
    ) override {
        std::vector<Digest> leaves_vec(leaves, leaves + num_leaves);
        MerkleTree mt(leaves_vec);
        
        // Copy leaves to bottom half
        std::memcpy(tree + num_leaves - 1, leaves, num_leaves * sizeof(Digest));
        
        // Build tree bottom-up
        for (size_t i = num_leaves - 2; i < num_leaves; --i) {
            size_t left_idx = 2 * i + 1;
            size_t right_idx = 2 * i + 2;
            tree[i] = Tip5::hash_pair(tree[left_idx], tree[right_idx]);
        }
    }
    
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
    ) override {
        // Interpolate over trace domain
        std::vector<BFieldElement> coeffs(trace_column, trace_column + trace_len);
        
        // Coset INTT: unscale by offset^{-i}
        BFieldElement offset_inv = trace_offset.inverse();
        BFieldElement pow = BFieldElement::one();
        for (size_t i = 0; i < trace_len; ++i) {
            coeffs[i] = coeffs[i] * pow;
            pow = pow * offset_inv;
        }
        NTT::inverse(coeffs);
        
        // Pad to extended length
        coeffs.resize(extended_len, BFieldElement::zero());
        
        // Coset NTT with extended offset
        NTT::forward(coeffs);
        pow = BFieldElement::one();
        for (size_t i = 0; i < extended_len; ++i) {
            extended_column[i] = coeffs[i] * pow;
            pow = pow * extended_offset;
        }
    }
    
    void lde_batch(
        const BFieldElement** columns,
        size_t num_columns,
        size_t trace_len,
        BFieldElement** extended,
        size_t extended_len,
        BFieldElement trace_offset,
        BFieldElement extended_offset
    ) override {
        for (size_t c = 0; c < num_columns; ++c) {
            lde_column(columns[c], trace_len, extended[c], extended_len,
                      trace_offset, extended_offset);
        }
    }
    
    // =========================================================================
    // Quotient Operations
    // =========================================================================
    
    void evaluate_constraints_batch(
        const BFieldElement* main_rows,
        const XFieldElement* aux_rows,
        size_t num_rows,
        const XFieldElement* challenges,
        XFieldElement* outputs
    ) override {
        // Uses Rust FFI for constraint evaluation (existing implementation)
        // This will be replaced with pure C++/CUDA implementation
        (void)main_rows;
        (void)aux_rows;
        (void)num_rows;
        (void)challenges;
        (void)outputs;
        throw std::runtime_error("evaluate_constraints_batch not yet implemented in CPU backend");
    }
    
    // =========================================================================
    // FRI Operations
    // =========================================================================
    
    void fri_fold(
        const XFieldElement* codeword,
        size_t codeword_len,
        const XFieldElement& challenge,
        XFieldElement* folded
    ) override {
        size_t half_len = codeword_len / 2;
        for (size_t i = 0; i < half_len; ++i) {
            // Standard FRI folding: f'(x) = (f(x) + f(-x))/2 + challenge * (f(x) - f(-x))/(2x)
            const XFieldElement& left = codeword[i];
            const XFieldElement& right = codeword[i + half_len];
            
            XFieldElement sum = left + right;
            XFieldElement diff = left - right;
            
            // Simplified folding (assuming standard evaluation domain structure)
            folded[i] = sum + challenge * diff;
        }
    }
};

} // namespace triton_vm

