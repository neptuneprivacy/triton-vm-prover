#pragma once

#include "types/digest.hpp"
#include "hash/tip5.hpp"
#include <vector>
#include <stdexcept>

namespace triton_vm {

/**
 * MerkleTree - Binary tree of digests for commitment
 * 
 * A Merkle tree commits to a set of leaves and allows efficient
 * inclusion proofs. Uses Tip5 hash function.
 */
class MerkleTree {
public:
    /**
     * Build a Merkle tree from the given leaves.
     * 
     * @param leaves The leaves of the tree (must be power of 2)
     * @throws std::invalid_argument if leaves is empty or not power of 2
     */
    explicit MerkleTree(const std::vector<Digest>& leaves);
    
    /**
     * Build a Merkle tree in parallel (for large trees).
     * For now, same as sequential constructor.
     */
    static MerkleTree parallel_new(const std::vector<Digest>& leaves);
    
    // Get the root digest
    Digest root() const;
    
    // Get the number of leaves
    size_t num_leaves() const { return num_leaves_; }
    
    // Get the height of the tree (log2 of num_leaves)
    size_t height() const;
    
    // Get a leaf by index
    Digest leaf(size_t index) const;
    
    // Get authentication path for a leaf
    std::vector<Digest> authentication_path(size_t leaf_index) const;
    
    // Get authentication structure for multiple leaves (batched)
    // This is the combined set of digests needed to verify all given leaves
    std::vector<Digest> authentication_structure(const std::vector<size_t>& leaf_indices) const;
    
    // Verify a leaf with authentication path
    static bool verify(
        const Digest& root,
        size_t leaf_index,
        const Digest& leaf,
        const std::vector<Digest>& auth_path
    );

    // Verify multiple leaves using a compact authentication structure.
    //
    // The `auth_structure` is the output of `authentication_structure(leaf_indices)`.
    // Verifier recomputes the corresponding node indices deterministically from `leaf_indices`,
    // then reconstructs the root by hashing upwards.
    static bool verify_authentication_structure(
        const Digest& root,
        size_t num_leaves,
        const std::vector<size_t>& leaf_indices,
        const std::vector<Digest>& leaves,
        const std::vector<Digest>& auth_structure
    );
    
    // Get the sibling index of a node
    static size_t sibling_index(size_t node_index);
    
    // Get the parent index of a node
    static size_t parent_index(size_t node_index);

private:
    std::vector<Digest> nodes_;
    size_t num_leaves_;
    
    // Build the tree from leaves
    void build_tree();
    
    // Get node by index (1-indexed, root = 1)
    Digest node(size_t index) const;

    // Deterministically compute the node indices that form the authentication structure
    // for the given leaf indices, sorted descending (matches Rust).
    static std::vector<size_t> authentication_structure_node_indices(
        size_t num_leaves,
        const std::vector<size_t>& leaf_indices
    );
};

/**
 * Hash a row of BFieldElements to a Digest for Merkle tree leaves.
 */
Digest hash_row(const std::vector<BFieldElement>& row);

} // namespace triton_vm

