#include "merkle/merkle_tree.hpp"
#include <cmath>
#include <algorithm>
#include <unordered_set>
#include <unordered_map>
#include <cstdlib>
#include <iostream>
#include <omp.h>

namespace triton_vm {

MerkleTree::MerkleTree(const std::vector<Digest>& leaves) {
    if (leaves.empty()) {
        throw std::invalid_argument("Cannot create Merkle tree with no leaves");
    }
    
    // Check power of 2
    if ((leaves.size() & (leaves.size() - 1)) != 0) {
        throw std::invalid_argument("Number of leaves must be a power of 2");
    }
    
    num_leaves_ = leaves.size();
    
    // Allocate space for all nodes (2 * num_leaves)
    // Index 0 is unused, index 1 is root
    nodes_.resize(2 * num_leaves_);
    
    // Copy leaves to second half
    std::copy(leaves.begin(), leaves.end(), nodes_.begin() + num_leaves_);
    
    // Build the tree
    build_tree();
}

MerkleTree MerkleTree::parallel_new(const std::vector<Digest>& leaves) {
    // For now, same as sequential
    // TODO: Add parallel construction for large trees
    return MerkleTree(leaves);
}

void MerkleTree::build_tree() {
    // Build level by level from leaves up to root
    // Each level can be parallelized since nodes at the same level are independent
    size_t level_size = num_leaves_ / 2;  // Number of nodes to compute at current level
    size_t level_start = num_leaves_ / 2; // First node index at current level
    
    while (level_size >= 1) {
        // Parallel process all nodes at this level
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < level_size; ++i) {
            size_t node_idx = level_start + i;
            size_t left_child = 2 * node_idx;
            size_t right_child = 2 * node_idx + 1;
            nodes_[node_idx] = Tip5::hash_pair(nodes_[left_child], nodes_[right_child]);
        }
        
        // Move up one level
        level_start /= 2;
        level_size /= 2;
    }
}

Digest MerkleTree::root() const {
    return nodes_[1];
}

size_t MerkleTree::height() const {
    return static_cast<size_t>(std::log2(num_leaves_));
}

Digest MerkleTree::leaf(size_t index) const {
    if (index >= num_leaves_) {
        throw std::out_of_range("Leaf index out of range");
    }
    return nodes_[num_leaves_ + index];
}

Digest MerkleTree::node(size_t index) const {
    if (index == 0 || index >= nodes_.size()) {
        throw std::out_of_range("Node index out of range");
    }
    return nodes_[index];
}

size_t MerkleTree::sibling_index(size_t node_index) {
    return node_index ^ 1;
}

size_t MerkleTree::parent_index(size_t node_index) {
    return node_index / 2;
}

std::vector<Digest> MerkleTree::authentication_path(size_t leaf_index) const {
    if (leaf_index >= num_leaves_) {
        throw std::out_of_range("Leaf index out of range");
    }
    
    std::vector<Digest> path;
    size_t h = height();
    path.reserve(h);
    
    // Start from leaf node
    size_t current_index = num_leaves_ + leaf_index;
    
    // Walk up to root
    while (current_index > 1) {
        size_t sibling = sibling_index(current_index);
        path.push_back(nodes_[sibling]);
        current_index = parent_index(current_index);
    }
    
    return path;
}

std::vector<Digest> MerkleTree::authentication_structure(const std::vector<size_t>& leaf_indices) const {
    // Match Rust's authentication_structure_node_indices algorithm exactly
    // Algorithm:
    // 1. For each leaf, walk up to root, marking nodes as "can be computed" and siblings as "needed"
    // 2. Take set difference: needed - can_be_computed
    // 3. Return digests for the resulting nodes
    
    constexpr size_t ROOT_INDEX = 1;
    static bool debug_enabled = std::getenv("TVM_DEBUG_AUTH_STRUCT") != nullptr;
    
    // Set of node indices that are needed (siblings of nodes on paths)
    std::unordered_set<size_t> node_is_needed;
    
    // Set of node indices that can be computed (nodes on direct paths from leaves to root)
    std::unordered_set<size_t> node_can_be_computed;
    
    if (debug_enabled) {
        std::cout << "DEBUG: authentication_structure called with " << leaf_indices.size() 
                  << " leaf indices, num_leaves_=" << num_leaves_ << std::endl;
    }
    
    // For each leaf index, walk up to root
    for (size_t leaf_idx : leaf_indices) {
        if (leaf_idx >= num_leaves_) {
            throw std::out_of_range("Leaf index out of range");
        }
        
        // Start from leaf node (1-indexed: leaf_index + num_leaves_)
        size_t node_index = leaf_idx + num_leaves_;
        
        if (debug_enabled && leaf_idx < 5) {
            std::cout << "  Leaf " << leaf_idx << " -> node " << node_index << ":" << std::endl;
        }
        
        // Walk up to root
        while (node_index > ROOT_INDEX) {
            // Mark current node as "can be computed"
            node_can_be_computed.insert(node_index);
            
            // Mark sibling as "needed"
            size_t sibling = sibling_index(node_index);
            node_is_needed.insert(sibling);
            
            if (debug_enabled && leaf_idx < 5) {
                std::cout << "    node " << node_index << " (can compute), sibling " 
                          << sibling << " (needed)" << std::endl;
            }
            
            // Move to parent
            node_index = parent_index(node_index);
        }
    }
    
    if (debug_enabled) {
        std::cout << "  node_is_needed size: " << node_is_needed.size() << std::endl;
        std::cout << "  node_can_be_computed size: " << node_can_be_computed.size() << std::endl;

        // Count overlap
        size_t overlap_count = 0;
        for (size_t idx : node_is_needed) {
            if (node_can_be_computed.find(idx) != node_can_be_computed.end()) {
                overlap_count++;
            }
        }
        std::cout << "  Overlap (needed ∩ can_compute): " << overlap_count << std::endl;
        std::cout << "  Result size (needed - can_compute): " << (node_is_needed.size() - overlap_count) << std::endl;
    }
    
    // Compute set difference: needed - can_be_computed
    std::vector<size_t> result_indices;
    for (size_t idx : node_is_needed) {
        if (node_can_be_computed.find(idx) == node_can_be_computed.end()) {
            result_indices.push_back(idx);
        }
    }
    
    if (debug_enabled) {
        std::cout << "  Result indices size (needed - can_compute): " << result_indices.size() << std::endl;
        if (result_indices.size() <= 20) {
            std::cout << "  Result indices: ";
            for (size_t idx : result_indices) {
                std::cout << idx << " ";
            }
            std::cout << std::endl;
        } else {
            std::cout << "  First 10 result indices: ";
            for (size_t i = 0; i < 10 && i < result_indices.size(); ++i) {
                std::cout << result_indices[i] << " ";
            }
            std::cout << std::endl;
        }
    }
    
    // Sort in descending order (matching Rust's .sorted_unstable().rev())
    std::sort(result_indices.begin(), result_indices.end(), std::greater<size_t>());
    
    // Extract digests for result nodes
    std::vector<Digest> auth_structure;
    auth_structure.reserve(result_indices.size());
    for (size_t idx : result_indices) {
        if (idx >= nodes_.size()) {
            throw std::out_of_range("Node index out of range");
        }
        auth_structure.push_back(nodes_[idx]);
    }
    
    if (debug_enabled) {
        std::cout << "  Final auth_structure size: " << auth_structure.size() << std::endl;
    }
    
    return auth_structure;
}

std::vector<size_t> MerkleTree::authentication_structure_node_indices(
    size_t num_leaves,
    const std::vector<size_t>& leaf_indices
) {
    constexpr size_t ROOT_INDEX = 1;
    if (num_leaves == 0) {
        throw std::invalid_argument("num_leaves must be > 0");
    }
    if ((num_leaves & (num_leaves - 1)) != 0) {
        throw std::invalid_argument("num_leaves must be a power of 2");
    }

    std::unordered_set<size_t> node_is_needed;
    std::unordered_set<size_t> node_can_be_computed;

    for (size_t leaf_idx : leaf_indices) {
        if (leaf_idx >= num_leaves) {
            throw std::out_of_range("Leaf index out of range");
        }
        size_t node_index = leaf_idx + num_leaves;
        while (node_index > ROOT_INDEX) {
            node_can_be_computed.insert(node_index);
            node_is_needed.insert(sibling_index(node_index));
            node_index = parent_index(node_index);
        }
    }

    std::vector<size_t> result_indices;
    result_indices.reserve(node_is_needed.size());
    for (size_t idx : node_is_needed) {
        if (node_can_be_computed.find(idx) == node_can_be_computed.end()) {
            result_indices.push_back(idx);
        }
    }

    std::sort(result_indices.begin(), result_indices.end(), std::greater<size_t>());
    return result_indices;
}

bool MerkleTree::verify_authentication_structure(
    const Digest& root,
    size_t num_leaves,
    const std::vector<size_t>& leaf_indices,
    const std::vector<Digest>& leaves,
    const std::vector<Digest>& auth_structure
) {
    if (leaf_indices.size() != leaves.size()) {
        return false;
    }
    if (num_leaves == 0 || (num_leaves & (num_leaves - 1)) != 0) {
        return false;
    }
    if (num_leaves == 1) {
        // Tree of height 0: root equals the only leaf.
        if (leaf_indices.size() != 1 || leaf_indices[0] != 0) return false;
        return leaves[0] == root;
    }

    std::vector<size_t> auth_indices;
    try {
        auth_indices = authentication_structure_node_indices(num_leaves, leaf_indices);
    } catch (...) {
        return false;
    }
    if (auth_indices.size() != auth_structure.size()) {
        return false;
    }

    // Map known nodes -> digest (node indices are 1-indexed).
    std::unordered_map<size_t, Digest> known;
    known.reserve(leaf_indices.size() + auth_structure.size());

    for (size_t i = 0; i < leaf_indices.size(); ++i) {
        known[num_leaves + leaf_indices[i]] = leaves[i];
    }
    for (size_t i = 0; i < auth_structure.size(); ++i) {
        known[auth_indices[i]] = auth_structure[i];
    }

    // Propagate upwards until root is computed or no progress.
    bool progressed = true;
    while (progressed) {
        progressed = false;
        if (known.find(1) != known.end()) break;

        // Snapshot keys to iterate safely while inserting.
        std::vector<size_t> keys;
        keys.reserve(known.size());
        for (const auto& [k, _] : known) keys.push_back(k);

        for (size_t node_idx : keys) {
            if (node_idx <= 1) continue;
            size_t sibling = sibling_index(node_idx);
            auto it_sib = known.find(sibling);
            if (it_sib == known.end()) continue;

            size_t parent = parent_index(node_idx);
            if (known.find(parent) != known.end()) continue;

            const Digest& d_node = known[node_idx];
            const Digest& d_sib = it_sib->second;
            Digest parent_digest = (node_idx % 2 == 0)
                ? Tip5::hash_pair(d_node, d_sib)   // node is left child
                : Tip5::hash_pair(d_sib, d_node);  // node is right child

            known[parent] = parent_digest;
            progressed = true;
        }
    }

    auto it_root = known.find(1);
    if (it_root == known.end()) return false;
    return it_root->second == root;
}

bool MerkleTree::verify(
    const Digest& root,
    size_t leaf_index,
    const Digest& leaf,
    const std::vector<Digest>& auth_path
) {
    Digest current = leaf;
    size_t index = leaf_index;
    
    for (const Digest& sibling : auth_path) {
        if (index % 2 == 0) {
            // Current is left child
            current = Tip5::hash_pair(current, sibling);
        } else {
            // Current is right child
            current = Tip5::hash_pair(sibling, current);
        }
        index /= 2;
    }
    
    return current == root;
}

Digest hash_row(const std::vector<BFieldElement>& row) {
    if (row.empty()) {
        return Digest::zero();
    }
    
    // Hash the row using Tip5's variable-length hash
    return Tip5::hash_varlen(row);
}

} // namespace triton_vm

