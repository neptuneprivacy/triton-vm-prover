#pragma once

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"
#include "merkle/merkle_tree.hpp"
#include "table/master_table.hpp"
#include <vector>
#include <memory>

namespace triton_vm {

/**
 * TableCommitment - Merkle tree commitment to a table
 * 
 * Commits to the rows of a table by hashing each row to a Digest
 * and building a Merkle tree over the row digests.
 */
class TableCommitment {
public:
    /**
     * Create a commitment to the given table (main table with BFieldElements).
     * 
     * @param table The table to commit to
     * @return TableCommitment with the Merkle root and tree
     */
    static TableCommitment commit(const MasterMainTable& table);
    
    /**
     * Create a commitment to the given table (aux table with XFieldElements).
     * 
     * @param table The auxiliary table to commit to
     * @return TableCommitment with the Merkle root and tree
     */
    static TableCommitment commit(const MasterAuxTable& table);
    
    /**
     * Create a commitment from pre-computed row digests.
     * 
     * @param row_digests The digest of each row (must be power of 2)
     * @return TableCommitment with the Merkle root and tree
     */
    static TableCommitment from_digests(const std::vector<Digest>& row_digests);
    
    // Get the Merkle root (commitment)
    Digest root() const;
    
    // Get the number of rows
    size_t num_rows() const;
    
    // Get authentication path for a specific row
    std::vector<Digest> authentication_path(size_t row_index) const;
    
    // Get authentication structure for multiple rows (batched)
    // Returns the combined authentication structure for all given indices
    std::vector<Digest> authentication_structure(const std::vector<size_t>& row_indices) const;
    
    // Verify an inclusion proof
    static bool verify(
        const Digest& root,
        size_t row_index,
        const Digest& row_digest,
        const std::vector<Digest>& auth_path
    );
    
    // Get the underlying Merkle tree
    const MerkleTree& tree() const { return *tree_; }

private:
    std::unique_ptr<MerkleTree> tree_;
    
    explicit TableCommitment(std::unique_ptr<MerkleTree> tree);
};

/**
 * Hash a row of BFieldElements to a Digest.
 * Uses Tip5 variable-length hash.
 */
Digest hash_bfield_row(const std::vector<BFieldElement>& row);

/**
 * Hash a row of XFieldElements to a Digest.
 * First flattens to BFieldElements, then hashes.
 */
Digest hash_xfield_row(const std::vector<XFieldElement>& row);

} // namespace triton_vm

