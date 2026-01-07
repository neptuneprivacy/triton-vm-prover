#include "table/table_commitment.hpp"
#include "hash/tip5.hpp"

namespace triton_vm {

TableCommitment::TableCommitment(std::unique_ptr<MerkleTree> tree)
    : tree_(std::move(tree)) {}

TableCommitment TableCommitment::commit(const MasterMainTable& table) {
    std::vector<Digest> row_digests;
    row_digests.reserve(table.num_rows());
    
    for (size_t i = 0; i < table.num_rows(); ++i) {
        row_digests.push_back(hash_bfield_row(table.row(i)));
    }
    
    return from_digests(row_digests);
}

TableCommitment TableCommitment::commit(const MasterAuxTable& table) {
    std::vector<Digest> row_digests;
    row_digests.reserve(table.num_rows());
    
    for (size_t i = 0; i < table.num_rows(); ++i) {
        row_digests.push_back(hash_xfield_row(table.row(i)));
    }
    
    return from_digests(row_digests);
}

TableCommitment TableCommitment::from_digests(const std::vector<Digest>& row_digests) {
    auto tree = std::make_unique<MerkleTree>(row_digests);
    return TableCommitment(std::move(tree));
}

Digest TableCommitment::root() const {
    return tree_->root();
}

size_t TableCommitment::num_rows() const {
    return tree_->num_leaves();
}

std::vector<Digest> TableCommitment::authentication_path(size_t row_index) const {
    return tree_->authentication_path(row_index);
}

std::vector<Digest> TableCommitment::authentication_structure(const std::vector<size_t>& row_indices) const {
    return tree_->authentication_structure(row_indices);
}

bool TableCommitment::verify(
    const Digest& root,
    size_t row_index,
    const Digest& row_digest,
    const std::vector<Digest>& auth_path
) {
    return MerkleTree::verify(root, row_index, row_digest, auth_path);
}

Digest hash_bfield_row(const std::vector<BFieldElement>& row) {
    return Tip5::hash_varlen(row);
}

Digest hash_xfield_row(const std::vector<XFieldElement>& row) {
    // Flatten XFieldElements to BFieldElements (3 per XFieldElement)
    std::vector<BFieldElement> flat;
    flat.reserve(row.size() * 3);
    
    for (const auto& x : row) {
        flat.push_back(x.coeff(0));
        flat.push_back(x.coeff(1));
        flat.push_back(x.coeff(2));
    }
    
    return Tip5::hash_varlen(flat);
}

} // namespace triton_vm

