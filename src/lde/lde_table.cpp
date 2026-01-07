#include "lde/lde_table.hpp"
#include "lde/lde_randomized.hpp"
#include "merkle/merkle_tree.hpp"
#include "hash/tip5.hpp"

namespace triton_vm {

std::vector<std::vector<BFieldElement>> LDETable::extend_table_with_randomizers(
    const std::vector<std::vector<BFieldElement>>& trace_table,
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain,
    const std::vector<std::vector<BFieldElement>>& randomizer_coeffs
) {
    if (trace_table.empty()) {
        return {};
    }
    
    size_t num_rows = trace_table.size();
    size_t num_cols = trace_table[0].size();
    size_t output_rows = quotient_domain.length;
    
    // Validate randomizer coefficients
    if (randomizer_coeffs.size() != num_cols) {
        throw std::invalid_argument("Randomizer coefficients count must match column count");
    }
    
    // Initialize output table
    std::vector<std::vector<BFieldElement>> lde_table(output_rows);
    for (size_t r = 0; r < output_rows; r++) {
        lde_table[r].resize(num_cols);
    }
    
    // Process each column
    for (size_t c = 0; c < num_cols; c++) {
        // Extract column from trace table
        std::vector<BFieldElement> trace_column(num_rows);
        for (size_t r = 0; r < num_rows; r++) {
            trace_column[r] = trace_table[r][c];
        }
        
        // Get randomizer coefficients for this column
        const auto& col_randomizer = randomizer_coeffs[c];
        
        // Extend column with randomizer
        std::vector<BFieldElement> lde_column = RandomizedLDE::extend_column_with_randomizer(
            trace_column, trace_domain, quotient_domain, col_randomizer
        );
        
        // Copy to output table
        for (size_t r = 0; r < output_rows; r++) {
            lde_table[r][c] = lde_column[r];
        }
    }
    
    return lde_table;
}

std::pair<std::vector<Digest>, Digest> LDETable::compute_merkle_tree(
    const std::vector<std::vector<BFieldElement>>& lde_table
) {
    if (lde_table.empty()) {
        return {{}, Digest()};
    }
    
    size_t num_rows = lde_table.size();
    size_t num_cols = lde_table[0].size();
    
    // Hash each row
    Tip5 hasher;
    std::vector<Digest> row_hashes;
    row_hashes.reserve(num_rows);
    
    for (size_t r = 0; r < num_rows; r++) {
        // Hash the row
        Digest row_hash = hasher.hash_varlen(lde_table[r]);
        row_hashes.push_back(row_hash);
    }
    
    // Build Merkle tree
    MerkleTree tree(row_hashes);
    Digest root = tree.root();
    
    return {row_hashes, root};
}

Digest LDETable::compute_lde_and_merkle(
    const std::vector<std::vector<BFieldElement>>& trace_table,
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain,
    const std::vector<std::vector<BFieldElement>>& randomizer_coeffs
) {
    // Step 1: Extend table
    auto lde_table = extend_table_with_randomizers(
        trace_table, trace_domain, quotient_domain, randomizer_coeffs
    );
    
    // Step 2: Build Merkle tree
    auto [row_hashes, root] = compute_merkle_tree(lde_table);
    
    return root;
}

} // namespace triton_vm

