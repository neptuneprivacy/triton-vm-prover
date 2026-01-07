#pragma once

#include "types/b_field_element.hpp"
#include "types/digest.hpp"
#include "table/master_table.hpp"
#include "hash/tip5.hpp"
#include <vector>

namespace triton_vm {

/**
 * Full table LDE computation and Merkle tree construction
 */
class LDETable {
public:
    /**
     * Perform LDE on entire table (all columns)
     * 
     * @param trace_table Input table on trace domain [rows × cols]
     * @param trace_domain Trace domain parameters
     * @param quotient_domain Quotient domain parameters
     * @param randomizer_coeffs Randomizer coefficients for each column [num_cols × num_randomizers]
     * @return Extended table on quotient domain [quotient_rows × cols]
     */
    static std::vector<std::vector<BFieldElement>> extend_table_with_randomizers(
        const std::vector<std::vector<BFieldElement>>& trace_table,
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain,
        const std::vector<std::vector<BFieldElement>>& randomizer_coeffs
    );
    
    /**
     * Hash all rows of LDE table and build Merkle tree
     * 
     * @param lde_table LDE table [rows × cols]
     * @return Pair of (row_hashes, merkle_root)
     */
    static std::pair<std::vector<Digest>, Digest> compute_merkle_tree(
        const std::vector<std::vector<BFieldElement>>& lde_table
    );
    
    /**
     * Complete LDE pipeline: extend table → hash rows → build Merkle tree
     * 
     * @param trace_table Input table
     * @param trace_domain Trace domain
     * @param quotient_domain Quotient domain
     * @param randomizer_coeffs Randomizer coefficients per column
     * @return Merkle root of LDE table
     */
    static Digest compute_lde_and_merkle(
        const std::vector<std::vector<BFieldElement>>& trace_table,
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain,
        const std::vector<std::vector<BFieldElement>>& randomizer_coeffs
    );
};

} // namespace triton_vm

