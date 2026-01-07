#pragma once

#include "types/b_field_element.hpp"
#include "types/x_field_element.hpp"
#include "types/digest.hpp"
#include "polynomial/polynomial.hpp"
#include <vector>
#include <array>
#include <map>
#include <optional>

namespace triton_vm {

// Forward declarations
class Challenges;
class MasterAuxTable;
class AlgebraicExecutionTrace;

// Forward declaration for degree lowering computation (for testing)
void compute_degree_lowering_aux_columns(
    const std::vector<std::vector<BFieldElement>>& main_data,
    std::vector<std::vector<XFieldElement>>& aux_data,
    const Challenges& challenges);

/**
 * Table dimensions and column counts matching the Rust implementation
 */
struct TableDimensions {
    static constexpr size_t PROCESSOR_TABLE_WIDTH = 39;
    static constexpr size_t OP_STACK_TABLE_WIDTH = 4;
    static constexpr size_t RAM_TABLE_WIDTH = 7;
    static constexpr size_t JUMP_STACK_TABLE_WIDTH = 5;
    static constexpr size_t HASH_TABLE_WIDTH = 67;
    static constexpr size_t CASCADE_TABLE_WIDTH = 6;
    static constexpr size_t LOOKUP_TABLE_WIDTH = 4;
    static constexpr size_t U32_TABLE_WIDTH = 10;
    static constexpr size_t PROGRAM_TABLE_WIDTH = 7;
    
    // Total main columns (without degree lowering)
    static constexpr size_t TOTAL_MAIN_COLUMNS = 
        PROCESSOR_TABLE_WIDTH + OP_STACK_TABLE_WIDTH + RAM_TABLE_WIDTH +
        JUMP_STACK_TABLE_WIDTH + HASH_TABLE_WIDTH + CASCADE_TABLE_WIDTH +
        LOOKUP_TABLE_WIDTH + U32_TABLE_WIDTH + PROGRAM_TABLE_WIDTH;
};

/**
 * ArithmeticDomain - Domain for polynomial operations
 */
struct ArithmeticDomain {
    size_t length;
    BFieldElement offset;
    BFieldElement generator;
    
    static ArithmeticDomain of_length(size_t length);
    ArithmeticDomain with_offset(BFieldElement offset) const;
    ArithmeticDomain halve() const;
    BFieldElement element(size_t index) const;
    std::vector<BFieldElement> values() const;
};

/**
 * ProverDomains - All domains used during proving
 */
struct ProverDomains {
    ArithmeticDomain trace;
    ArithmeticDomain randomized_trace;
    ArithmeticDomain quotient;
    ArithmeticDomain fri;
    
    static ProverDomains derive(
        size_t padded_height,
        size_t num_trace_randomizers,
        const ArithmeticDomain& fri_domain,
        int64_t max_degree
    );
};

/**
 * MasterMainTable - Main execution trace table
 *
 * Contains all main columns from all AIR tables.
 */
class MasterMainTable {
public:
    MasterMainTable(size_t num_rows, size_t num_columns);

    // Constructor with domains (for LDE support)
    MasterMainTable(
        size_t num_rows,
        size_t num_columns,
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain);

    MasterMainTable(
        size_t num_rows,
        size_t num_columns,
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain,
        const ArithmeticDomain& fri_domain);

    MasterMainTable(
        size_t num_rows,
        size_t num_columns,
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain,
        const ArithmeticDomain& fri_domain,
        const std::array<uint8_t, 32>& trace_randomizer_seed);

    // Accessors
    size_t num_rows() const { return num_rows_; }
    size_t num_columns() const { return num_columns_; }
    const ArithmeticDomain& trace_domain() const { return trace_domain_; }
    const ArithmeticDomain& quotient_domain() const { return quotient_domain_; }
    const ArithmeticDomain& fri_domain() const { return fri_domain_; }
    
    // Row access
    const std::vector<BFieldElement>& row(size_t i) const;
    BFieldElement get(size_t row, size_t col) const;
    void set(size_t row, size_t col, BFieldElement value);
    
    // Padding
    void pad(size_t padded_height);
    void pad(size_t padded_height, const std::array<size_t, 9>& table_lengths);
    
    // Create MasterMainTable from AET (main table creation step)
    // This fills all tables from the AlgebraicExecutionTrace
    // NOTE: Currently only fills processor table; other tables will be filled incrementally
    static MasterMainTable from_aet(
        const AlgebraicExecutionTrace& aet,
        const ProverDomains& domains,
        size_t num_trace_randomizers,
        const std::array<uint8_t, 32>& trace_randomizer_seed
    );
    
    // Low-degree extension
    void low_degree_extend(const ArithmeticDomain& domain);
    bool has_lde() const { return !lde_table_.empty(); }
    const std::vector<std::vector<BFieldElement>>& lde_table() const { return lde_table_; }
    void set_lde_table(std::vector<std::vector<BFieldElement>> lde_table) { lde_table_ = std::move(lde_table); }

    // Trace randomizers (for zero-knowledge)
    void set_trace_randomizer_seed(const std::array<uint8_t, 32>& seed) { trace_randomizer_seed_ = seed; }
    const std::array<uint8_t, 32>& trace_randomizer_seed() const { return trace_randomizer_seed_; }
    void set_num_trace_randomizers(size_t num) { num_trace_randomizers_ = num; }
    size_t num_trace_randomizers() const { return num_trace_randomizers_; }
    
    // Set pre-computed randomizer coefficients for a column (from Rust test data)
    // This allows matching Rust-generated coefficients when RNG implementations differ
    void set_trace_randomizer_coefficients(size_t column_idx, const std::vector<BFieldElement>& coefficients);
    
    // Check if pre-computed coefficients are available for a column
    bool has_trace_randomizer_coefficients(size_t column_idx) const;
    
    std::vector<BFieldElement> trace_randomizer_for_column(size_t column_idx) const;

    // Extend to create MasterAuxTable (after Fiat-Shamir challenges)
    MasterAuxTable extend(const Challenges& challenges, 
                          const std::optional<std::vector<std::vector<XFieldElement>>>& randomizer_values = std::nullopt) const;
    
    // Get table slice for extending (helper for extend functions)
    std::vector<std::vector<BFieldElement>> get_table_slice(size_t start_col, size_t num_cols) const;
    
    // Out-of-domain row evaluation (barycentric Lagrangian interpolation)
    // Evaluates the table at an arbitrary point (not necessarily in the domain)
    // Returns XFieldElement row (for consistency with Rust, even though main table is BFieldElement)
    std::vector<XFieldElement> out_of_domain_row(const XFieldElement& indeterminate) const;
    
    // Weighted sum of columns - creates a polynomial from weighted sum of columns
    // Returns XFieldElement polynomial (matching Rust's weighted_sum_of_columns)
    Polynomial<XFieldElement> weighted_sum_of_columns(const std::vector<XFieldElement>& weights) const;
    
    // Pre-computed FRI domain digests (for GPU-accelerated hashing)
    void set_fri_digests(std::vector<Digest> digests) { fri_digests_ = std::move(digests); }
    bool has_fri_digests() const { return !fri_digests_.empty(); }
    const std::vector<Digest>& fri_digests() const { return fri_digests_; }
    
    // FLAT BUFFER MODE - Direct GPU-ready memory layout
    // When enabled, data is stored in row-major flat buffer instead of vector<vector>
    void enable_flat_buffer(bool use_pinned_memory = false);
    bool is_flat_buffer() const { return use_flat_buffer_; }
    
    // Direct access to flat buffer (row-major: data[row * num_cols + col])
    uint64_t* flat_data() { return flat_data_; }
    const uint64_t* flat_data() const { return flat_data_; }
    
    // Fast flat buffer access (no bounds checking)
    // Cast before multiply to prevent 32-bit overflow (critical for large inputs like input21)
    void set_flat(size_t row, size_t col, BFieldElement value) {
        flat_data_[static_cast<size_t>(row) * static_cast<size_t>(num_columns_) + static_cast<size_t>(col)] = value.value();
    }
    BFieldElement get_flat(size_t row, size_t col) const {
        return BFieldElement(flat_data_[static_cast<size_t>(row) * static_cast<size_t>(num_columns_) + static_cast<size_t>(col)]);
    }
    
    // Destructor (needed for pinned memory cleanup)
    ~MasterMainTable();
    
    // Move constructor/assignment (handle pinned memory correctly)
    MasterMainTable(MasterMainTable&& other) noexcept;
    MasterMainTable& operator=(MasterMainTable&& other) noexcept;
    
    // Disable copy (pinned memory can't be copied easily)
    MasterMainTable(const MasterMainTable&) = delete;
    MasterMainTable& operator=(const MasterMainTable&) = delete;

private:
    size_t num_rows_;
    size_t num_columns_;
    std::vector<std::vector<BFieldElement>> data_;
    
    // Flat buffer mode (GPU-optimized)
    bool use_flat_buffer_ = false;
    bool flat_data_pinned_ = false;
    uint64_t* flat_data_ = nullptr;

    // Domains for LDE
    ArithmeticDomain trace_domain_;
    ArithmeticDomain quotient_domain_;
    ArithmeticDomain fri_domain_;

    // Trace randomizer seed (for ZK)
    std::array<uint8_t, 32> trace_randomizer_seed_;
    
    // Number of trace randomizers (coefficients per column)
    size_t num_trace_randomizers_;
    
    // Map from column index to pre-computed randomizer coefficients (from Rust test data)
    // This allows matching Rust-generated coefficients when RNG implementations differ
    std::map<size_t, std::vector<BFieldElement>> precomputed_randomizer_coefficients_;

    // Low-degree extended table
    std::vector<std::vector<BFieldElement>> lde_table_;
    
    // Pre-computed FRI domain digests (from GPU hashing)
    std::vector<Digest> fri_digests_;
};

/**
 * MasterAuxTable - Auxiliary (extension) table
 * 
 * Contains all auxiliary columns using XFieldElements.
 */
class MasterAuxTable {
public:
    MasterAuxTable(size_t num_rows, size_t num_columns);

    MasterAuxTable(
        size_t num_rows,
        size_t num_columns,
        const ArithmeticDomain& trace_domain,
        const ArithmeticDomain& quotient_domain,
        const ArithmeticDomain& fri_domain);
    
    // Accessors
    size_t num_rows() const { return num_rows_; }
    size_t num_columns() const { return num_columns_; }
    
    // Row access
    const std::vector<XFieldElement>& row(size_t i) const;
    XFieldElement get(size_t row, size_t col) const;
    void set(size_t row, size_t col, XFieldElement value);

    // Domain access
    const ArithmeticDomain& trace_domain() const { return trace_domain_; }
    const ArithmeticDomain& quotient_domain() const { return quotient_domain_; }
    const ArithmeticDomain& fri_domain() const { return fri_domain_; }
    
    // LDE utilities
    void low_degree_extend(const ArithmeticDomain& target_domain);
    bool has_low_degree_extension() const { return !lde_table_.empty(); }
    const std::vector<std::vector<XFieldElement>>& lde_table() const { return lde_table_; }
    void clear_low_degree_extension();
    
    // Trace randomizer support (for randomized LDE)
    void set_trace_randomizer_seed(const std::array<uint8_t, 32>& seed) { trace_randomizer_seed_ = seed; }
    const std::array<uint8_t, 32>& trace_randomizer_seed() const { return trace_randomizer_seed_; }
    void set_num_trace_randomizers(size_t num) { num_trace_randomizers_ = num; }
    size_t num_trace_randomizers() const { return num_trace_randomizers_; }
    
    // Set pre-computed randomizer coefficients for a column (from Rust test data)
    // This allows matching Rust-generated coefficients when RNG implementations differ
    void set_trace_randomizer_coefficients(size_t column_idx, const std::vector<BFieldElement>& coefficients);
    
    // Set pre-computed XFieldElement randomizer coefficients (for aux table - full XFieldElement)
    void set_trace_randomizer_xfield_coefficients(size_t column_idx, const std::vector<XFieldElement>& coefficients);
    
    // Check if pre-computed coefficients are available for a column
    bool has_trace_randomizer_coefficients(size_t column_idx) const;
    
    // Get XFieldElement randomizer coefficients (for aux table)
    std::vector<XFieldElement> trace_randomizer_xfield_for_column(size_t column_idx) const;
    
    std::vector<BFieldElement> trace_randomizer_for_column(size_t column_idx) const;
    
    // For extend functions: get mutable reference to internal data
    std::vector<std::vector<XFieldElement>>& data_mut() { return data_; }
    const std::vector<std::vector<XFieldElement>>& data() const { return data_; }
    
    // Out-of-domain row evaluation (barycentric Lagrangian interpolation)
    // Evaluates the table at an arbitrary point (not necessarily in the domain)
    std::vector<XFieldElement> out_of_domain_row(const XFieldElement& indeterminate) const;
    
    // Weighted sum of columns - creates a polynomial from weighted sum of columns
    // Returns XFieldElement polynomial (matching Rust's weighted_sum_of_columns)
    Polynomial<XFieldElement> weighted_sum_of_columns(const std::vector<XFieldElement>& weights) const;

private:
    size_t num_rows_;
    size_t num_columns_;
    std::vector<std::vector<XFieldElement>> data_;

    ArithmeticDomain trace_domain_;
    ArithmeticDomain quotient_domain_;
    ArithmeticDomain fri_domain_;

    // Trace randomizer support (for randomized LDE)
    std::array<uint8_t, 32> trace_randomizer_seed_ = {0};
    size_t num_trace_randomizers_ = 0;
    
    // Map from column index to pre-computed randomizer coefficients (from Rust test data)
    // This allows matching Rust-generated coefficients when RNG implementations differ
    std::map<size_t, std::vector<BFieldElement>> precomputed_randomizer_coefficients_;
    
    // Map from column index to pre-computed XFieldElement randomizer coefficients (for aux table)
    std::map<size_t, std::vector<XFieldElement>> precomputed_xfield_randomizer_coefficients_;

    std::vector<std::vector<XFieldElement>> lde_table_;
    size_t lde_domain_length_ = 0;
};

} // namespace triton_vm

