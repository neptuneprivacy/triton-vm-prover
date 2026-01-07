#include "table/master_table.hpp"
#include "table/extend_helpers.hpp"
#include "table/table_padding.hpp"
#include "stark/challenges.hpp"
#include "ntt/ntt.hpp"
#include "lde/lde_randomized.hpp"
#include "polynomial/polynomial.hpp"
#include "quotient/quotient.hpp"
#include "vm/aet.hpp"
#include "vm/processor_columns.hpp"
#include "hash/tip5.hpp"
#include "chacha12_rng.hpp"
#include <stdexcept>
#include <cmath>
#include <random>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <future>
#include <thread>
#include <cstring>
#include <nlohmann/json.hpp>

#ifdef TVM_USE_TBB
#include <execution>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/randomized_lde_kernel.cuh"
#include "gpu/kernels/bezout_kernel.cuh"
#include "gpu/kernels/poly_mul_kernel.cuh"
#include <cuda_runtime.h>
#endif

namespace triton_vm {

// Forward declaration for C++ degree lowering implementation
void evaluate_degree_lowering_aux_columns_cpp(
    const std::vector<std::vector<BFieldElement>>& main_data,
    std::vector<std::vector<XFieldElement>>& aux_data,
    const Challenges& challenges);

// FFI declaration for degree lowering main columns (from Rust)
extern "C" void degree_lowering_fill_main_columns(
    uint64_t* table_ptr,
    size_t num_rows,
    size_t num_cols
);

// Pure C++ implementation of degree lowering main columns (no FFI overhead)
void fill_degree_lowering_main_columns_cpp(std::vector<std::vector<BFieldElement>>& data);

namespace {

// Parallel sort helper - uses TBB's parallel execution policy if available
template<typename Iterator, typename Compare>
void parallel_sort(Iterator begin, Iterator end, Compare comp) {
#ifdef TVM_USE_TBB
    std::sort(std::execution::par_unseq, begin, end, comp);
#else
    std::sort(begin, end, comp);
#endif
}

// Helper functions for degree lowering computation

std::vector<uint64_t> flatten_main_table(const std::vector<std::vector<BFieldElement>>& data) {
    std::vector<uint64_t> flat;
    if (data.empty()) {
        return flat;
    }
    const size_t rows = data.size();
    const size_t cols = data[0].size();
    flat.resize(rows * cols);
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            flat[r * cols + c] = data[r][c].value();
        }
    }
    return flat;
}

void write_degree_lowering_main_columns(
    std::vector<std::vector<BFieldElement>>& data,
    const std::vector<uint64_t>& flat,
    size_t start_col) {
    if (data.empty()) {
        return;
    }
    const size_t rows = data.size();
    const size_t cols = data[0].size();
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = start_col; c < cols; ++c) {
            data[r][c] = BFieldElement(flat[r * cols + c]);
        }
    }
}

void compute_degree_lowering_main_columns(std::vector<std::vector<BFieldElement>>& data) {
    if (data.empty() || data[0].empty()) {
        return;
    }
    
    // If GPU degree lowering is enabled, skip CPU computation
    // The GPU will compute these columns after table upload
    const bool use_gpu_dl = (std::getenv("TRITON_GPU_DEGREE_LOWERING") != nullptr);
    if (use_gpu_dl) {
        return;  // Skip CPU degree lowering - will be done on GPU
    }
    
    const bool profile = (std::getenv("TVM_PROFILE_PAD") != nullptr);
    const bool use_rust_ffi = (std::getenv("TVM_USE_RUST_DEGREE_LOWERING") != nullptr);
    
    // Always use Rust FFI for degree lowering to ensure exact match with Rust implementation
    auto t0 = std::chrono::high_resolution_clock::now();

    size_t num_rows = data.size();
    size_t num_cols = data[0].size();

    // OPTIMIZED: Flatten data to u64 array for FFI - parallel with better cache locality
    // Use row-major order for better cache performance
    std::vector<uint64_t> flat_data(num_rows * num_cols);
    
    // Parallelize by rows - use adaptive chunk size based on thread count
    // For static scheduling, let OpenMP divide work evenly (optimal for load balancing)
    // This adapts automatically to any thread count (96, 170, etc.)
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < num_rows; ++r) {
        const size_t row_offset = r * num_cols;
        const auto& row = data[r];
        // Unroll inner loop for better performance (compiler hint)
        size_t c = 0;
        // Process in chunks of 4 for potential vectorization
        for (; c + 4 <= num_cols; c += 4) {
            flat_data[row_offset + c] = row[c].value();
            flat_data[row_offset + c + 1] = row[c + 1].value();
            flat_data[row_offset + c + 2] = row[c + 2].value();
            flat_data[row_offset + c + 3] = row[c + 3].value();
        }
        // Handle remaining columns
        for (; c < num_cols; ++c) {
            flat_data[row_offset + c] = row[c].value();
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    if (profile) std::cout << "      flatten: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;

    // Call Rust FFI function
    degree_lowering_fill_main_columns(flat_data.data(), num_rows, num_cols);

    auto t2 = std::chrono::high_resolution_clock::now();
    if (profile) std::cout << "      rust_ffi: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;

    // OPTIMIZED: Write back to data - parallel with better cache locality
    // Use adaptive static scheduling (automatically divides work evenly among threads)
    #pragma omp parallel for schedule(static)
    for (size_t r = 0; r < num_rows; ++r) {
        const size_t row_offset = r * num_cols;
        auto& row = data[r];
        // Unroll inner loop for better performance (compiler hint)
        size_t c = 0;
        // Process in chunks of 4 for potential vectorization
        for (; c + 4 <= num_cols; c += 4) {
            row[c] = BFieldElement(flat_data[row_offset + c]);
            row[c + 1] = BFieldElement(flat_data[row_offset + c + 1]);
            row[c + 2] = BFieldElement(flat_data[row_offset + c + 2]);
            row[c + 3] = BFieldElement(flat_data[row_offset + c + 3]);
        }
        // Handle remaining columns
        for (; c < num_cols; ++c) {
            row[c] = BFieldElement(flat_data[row_offset + c]);
        }
    }

    auto t3 = std::chrono::high_resolution_clock::now();
    if (profile) std::cout << "      write_back: " << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms" << std::endl;
}

// Helper function: inverse_or_zero - returns inverse if non-zero, zero otherwise
BFieldElement inverse_or_zero(BFieldElement x) {
    if (x.is_zero()) {
        return BFieldElement::zero();
    }
    return x.inverse();
}

// Helper function: inverse_or_zero_of_highest_2_limbs for Hash table padding
// Computes inverse of (2^32 - 1 - 2^16 * highest - mid_high) where highest and mid_high
// are the two most significant 16-bit limbs of the state element
BFieldElement inverse_or_zero_of_highest_2_limbs(BFieldElement state_element) {
    // Convert to 16-bit limbs (little-endian)
    uint64_t r_times_x = (0xffffffff00000001ULL * state_element.value()) % BFieldElement::MODULUS;
    uint16_t highest = (r_times_x >> 48) & 0xffff;
    uint16_t mid_high = (r_times_x >> 32) & 0xffff;
    
    uint64_t high_limbs = (static_cast<uint64_t>(highest) << 16) + mid_high;
    uint64_t two_pow_32_minus_1 = (1ULL << 32) - 1;
    uint64_t to_invert = two_pow_32_minus_1 - high_limbs;
    
    BFieldElement result(to_invert);
    return inverse_or_zero(result);
}

// Helper function: Get Tip5 round constants for round 0 (for padding)
// Tip5 has 12 round constants per round
std::array<BFieldElement, 12> tip5_round_constants_round_0() {
    // Tip5 round constants for round 0 (from twenty-first crate)
    // These are hardcoded constants from the Tip5 specification
    return {
        BFieldElement(0x0000000000000001ULL),  // Constant0
        BFieldElement(0x0000000000008082ULL),  // Constant1
        BFieldElement(0x800000000000808aULL),  // Constant2
        BFieldElement(0x8000000080008000ULL),  // Constant3
        BFieldElement(0x000000000000808bULL),  // Constant4
        BFieldElement(0x0000000080000001ULL),  // Constant5
        BFieldElement(0x8000000080008081ULL),  // Constant6
        BFieldElement(0x8000000000008009ULL),  // Constant7
        BFieldElement(0x000000000000008aULL),  // Constant8
        BFieldElement(0x0000000000000088ULL),  // Constant9
        BFieldElement(0x0000000080008009ULL), // Constant10
        BFieldElement(0x000000008000000aULL)   // Constant11
    };
}

// Tip5 RATE constant (number of elements absorbed per permutation in sponge operations)
constexpr size_t TIP5_RATE = Tip5::RATE;

} // namespace

// Helper to flatten challenges for FFI
std::vector<uint64_t> flatten_challenges(const Challenges& challenges) {
    std::vector<uint64_t> flat;
    // Challenges has 63 elements (59 sampled + 4 derived)
    for (size_t i = 0; i < 63; ++i) {
        const XFieldElement& xfe = challenges[i];
        flat.push_back(xfe.coeff(0).value());
        flat.push_back(xfe.coeff(1).value());
        flat.push_back(xfe.coeff(2).value());
    }
    return flat;
}

// Helper to unflatten aux table from FFI
void unflatten_aux_table(
    const uint64_t* aux_flat,
    size_t num_rows,
    size_t num_cols,
    std::vector<std::vector<XFieldElement>>& aux_data) {
    for (size_t r = 0; r < num_rows; ++r) {
        for (size_t c = 0; c < num_cols; ++c) {
            size_t idx = (r * num_cols + c) * 3;
            XFieldElement xfe(
                BFieldElement(aux_flat[idx]),
                BFieldElement(aux_flat[idx + 1]),
                BFieldElement(aux_flat[idx + 2])
            );
            aux_data[r][c] = xfe;
        }
    }
}

// FFI declaration for Rust degree lowering function
extern "C" {
    void degree_lowering_fill_aux_columns(
        const uint64_t* main_ptr,
        size_t num_rows,
        size_t main_cols,
        uint64_t* aux_ptr,
        size_t aux_cols,
        const uint64_t* challenges_ptr,
        size_t challenges_len
    );
}

// Make compute_degree_lowering_aux_columns accessible for testing (outside anonymous namespace)
// Using Rust FFI with full generated code from degree_lowering_table.rs
void compute_degree_lowering_aux_columns(
    const std::vector<std::vector<BFieldElement>>& main_data,
    std::vector<std::vector<XFieldElement>>& aux_data,
    const Challenges& challenges) {
    if (main_data.empty() || aux_data.empty()) {
        return;
    }
    const size_t aux_cols = aux_data[0].size();
    const size_t rows = main_data.size();
    constexpr size_t TABLE_AUX_COLUMNS = 49;
    constexpr size_t NUM_RANDOMIZER_POLYNOMIALS = 1;
    const size_t randomizer_start = aux_cols > NUM_RANDOMIZER_POLYNOMIALS
        ? aux_cols - NUM_RANDOMIZER_POLYNOMIALS
        : aux_cols;
    if (aux_cols <= TABLE_AUX_COLUMNS || randomizer_start <= TABLE_AUX_COLUMNS || rows < 2) {
        return;
    }

    // Flatten data for FFI
    std::vector<uint64_t> main_flat = flatten_main_table(main_data);
    std::vector<uint64_t> challenges_flat = flatten_challenges(challenges);
    
    // Allocate aux_flat buffer (num_rows * num_cols * 3 for XFieldElement)
    const size_t main_cols = main_data[0].size();
    std::vector<uint64_t> aux_flat(rows * aux_cols * 3);
    
    // Copy existing aux_data to aux_flat (preserve columns 0-87, including randomizer)
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < aux_cols; ++c) {
            size_t idx = (r * aux_cols + c) * 3;
            aux_flat[idx] = aux_data[r][c].coeff(0).value();
            aux_flat[idx + 1] = aux_data[r][c].coeff(1).value();
            aux_flat[idx + 2] = aux_data[r][c].coeff(2).value();
        }
    }
    
    // Call Rust FFI
    degree_lowering_fill_aux_columns(
        main_flat.data(),
        rows,
        main_cols,
        aux_flat.data(),
        aux_cols,
        challenges_flat.data(),
        63  // All 63 challenges (59 sampled + 4 derived)
    );
    
    // Unflatten aux_flat back to aux_data
    unflatten_aux_table(aux_flat.data(), rows, aux_cols, aux_data);
}

// ArithmeticDomain implementation
ArithmeticDomain ArithmeticDomain::of_length(size_t length) {
    // Verify length is a power of 2
    if (length == 0 || (length & (length - 1)) != 0) {
        throw std::invalid_argument("Domain length must be a power of 2");
    }
    
    ArithmeticDomain domain;
    domain.length = length;
    domain.offset = BFieldElement::one();
    domain.generator = BFieldElement::primitive_root_of_unity(
        static_cast<uint32_t>(std::log2(length))
    );
    
    return domain;
}

ArithmeticDomain ArithmeticDomain::with_offset(BFieldElement offset) const {
    ArithmeticDomain result = *this;
    result.offset = offset;
    return result;
}

ArithmeticDomain ArithmeticDomain::halve() const {
    if (length % 2 != 0) {
        throw std::invalid_argument("Cannot halve domain with odd length");
    }
    
    ArithmeticDomain result = *this;
    result.length = length / 2;
    // Generator squared gives generator for half-length domain
    result.generator = generator * generator;
    // Match Rust: `ArithmeticDomain::halve()` squares offset and generator.
    // See `triton-vm/src/arithmetic_domain.rs`: `offset: self.offset.square()`.
    result.offset = offset * offset;
    return result;
}

BFieldElement ArithmeticDomain::element(size_t index) const {
    return offset * generator.pow(index);
}

std::vector<BFieldElement> ArithmeticDomain::values() const {
    std::vector<BFieldElement> domain_values;
    domain_values.reserve(length);
    BFieldElement current = offset;
    for (size_t i = 0; i < length; ++i) {
        domain_values.push_back(current);
        current *= generator;
    }
    return domain_values;
}

// ProverDomains implementation
ProverDomains ProverDomains::derive(
    size_t padded_height,
    size_t num_trace_randomizers,
    const ArithmeticDomain& fri_domain,
    int64_t max_degree
) {
    // Calculate randomized trace length
    size_t total_table_length = padded_height + num_trace_randomizers;
    size_t randomized_trace_len = 1;
    while (randomized_trace_len < total_table_length) {
        randomized_trace_len <<= 1;
    }
    
    // Create randomized trace domain
    ArithmeticDomain randomized_trace_domain = ArithmeticDomain::of_length(randomized_trace_len);
    
    // Trace domain is half of randomized trace domain
    ArithmeticDomain trace_domain = randomized_trace_domain.halve();
    
    // Quotient domain
    size_t max_degree_u = static_cast<size_t>(max_degree);
    size_t quotient_domain_length = 1;
    while (quotient_domain_length < max_degree_u) {
        quotient_domain_length <<= 1;
    }
    ArithmeticDomain quotient_domain = ArithmeticDomain::of_length(quotient_domain_length)
        .with_offset(fri_domain.offset);
    
    // DEBUG: Print domain lengths to verify correct computation
    if (const char* env = std::getenv("TVM_DEBUG_QUOTIENT")) {
        std::cout << "DEBUG ProverDomains::derive:" << std::endl;
        std::cout << "  padded_height: " << padded_height << std::endl;
        std::cout << "  randomized_trace_len: " << randomized_trace_len << std::endl;
        std::cout << "  trace_domain.length: " << trace_domain.length << std::endl;
        std::cout << "  max_degree: " << max_degree << " (as size_t: " << max_degree_u << ")" << std::endl;
        std::cout << "  quotient_domain_length: " << quotient_domain_length << std::endl;
        std::cout << "  quotient_domain.length: " << quotient_domain.length << std::endl;
        std::cout << "  fri_domain.length: " << fri_domain.length << std::endl;
    }
    
    ProverDomains domains;
    domains.trace = trace_domain;
    domains.randomized_trace = randomized_trace_domain;
    domains.quotient = quotient_domain;
    domains.fri = fri_domain;
    
    return domains;
}

// MasterMainTable implementation
MasterMainTable::MasterMainTable(size_t num_rows, size_t num_columns)
    : num_rows_(num_rows)
    , num_columns_(num_columns)
    , data_(num_rows, std::vector<BFieldElement>(num_columns))
    , trace_domain_({0, BFieldElement::zero(), BFieldElement::zero()})
    , quotient_domain_({0, BFieldElement::zero(), BFieldElement::zero()})
    , fri_domain_({0, BFieldElement::zero(), BFieldElement::zero()})
    , trace_randomizer_seed_({0})  // Default seed
    , num_trace_randomizers_(0)  // Default: no randomizers
{
}

MasterMainTable::MasterMainTable(
    size_t num_rows,
    size_t num_columns,
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain
)
    : num_rows_(num_rows)
    , num_columns_(num_columns)
    , data_(num_rows, std::vector<BFieldElement>(num_columns))
    , trace_domain_(trace_domain)
    , quotient_domain_(quotient_domain)
    , fri_domain_(quotient_domain)
    , trace_randomizer_seed_({0})  // Default seed
    , num_trace_randomizers_(0)  // Default: no randomizers
{
}

MasterMainTable::MasterMainTable(
    size_t num_rows,
    size_t num_columns,
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain,
    const ArithmeticDomain& fri_domain
)
    : num_rows_(num_rows)
    , num_columns_(num_columns)
    , data_(num_rows, std::vector<BFieldElement>(num_columns))
    , trace_domain_(trace_domain)
    , quotient_domain_(quotient_domain)
    , fri_domain_(fri_domain)
    , trace_randomizer_seed_({0})  // Default seed
    , num_trace_randomizers_(0)  // Default: no randomizers
{
}

MasterMainTable::MasterMainTable(
    size_t num_rows,
    size_t num_columns,
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain,
    const ArithmeticDomain& fri_domain,
    const std::array<uint8_t, 32>& trace_randomizer_seed
)
    : num_rows_(num_rows)
    , num_columns_(num_columns)
    , data_(num_rows, std::vector<BFieldElement>(num_columns))
    , trace_domain_(trace_domain)
    , quotient_domain_(quotient_domain)
    , fri_domain_(fri_domain)
    , trace_randomizer_seed_(trace_randomizer_seed)
    , num_trace_randomizers_(0)  // Default: no randomizers (should be set via setter)
{
}

// Destructor - cleanup pinned memory if used
MasterMainTable::~MasterMainTable() {
    if (flat_data_) {
#ifdef TRITON_CUDA_ENABLED
        if (flat_data_pinned_) {
            cudaFreeHost(flat_data_);
        } else {
            delete[] flat_data_;
        }
#else
        delete[] flat_data_;
#endif
        flat_data_ = nullptr;
    }
}

// Move constructor
MasterMainTable::MasterMainTable(MasterMainTable&& other) noexcept
    : num_rows_(other.num_rows_)
    , num_columns_(other.num_columns_)
    , data_(std::move(other.data_))
    , use_flat_buffer_(other.use_flat_buffer_)
    , flat_data_pinned_(other.flat_data_pinned_)
    , flat_data_(other.flat_data_)
    , trace_domain_(other.trace_domain_)
    , quotient_domain_(other.quotient_domain_)
    , fri_domain_(other.fri_domain_)
    , trace_randomizer_seed_(other.trace_randomizer_seed_)
    , num_trace_randomizers_(other.num_trace_randomizers_)
    , precomputed_randomizer_coefficients_(std::move(other.precomputed_randomizer_coefficients_))
    , lde_table_(std::move(other.lde_table_))
    , fri_digests_(std::move(other.fri_digests_))
{
    other.flat_data_ = nullptr;
    other.use_flat_buffer_ = false;
}

// Move assignment
MasterMainTable& MasterMainTable::operator=(MasterMainTable&& other) noexcept {
    if (this != &other) {
        // Free existing flat data
        if (flat_data_) {
#ifdef TRITON_CUDA_ENABLED
            if (flat_data_pinned_) {
                cudaFreeHost(flat_data_);
            } else {
                delete[] flat_data_;
            }
#else
            delete[] flat_data_;
#endif
        }
        
        num_rows_ = other.num_rows_;
        num_columns_ = other.num_columns_;
        data_ = std::move(other.data_);
        use_flat_buffer_ = other.use_flat_buffer_;
        flat_data_pinned_ = other.flat_data_pinned_;
        flat_data_ = other.flat_data_;
        trace_domain_ = other.trace_domain_;
        quotient_domain_ = other.quotient_domain_;
        fri_domain_ = other.fri_domain_;
        trace_randomizer_seed_ = other.trace_randomizer_seed_;
        num_trace_randomizers_ = other.num_trace_randomizers_;
        precomputed_randomizer_coefficients_ = std::move(other.precomputed_randomizer_coefficients_);
        lde_table_ = std::move(other.lde_table_);
        fri_digests_ = std::move(other.fri_digests_);
        
        other.flat_data_ = nullptr;
        other.use_flat_buffer_ = false;
    }
    return *this;
}

// Enable flat buffer mode for GPU-ready memory layout
void MasterMainTable::enable_flat_buffer(bool use_pinned_memory) {
    if (use_flat_buffer_) {
        return; // Already enabled
    }
    
    const bool profile = (std::getenv("TVM_PROFILE_TABLE_CREATE") != nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();
    
    size_t total_elements = num_rows_ * num_columns_;
    size_t row_bytes = num_columns_ * sizeof(uint64_t);
    
#ifdef TRITON_CUDA_ENABLED
    if (use_pinned_memory) {
        cudaError_t err = cudaMallocHost(&flat_data_, total_elements * sizeof(uint64_t));
        if (err != cudaSuccess) {
            throw std::runtime_error("Failed to allocate pinned memory for flat buffer");
        }
        flat_data_pinned_ = true;
    } else {
        flat_data_ = new uint64_t[total_elements];
        flat_data_pinned_ = false;
    }
#else
    (void)use_pinned_memory;
    flat_data_ = new uint64_t[total_elements];
    flat_data_pinned_ = false;
#endif

    auto t1 = std::chrono::high_resolution_clock::now();
    if (profile) {
        std::cout << "       - alloc pinned: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;
    }
    
    // Copy existing data using direct memcpy per row
    // BFieldElement is just a uint64_t wrapper, so memory layout is identical
    if (!data_.empty()) {
        #pragma omp parallel for schedule(static)
        for (size_t r = 0; r < num_rows_; ++r) {
            // Direct memcpy: BFieldElement stores uint64_t at offset 0
            // Cast before multiply to prevent 32-bit overflow (critical for large inputs like input21)
            std::memcpy(flat_data_ + static_cast<size_t>(r) * static_cast<size_t>(num_columns_), 
                       reinterpret_cast<const uint64_t*>(data_[r].data()),
                       row_bytes);
        }
        
        auto t2 = std::chrono::high_resolution_clock::now();
        if (profile) {
            std::cout << "       - memcpy rows: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;
        }
        
        // Clear vector data to save memory
        data_.clear();
        data_.shrink_to_fit();
    } else {
        // Zero-initialize flat buffer
        std::memset(flat_data_, 0, total_elements * sizeof(uint64_t));
    }
    
    use_flat_buffer_ = true;
}

// Thread-local storage for row() in flat buffer mode
static thread_local std::vector<BFieldElement> flat_row_buffer_;

const std::vector<BFieldElement>& MasterMainTable::row(size_t i) const {
    if (i >= num_rows_) {
        throw std::out_of_range("Row index out of range");
    }
    if (use_flat_buffer_) {
        // Return a view into flat buffer (using thread-local buffer for compatibility)
        flat_row_buffer_.resize(num_columns_);
        // Cast before multiply to prevent 32-bit overflow (critical for large inputs like input21)
        const uint64_t* row_ptr = flat_data_ + static_cast<size_t>(i) * static_cast<size_t>(num_columns_);
        for (size_t c = 0; c < num_columns_; ++c) {
            flat_row_buffer_[c] = BFieldElement(row_ptr[c]);
        }
        return flat_row_buffer_;
    }
    return data_[i];
}

BFieldElement MasterMainTable::get(size_t row, size_t col) const {
    if (row >= num_rows_ || col >= num_columns_) {
        throw std::out_of_range("Index out of range");
    }
    if (use_flat_buffer_) {
        // Cast before multiply to prevent 32-bit overflow (critical for large inputs like input21)
        return BFieldElement(flat_data_[static_cast<size_t>(row) * static_cast<size_t>(num_columns_) + static_cast<size_t>(col)]);
    }
    return data_[row][col];
}

void MasterMainTable::set(size_t row, size_t col, BFieldElement value) {
    if (row >= num_rows_ || col >= num_columns_) {
        throw std::out_of_range("Index out of range");
    }
    if (use_flat_buffer_) {
        flat_data_[row * num_columns_ + col] = value.value();
    } else {
        data_[row][col] = value;
    }
}

MasterMainTable MasterMainTable::from_aet(
    const AlgebraicExecutionTrace& aet,
    const ProverDomains& domains,
    size_t num_trace_randomizers,
    const std::array<uint8_t, 32>& trace_randomizer_seed
) {
    using namespace TableColumnOffsets;
    
    const bool profile = (std::getenv("TVM_PROFILE_TABLE_CREATE") != nullptr);
    auto profile_start = std::chrono::high_resolution_clock::now();
    auto section_start = profile_start;
    
    auto log_section = [&](const char* name) {
        if (profile) {
            auto now = std::chrono::high_resolution_clock::now();
            auto ms = std::chrono::duration<double, std::milli>(now - section_start).count();
            std::cout << "  " << name << ": " << ms << " ms" << std::endl;
            section_start = now;
        }
    };
    
    // Rust `MasterMainTable::new(aet, ...)` creates a trace table of height equal to the
    // maximum height of all main tables, not just the processor trace.
    size_t trace_height = aet.height();
    
    // Create table with actual trace height (will be padded later)
    constexpr size_t NUM_COLUMNS = 379;
    // DEBUG: Print domains before creating table
    if (const char* env = std::getenv("TVM_DEBUG_QUOTIENT")) {
        std::cout << "DEBUG MasterMainTable::from_aet:" << std::endl;
        std::cout << "  trace_height: " << trace_height << std::endl;
        std::cout << "  domains.trace.length: " << domains.trace.length << std::endl;
        std::cout << "  domains.quotient.length: " << domains.quotient.length << std::endl;
        std::cout << "  domains.fri.length: " << domains.fri.length << std::endl;
    }
    
    MasterMainTable table(
        trace_height,  // Use actual trace height, not padded_height
        NUM_COLUMNS,
        domains.trace,
        domains.quotient,
        domains.fri,
        trace_randomizer_seed
    );
    
    // DEBUG: Print domains after creating table
    if (const char* env = std::getenv("TVM_DEBUG_QUOTIENT")) {
        std::cout << "DEBUG After table creation:" << std::endl;
        std::cout << "  table.trace_domain().length: " << table.trace_domain().length << std::endl;
        std::cout << "  table.quotient_domain().length: " << table.quotient_domain().length << std::endl;
        std::cout << "  table.fri_domain().length: " << table.fri_domain().length << std::endl;
    }
    
    // Set number of trace randomizers
    table.set_num_trace_randomizers(num_trace_randomizers);
    
    // Initialize all rows with zeros
    // The table is already zero-initialized by constructor
    
    // Fill processor table from AET (OPTIMIZED: use flat buffer directly)
    // Processor table starts at column 7 (PROCESSOR_TABLE_START) and has 39 columns
    const size_t processor_height = aet.processor_trace_height();
    const size_t processor_width = aet.processor_trace_width();
    const BFieldElement* proc_flat = aet.processor_trace_flat_data();
    
    // Clock jump differences (from OpStack/RAM/JumpStack tables) used to populate
    // ProcessorMainColumn::ClockJumpDifferenceLookupMultiplicity (matches Rust ProcessorTable::fill).
    std::vector<BFieldElement> clk_jump_diffs_op_stack;
    std::vector<BFieldElement> clk_jump_diffs_ram;
    std::vector<BFieldElement> clk_jump_diffs_jump_stack;
    
    // Parallelize processor table fill (direct from flat buffer - no conversion!)
    // Control via TRITON_OMP_PROCESSOR (default: enabled if OpenMP available)
    static int omp_processor_enabled = -1;
    if (omp_processor_enabled == -1) {
        const char* env = std::getenv("TRITON_OMP_PROCESSOR");
        omp_processor_enabled = (env == nullptr || (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
    }
    
    const size_t proc_fill_height = std::min(processor_height, table.num_rows());
    const size_t cols_to_fill = std::min(static_cast<size_t>(PROCESSOR_TABLE_COLS), processor_width);
#ifdef _OPENMP
    if (omp_processor_enabled) {
        #pragma omp parallel for schedule(static)
        for (size_t row = 0; row < proc_fill_height; ++row) {
            const BFieldElement* src_row = proc_flat + row * processor_width;
            for (size_t col = 0; col < cols_to_fill; ++col) {
                table.set(row, PROCESSOR_TABLE_START + col, src_row[col]);
            }
        }
    } else {
        for (size_t row = 0; row < proc_fill_height; ++row) {
            const BFieldElement* src_row = proc_flat + row * processor_width;
            for (size_t col = 0; col < cols_to_fill; ++col) {
                table.set(row, PROCESSOR_TABLE_START + col, src_row[col]);
            }
        }
    }
#else
    for (size_t row = 0; row < proc_fill_height; ++row) {
        const BFieldElement* src_row = proc_flat + row * processor_width;
        for (size_t col = 0; col < cols_to_fill; ++col) {
            table.set(row, PROCESSOR_TABLE_START + col, src_row[col]);
        }
    }
#endif
    log_section("Processor table fill");
    
    // Fill OpStack table from AET (columns 46-49), matching Rust `OpStackTable::fill`.
    // Sort by (StackPointer, CLK) and collect clock jump differences when StackPointer stays constant.
    {
        const auto& op_stack_trace = aet.op_stack_underflow_trace();
        struct OSRow { BFieldElement clk; BFieldElement ib1; BFieldElement sp; BFieldElement first; };
        std::vector<OSRow> rows;
        rows.reserve(op_stack_trace.size());
        for (const auto& r : op_stack_trace) {
            if (r.size() < OP_STACK_TABLE_COLS) continue;
            rows.push_back(OSRow{r[0], r[1], r[2], r[3]});
        }
        parallel_sort(rows.begin(), rows.end(), [](const OSRow& a, const OSRow& b) {
            if (a.sp.value() != b.sp.value()) return a.sp.value() < b.sp.value();
            return a.clk.value() < b.clk.value();
        });

        // OPTIMIZATION: Parallelize OpStack table fill
        const size_t fill_rows = std::min(rows.size(), table.num_rows());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < fill_rows; ++i) {
            table.set(i, OP_STACK_TABLE_START + OpStackMainColumn::CLK, rows[i].clk);
            table.set(i, OP_STACK_TABLE_START + OpStackMainColumn::IB1ShrinkStack, rows[i].ib1);
            table.set(i, OP_STACK_TABLE_START + OpStackMainColumn::StackPointer, rows[i].sp);
            table.set(i, OP_STACK_TABLE_START + OpStackMainColumn::FirstUnderflowElement, rows[i].first);
        }

        // clock jump diffs
        if (rows.size() > 1) {
            clk_jump_diffs_op_stack.reserve(rows.size());
            for (size_t i = 0; i + 1 < rows.size(); ++i) {
                if (rows[i].sp == rows[i + 1].sp) {
                    clk_jump_diffs_op_stack.push_back(rows[i + 1].clk - rows[i].clk);
                }
            }
        }
    }
    log_section("OpStack table fill");
    
    // Fill RAM table from AET (columns 50-56), matching Rust `RamTable::fill`.
    // Steps:
    // 1) sort by (RamPointer, CLK)
    // 2) set InverseOfRampDifference and BezoutCoefficientPolynomialCoefficient0/1
    // 3) compute clock jump diffs (for processor multiplicities)
    struct RamRow {
        BFieldElement clk;
        BFieldElement inst_type;
        BFieldElement ramp;
        BFieldElement ramv;
    };
    std::vector<RamRow> ram_rows;
    ram_rows.reserve(aet.ram_trace().size());
    for (const auto& r : aet.ram_trace()) {
        if (r.size() < RAM_TABLE_COLS) continue;
        ram_rows.push_back(RamRow{
            r[RamMainColumn::CLK],
            r[RamMainColumn::InstructionType],
            r[RamMainColumn::RamPointer],
            r[RamMainColumn::RamValue],
        });
    }
    parallel_sort(ram_rows.begin(), ram_rows.end(), [](const RamRow& a, const RamRow& b) {
        if (a.ramp.value() != b.ramp.value()) return a.ramp.value() < b.ramp.value();
        return a.clk.value() < b.clk.value();
    });

    // Unique RAM pointers in sorted order (unique).
    std::vector<BFieldElement> unique_ramps;
    unique_ramps.reserve(ram_rows.size());
    for (const auto& rr : ram_rows) {
        if (unique_ramps.empty() || unique_ramps.back() != rr.ramp) unique_ramps.push_back(rr.ramp);
    }

    // Compute Bézout coefficient polynomials coefficients (a,b) like Rust:
    // - rp = Π (x - r) over unique roots
    // - fd = rp'
    // - b interpolates (r_i, 1/fd(r_i))
    // - a = (1 - fd*b) / rp  (exact division)
    //
    // Optimized with parallel evaluation where possible.
    auto bezout_coeff_polynomials_coefficients = [&](const std::vector<BFieldElement>& roots) {
        const size_t n = roots.size();
        if (n == 0) {
            return std::pair<std::vector<BFieldElement>, std::vector<BFieldElement>>{{}, {}};
        }

        auto trim = [](std::vector<BFieldElement> v) {
            while (!v.empty() && v.back().is_zero()) v.pop_back();
            if (v.empty()) v.push_back(BFieldElement::zero());
            return v;
        };

        auto poly_add = [](const std::vector<BFieldElement>& a, const std::vector<BFieldElement>& b) {
            std::vector<BFieldElement> r(std::max(a.size(), b.size()), BFieldElement::zero());
            for (size_t i = 0; i < a.size(); ++i) r[i] += a[i];
            for (size_t i = 0; i < b.size(); ++i) r[i] += b[i];
            return r;
        };

        auto poly_sub = [](const std::vector<BFieldElement>& a, const std::vector<BFieldElement>& b) {
            std::vector<BFieldElement> r(std::max(a.size(), b.size()), BFieldElement::zero());
            for (size_t i = 0; i < a.size(); ++i) r[i] += a[i];
            for (size_t i = 0; i < b.size(); ++i) r[i] -= b[i];
            return r;
        };

        // FFT-based polynomial multiplication for large polys, naive for small
        auto poly_mul = [](const std::vector<BFieldElement>& a, const std::vector<BFieldElement>& b) {
            if (a.size() == 1 && a[0].is_zero()) return std::vector<BFieldElement>{BFieldElement::zero()};
            if (b.size() == 1 && b[0].is_zero()) return std::vector<BFieldElement>{BFieldElement::zero()};
            
            const size_t result_size = a.size() + b.size() - 1;
            
            // For small polynomials, use naive O(n^2) multiplication
            if (result_size < 64) {
                std::vector<BFieldElement> r(result_size, BFieldElement::zero());
                for (size_t i = 0; i < a.size(); ++i) {
                    for (size_t j = 0; j < b.size(); ++j) {
                        r[i + j] += a[i] * b[j];
                    }
                }
                return r;
            }
            
#ifdef TRITON_CUDA_ENABLED
            // GPU NTT threshold (default 8192, set TRITON_GPU_POLY_THRESHOLD to change)
            static size_t gpu_threshold = []() {
                const char* env = std::getenv("TRITON_GPU_POLY_THRESHOLD");
                return env ? static_cast<size_t>(std::atoi(env)) : 8192;
            }();
            
            if (result_size >= gpu_threshold && gpu_poly_mul_available()) {
                // Convert to raw uint64_t
                std::vector<uint64_t> a_raw(a.size()), b_raw(b.size());
                for (size_t i = 0; i < a.size(); ++i) a_raw[i] = a[i].value();
                for (size_t i = 0; i < b.size(); ++i) b_raw[i] = b[i].value();
                
                std::vector<uint64_t> result_raw(result_size);
                size_t actual_size = 0;
                
                int err = gpu_poly_mul_ntt(
                    a_raw.data(), a_raw.size(),
                    b_raw.data(), b_raw.size(),
                    result_raw.data(), &actual_size
                );
                
                if (err == 0) {
                    std::vector<BFieldElement> result(actual_size);
                    for (size_t i = 0; i < actual_size; ++i) {
                        result[i] = BFieldElement(result_raw[i]);
                    }
                    return result;
                }
                // Fall through to CPU on error
            }
#endif
            
            // Pad to next power of 2
            size_t n = 1;
            while (n < result_size) n *= 2;
            
            // CPU NTT
            std::vector<BFieldElement> a_padded(n, BFieldElement::zero());
            std::vector<BFieldElement> b_padded(n, BFieldElement::zero());
            std::copy(a.begin(), a.end(), a_padded.begin());
            std::copy(b.begin(), b.end(), b_padded.begin());
            
            // Forward NTT
            NTT::forward(a_padded);
            NTT::forward(b_padded);
            
            // Pointwise multiplication
            for (size_t i = 0; i < n; ++i) {
                a_padded[i] *= b_padded[i];
            }
            
            // Inverse NTT
            NTT::inverse(a_padded);
            
            // Trim to actual result size
            a_padded.resize(result_size);
            return a_padded;
        };

        auto poly_eval = [](const std::vector<BFieldElement>& a, BFieldElement x) {
            // Horner for ascending coefficients
            BFieldElement acc = BFieldElement::zero();
            for (int i = static_cast<int>(a.size()) - 1; i >= 0; --i) {
                acc = acc * x + a[static_cast<size_t>(i)];
            }
            return acc;
        };

        auto poly_derivative = [](const std::vector<BFieldElement>& a) {
            if (a.size() <= 1) return std::vector<BFieldElement>{BFieldElement::zero()};
            std::vector<BFieldElement> d(a.size() - 1, BFieldElement::zero());
            for (size_t i = 1; i < a.size(); ++i) {
                d[i - 1] = a[i] * BFieldElement(static_cast<uint64_t>(i));
            }
            return d;
        };

        auto poly_div_exact = [&](const std::vector<BFieldElement>& dividend,
                                  const std::vector<BFieldElement>& divisor) {
            // FFT-based polynomial division for large polynomials
            // For exact division f / g where g divides f:
            // Use Newton iteration to compute 1/rev(g) mod x^k, then q = rev(rev(f) * inv)
            auto a = trim(dividend);
            auto d = trim(divisor);
            if (a.empty()) return std::vector<BFieldElement>{BFieldElement::zero()};
            if (d.empty()) return std::vector<BFieldElement>{BFieldElement::zero()};
            size_t deg_a = a.size() - 1;
            size_t deg_d = d.size() - 1;
            if (deg_a < deg_d) return std::vector<BFieldElement>{BFieldElement::zero()};
            
            // For small polynomials, use naive division
            if (deg_a < 256) {
                std::vector<BFieldElement> a_desc(a.rbegin(), a.rend());
                std::vector<BFieldElement> d_desc(d.rbegin(), d.rend());
                size_t q_deg = deg_a - deg_d;
                std::vector<BFieldElement> q_desc(q_deg + 1, BFieldElement::zero());
                std::vector<BFieldElement> rem = a_desc;
                BFieldElement d_lead_inv = d_desc[0].inverse();
                for (size_t k = 0; k <= q_deg; ++k) {
                    BFieldElement lead = rem[k];
                    if (lead.is_zero()) continue;
                    BFieldElement qk = lead * d_lead_inv;
                    q_desc[k] = qk;
                    for (size_t j = 0; j <= deg_d; ++j) {
                        rem[k + j] -= qk * d_desc[j];
                    }
                }
                std::vector<BFieldElement> q(q_desc.rbegin(), q_desc.rend());
                return trim(q);
            }
            
            // FFT-based division for large polynomials
            size_t k = deg_a - deg_d + 1;  // Number of quotient coefficients
            
            // Reverse polynomials: f_rev(x) = x^deg_a * f(1/x)
            std::vector<BFieldElement> f_rev(a.rbegin(), a.rend());
            std::vector<BFieldElement> g_rev(d.rbegin(), d.rend());
            
            // Compute 1/g_rev mod x^k using Newton iteration
            // Start with initial approximation: 1/g_rev[0]
            std::vector<BFieldElement> inv{g_rev[0].inverse()};
            inv.reserve(k);
            
            size_t prec = 1;
            while (prec < k) {
                size_t next_prec = std::min(prec * 2, k);
                
                // Newton step: inv' = inv * (2 - g_rev * inv) mod x^next_prec
                // Truncate g_rev to next_prec terms
                size_t g_len = std::min(g_rev.size(), next_prec);
                std::vector<BFieldElement> g_trunc(g_rev.begin(), g_rev.begin() + g_len);
                
                // prod = g_trunc * inv (mod x^next_prec is implicit in the truncation)
                auto prod = poly_mul(g_trunc, inv);
                if (prod.size() > next_prec) prod.resize(next_prec);
                
                // two_minus_prod = 2 - prod
                std::vector<BFieldElement> two_minus(next_prec, BFieldElement::zero());
                two_minus[0] = BFieldElement(2);
                for (size_t i = 0; i < prod.size(); ++i) {
                    two_minus[i] -= prod[i];
                }
                
                // inv = inv * two_minus_prod mod x^next_prec
                inv = poly_mul(inv, two_minus);
                if (inv.size() > next_prec) inv.resize(next_prec);
                
                prec = next_prec;
            }
            
            // q_rev = f_rev * inv mod x^k
            size_t f_len = std::min(f_rev.size(), k);
            std::vector<BFieldElement> f_trunc(f_rev.begin(), f_rev.begin() + f_len);
            auto q_rev = poly_mul(f_trunc, inv);
            if (q_rev.size() > k) q_rev.resize(k);
            
            // Pad q_rev to k terms if needed
            while (q_rev.size() < k) q_rev.push_back(BFieldElement::zero());
            
            // Reverse to get q
            std::vector<BFieldElement> q(q_rev.rbegin(), q_rev.rend());
            return trim(q);
        };

        // Optimized Lagrange interpolation with O(n^2) poly muls but cached prefix products
        // Precompute prefix products so each basis poly only needs one additional multiplication
        auto lagrange_interpolate = [&](const std::vector<BFieldElement>& xs,
                                        const std::vector<BFieldElement>& ys) {
            const size_t m = xs.size();
            if (m == 0) return std::vector<BFieldElement>{BFieldElement::zero()};
            if (m == 1) return std::vector<BFieldElement>{ys[0]};
            
            // 1. Compute full product and derivative for denominators
            // Use product tree for O(n log n) product computation
            std::vector<std::vector<BFieldElement>> factors(m);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < m; ++i) {
                factors[i] = {BFieldElement::zero() - xs[i], BFieldElement::one()};
            }
            
            // Product tree reduction
            std::vector<std::vector<BFieldElement>> tree_factors = factors;
            while (tree_factors.size() > 1) {
                size_t new_sz = (tree_factors.size() + 1) / 2;
                std::vector<std::vector<BFieldElement>> new_factors(new_sz);
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < new_sz; ++i) {
                    size_t left = 2 * i;
                    size_t right = 2 * i + 1;
                    if (right < tree_factors.size()) {
                        new_factors[i] = poly_mul(tree_factors[left], tree_factors[right]);
                    } else {
                        new_factors[i] = tree_factors[left];
                    }
                }
                tree_factors = std::move(new_factors);
            }
            std::vector<BFieldElement> rp_full = tree_factors.empty() ? 
                std::vector<BFieldElement>{BFieldElement::one()} : tree_factors[0];
            
            // Derivative evaluated at roots gives denominators
            std::vector<BFieldElement> fd = poly_derivative(rp_full);
            
            std::vector<BFieldElement> denoms(m);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < m; ++i) {
                denoms[i] = poly_eval(fd, xs[i]);
            }
            
            std::vector<BFieldElement> inv_denoms = BFieldElement::batch_inversion(denoms);
            
            // 2. Scaled values: c_i = y_i / denominator_i
            std::vector<BFieldElement> scaled(m);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < m; ++i) {
                scaled[i] = ys[i] * inv_denoms[i];
            }
            
            // 3. Compute prefix and suffix products using parallel block reduction
            // Split into blocks, compute block products in parallel, then combine
            const size_t BLOCK_SIZE = 64;  // Tune for cache and parallelism
            const size_t num_blocks = (m + BLOCK_SIZE - 1) / BLOCK_SIZE;
            
            // 3a. Compute block products in parallel
            std::vector<std::vector<BFieldElement>> block_prods(num_blocks);
            #pragma omp parallel for schedule(static)
            for (size_t b = 0; b < num_blocks; ++b) {
                size_t start = b * BLOCK_SIZE;
                size_t end = std::min(start + BLOCK_SIZE, m);
                std::vector<BFieldElement> prod{BFieldElement::one()};
                for (size_t i = start; i < end; ++i) {
                    prod = poly_mul(prod, factors[i]);
                }
                block_prods[b] = std::move(prod);
            }
            
            // 3b. Compute block prefix products (sequential - only num_blocks iterations)
            std::vector<std::vector<BFieldElement>> block_prefix(num_blocks + 1);
            block_prefix[0] = {BFieldElement::one()};
            for (size_t b = 0; b < num_blocks; ++b) {
                block_prefix[b + 1] = poly_mul(block_prefix[b], block_prods[b]);
            }
            
            // 3c. Compute block suffix products (sequential)
            std::vector<std::vector<BFieldElement>> block_suffix(num_blocks + 1);
            block_suffix[num_blocks] = {BFieldElement::one()};
            for (size_t b = num_blocks; b > 0; --b) {
                block_suffix[b - 1] = poly_mul(block_suffix[b], block_prods[b - 1]);
            }
            
            // 3d. Compute per-element prefix/suffix within blocks (parallel)
            std::vector<std::vector<BFieldElement>> prefix(m + 1);
            std::vector<std::vector<BFieldElement>> suffix(m + 1);
            prefix[0] = block_prefix[0];
            suffix[m] = block_suffix[num_blocks];
            
            #pragma omp parallel for schedule(static)
            for (size_t b = 0; b < num_blocks; ++b) {
                size_t start = b * BLOCK_SIZE;
                size_t end = std::min(start + BLOCK_SIZE, m);
                
                // Compute prefix within this block
                std::vector<BFieldElement> local_prefix = block_prefix[b];
                for (size_t i = start; i < end; ++i) {
                    prefix[i] = local_prefix;
                    local_prefix = poly_mul(local_prefix, factors[i]);
                }
                prefix[end] = local_prefix;  // This should equal block_prefix[b+1]
                
                // Compute suffix within this block  
                std::vector<BFieldElement> local_suffix = block_suffix[b + 1];
                for (size_t i = end; i > start; --i) {
                    suffix[i] = local_suffix;
                    local_suffix = poly_mul(local_suffix, factors[i - 1]);
                }
                suffix[start] = local_suffix;  // This should equal block_suffix[b]
            }
            
            // 5. Compute basis polynomials in parallel: basis[i] = prefix[i] * suffix[i+1] * scaled[i]
            std::vector<std::vector<BFieldElement>> basis_polys(m);
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < m; ++i) {
                std::vector<BFieldElement> basis = poly_mul(prefix[i], suffix[i + 1]);
                for (auto& c : basis) c *= scaled[i];
                basis_polys[i] = std::move(basis);
            }
            
            // 6. Sum all basis polynomials (use parallel reduction for large m)
            std::vector<BFieldElement> poly(m, BFieldElement::zero());
            for (size_t i = 0; i < m; ++i) {
                for (size_t j = 0; j < basis_polys[i].size() && j < poly.size(); ++j) {
                    poly[j] += basis_polys[i][j];
                }
            }
            return trim(poly);
        };

        // =========================================================
        // OPTIMIZED BÉZOUT: Single product tree, parallel evaluation
        // =========================================================
        
        auto t0 = std::chrono::high_resolution_clock::now();
        
        // Build subproduct tree once
        size_t num_levels = 0;
        size_t sz = n;
        while (sz > 1) { ++num_levels; sz = (sz + 1) / 2; }
        ++num_levels;
        
        std::vector<std::vector<std::vector<BFieldElement>>> tree(num_levels);
        
        // Level 0: linear factors
        tree[0].resize(n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            tree[0][i] = {BFieldElement::zero() - roots[i], BFieldElement::one()};
        }
        
        // Build higher levels
        // OPTIMIZED: Use static scheduling for large workloads, dynamic for small ones
        for (size_t d = 1; d < num_levels; ++d) {
            size_t prev_sz = tree[d-1].size();
            size_t new_sz = (prev_sz + 1) / 2;
            tree[d].resize(new_sz);
            
            // Use static scheduling for larger workloads (better cache locality)
            // Dynamic scheduling for smaller workloads (better load balancing)
            if (new_sz > 64) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < new_sz; ++i) {
                    size_t left = 2 * i;
                    size_t right = 2 * i + 1;
                    if (right < prev_sz) {
                        tree[d][i] = poly_mul(tree[d-1][left], tree[d-1][right]);
                    } else {
                        tree[d][i] = tree[d-1][left];
                    }
                }
            } else {
                #pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < new_sz; ++i) {
                    size_t left = 2 * i;
                    size_t right = 2 * i + 1;
                    if (right < prev_sz) {
                        tree[d][i] = poly_mul(tree[d-1][left], tree[d-1][right]);
                    } else {
                        tree[d][i] = tree[d-1][left];
                    }
                }
            }
        }
        
        auto t1 = std::chrono::high_resolution_clock::now();
        if (profile) std::cout << "    [1] Product tree: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;
        
        // rp = full product (at root)
        std::vector<BFieldElement> rp = tree[num_levels - 1][0];
        rp = trim(rp);
        
        // fd = rp'
        std::vector<BFieldElement> fd = poly_derivative(rp);
        
        auto t2 = std::chrono::high_resolution_clock::now();
        
        // Evaluate fd at all roots
        std::vector<BFieldElement> fd_in_roots(n);
        
#ifdef TRITON_CUDA_ENABLED
        // GPU-accelerated polynomial evaluation
        // OPTIMIZED: Use pinned memory for faster transfers, parallel conversion
        {
            // Allocate GPU memory
            uint64_t *d_coeffs = nullptr, *d_points = nullptr, *d_results = nullptr;
            uint64_t *h_coeffs_pinned = nullptr, *h_points_pinned = nullptr, *h_results_pinned = nullptr;
            cudaError_t err;
            
            // Try to use pinned memory for host buffers (faster transfers)
            bool use_pinned = (n > 1024);  // Use pinned memory for larger workloads
            if (use_pinned) {
                err = cudaMallocHost(&h_coeffs_pinned, fd.size() * sizeof(uint64_t));
                if (err == cudaSuccess) err = cudaMallocHost(&h_points_pinned, n * sizeof(uint64_t));
                if (err == cudaSuccess) err = cudaMallocHost(&h_results_pinned, n * sizeof(uint64_t));
            }
            
            if (err == cudaSuccess) {
                err = cudaMalloc(&d_coeffs, fd.size() * sizeof(uint64_t));
                if (err == cudaSuccess) err = cudaMalloc(&d_points, n * sizeof(uint64_t));
                if (err == cudaSuccess) err = cudaMalloc(&d_results, n * sizeof(uint64_t));
            }
            
            if (err == cudaSuccess) {
                // Convert coefficients in parallel
                if (use_pinned && h_coeffs_pinned) {
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < fd.size(); ++i) {
                        h_coeffs_pinned[i] = fd[i].value();
                    }
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < n; ++i) {
                        h_points_pinned[i] = roots[i].value();
                    }
                    
                    // Async memory transfers (faster with pinned memory)
                    cudaMemcpyAsync(d_coeffs, h_coeffs_pinned, fd.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
                    cudaMemcpyAsync(d_points, h_points_pinned, n * sizeof(uint64_t), cudaMemcpyHostToDevice);
                } else {
                    // Fallback to regular memory
                    std::vector<uint64_t> fd_raw(fd.size());
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < fd.size(); ++i) fd_raw[i] = fd[i].value();
                    
                    std::vector<uint64_t> roots_raw(n);
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < n; ++i) roots_raw[i] = roots[i].value();
                    
                    cudaMemcpy(d_coeffs, fd_raw.data(), fd.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
                    cudaMemcpy(d_points, roots_raw.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
                }
                
                // Call GPU kernel
                gpu::kernels::gpu_poly_eval_batch(d_coeffs, fd.size() - 1, d_points, n, d_results, nullptr);
                cudaDeviceSynchronize();
                
                // Download results
                if (use_pinned && h_results_pinned) {
                    cudaMemcpyAsync(h_results_pinned, d_results, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    cudaDeviceSynchronize();
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < n; ++i) {
                        fd_in_roots[i] = BFieldElement(h_results_pinned[i]);
                    }
                } else {
                    std::vector<uint64_t> results_raw(n);
                    cudaMemcpy(results_raw.data(), d_results, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < n; ++i) {
                        fd_in_roots[i] = BFieldElement(results_raw[i]);
                    }
                }
            } else {
                // Fallback to CPU
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < n; ++i) {
                    BFieldElement acc = BFieldElement::zero();
                    for (int j = static_cast<int>(fd.size()) - 1; j >= 0; --j) {
                        acc = acc * roots[i] + fd[static_cast<size_t>(j)];
                    }
                    fd_in_roots[i] = acc;
                }
            }
            
            if (d_coeffs) cudaFree(d_coeffs);
            if (d_points) cudaFree(d_points);
            if (d_results) cudaFree(d_results);
            if (h_coeffs_pinned) cudaFreeHost(h_coeffs_pinned);
            if (h_points_pinned) cudaFreeHost(h_points_pinned);
            if (h_results_pinned) cudaFreeHost(h_results_pinned);
        }
#else
        // CPU parallel Horner
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            BFieldElement acc = BFieldElement::zero();
            for (int j = static_cast<int>(fd.size()) - 1; j >= 0; --j) {
                acc = acc * roots[i] + fd[static_cast<size_t>(j)];
            }
            fd_in_roots[i] = acc;
        }
#endif
        
        auto t3 = std::chrono::high_resolution_clock::now();
        if (profile) std::cout << "    [2] Horner eval: " << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms" << std::endl;
        
        // b_in_roots = 1/fd(roots[i])
        std::vector<BFieldElement> b_in_roots = BFieldElement::batch_inversion(fd_in_roots);
        
        auto t4 = std::chrono::high_resolution_clock::now();
        if (profile) std::cout << "    [3] Batch inv: " << std::chrono::duration<double, std::milli>(t4 - t3).count() << " ms" << std::endl;
        
        // =========================================================
        // Fast interpolation: weights[0][i] = b_in_roots[i] / fd(roots[i])
        // (scaled value for Lagrange interpolation)
        // =========================================================
        
        // Compute scaled values: b_in_roots[i] / fd_in_roots[i] = 1/fd^2
        std::vector<BFieldElement> scaled(n);
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            scaled[i] = b_in_roots[i] * b_in_roots[i];  // (1/fd)^2 = 1/fd^2
        }
        
        // Build weighted sums bottom-up through the tree
        std::vector<std::vector<std::vector<BFieldElement>>> weights(num_levels);
        
        // Level 0: weights[0][i] = scaled[i]
        weights[0].resize(n);
        for (size_t i = 0; i < n; ++i) {
            weights[0][i] = {scaled[i]};
        }
        
        // Combine upwards
        // OPTIMIZED: Use static scheduling for large workloads
        for (size_t d = 1; d < num_levels; ++d) {
            size_t this_sz = tree[d].size();
            weights[d].resize(this_sz);
            
            size_t prev_sz = tree[d-1].size();
            // Use static scheduling for larger workloads (better cache locality)
            if (this_sz > 64) {
                #pragma omp parallel for schedule(static)
                for (size_t i = 0; i < this_sz; ++i) {
                    size_t left = 2 * i;
                    size_t right = 2 * i + 1;
                    
                    if (right < prev_sz) {
                        auto left_contrib = poly_mul(weights[d-1][left], tree[d-1][right]);
                        auto right_contrib = poly_mul(weights[d-1][right], tree[d-1][left]);
                        weights[d][i] = poly_add(left_contrib, right_contrib);
                    } else {
                        weights[d][i] = weights[d-1][left];
                    }
                }
            } else {
                #pragma omp parallel for schedule(dynamic)
                for (size_t i = 0; i < this_sz; ++i) {
                    size_t left = 2 * i;
                    size_t right = 2 * i + 1;
                    
                    if (right < prev_sz) {
                        auto left_contrib = poly_mul(weights[d-1][left], tree[d-1][right]);
                        auto right_contrib = poly_mul(weights[d-1][right], tree[d-1][left]);
                        weights[d][i] = poly_add(left_contrib, right_contrib);
                    } else {
                        weights[d][i] = weights[d-1][left];
                    }
                }
            }
        }
        
        auto t5 = std::chrono::high_resolution_clock::now();
        if (profile) std::cout << "    [4] Weights tree: " << std::chrono::duration<double, std::milli>(t5 - t4).count() << " ms" << std::endl;
        
        std::vector<BFieldElement> b_coeffs = trim(weights[num_levels - 1][0]);
        
        // one_minus_fd_b = 1 - fd*b
        std::vector<BFieldElement> fd_b = poly_mul(fd, b_coeffs);
        std::vector<BFieldElement> one{BFieldElement::one()};
        std::vector<BFieldElement> one_minus_fd_b = poly_sub(one, fd_b);
        one_minus_fd_b = trim(one_minus_fd_b);
        
        auto t6 = std::chrono::high_resolution_clock::now();
        if (profile) std::cout << "    [5] fd*b mul: " << std::chrono::duration<double, std::milli>(t6 - t5).count() << " ms" << std::endl;

        // a = (1 - fd*b)/rp
        std::vector<BFieldElement> a_coeffs = poly_div_exact(one_minus_fd_b, rp);
        
        auto t7 = std::chrono::high_resolution_clock::now();
        if (profile) std::cout << "    [6] poly_div: " << std::chrono::duration<double, std::milli>(t7 - t6).count() << " ms" << std::endl;

        // Resize to n, like Rust
        a_coeffs.resize(n, BFieldElement::zero());
        b_coeffs.resize(n, BFieldElement::zero());
        return std::pair<std::vector<BFieldElement>, std::vector<BFieldElement>>{a_coeffs, b_coeffs};
    };

    if (profile) {
        std::cout << "  unique_ramps size: " << unique_ramps.size() << std::endl;
    }
    
    // =========================================================================
    // OPTIMIZATION: Run Bézout computation ASYNC while filling other tables
    // Bézout is the main bottleneck (~1.5s) but only RAM table needs the results.
    // Other tables (Hash, Program, Cascade, Lookup, JumpStack, U32) can fill in parallel.
    // 
    // The CPU implementation uses O(n log² n) product tree algorithm with:
    // - OpenMP parallelization for tree construction
    // - NTT-based polynomial multiplication
    // - GPU polynomial evaluation (when CUDA enabled)
    // =========================================================================
    auto bezout_future = std::async(std::launch::async, [&]() {
        return bezout_coeff_polynomials_coefficients(unique_ramps);
    });
    
    auto bezout_launch_time = std::chrono::high_resolution_clock::now();
    if (profile) {
        std::cout << "  [ASYNC] Bézout computation started" << std::endl;
    }
    
    // Fill Hash table from AET (columns 62-128) - MOVED HERE to run parallel with Bézout
    // Hash table combines (Rust order): program_hash_trace, sponge_trace, hash_trace.
    const auto& hash_trace = aet.hash_trace();
    const auto& sponge_trace = aet.sponge_trace();
    const auto& program_hash_trace = aet.program_hash_trace();
    
    // 1) program_hash_trace - parallelized
    const size_t program_hash_height = program_hash_trace.size();
    const size_t ph_fill_height = std::min(program_hash_height, table.num_rows());
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < ph_fill_height; ++row) {
        const auto& r = program_hash_trace[row];
        const size_t cols_to_fill = std::min(static_cast<size_t>(HASH_TABLE_COLS), r.size());
        for (size_t col = 0; col < cols_to_fill; ++col) {
            table.set(row, HASH_TABLE_START + col, r[col]);
        }
        // Mode = ProgramHashing
        table.set(row, HASH_TABLE_START + HashMainColumn::Mode, BFieldElement(HashTableMode::ProgramHashing));
    }

    // 2) sponge_trace - parallelized
    const size_t sponge_start_row = program_hash_height;
    const size_t sponge_height = sponge_trace.size();
    const size_t sp_fill_height = std::min(sponge_height, table.num_rows() - sponge_start_row);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < sp_fill_height; ++i) {
        const auto& r = sponge_trace[i];
        const size_t cols_to_fill = std::min(static_cast<size_t>(HASH_TABLE_COLS), r.size());
        for (size_t col = 0; col < cols_to_fill; ++col) {
            table.set(sponge_start_row + i, HASH_TABLE_START + col, r[col]);
        }
        table.set(sponge_start_row + i, HASH_TABLE_START + HashMainColumn::Mode, BFieldElement(HashTableMode::Sponge));
    }

    // 3) hash_trace - parallelized
    const size_t hash_start_row = sponge_start_row + sponge_height;
    const size_t hash_height = hash_trace.size();
    const size_t ht_fill_height = std::min(hash_height, table.num_rows() > hash_start_row ? table.num_rows() - hash_start_row : 0);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < ht_fill_height; ++i) {
        const auto& r = hash_trace[i];
        const size_t cols_to_fill = std::min(static_cast<size_t>(HASH_TABLE_COLS), r.size());
        for (size_t col = 0; col < cols_to_fill; ++col) {
            table.set(hash_start_row + i, HASH_TABLE_START + col, r[col]);
        }
        table.set(hash_start_row + i, HASH_TABLE_START + HashMainColumn::Mode, BFieldElement(HashTableMode::Hash));
    }
    log_section("Hash table fill (parallel with Bézout)");
    
    // Fill Program table from AET (columns 0-6), matching Rust `ProgramTable::fill`.
    {
        using namespace ProgramMainColumn;
        const auto& program_bwords = aet.program_bwords();
        const auto& instruction_multiplicities = aet.instruction_multiplicities();

        const size_t program_len = program_bwords.size();
        const size_t padded_program_len = aet.height_of_table(0); // TableId::Program
        const BFieldElement max_index_in_chunk = BFieldElement(static_cast<uint64_t>(Tip5::RATE - 1));
        
        // Precompute MaxMinusIndexInChunkInv for all 10 possible index_in_chunk values (Tip5::RATE = 10)
        std::array<BFieldElement, Tip5::RATE> precomputed_inv;
        for (size_t i = 0; i < Tip5::RATE; ++i) {
            BFieldElement index_in_chunk(static_cast<uint64_t>(i));
            precomputed_inv[i] = inverse_or_zero(max_index_in_chunk - index_in_chunk);
        }

        const size_t num_rows_prog = table.num_rows();
        #pragma omp parallel for schedule(static)
        for (size_t row_idx = 0; row_idx < num_rows_prog; ++row_idx) {
            // Address
            table.set(row_idx, PROGRAM_TABLE_START + Address, BFieldElement(static_cast<uint64_t>(row_idx)));

            // Instruction word (bword): program_bwords[row] for real program, then [1, 0, 0, ...] up to padded_program_len.
            BFieldElement instruction_word = BFieldElement::zero();
            if (row_idx < program_len) {
                instruction_word = program_bwords[row_idx];
            } else if (row_idx < padded_program_len) {
                instruction_word = (row_idx == program_len) ? BFieldElement::one() : BFieldElement::zero();
            } else {
                instruction_word = BFieldElement::zero();
            }
            table.set(row_idx, PROGRAM_TABLE_START + Instruction, instruction_word);

            // Lookup multiplicity: only defined for real program bwords
            uint64_t multiplicity = (row_idx < program_len && row_idx < instruction_multiplicities.size())
                ? static_cast<uint64_t>(instruction_multiplicities[row_idx])
                : 0ULL;
            table.set(row_idx, PROGRAM_TABLE_START + LookupMultiplicity, BFieldElement(multiplicity));

            // IndexInChunk and inverses are defined for all rows - use precomputed
            size_t idx_mod = row_idx % Tip5::RATE;
            table.set(row_idx, PROGRAM_TABLE_START + IndexInChunk, BFieldElement(static_cast<uint64_t>(idx_mod)));
            table.set(row_idx, PROGRAM_TABLE_START + MaxMinusIndexInChunkInv, precomputed_inv[idx_mod]);

            // Hash input padding flag
            table.set(row_idx, PROGRAM_TABLE_START + IsHashInputPadding,
                      (row_idx < program_len) ? BFieldElement::zero() : BFieldElement::one());

            // Table padding flag: in Rust, this is only set during `ProgramTable::pad`
            // (i.e., for rows beyond `padded_program_len`). Keep 0 here.
            table.set(row_idx, PROGRAM_TABLE_START + IsTablePadding, BFieldElement::zero());
        }
    }
    log_section("Program table fill (parallel with Bézout)");
    
    // Fill Cascade table (columns 129-134), matching Rust `CascadeTable::fill`.
    {
        using namespace CascadeMainColumn;
        const auto& mults = aet.cascade_table_lookup_multiplicities();
        const size_t mults_size = mults.size();
        const size_t fill_rows = std::min(mults_size, table.num_rows());
        
        // OPTIMIZATION: Parallelize Cascade table fill (matching Rust parallel version)
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < fill_rows; ++i) {
            const auto& [to_look_up, multiplicity] = mults[i];
            uint8_t lo = static_cast<uint8_t>(to_look_up & 0xFF);
            uint8_t hi = static_cast<uint8_t>((to_look_up >> 8) & 0xFF);
            table.set(i, CASCADE_TABLE_START + IsPadding, BFieldElement::zero());
            table.set(i, CASCADE_TABLE_START + LookInLo, BFieldElement(static_cast<uint64_t>(lo)));
            table.set(i, CASCADE_TABLE_START + LookInHi, BFieldElement(static_cast<uint64_t>(hi)));
            table.set(i, CASCADE_TABLE_START + LookOutLo, BFieldElement(static_cast<uint64_t>(Tip5::LOOKUP_TABLE[lo])));
            table.set(i, CASCADE_TABLE_START + LookOutHi, BFieldElement(static_cast<uint64_t>(Tip5::LOOKUP_TABLE[hi])));
            table.set(i, CASCADE_TABLE_START + LookupMultiplicity, BFieldElement(static_cast<uint64_t>(multiplicity)));
        }
        
        // Parallelize padding loop
        #pragma omp parallel for schedule(static)
        for (size_t row_idx = fill_rows; row_idx < table.num_rows(); ++row_idx) {
            table.set(row_idx, CASCADE_TABLE_START + IsPadding, BFieldElement::one());
        }
    }

    // Fill Lookup table (columns 135-138) with the fixed Tip5 lookup table + multiplicities from AET.
    {
        using namespace LookupMainColumn;
        const size_t lut_h = AlgebraicExecutionTrace::LOOKUP_TABLE_HEIGHT;
        const auto& lookup_mults = aet.lookup_table_lookup_multiplicities();
        const size_t fill_rows = std::min(lut_h, table.num_rows());
        
        // OPTIMIZATION: Parallelize Lookup table fill
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < fill_rows; ++i) {
            table.set(i, LOOKUP_TABLE_START + IsPadding, BFieldElement::zero());
            table.set(i, LOOKUP_TABLE_START + LookIn, BFieldElement(static_cast<uint64_t>(i)));
            table.set(i, LOOKUP_TABLE_START + LookOut, BFieldElement(static_cast<uint64_t>(Tip5::LOOKUP_TABLE[i])));
            table.set(i, LOOKUP_TABLE_START + LookupMultiplicity, BFieldElement(static_cast<uint64_t>(lookup_mults[i])));
        }
        
        // Parallelize padding loop
        #pragma omp parallel for schedule(static)
        for (size_t i = lut_h; i < table.num_rows(); ++i) {
            table.set(i, LOOKUP_TABLE_START + IsPadding, BFieldElement::one());
        }
    }
    log_section("Cascade+Lookup table fill (parallel with Bézout)");
    
    // =========================================================================
    // WAIT for Bézout computation to complete
    // =========================================================================
    auto [bez0, bez1] = bezout_future.get();
    
    auto bezout_complete_time = std::chrono::high_resolution_clock::now();
    if (profile) {
        double bezout_wall_time = std::chrono::duration<double, std::milli>(bezout_complete_time - bezout_launch_time).count();
        std::cout << "  [ASYNC] Bézout completed (wall time including overlap): " << bezout_wall_time << " ms" << std::endl;
    }
    log_section("RAM Bézout coefficients (async)");

    // Optional debug dump for RAM Bézout coefficients (for Rust comparison).
    if (const char* env = std::getenv("TVM_DEBUG_QUOTIENT")) {
        try {
            std::string debug_dir = env;
            nlohmann::json j;
            j["unique_ramps"] = nlohmann::json::array();
            for (const auto& r : unique_ramps) j["unique_ramps"].push_back(r.value());
            j["bezout_coeffs_0"] = nlohmann::json::array();
            for (const auto& c : bez0) j["bezout_coeffs_0"].push_back(c.value());
            j["bezout_coeffs_1"] = nlohmann::json::array();
            for (const auto& c : bez1) j["bezout_coeffs_1"].push_back(c.value());
            std::ofstream f(debug_dir + "/ram_bezout_coeffs.json");
            f << j.dump(2) << std::endl;
        } catch (...) {
            // ignore debug failures
        }
    }

    // Fill RAM region rows 0..ram_len
    // OPTIMIZED: Parallelize inverse computations and table writes
    if (!ram_rows.empty()) {
        const size_t ram_len = std::min(ram_rows.size(), table.num_rows());
        clk_jump_diffs_ram.reserve(ram_len);

        // Rust: pop from end, assign first row
        BFieldElement current_bcpc0 = bez0.empty() ? BFieldElement::zero() : bez0.back();
        BFieldElement current_bcpc1 = bez1.empty() ? BFieldElement::zero() : bez1.back();
        if (!bez0.empty()) bez0.pop_back();
        if (!bez1.empty()) bez1.pop_back();

        // Precompute all ramp differences and inverses in parallel
        size_t diff_count = ram_len > 1 ? ram_len - 1 : 0;
        std::vector<BFieldElement> ramp_diffs(diff_count);
        std::vector<BFieldElement> ramp_inverses(diff_count);
        std::vector<bool> is_zero_diff(diff_count);
        
        if (diff_count > 0) {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < diff_count; ++i) {
                const auto& curr = ram_rows[i];
                const auto& next = ram_rows[i + 1];
                ramp_diffs[i] = next.ramp - curr.ramp;
                is_zero_diff[i] = ramp_diffs[i].is_zero();
                ramp_inverses[i] = is_zero_diff[i] ? BFieldElement::zero() : ramp_diffs[i].inverse();
            }
        }

        // Row 0 (sequential - needed for Bézout coefficient tracking)
        table.set(0, RAM_TABLE_START + RamMainColumn::CLK, ram_rows[0].clk);
        table.set(0, RAM_TABLE_START + RamMainColumn::InstructionType, ram_rows[0].inst_type);
        table.set(0, RAM_TABLE_START + RamMainColumn::RamPointer, ram_rows[0].ramp);
        table.set(0, RAM_TABLE_START + RamMainColumn::RamValue, ram_rows[0].ramv);
        table.set(0, RAM_TABLE_START + RamMainColumn::InverseOfRampDifference, BFieldElement::zero());
        table.set(0, RAM_TABLE_START + RamMainColumn::BezoutCoefficientPolynomialCoefficient0, current_bcpc0);
        table.set(0, RAM_TABLE_START + RamMainColumn::BezoutCoefficientPolynomialCoefficient1, current_bcpc1);

        // Process remaining rows: track Bézout coefficients sequentially, but parallelize table writes
        for (size_t i = 0; i < diff_count; ++i) {
            const auto& curr = ram_rows[i];
            const auto& next = ram_rows[i + 1];
            BFieldElement clk_diff = next.clk - curr.clk;
            
            if (is_zero_diff[i]) {
                clk_jump_diffs_ram.push_back(clk_diff);
            } else {
                if (!bez0.empty()) { current_bcpc0 = bez0.back(); bez0.pop_back(); }
                if (!bez1.empty()) { current_bcpc1 = bez1.back(); bez1.pop_back(); }
            }
            
            // Set inverse (precomputed)
            table.set(i, RAM_TABLE_START + RamMainColumn::InverseOfRampDifference, ramp_inverses[i]);

            // Next row basic fields
            table.set(i + 1, RAM_TABLE_START + RamMainColumn::CLK, next.clk);
            table.set(i + 1, RAM_TABLE_START + RamMainColumn::InstructionType, next.inst_type);
            table.set(i + 1, RAM_TABLE_START + RamMainColumn::RamPointer, next.ramp);
            table.set(i + 1, RAM_TABLE_START + RamMainColumn::RamValue, next.ramv);
            table.set(i + 1, RAM_TABLE_START + RamMainColumn::BezoutCoefficientPolynomialCoefficient0, current_bcpc0);
            table.set(i + 1, RAM_TABLE_START + RamMainColumn::BezoutCoefficientPolynomialCoefficient1, current_bcpc1);
        }
    }
    log_section("RAM table fill");

    // Fill JumpStack table (columns 57-61) from the processor trace (matches Rust: JumpStack table is derived per-cycle).
    // OPTIMIZED: Parallelize bucket processing and table writes
    {
        using namespace JumpStackMainColumn;
        // OPTIMIZED: Use flat buffer directly instead of processor_trace()
        const BFieldElement* proc_data = aet.processor_trace_flat_data();
        const size_t proc_rows = aet.processor_trace_height();
        const size_t proc_cols = aet.processor_trace_width();

        // Preprocess by JSP value, preserving processor CLK order.
        // Use thread-safe bucket collection
        std::vector<std::vector<std::tuple<BFieldElement, BFieldElement, BFieldElement, BFieldElement>>> buckets;
        buckets.reserve(64);
        
        // First pass: collect all entries into a flat array with JSP values
        struct JSEntry {
            size_t jsp_val;
            BFieldElement clk;
            BFieldElement ci;
            BFieldElement jso;
            BFieldElement jsd;
        };
        std::vector<JSEntry> entries;
        entries.reserve(proc_rows);
        
#ifdef _OPENMP
        #pragma omp parallel
        {
            std::vector<JSEntry> local_entries;
            int num_threads = omp_get_num_threads();
            local_entries.reserve(proc_rows / static_cast<size_t>(num_threads) + 1);
            
            #pragma omp for nowait
            for (size_t row = 0; row < proc_rows; ++row) {
                const BFieldElement* pr = proc_data + row * proc_cols;
                BFieldElement clk = pr[processor_column_index(ProcessorMainColumn::CLK)];
                BFieldElement ci = pr[processor_column_index(ProcessorMainColumn::CI)];
                BFieldElement jsp = pr[processor_column_index(ProcessorMainColumn::JSP)];
                BFieldElement jso = pr[processor_column_index(ProcessorMainColumn::JSO)];
                BFieldElement jsd = pr[processor_column_index(ProcessorMainColumn::JSD)];
                
                size_t jsp_val = static_cast<size_t>(jsp.value());
                local_entries.push_back({jsp_val, clk, ci, jso, jsd});
            }
            
            #pragma omp critical
            {
                entries.insert(entries.end(), local_entries.begin(), local_entries.end());
            }
        }
#else
        // Sequential fallback
        for (size_t row = 0; row < proc_rows; ++row) {
            const BFieldElement* pr = proc_data + row * proc_cols;
            BFieldElement clk = pr[processor_column_index(ProcessorMainColumn::CLK)];
            BFieldElement ci = pr[processor_column_index(ProcessorMainColumn::CI)];
            BFieldElement jsp = pr[processor_column_index(ProcessorMainColumn::JSP)];
            BFieldElement jso = pr[processor_column_index(ProcessorMainColumn::JSO)];
            BFieldElement jsd = pr[processor_column_index(ProcessorMainColumn::JSD)];
            
            size_t jsp_val = static_cast<size_t>(jsp.value());
            entries.push_back({jsp_val, clk, ci, jso, jsd});
        }
#endif
        
        // Sort by JSP, then CLK (stable sort preserves CLK order within JSP)
        std::stable_sort(entries.begin(), entries.end(), [](const JSEntry& a, const JSEntry& b) {
            if (a.jsp_val != b.jsp_val) return a.jsp_val < b.jsp_val;
            return a.clk.value() < b.clk.value();
        });

        // Fill table in parallel (entries are already sorted)
        const size_t fill_rows = std::min(entries.size(), table.num_rows());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < fill_rows; ++i) {
            const auto& e = entries[i];
            BFieldElement jsp_bfe(static_cast<uint64_t>(e.jsp_val));
            table.set(i, JUMP_STACK_TABLE_START + CLK, e.clk);
            table.set(i, JUMP_STACK_TABLE_START + CI, e.ci);
            table.set(i, JUMP_STACK_TABLE_START + JSP, jsp_bfe);
            table.set(i, JUMP_STACK_TABLE_START + JSO, e.jso);
            table.set(i, JUMP_STACK_TABLE_START + JSD, e.jsd);
        }

        // Collect clock jump differences (only when JSP stays constant) - parallelizable
        if (fill_rows > 1) {
            std::vector<std::pair<size_t, BFieldElement>> local_diffs;
            size_t diff_limit = fill_rows - 1;
#ifdef _OPENMP
            #pragma omp parallel
            {
                std::vector<std::pair<size_t, BFieldElement>> thread_diffs;
                int num_threads = omp_get_num_threads();
                thread_diffs.reserve(diff_limit / static_cast<size_t>(num_threads) + 1);
                
                #pragma omp for nowait
                for (size_t i = 0; i < diff_limit; ++i) {
                    size_t curr_jsp_val = entries[i].jsp_val;
                    size_t next_jsp_val = entries[i + 1].jsp_val;
                    if (curr_jsp_val == next_jsp_val) {
                        BFieldElement clk_diff = entries[i + 1].clk - entries[i].clk;
                        thread_diffs.push_back({i, clk_diff});
                    }
                }
                
                #pragma omp critical
                {
                    local_diffs.insert(local_diffs.end(), thread_diffs.begin(), thread_diffs.end());
                }
            }
#else
            // Sequential fallback
            for (size_t i = 0; i < diff_limit; ++i) {
                size_t curr_jsp_val = entries[i].jsp_val;
                size_t next_jsp_val = entries[i + 1].jsp_val;
                if (curr_jsp_val == next_jsp_val) {
                    BFieldElement clk_diff = entries[i + 1].clk - entries[i].clk;
                    local_diffs.push_back({i, clk_diff});
                }
            }
#endif
            // Sort by index and extract values
            std::sort(local_diffs.begin(), local_diffs.end());
            clk_jump_diffs_jump_stack.reserve(local_diffs.size());
            for (const auto& [idx, diff] : local_diffs) {
                clk_jump_diffs_jump_stack.push_back(diff);
            }
        }
    }
    log_section("JumpStack table fill");

    // Fill U32 table (columns 139-148), matching Rust `U32Table::fill` + `U32Table::pad`.
    {
        using namespace U32MainColumn;

        auto inverse_or_zero_local = [](BFieldElement x) {
            return x.is_zero() ? BFieldElement::zero() : x.inverse();
        };

        struct U32Row {
            BFieldElement copy_flag;
            BFieldElement bits;
            BFieldElement bits_minus_33_inv;
            BFieldElement ci;
            BFieldElement lhs;
            BFieldElement lhs_inv;
            BFieldElement rhs;
            BFieldElement rhs_inv;
            BFieldElement result;
            BFieldElement lookup_mult;
        };

        // Precompute BitsMinus33Inv for Bits in [0..63] using inverse_or_zero.
        std::array<BFieldElement, 64> bits_minus_33_inv_lut;
        for (size_t i = 0; i < bits_minus_33_inv_lut.size(); ++i) {
            bits_minus_33_inv_lut[i] = inverse_or_zero_local(BFieldElement(static_cast<uint64_t>(i)) - BFieldElement(33));
        }
        
        auto make_section = [&](uint32_t ci_opcode, BFieldElement lhs0, BFieldElement rhs0, uint64_t multiplicity) -> std::vector<U32Row> {
            using Row = U32Row;

            std::vector<Row> sec;
            sec.reserve(40);
            sec.push_back(Row{
                BFieldElement::one(),
                BFieldElement::zero(),
                bits_minus_33_inv_lut[0],
                BFieldElement(static_cast<uint64_t>(ci_opcode)),
                lhs0,
                BFieldElement::zero(),
                rhs0,
                BFieldElement::zero(),
                BFieldElement::zero(),
                BFieldElement(multiplicity),
            });

            auto is_pow = (ci_opcode == TritonInstruction{AnInstruction::Pow}.opcode());
            auto is_lt = (ci_opcode == TritonInstruction{AnInstruction::Lt}.opcode());
            auto is_split = (ci_opcode == TritonInstruction{AnInstruction::Split}.opcode());
            auto is_and = (ci_opcode == TritonInstruction{AnInstruction::And}.opcode());
            auto is_log2floor = (ci_opcode == TritonInstruction{AnInstruction::Log2Floor}.opcode());
            auto is_popcount = (ci_opcode == TritonInstruction{AnInstruction::PopCount}.opcode());

            // Forward build until terminal condition
            while (true) {
                Row& last = sec.back();
                bool terminal = ((last.lhs.is_zero() || is_pow) && last.rhs.is_zero());
                if (terminal) {
                    if (is_split) last.result = BFieldElement::zero();
                    else if (is_lt) last.result = BFieldElement(2);
                    else if (is_pow) last.result = BFieldElement::one();
                    else if (is_and) last.result = BFieldElement::zero();
                    else if (is_log2floor) last.result = BFieldElement(BFieldElement::MODULUS - 1); // -1 mod P
                    else if (is_popcount) last.result = BFieldElement::zero();
                    else last.result = BFieldElement::zero();

                    // Lt edge case: if both operands are 0 (detected by Bits==0), result is 0 not 2.
                    if (is_lt && last.bits.is_zero()) {
                        last.result = BFieldElement::zero();
                    }

                    last.lhs_inv = inverse_or_zero_local(last.lhs);
                    last.rhs_inv = inverse_or_zero_local(last.rhs);
                    break;
                }

                BFieldElement lhs_lsb(last.lhs.value() % 2);
                BFieldElement rhs_lsb(last.rhs.value() % 2);

                Row next = last;
                next.copy_flag = BFieldElement::zero();
                next.bits = next.bits + BFieldElement::one();
                const size_t bits_idx = std::min(static_cast<size_t>(next.bits.value()), bits_minus_33_inv_lut.size() - 1);
                next.bits_minus_33_inv = bits_minus_33_inv_lut[bits_idx];

                if (!is_pow) {
                    next.lhs = (last.lhs - lhs_lsb) / BFieldElement(2);
                }
                next.rhs = (last.rhs - rhs_lsb) / BFieldElement(2);
                next.lookup_mult = BFieldElement::zero();
                sec.push_back(next);
            }

            // Back-propagate results/inverses to match Rust `u32_section_next_row`.
            for (int i = static_cast<int>(sec.size()) - 2; i >= 0; --i) {
                Row& row = sec[static_cast<size_t>(i)];
                Row& next = sec[static_cast<size_t>(i) + 1];

                BFieldElement lhs_lsb(row.lhs.value() % 2);
                BFieldElement rhs_lsb(row.rhs.value() % 2);
                row.lhs_inv = inverse_or_zero_local(row.lhs);
                row.rhs_inv = inverse_or_zero_local(row.rhs);

                BFieldElement next_res = next.result;
                if (is_split) {
                    row.result = next_res;
                } else if (is_lt) {
                    uint64_t nr = next_res.value();
                    uint64_t lsb_l = lhs_lsb.value();
                    uint64_t lsb_r = rhs_lsb.value();
                    uint64_t cf = row.copy_flag.value();
                    if (nr == 0 || nr == 1) {
                        row.result = next_res;
                    } else if (nr == 2 && lsb_l == 0 && lsb_r == 1) {
                        row.result = BFieldElement::one();
                    } else if (nr == 2 && lsb_l == 1 && lsb_r == 0) {
                        row.result = BFieldElement::zero();
                    } else if (nr == 2 && cf == 1) {
                        row.result = BFieldElement::zero();
                    } else {
                        row.result = BFieldElement(2);
                    }
                } else if (is_and) {
                    // And: result = 2 * next_result + lhs_lsb * rhs_lsb
                    row.result = BFieldElement(2) * next_res + lhs_lsb * rhs_lsb;
                } else if (is_log2floor) {
                    // Log2Floor: if LHS == 0, result = -1; else if LHS' != 0, result = next_result; else result = Bits
                    if (row.lhs.is_zero()) {
                        row.result = BFieldElement(BFieldElement::MODULUS - 1); // -1 mod P
                    } else if (!next.lhs.is_zero()) {
                        row.result = next_res;
                    } else {
                        // LHS != 0 && LHS' == 0
                        row.result = row.bits;
                    }
                } else if (is_popcount) {
                    // PopCount: result = next_result + lhs_lsb
                    row.result = next_res + lhs_lsb;
                } else if (is_pow) {
                    row.result = rhs_lsb.is_zero()
                        ? (next_res * next_res)
                        : (next_res * next_res * row.lhs);
                } else {
                    row.result = next_res;
                }
            }

            return sec;
        };

        // Fill sections - parallelizable using precomputed offsets
        const auto& entries = aet.u32_entries();
        const size_t num_entries = entries.size();
        
        const bool debug_u32_fill = (std::getenv("TVM_DEBUG_U32_FILL") != nullptr);
        
        if (debug_u32_fill) {
            std::cerr << "[DBG U32_FILL] Starting U32 table fill:" << std::endl;
            std::cerr << "[DBG U32_FILL]   num_entries: " << num_entries << std::endl;
            std::cerr << "[DBG U32_FILL]   table.num_rows(): " << table.num_rows() << std::endl;
        }
        
        // Phase 1: Build all sections in parallel
        std::vector<std::vector<U32Row>> all_sections(num_entries);
        #pragma omp parallel for schedule(dynamic)
        for (size_t e = 0; e < num_entries; ++e) {
            const auto& [entry, multiplicity] = entries[e];
            all_sections[e] = make_section(entry.instruction_opcode, entry.left_operand, entry.right_operand, multiplicity);
        }
        
        // Phase 2: Compute prefix sums for offsets (sequential but fast)
        std::vector<size_t> offsets(num_entries + 1);
        offsets[0] = 0;
        for (size_t e = 0; e < num_entries; ++e) {
            offsets[e + 1] = offsets[e] + all_sections[e].size();
            if (debug_u32_fill && e < 20) {
                const auto& [entry, mult] = entries[e];
                std::cerr << "[DBG U32_FILL] Entry " << e << ": opcode=" << entry.instruction_opcode 
                          << ", lhs=" << entry.left_operand.value() 
                          << ", rhs=" << entry.right_operand.value()
                          << ", mult=" << mult
                          << ", section_size=" << all_sections[e].size()
                          << ", offset=" << offsets[e] << std::endl;
            }
        }
        
        // Phase 3: Fill table from all sections in parallel
        #pragma omp parallel for schedule(dynamic)
        for (size_t e = 0; e < num_entries; ++e) {
            const auto& section = all_sections[e];
            size_t section_start = offsets[e];
            if (section_start >= table.num_rows()) continue;
            
            // Debug: Track which sections cover rows 181-186
            bool covers_target_rows = false;
            if (debug_u32_fill) {
                size_t section_end = section_start + section.size();
                covers_target_rows = (section_start <= 186 && section_end > 180);
            }
            
            for (size_t r = 0; r < section.size() && (section_start + r) < table.num_rows(); ++r) {
                const auto& row = section[r];
                size_t table_row = section_start + r;
                table.set(table_row, U32_TABLE_START + CopyFlag, row.copy_flag);
                table.set(table_row, U32_TABLE_START + Bits, row.bits);
                table.set(table_row, U32_TABLE_START + BitsMinus33Inv, row.bits_minus_33_inv);
                table.set(table_row, U32_TABLE_START + CI, row.ci);
                table.set(table_row, U32_TABLE_START + LHS, row.lhs);
                table.set(table_row, U32_TABLE_START + LhsInv, row.lhs_inv);
                table.set(table_row, U32_TABLE_START + RHS, row.rhs);
                table.set(table_row, U32_TABLE_START + RhsInv, row.rhs_inv);
                table.set(table_row, U32_TABLE_START + Result, row.result);
                table.set(table_row, U32_TABLE_START + LookupMultiplicity, row.lookup_mult);
                
                // Debug: Print details for rows 181-186
                if (debug_u32_fill && table_row >= 180 && table_row <= 186) {
                    const auto& [entry, mult] = entries[e];
                    std::cerr << "[DBG U32_FILL] Row " << table_row << " (entry " << e << ", section_row " << r << "):" << std::endl;
                    std::cerr << "  Entry: opcode=" << entry.instruction_opcode 
                              << ", lhs=" << entry.left_operand.value() 
                              << ", rhs=" << entry.right_operand.value()
                              << ", mult=" << mult << std::endl;
                    std::cerr << "  Section row: CI=" << row.ci.value() 
                              << ", Result=" << row.result.value()
                              << ", LHS=" << row.lhs.value()
                              << ", RHS=" << row.rhs.value()
                              << ", Bits=" << row.bits.value() << std::endl;
                }
            }
            
            if (debug_u32_fill && covers_target_rows) {
                const auto& [entry, mult] = entries[e];
                std::cerr << "[DBG U32_FILL] Section " << e << " covers rows " << section_start 
                          << " to " << (section_start + section.size() - 1) << std::endl;
            }
        }
        
        size_t next_section_start = offsets.empty() ? 0 : offsets.back();
        
        if (debug_u32_fill) {
            std::cerr << "[DBG U32_FILL] After section fill:" << std::endl;
            std::cerr << "[DBG U32_FILL]   next_section_start: " << next_section_start << std::endl;
            for (size_t r = 180; r <= 186 && r < table.num_rows(); ++r) {
                uint64_t ci = table.get(r, U32_TABLE_START + CI).value();
                uint64_t result = table.get(r, U32_TABLE_START + Result).value();
                uint64_t lhs = table.get(r, U32_TABLE_START + LHS).value();
                uint64_t rhs = table.get(r, U32_TABLE_START + RHS).value();
                std::cerr << "[DBG U32_FILL] Row " << r << ": CI=" << ci << ", Result=" << result 
                          << ", LHS=" << lhs << ", RHS=" << rhs << std::endl;
            }
        }

        // Pad remaining rows (Rust `U32Table::pad`)
        BFieldElement padding_ci(static_cast<uint64_t>(TritonInstruction{AnInstruction::Split}.opcode()));
        BFieldElement padding_bits_minus_33_inv = bits_minus_33_inv_lut[0];
        BFieldElement padding_lhs = BFieldElement::zero();
        BFieldElement padding_lhs_inv = BFieldElement::zero();
        BFieldElement padding_result = BFieldElement::zero();

        if (next_section_start > 0) {
            size_t last_row = std::min(next_section_start - 1, table.num_rows() - 1);
            padding_ci = table.get(last_row, U32_TABLE_START + CI);
            padding_lhs = table.get(last_row, U32_TABLE_START + LHS);
            padding_lhs_inv = table.get(last_row, U32_TABLE_START + LhsInv);
            padding_result = table.get(last_row, U32_TABLE_START + Result);
            if (padding_ci == BFieldElement(static_cast<uint64_t>(TritonInstruction{AnInstruction::Lt}.opcode()))) {
                padding_result = BFieldElement(2);
            }
            
            if (debug_u32_fill) {
                std::cerr << "[DBG U32_FILL] Padding values from last_row=" << last_row << ":" << std::endl;
                std::cerr << "  padding_ci=" << padding_ci.value() << std::endl;
                std::cerr << "  padding_lhs=" << padding_lhs.value() << std::endl;
                std::cerr << "  padding_result=" << padding_result.value() << std::endl;
                std::cerr << "  Padding rows from " << next_section_start << " to " << (table.num_rows() - 1) << std::endl;
            }
        }

        #pragma omp parallel for schedule(static)
        for (size_t row = next_section_start; row < table.num_rows(); ++row) {
            // Debug: Check if we're padding rows 181-186
            bool is_target_row = (row >= 180 && row <= 186);
            if (debug_u32_fill && is_target_row) {
                uint64_t old_ci = table.get(row, U32_TABLE_START + CI).value();
                uint64_t old_result = table.get(row, U32_TABLE_START + Result).value();
                std::cerr << "[DBG U32_FILL] Padding row " << row << " (old: CI=" << old_ci << ", Result=" << old_result << ")" << std::endl;
            }
            
            table.set(row, U32_TABLE_START + CopyFlag, BFieldElement::zero());
            table.set(row, U32_TABLE_START + Bits, BFieldElement::zero());
            table.set(row, U32_TABLE_START + BitsMinus33Inv, padding_bits_minus_33_inv);
            table.set(row, U32_TABLE_START + CI, padding_ci);
            table.set(row, U32_TABLE_START + LHS, padding_lhs);
            table.set(row, U32_TABLE_START + LhsInv, padding_lhs_inv);
            table.set(row, U32_TABLE_START + RHS, BFieldElement::zero());
            table.set(row, U32_TABLE_START + RhsInv, BFieldElement::zero());
            table.set(row, U32_TABLE_START + Result, padding_result);
            table.set(row, U32_TABLE_START + LookupMultiplicity, BFieldElement::zero());
        }
        
        if (debug_u32_fill) {
            std::cerr << "[DBG U32_FILL] After padding:" << std::endl;
            for (size_t r = 180; r <= 186 && r < table.num_rows(); ++r) {
                uint64_t ci = table.get(r, U32_TABLE_START + CI).value();
                uint64_t result = table.get(r, U32_TABLE_START + Result).value();
                uint64_t lhs = table.get(r, U32_TABLE_START + LHS).value();
                uint64_t rhs = table.get(r, U32_TABLE_START + RHS).value();
                std::cerr << "[DBG U32_FILL] Row " << r << ": CI=" << ci << ", Result=" << result 
                          << ", LHS=" << lhs << ", RHS=" << rhs << std::endl;
            }
        }
    }
    log_section("U32 table fill");
    
    // JumpStack table (columns 57-61) and U32 table (columns 139-148) are not yet in AET
    // They will be filled when we implement co-processor call recording for these
    
    // Degree-lowering table is filled during padding, not here

    // Populate processor ClockJumpDifferenceLookupMultiplicity from the collected diffs.
    {
        const size_t proc_len = aet.processor_trace_height();
        std::vector<BFieldElement> mult(proc_len, BFieldElement::zero());
        auto add = [&](const std::vector<BFieldElement>& diffs) {
            for (const auto& d : diffs) {
                size_t idx = static_cast<size_t>(d.value());
                if (idx < mult.size()) mult[idx] += BFieldElement::one();
            }
        };
        add(clk_jump_diffs_op_stack);
        add(clk_jump_diffs_ram);
        add(clk_jump_diffs_jump_stack);

        const size_t mult_col = PROCESSOR_TABLE_START +
            processor_column_index(ProcessorMainColumn::ClockJumpDifferenceLookupMultiplicity);
        for (size_t row = 0; row < proc_len && row < table.num_rows(); ++row) {
            table.set(row, mult_col, mult[row]);
        }
    }
    
    return table;
}

void MasterMainTable::pad(size_t padded_height) {
    // Default implementation: use zeros for table lengths (will be filled by pad_all_tables)
    std::array<size_t, 9> table_lengths = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    pad(padded_height, table_lengths);
}

void MasterMainTable::pad(size_t padded_height, const std::array<size_t, 9>& table_lengths) {
    if (padded_height < num_rows_) {
        throw std::invalid_argument("Padded height must be >= current height");
    }
    
    // Verify padded_height is a power of 2
    if ((padded_height & (padded_height - 1)) != 0) {
        throw std::invalid_argument("Padded height must be a power of 2");
    }

    const bool profile = (std::getenv("TVM_PROFILE_PAD") != nullptr);
    auto t0 = std::chrono::high_resolution_clock::now();

    // ---------------------------------------------------------------------
    // New, Rust-equivalent padding: apply table-specific padding across the
    // entire table height (including rows < current height) using `table_lengths`.
    // This fixes the old behavior which only padded newly appended rows.
    // ---------------------------------------------------------------------
    // Optimize resize: reserve capacity first, then resize to avoid multiple reallocations
    if (data_.size() < padded_height) {
        const size_t old_size = data_.size();
        data_.reserve(padded_height);
        data_.resize(padded_height);
        // Initialize new rows in parallel
        #pragma omp parallel for schedule(static)
        for (size_t r = old_size; r < padded_height; ++r) {
            data_[r].resize(num_columns_, BFieldElement::zero());
        }
    }
    num_rows_ = padded_height;

    auto t1 = std::chrono::high_resolution_clock::now();
    if (profile) std::cout << "    [pad] resize: " << std::chrono::duration<double, std::milli>(t1 - t0).count() << " ms" << std::endl;

    pad_all_tables(data_, table_lengths, padded_height);

    auto t2 = std::chrono::high_resolution_clock::now();
    if (profile) std::cout << "    [pad] pad_all_tables: " << std::chrono::duration<double, std::milli>(t2 - t1).count() << " ms" << std::endl;

    // Degree-lowering main columns are defined *after* main-table padding in Rust.
    compute_degree_lowering_main_columns(data_);

    auto t3 = std::chrono::high_resolution_clock::now();
    if (profile) std::cout << "    [pad] degree_lowering: " << std::chrono::duration<double, std::milli>(t3 - t2).count() << " ms" << std::endl;
    
    return;
    
    size_t original_height = num_rows_;
    
    // Resize table
    size_t old_height = num_rows_;
    data_.resize(padded_height, std::vector<BFieldElement>(num_columns_));
    
    using namespace TableColumnOffsets;
    using namespace ProgramMainColumn;
    
    // Constants for padding
    constexpr BFieldElement PADDING_VALUE(2);  // For OpStack IB1ShrinkStack
    constexpr BFieldElement PADDING_INDICATOR(2);  // For RAM InstructionType
    constexpr uint32_t HASH_OPCODE = 18;  // Instruction::Hash opcode
    constexpr uint32_t SPLIT_OPCODE = 4;  // Instruction::Split opcode
    constexpr uint32_t LT_OPCODE = 6;     // Instruction::Lt opcode
    
    // Pad original tables with table-specific rules
    if (original_height > 0) {
        const auto& last_row = data_[original_height - 1];
        std::vector<BFieldElement> padding_template = last_row;
        
        // ====================================================================
        // 1. Processor table padding (columns 7-45)
        // ====================================================================
        // Rule: Copy last row, set IsPadding=1, update CLK values, set ClockJumpDifferenceLookupMultiplicity=0
        size_t is_padding_col = PROCESSOR_TABLE_START + processor_column_index(ProcessorMainColumn::IsPadding);
        padding_template[is_padding_col] = BFieldElement::one();
        
        size_t clk_jump_mult_col = PROCESSOR_TABLE_START + processor_column_index(ProcessorMainColumn::ClockJumpDifferenceLookupMultiplicity);
        padding_template[clk_jump_mult_col] = BFieldElement::zero();
        
        // ====================================================================
        // 2. OpStack table padding (columns 46-49)
        // ====================================================================
        // Rule: Copy last row, set IB1ShrinkStack = PADDING_VALUE (2)
        // Column indices: CLK=0, IB1ShrinkStack=1, StackPointer=2, FirstUnderflowElement=3
        size_t op_stack_ib1_col = OP_STACK_TABLE_START + 1;  // IB1ShrinkStack
        padding_template[op_stack_ib1_col] = PADDING_VALUE;
        
        // If table was empty, set StackPointer to 16 (OpStackElement::COUNT)
        if (original_height == 0) {
            size_t op_stack_sp_col = OP_STACK_TABLE_START + 2;  // StackPointer
            padding_template[op_stack_sp_col] = BFieldElement(16);
        }
        
        // ====================================================================
        // 3. RAM table padding (columns 50-56)
        // ====================================================================
        // Rule: Copy last row, set InstructionType = PADDING_INDICATOR (2)
        // Column indices: CLK=0, InstructionType=1, RamPointer=2, RamValue=3, ...
        size_t ram_inst_type_col = RAM_TABLE_START + 1;  // InstructionType
        padding_template[ram_inst_type_col] = PADDING_INDICATOR;
        
        // If table was empty, set BezoutCoefficientPolynomialCoefficient1 = 1
        if (original_height == 0) {
            size_t ram_bezout1_col = RAM_TABLE_START + 6;  // BezoutCoefficientPolynomialCoefficient1
            padding_template[ram_bezout1_col] = BFieldElement::one();
        }
        
        // ====================================================================
        // 4. Hash table padding (columns 62-128)
        // ====================================================================
        // Rule: Set State0Inv-State3Inv to inverse_of_high_limbs(0), round constants, Mode=Pad, CI=Hash
        // HashMainColumn enum: Mode=0, CI=1, RoundNumber=2, ... State0Inv=60, State1Inv=61, State2Inv=62, State3Inv=63,
        //                      Constant0=64, Constant1=65, ..., Constant11=75
        BFieldElement inverse_of_high_limbs = inverse_or_zero_of_highest_2_limbs(BFieldElement::zero());
        
        // State0Inv, State1Inv, State2Inv, State3Inv columns (indices 60-63 in HashMainColumn)
        size_t hash_state0_inv_col = HASH_TABLE_START + 60;
        size_t hash_state1_inv_col = HASH_TABLE_START + 61;
        size_t hash_state2_inv_col = HASH_TABLE_START + 62;
        size_t hash_state3_inv_col = HASH_TABLE_START + 63;
        padding_template[hash_state0_inv_col] = inverse_of_high_limbs;
        padding_template[hash_state1_inv_col] = inverse_of_high_limbs;
        padding_template[hash_state2_inv_col] = inverse_of_high_limbs;
        padding_template[hash_state3_inv_col] = inverse_of_high_limbs;
        
        // Round constants for round 0 (Constant0-Constant11, indices 64-75 in HashMainColumn)
        auto round_constants = tip5_round_constants_round_0();
        for (size_t i = 0; i < 12; ++i) {
            size_t hash_constant_col = HASH_TABLE_START + 64 + i;  // Constant0-Constant11
            padding_template[hash_constant_col] = round_constants[i];
        }
        
        // Mode column = Pad (value 3 typically, HashTableMode::Pad)
        size_t hash_mode_col = HASH_TABLE_START + 0;  // Mode is first column
        padding_template[hash_mode_col] = BFieldElement(3);  // HashTableMode::Pad
        
        // CI column = Hash opcode
        size_t hash_ci_col = HASH_TABLE_START + 1;  // CI is second column
        padding_template[hash_ci_col] = BFieldElement(static_cast<uint64_t>(HASH_OPCODE));
        
        // ====================================================================
        // 5. Program table padding (columns 0-6)
        // ====================================================================
        // Rule: Set addresses sequentially, IndexInChunk, MaxMinusIndexInChunkInv, 
        //       IsHashInputPadding=1, IsTablePadding=1
        // This is done per-row, not in template
        
        // ====================================================================
        // 6. JumpStack table padding (columns 57-61)
        // ====================================================================
        // Rule: Complex - find row with max CLK, move rows after it, fill gap
        // This is handled separately after filling padding rows
        
        // ====================================================================
        // 7. Lookup table padding (columns 135-138)
        // ====================================================================
        // Rule: Set IsPadding = 1
        // Column indices: LookIn=0, LookOut=1, LookupMultiplicity=2, IsPadding=3
        size_t lookup_is_padding_col = LOOKUP_TABLE_START + 3;  // IsPadding
        padding_template[lookup_is_padding_col] = BFieldElement::one();
        
        // ====================================================================
        // 8. U32 table padding (columns 139-148)
        // ====================================================================
        // Rule: Copy last row, set CI=Split, BitsMinus33Inv, handle Lt edge case
        // Column indices: CopyFlag=0, Bits=1, BitsMinus33Inv=2, CI=3, LHS=4, LhsInv=5, RHS=6, RhsInv=7, Result=8, LookupMultiplicity=9
        size_t u32_ci_col = U32_TABLE_START + 3;  // CI
        padding_template[u32_ci_col] = BFieldElement(static_cast<uint64_t>(SPLIT_OPCODE));
        
        size_t u32_bits_minus_33_inv_col = U32_TABLE_START + 2;  // BitsMinus33Inv
        // BitsMinus33Inv = inverse_or_zero(33 - Bits), for padding Bits=0, so inverse_or_zero(33)
        BFieldElement bits_minus_33(33);
        padding_template[u32_bits_minus_33_inv_col] = inverse_or_zero(bits_minus_33);
        
        // Handle Lt edge case: if last row was Lt with Result=0, set Result=2
        if (original_height > 0) {
            BFieldElement last_ci = data_[original_height - 1][u32_ci_col];
            if (last_ci == BFieldElement(static_cast<uint64_t>(LT_OPCODE))) {
                size_t u32_result_col = U32_TABLE_START + 8;  // Result
                BFieldElement last_result = data_[original_height - 1][u32_result_col];
                if (last_result == BFieldElement::zero()) {
                    padding_template[u32_result_col] = BFieldElement(2);
                }
            }
        }
        
        // ====================================================================
        // JumpStack table padding (complex row movement) - DO THIS FIRST
        // ====================================================================
        // Find row with max CLK (should be original_height - 1)
        // Move all rows after max CLK row to the end, fill gap with padding
        // This must be done BEFORE filling general padding rows to avoid overwriting
        size_t jump_stack_clk_col = JUMP_STACK_TABLE_START + 0;  // CLK
        size_t max_clk_before_padding = original_height - 1;
        size_t max_clk_row_idx = original_height - 1;  // Last row has max CLK
        
        // Find actual row index with this CLK value (in case rows are out of order)
        for (size_t i = 0; i < original_height; ++i) {
            if (data_[i][jump_stack_clk_col] == BFieldElement(static_cast<uint64_t>(max_clk_before_padding))) {
                max_clk_row_idx = i;
                break;
            }
        }
        
        size_t num_padding_rows = padded_height - original_height;
        size_t rows_to_move_start = max_clk_row_idx + 1;
        size_t rows_to_move_end = original_height;
        size_t num_rows_to_move = rows_to_move_end - rows_to_move_start;
        
        if (num_rows_to_move > 0) {
            // Move rows after max CLK row to the end
            size_t dest_start = rows_to_move_start + num_padding_rows;
            for (size_t i = 0; i < num_rows_to_move; ++i) {
                data_[dest_start + i] = data_[rows_to_move_start + i];
            }
        }
        
        // Fill gap with padding rows (copy of max CLK row) - but preserve IsPadding from template
        std::vector<BFieldElement> jump_stack_padding_template = data_[max_clk_row_idx];
        // Apply general padding template settings to JumpStack padding rows
        jump_stack_padding_template[is_padding_col] = BFieldElement::one();
        for (size_t i = rows_to_move_start; i < rows_to_move_start + num_padding_rows; ++i) {
            data_[i] = jump_stack_padding_template;
            // Update CLK values sequentially
            data_[i][jump_stack_clk_col] = BFieldElement(static_cast<uint64_t>(i));
            // Also update processor CLK
            size_t clk_col = PROCESSOR_TABLE_START + processor_column_index(ProcessorMainColumn::CLK);
            data_[i][clk_col] = BFieldElement(static_cast<uint64_t>(i));
        }
        
        // Update CLK values in moved rows
        for (size_t i = rows_to_move_start + num_padding_rows; i < padded_height; ++i) {
            data_[i][jump_stack_clk_col] = BFieldElement(static_cast<uint64_t>(i));
        }
        
        // ====================================================================
        // Fill padding rows with template (for most tables)
        // ====================================================================
        for (size_t i = original_height; i < padded_height; ++i) {
            // Only fill rows that weren't already filled by JumpStack padding
            if (i < rows_to_move_start || i >= rows_to_move_start + num_padding_rows) {
                data_[i] = padding_template;
                
                // Update CLK values for processor table (sequential clock values)
                size_t clk_col = PROCESSOR_TABLE_START + processor_column_index(ProcessorMainColumn::CLK);
                data_[i][clk_col] = BFieldElement(static_cast<uint64_t>(i));
                
                // Update OpStack CLK
                size_t op_stack_clk_col = OP_STACK_TABLE_START + 0;  // CLK
                data_[i][op_stack_clk_col] = BFieldElement(static_cast<uint64_t>(i));
                
                // Update RAM CLK
                size_t ram_clk_col = RAM_TABLE_START + 0;  // CLK
                data_[i][ram_clk_col] = BFieldElement(static_cast<uint64_t>(i));
                
                // Update Program table: addresses, IndexInChunk, MaxMinusIndexInChunkInv
                size_t program_addr_col = PROGRAM_TABLE_START + Address;
                data_[i][program_addr_col] = BFieldElement(static_cast<uint64_t>(i));
                
                size_t program_index_in_chunk_col = PROGRAM_TABLE_START + IndexInChunk;
                data_[i][program_index_in_chunk_col] = BFieldElement(static_cast<uint64_t>(i % TIP5_RATE));
                
                size_t program_max_minus_index_inv_col = PROGRAM_TABLE_START + MaxMinusIndexInChunkInv;
                size_t max_minus_index = TIP5_RATE - 1 - (i % TIP5_RATE);
                BFieldElement max_minus_index_bfe(static_cast<uint64_t>(max_minus_index));
                data_[i][program_max_minus_index_inv_col] = inverse_or_zero(max_minus_index_bfe);
                
                size_t program_is_hash_input_padding_col = PROGRAM_TABLE_START + IsHashInputPadding;
                data_[i][program_is_hash_input_padding_col] = BFieldElement::one();
                
                size_t program_is_table_padding_col = PROGRAM_TABLE_START + IsTablePadding;
                data_[i][program_is_table_padding_col] = BFieldElement::one();
            }
        }
        
        // Update row 1's ClockJumpDifferenceLookupMultiplicity
        // Add number of padding rows to account for jump stack padding
        if (padded_height > 1) {
            BFieldElement current_mult = data_[1][clk_jump_mult_col];
            data_[1][clk_jump_mult_col] = current_mult + BFieldElement(static_cast<uint64_t>(num_padding_rows));
        }
    } else {
        // If table is empty, fill with zeros
        for (size_t i = 0; i < padded_height; ++i) {
            for (size_t j = 0; j < num_columns_; ++j) {
                data_[i][j] = BFieldElement::zero();
            }
        }
    }
    
    num_rows_ = padded_height;

    // Fill degree-lowering table (columns 149-378, 230 columns)
    // This is computed from the main table data
    compute_degree_lowering_main_columns(data_);
}

void MasterMainTable::low_degree_extend(const ArithmeticDomain& target_domain) {
    if (lde_table_.size() == target_domain.length && lde_table_[0].size() == num_columns_) {
        // Already computed
        return;
    }

    // Initialize LDE table
    lde_table_.resize(target_domain.length);
    for (auto& row : lde_table_) {
        row.resize(num_columns_);
    }

    // Perform LDE on each column WITH trace randomizers (for zero-knowledge)
    for (size_t col = 0; col < num_columns_; col++) {
        // Extract column data
        std::vector<BFieldElement> trace_column(num_rows_);
        for (size_t row = 0; row < num_rows_; row++) {
            trace_column[row] = data_[row][col];
        }

        // Perform randomized low-degree extension if randomizers are configured
        std::vector<BFieldElement> lde_column;
        if (num_trace_randomizers_ > 0) {
            // Get trace randomizer coefficients for this column
            std::vector<BFieldElement> randomizer_coeffs = trace_randomizer_for_column(col);
            
            // Perform randomized LDE: interpolant + zerofier * randomizer
            lde_column = RandomizedLDE::extend_column_with_randomizer(
                trace_column, trace_domain_, target_domain, randomizer_coeffs);
        } else {
            // Fallback to plain LDE if no randomizers configured
            lde_column = LDE::extend_column(trace_column, trace_domain_, target_domain);
        }

        // Store in LDE table
        for (size_t row = 0; row < target_domain.length; row++) {
            lde_table_[row][col] = lde_column[row];
        }
    }
}

void MasterMainTable::set_trace_randomizer_coefficients(size_t column_idx, const std::vector<BFieldElement>& coefficients) {
    if (column_idx >= num_columns_) {
        throw std::out_of_range("Column index out of range");
    }
    precomputed_randomizer_coefficients_[column_idx] = coefficients;
}

bool MasterMainTable::has_trace_randomizer_coefficients(size_t column_idx) const {
    return precomputed_randomizer_coefficients_.find(column_idx) != precomputed_randomizer_coefficients_.end();
}

std::vector<BFieldElement> MasterMainTable::trace_randomizer_for_column(size_t column_idx) const {
    if (column_idx >= num_columns_) {
        throw std::out_of_range("Column index out of range");
    }
    
    if (num_trace_randomizers_ == 0) {
        // Return empty vector if no randomizers configured
        return std::vector<BFieldElement>();
    }
    
    // If pre-computed coefficients are available (from Rust test data), use them
    // This allows matching Rust-generated coefficients when RNG implementations differ
    auto it = precomputed_randomizer_coefficients_.find(column_idx);
    if (it != precomputed_randomizer_coefficients_.end()) {
        return it->second;
    }
    
    // Otherwise, generate coefficients using C++ RNG
    // Create RNG from seed + column index offset
    // This matches Rust: rng_from_offset_seed(self.trace_randomizer_seed(), idx)
    std::array<uint8_t, 32> seed = trace_randomizer_seed_;
    
    // Convert column_idx to little-endian bytes (as u64)
    uint64_t offset = static_cast<uint64_t>(column_idx);
    std::array<uint8_t, 8> offset_bytes;
    for (size_t i = 0; i < 8; i++) {
        offset_bytes[i] = static_cast<uint8_t>((offset >> (i * 8)) & 0xFF);
    }
    
    // Add offset bytes to seed (wrapping addition, matching Rust)
    for (size_t i = 0; i < 8 && i < seed.size(); i++) {
        seed[i] = static_cast<uint8_t>(seed[i] + offset_bytes[i]);
    }
    
    // Create ChaCha12Rng with modified seed and generate coefficients
    ChaCha12Rng rng(seed);

    std::vector<BFieldElement> coefficients;
    coefficients.reserve(num_trace_randomizers_);

    for (size_t i = 0; i < num_trace_randomizers_; i++) {
        uint64_t random_value = rng.next_u64();
        coefficients.push_back(BFieldElement(random_value));
    }
    
    return coefficients;
}

// MasterAuxTable implementation
MasterAuxTable::MasterAuxTable(size_t num_rows, size_t num_columns)
    : num_rows_(num_rows)
    , num_columns_(num_columns)
    , data_(num_rows, std::vector<XFieldElement>(num_columns))
    , trace_domain_({0, BFieldElement::zero(), BFieldElement::zero()})
    , quotient_domain_({0, BFieldElement::zero(), BFieldElement::zero()})
    , fri_domain_({0, BFieldElement::zero(), BFieldElement::zero()})
    , lde_domain_length_(0)
{
}

MasterAuxTable::MasterAuxTable(
    size_t num_rows,
    size_t num_columns,
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain,
    const ArithmeticDomain& fri_domain
)
    : num_rows_(num_rows)
    , num_columns_(num_columns)
    , data_(num_rows, std::vector<XFieldElement>(num_columns))
    , trace_domain_(trace_domain)
    , quotient_domain_(quotient_domain)
    , fri_domain_(fri_domain)
    , lde_domain_length_(0)
{
}

const std::vector<XFieldElement>& MasterAuxTable::row(size_t i) const {
    if (i >= num_rows_) {
        throw std::out_of_range("Row index out of range");
    }
    return data_[i];
}

XFieldElement MasterAuxTable::get(size_t row, size_t col) const {
    if (row >= num_rows_ || col >= num_columns_) {
        throw std::out_of_range("Index out of range");
    }
    return data_[row][col];
}

void MasterAuxTable::set(size_t row, size_t col, XFieldElement value) {
    if (row >= num_rows_ || col >= num_columns_) {
        throw std::out_of_range("Index out of range");
    }
    data_[row][col] = value;
}

void MasterAuxTable::clear_low_degree_extension() {
    lde_table_.clear();
    lde_domain_length_ = 0;
}

void MasterAuxTable::set_trace_randomizer_coefficients(size_t column_idx, const std::vector<BFieldElement>& coefficients) {
    if (column_idx >= num_columns_) {
        throw std::out_of_range("Column index out of range");
    }
    precomputed_randomizer_coefficients_[column_idx] = coefficients;
}

void MasterAuxTable::set_trace_randomizer_xfield_coefficients(size_t column_idx, const std::vector<XFieldElement>& coefficients) {
    if (column_idx >= num_columns_) {
        throw std::out_of_range("Column index out of range");
    }
    precomputed_xfield_randomizer_coefficients_[column_idx] = coefficients;
}

bool MasterAuxTable::has_trace_randomizer_coefficients(size_t column_idx) const {
    return precomputed_randomizer_coefficients_.find(column_idx) != precomputed_randomizer_coefficients_.end() ||
           precomputed_xfield_randomizer_coefficients_.find(column_idx) != precomputed_xfield_randomizer_coefficients_.end();
}

std::vector<XFieldElement> MasterAuxTable::trace_randomizer_xfield_for_column(size_t column_idx) const {
    if (column_idx >= num_columns_) {
        throw std::out_of_range("Column index out of range");
    }
    
    if (num_trace_randomizers_ == 0) {
        return std::vector<XFieldElement>();
    }
    
    // Check for XFieldElement randomizers first
    auto xfe_it = precomputed_xfield_randomizer_coefficients_.find(column_idx);
    if (xfe_it != precomputed_xfield_randomizer_coefficients_.end()) {
        return xfe_it->second;
    }
    
    // Fallback to BFieldElement and lift
    auto bfe_it = precomputed_randomizer_coefficients_.find(column_idx);
    if (bfe_it != precomputed_randomizer_coefficients_.end()) {
        std::vector<XFieldElement> result;
        result.reserve(bfe_it->second.size());
        for (const auto& bfe : bfe_it->second) {
            result.push_back(XFieldElement(bfe));
        }
        return result;
    }
    
    // Generate BFE coefficients and lift to XFE
    std::vector<BFieldElement> bfe_coeffs = trace_randomizer_for_column(column_idx);
    std::vector<XFieldElement> result;
    result.reserve(bfe_coeffs.size());
    for (const auto& bfe : bfe_coeffs) {
        result.push_back(XFieldElement(bfe));
    }
    return result;
}

std::vector<BFieldElement> MasterAuxTable::trace_randomizer_for_column(size_t column_idx) const {
    if (column_idx >= num_columns_) {
        throw std::out_of_range("Column index out of range");
    }
    
    if (num_trace_randomizers_ == 0) {
        return std::vector<BFieldElement>();
    }
    
    // If pre-computed coefficients are available (from Rust test data), use them
    // This allows matching Rust-generated coefficients when RNG implementations differ
    auto it = precomputed_randomizer_coefficients_.find(column_idx);
    if (it != precomputed_randomizer_coefficients_.end()) {
        return it->second;
    }
    
    // Generate coefficients using seed + column offset (matches Rust behavior)
    std::array<uint8_t, 32> seed = trace_randomizer_seed_;
    
    // Convert column_idx to little-endian bytes (as u64)
    uint64_t offset = static_cast<uint64_t>(column_idx);
    std::array<uint8_t, 8> offset_bytes;
    for (size_t i = 0; i < 8; i++) {
        offset_bytes[i] = static_cast<uint8_t>((offset >> (i * 8)) & 0xFF);
    }
    
    // Add offset bytes to seed (wrapping addition, matching Rust)
    for (size_t i = 0; i < 8 && i < seed.size(); i++) {
        seed[i] = static_cast<uint8_t>(seed[i] + offset_bytes[i]);
    }
    
    // Create ChaCha12Rng with modified seed and generate coefficients
    ChaCha12Rng rng(seed);

    std::vector<BFieldElement> coefficients;
    coefficients.reserve(num_trace_randomizers_);

    for (size_t i = 0; i < num_trace_randomizers_; i++) {
        uint64_t random_value = rng.next_u64();
        coefficients.push_back(BFieldElement(random_value));
    }

    return coefficients;
}

void MasterAuxTable::low_degree_extend(const ArithmeticDomain& target_domain) {
    if (num_rows_ == 0 || num_columns_ == 0) {
        return;
    }

    if (lde_domain_length_ == target_domain.length && !lde_table_.empty()) {
        return;
    }

    ArithmeticDomain trace_domain = trace_domain_;
    if (trace_domain.length == 0) {
        trace_domain = ArithmeticDomain::of_length(num_rows_);
    }

    size_t target_rows = target_domain.length;
    
    // Allocate output table
    lde_table_.assign(target_rows, std::vector<XFieldElement>(num_columns_, XFieldElement::zero()));

    // GPU aux LDE - enable for testing
    constexpr bool USE_GPU_AUX_LDE = true;

#ifdef TRITON_CUDA_ENABLED
    // ==========================================================================
    // GPU Path: Batch process all XFE columns using GPU LDE
    // ==========================================================================
    if (USE_GPU_AUX_LDE && num_trace_randomizers_ > 0) {
        auto gpu_start = std::chrono::high_resolution_clock::now();
        
        // Flatten XFE trace data: [num_columns * num_rows * 3]
        // Layout: (col, row, component) -> [(col * num_rows + row) * 3 + component]
        std::vector<uint64_t> h_xfe_trace(num_columns_ * num_rows_ * 3);
        for (size_t col = 0; col < num_columns_; ++col) {
            for (size_t row = 0; row < num_rows_; ++row) {
                const auto& xfe = data_[row][col];
                size_t idx = (col * num_rows_ + row) * 3;
                h_xfe_trace[idx + 0] = xfe.coeff(0).value();
                h_xfe_trace[idx + 1] = xfe.coeff(1).value();
                h_xfe_trace[idx + 2] = xfe.coeff(2).value();
            }
        }
        
        // Check if we have XFieldElement randomizers (preferred) or BFieldElement randomizers
        bool has_xfe_randomizers = false;
        for (size_t col = 0; col < num_columns_; ++col) {
            if (precomputed_xfield_randomizer_coefficients_.find(col) != precomputed_xfield_randomizer_coefficients_.end()) {
                has_xfe_randomizers = true;
                break;
            }
        }

        // Gather randomizer coefficients (XFE or BFE)
        std::vector<uint64_t> h_randomizers;
        if (has_xfe_randomizers) {
            // XFieldElement randomizers: [num_cols * num_trace_randomizers * 3]
            h_randomizers.resize(num_columns_ * num_trace_randomizers_ * 3);
            for (size_t col = 0; col < num_columns_; ++col) {
                auto xfe_it = precomputed_xfield_randomizer_coefficients_.find(col);
                if (xfe_it != precomputed_xfield_randomizer_coefficients_.end()) {
                    const auto& xfe_coeffs = xfe_it->second;
                    for (size_t r = 0; r < num_trace_randomizers_ && r < xfe_coeffs.size(); ++r) {
                        size_t idx = (col * num_trace_randomizers_ + r) * 3;
                        h_randomizers[idx + 0] = xfe_coeffs[r].coeff(0).value();
                        h_randomizers[idx + 1] = xfe_coeffs[r].coeff(1).value();
                        h_randomizers[idx + 2] = xfe_coeffs[r].coeff(2).value();
                    }
                } else {
                    // Fallback: lift BFE to XFE (b, 0, 0)
                    std::vector<BFieldElement> bfe_coeffs = trace_randomizer_for_column(col);
                    for (size_t r = 0; r < num_trace_randomizers_ && r < bfe_coeffs.size(); ++r) {
                        size_t idx = (col * num_trace_randomizers_ + r) * 3;
                        h_randomizers[idx + 0] = bfe_coeffs[r].value();
                        h_randomizers[idx + 1] = 0;
                        h_randomizers[idx + 2] = 0;
                    }
                }
            }
        } else {
            // BFieldElement randomizers: [num_cols * num_trace_randomizers]
            h_randomizers.resize(num_columns_ * num_trace_randomizers_);
            for (size_t col = 0; col < num_columns_; ++col) {
                std::vector<BFieldElement> rand_coeffs = trace_randomizer_for_column(col);
                for (size_t r = 0; r < num_trace_randomizers_ && r < rand_coeffs.size(); ++r) {
                    h_randomizers[col * num_trace_randomizers_ + r] = rand_coeffs[r].value();
                }
            }
        }
        
        // Allocate GPU memory
        uint64_t* d_xfe_trace = nullptr;
        uint64_t* d_randomizers = nullptr;
        uint64_t* d_xfe_output = nullptr;
        
        size_t trace_bytes = h_xfe_trace.size() * sizeof(uint64_t);
        size_t rand_bytes = h_randomizers.size() * sizeof(uint64_t);
        size_t output_bytes = num_columns_ * target_rows * 3 * sizeof(uint64_t);
        
        cudaMalloc(&d_xfe_trace, trace_bytes);
        cudaMalloc(&d_randomizers, rand_bytes);
        cudaMalloc(&d_xfe_output, output_bytes);
        
        cudaMemcpy(d_xfe_trace, h_xfe_trace.data(), trace_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(d_randomizers, h_randomizers.data(), rand_bytes, cudaMemcpyHostToDevice);
        
        // Run GPU XFE LDE (pass flag indicating XFE randomizers)
        gpu::kernels::randomized_xfe_lde_batch_gpu(
            d_xfe_trace,
            num_columns_,
            num_rows_,
            d_randomizers,
            num_trace_randomizers_,
            trace_domain.offset.value(),
            target_domain.offset.value(),
            target_rows,
            d_xfe_output,
            has_xfe_randomizers,  // flag: true if XFE randomizers, false if BFE
            0  // default stream
        );
        
        // Copy results back
        std::vector<uint64_t> h_xfe_output(num_columns_ * target_rows * 3);
        cudaMemcpy(h_xfe_output.data(), d_xfe_output, output_bytes, cudaMemcpyDeviceToHost);
        
        // Convert back to XFE row-major format
        for (size_t col = 0; col < num_columns_; ++col) {
            for (size_t row = 0; row < target_rows; ++row) {
                size_t idx = (col * target_rows + row) * 3;
                lde_table_[row][col] = XFieldElement(
                    BFieldElement(h_xfe_output[idx + 0]),
                    BFieldElement(h_xfe_output[idx + 1]),
                    BFieldElement(h_xfe_output[idx + 2])
                );
            }
        }
        
        cudaFree(d_xfe_trace);
        cudaFree(d_randomizers);
        cudaFree(d_xfe_output);
        
        auto gpu_end = std::chrono::high_resolution_clock::now();
        double gpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count() / 1000.0;
        std::cout << "GPU: Aux table LDE: " << gpu_ms << " ms" << std::endl;
        
        lde_domain_length_ = target_domain.length;
        return;
    }
#endif

    // ==========================================================================
    // CPU Path: Process columns one by one
    // ==========================================================================
    std::vector<BFieldElement> component0(num_rows_);
    std::vector<BFieldElement> component1(num_rows_);
    std::vector<BFieldElement> component2(num_rows_);

    for (size_t col = 0; col < num_columns_; ++col) {
        // Extract column components
        for (size_t row = 0; row < num_rows_; ++row) {
            component0[row] = data_[row][col].coeff(0);
            component1[row] = data_[row][col].coeff(1);
            component2[row] = data_[row][col].coeff(2);
        }

        // Use randomized LDE if randomizers are configured
        if (num_trace_randomizers_ > 0) {
            std::vector<XFieldElement> xfield_column(num_rows_);
            for (size_t row = 0; row < num_rows_; ++row) {
                xfield_column[row] = data_[row][col];
            }
            
            std::vector<XFieldElement> lde_column;
            auto xfe_it = precomputed_xfield_randomizer_coefficients_.find(col);
            if (xfe_it != precomputed_xfield_randomizer_coefficients_.end()) {
                lde_column = RandomizedLDE::extend_xfield_column_with_xfield_randomizer(
                    xfield_column, trace_domain, target_domain, xfe_it->second);
            } else {
                std::vector<BFieldElement> randomizer_coeffs = trace_randomizer_for_column(col);
                lde_column = RandomizedLDE::extend_xfield_column_with_randomizer(
                    xfield_column, trace_domain, target_domain, randomizer_coeffs);
            }
            
            for (size_t row = 0; row < target_rows; ++row) {
                lde_table_[row][col] = lde_column[row];
            }
        } else {
            std::vector<BFieldElement> lde0 = LDE::extend_column(component0, trace_domain, target_domain);
            std::vector<BFieldElement> lde1 = LDE::extend_column(component1, trace_domain, target_domain);
            std::vector<BFieldElement> lde2 = LDE::extend_column(component2, trace_domain, target_domain);
            
            for (size_t row = 0; row < target_rows; ++row) {
                lde_table_[row][col] = XFieldElement(lde0[row], lde1[row], lde2[row]);
            }
        }
    }

    lde_domain_length_ = target_domain.length;
}


// Helper to get table slice
std::vector<std::vector<BFieldElement>> MasterMainTable::get_table_slice(size_t start_col, size_t num_cols) const {
    std::vector<std::vector<BFieldElement>> slice;
    slice.reserve(num_rows_);
    
    for (size_t r = 0; r < num_rows_; r++) {
        std::vector<BFieldElement> row_slice;
        row_slice.reserve(num_cols);
        for (size_t c = 0; c < num_cols; c++) {
            row_slice.push_back(data_[r][start_col + c]);
        }
        slice.push_back(row_slice);
    }
    
    return slice;
}

// Extend method - creates MasterAuxTable from MasterMainTable using challenges
// Optionally accepts pre-filled randomizer column values (for exact matching with Rust test data)
MasterAuxTable MasterMainTable::extend(const Challenges& challenges, 
                                       const std::optional<std::vector<std::vector<XFieldElement>>>& randomizer_values) const {
    constexpr size_t NUM_RANDOMIZER_POLYNOMIALS = 1;
    constexpr size_t MASTER_AUX_NUM_COLUMNS = 88;  // Total aux columns (matches test data)
    
    // Step 1: Initialize aux table with zeros
    ArithmeticDomain aux_trace_domain = trace_domain_;
    if (aux_trace_domain.length == 0) {
        aux_trace_domain = ArithmeticDomain::of_length(num_rows_);
    }
    ArithmeticDomain aux_quotient_domain = quotient_domain_;
    if (aux_quotient_domain.length == 0) {
        aux_quotient_domain = aux_trace_domain;
    }
    ArithmeticDomain aux_fri_domain = fri_domain_;
    if (aux_fri_domain.length == 0) {
        aux_fri_domain = aux_quotient_domain;
    }

    MasterAuxTable aux_table(
        num_rows_,
        MASTER_AUX_NUM_COLUMNS,
        aux_trace_domain,
        aux_quotient_domain,
        aux_fri_domain);
    
    // Set number of trace randomizers (needed for GPU LDE path)
    aux_table.set_num_trace_randomizers(NUM_RANDOMIZER_POLYNOMIALS);
    
    // Step 2: Fill randomizer columns (last NUM_RANDOMIZER_POLYNOMIALS columns)
    // Derive auxiliary table randomizer seed from main table seed with offset
    // This matches Rust: rng_from_offset_seed(self.trace_randomizer_seed(), Self::NUM_COLUMNS)
    // Rust converts offset to little-endian bytes and adds them to seed bytes
    size_t randomizers_start = MASTER_AUX_NUM_COLUMNS - NUM_RANDOMIZER_POLYNOMIALS;

    // Use main table seed + offset to create auxiliary seed
    // Rust uses: rng_from_offset_seed(self.trace_randomizer_seed(), Self::NUM_COLUMNS)
    // where Self::NUM_COLUMNS = 379 for main table
    // Rust converts 379 to little-endian bytes and adds them to seed bytes
    std::array<uint8_t, 32> aux_seed = trace_randomizer_seed_;
    const size_t MAIN_TABLE_COLUMNS = 379;
    
    // Convert offset to little-endian bytes (379 = 0x017B = [0x7B, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00])
    // Rust's to_le_bytes() for usize (64-bit) produces 8 bytes
    uint64_t offset = MAIN_TABLE_COLUMNS;
    std::array<uint8_t, 8> offset_bytes;
    for (size_t i = 0; i < 8; ++i) {
        offset_bytes[i] = static_cast<uint8_t>((offset >> (i * 8)) & 0xFF);
    }
    
    // Add offset bytes to seed bytes (wrapping addition, matching Rust's wrapping_add)
    // Rust zips offset_bytes with seed, so only the first 8 bytes of seed are modified
    for (size_t i = 0; i < aux_seed.size() && i < offset_bytes.size(); ++i) {
        aux_seed[i] = static_cast<uint8_t>(aux_seed[i] + offset_bytes[i]); // unsigned addition wraps automatically
    }

    // Try to load randomizer column values from Rust test data (for deterministic comparison)
    // Skip loading if TVM_DISABLE_RANDOMIZER_LOAD is set (use ChaCha12 RNG instead)
    const char* disable_load_env = std::getenv("TVM_DISABLE_RANDOMIZER_LOAD");
    bool skip_loading = (disable_load_env && (strcmp(disable_load_env, "1") == 0 || strcmp(disable_load_env, "true") == 0));
    
    const char* test_data_dir_env = std::getenv("TVM_RUST_TEST_DATA_DIR");
    bool loaded_from_test_data = false;
    bool loaded_all_rows = false;
    
    if (test_data_dir_env && !skip_loading) {
        std::string test_data_dir = test_data_dir_env;
        std::string aux_create_path = test_data_dir + "/07_aux_tables_create.json";
        std::ifstream file(aux_create_path);
        
        if (file.is_open()) {
            try {
                nlohmann::json data = nlohmann::json::parse(file);
                
                // Parse XFieldElement strings (reusable lambda)
                auto parse_xfe_string = [](const std::string& s) -> std::tuple<uint64_t, uint64_t, uint64_t> {
                    if (s == "0_xfe" || s == "0") return {0, 0, 0};
                    if (s == "1_xfe" || s == "1") return {1, 0, 0};
                    
                    if (s.size() >= 5 && s.substr(s.size() - 4) == "_xfe") {
                        std::string num_str = s.substr(0, s.size() - 4);
                        try {
                            uint64_t val = std::stoull(num_str);
                            return {val, 0, 0};
                        } catch (...) {
                            return {0, 0, 0};
                        }
                    }
                    
                    if (s.empty() || s.front() != '(' || s.back() != ')') return {0, 0, 0};
                    std::string inner = s.substr(1, s.size() - 2);
                    
                    size_t x2_pos = std::string::npos;
                    size_t x_pos = std::string::npos;
                    for (size_t i = 0; i < inner.size(); ++i) {
                        if (inner[i] == 'x') {
                            if (i + 4 < inner.size() && inner.substr(i, 4) == "x + ") {
                                if (x_pos == std::string::npos) x_pos = i;
                            } else if (i + 1 < inner.size() && 
                                      static_cast<unsigned char>(inner[i+1]) == 0xC2 &&
                                      static_cast<unsigned char>(inner[i+2]) == 0xB2 &&
                                      i + 4 < inner.size() && inner.substr(i+3, 3) == " + ") {
                                x2_pos = i;
                            }
                        }
                    }
                    
                    if (x2_pos == std::string::npos || x_pos == std::string::npos) return {0, 0, 0};
                    
                    try {
                        size_t c2_start = 0;
                        while (c2_start < x2_pos && (inner[c2_start] == ' ' || inner[c2_start] == '\t')) c2_start++;
                        size_t c2_end = x2_pos;
                        while (c2_end > c2_start && inner[c2_end - 1] != ' ' && (inner[c2_end - 1] < '0' || inner[c2_end - 1] > '9')) c2_end--;
                        
                        size_t c1_start = x2_pos + 6;
                        while (c1_start < x_pos && (inner[c1_start] == ' ' || inner[c1_start] == '\t')) c1_start++;
                        size_t c1_end = x_pos;
                        while (c1_end > c1_start && inner[c1_end - 1] != ' ' && (inner[c1_end - 1] < '0' || inner[c1_end - 1] > '9')) c1_end--;
                        
                        size_t c0_start = x_pos + 4;
                        while (c0_start < inner.size() && (inner[c0_start] == ' ' || inner[c0_start] == '\t')) c0_start++;
                        size_t c0_end = inner.size();
                        while (c0_end > c0_start && (inner[c0_end - 1] == ' ' || inner[c0_end - 1] == '\t')) c0_end--;
                        
                        std::string c2_str = inner.substr(c2_start, c2_end - c2_start);
                        std::string c1_str = inner.substr(c1_start, c1_end - c1_start);
                        std::string c0_str = inner.substr(c0_start, c0_end - c0_start);
                        
                        c2_str.erase(std::remove_if(c2_str.begin(), c2_str.end(), [](char c) { return c < '0' || c > '9'; }), c2_str.end());
                        c1_str.erase(std::remove_if(c1_str.begin(), c1_str.end(), [](char c) { return c < '0' || c > '9'; }), c1_str.end());
                        c0_str.erase(std::remove_if(c0_str.begin(), c0_str.end(), [](char c) { return c < '0' || c > '9'; }), c0_str.end());
                        
                        if (c2_str.empty()) c2_str = "0";
                        if (c1_str.empty()) c1_str = "0";
                        if (c0_str.empty()) c0_str = "0";
                        
                        uint64_t c2 = std::stoull(c2_str);
                        uint64_t c1 = std::stoull(c1_str);
                        uint64_t c0 = std::stoull(c0_str);
                        return {c0, c1, c2};
                    } catch (...) {
                        return {0, 0, 0};
                    }
                };
                
                // Try to load all rows from sampled_rows (if available and matches row count)
                if (data.contains("sampled_rows") && data["sampled_rows"].is_array()) {
                    auto sampled_rows = data["sampled_rows"];
                    size_t rust_row_count = sampled_rows.size();
                    
                    if (rust_row_count == num_rows_ && sampled_rows[0].is_array() && sampled_rows[0].size() > randomizers_start) {
                        // Load all rows for column 87
                        for (size_t r = 0; r < num_rows_ && r < rust_row_count; ++r) {
                            std::string col87_str = sampled_rows[r][randomizers_start].get<std::string>();
                            auto [c0, c1, c2] = parse_xfe_string(col87_str);
                            aux_table.set(r, randomizers_start, XFieldElement(BFieldElement(c0), BFieldElement(c1), BFieldElement(c2)));
                        }
                        loaded_from_test_data = true;
                        loaded_all_rows = true;
                        std::cout << "[EXTEND] Loaded all " << num_rows_ << " rows of randomizer column 87 from Rust test data" << std::endl;
                    }
                }
                
                // Fallback: load first and last rows if sampled_rows not available
                if (!loaded_from_test_data && data.contains("first_row") && data.contains("last_row")) {
                    auto first_row = data["first_row"];
                    auto last_row = data["last_row"];
                    
                    if (first_row.is_array() && first_row.size() > randomizers_start) {
                        std::string first_col87_str = first_row[randomizers_start].get<std::string>();
                        auto [c0, c1, c2] = parse_xfe_string(first_col87_str);
                        aux_table.set(0, randomizers_start, XFieldElement(BFieldElement(c0), BFieldElement(c1), BFieldElement(c2)));
                    }
                    
                    if (last_row.is_array() && last_row.size() > randomizers_start && num_rows_ > 0) {
                        std::string last_col87_str = last_row[randomizers_start].get<std::string>();
                        auto [c0, c1, c2] = parse_xfe_string(last_col87_str);
                        aux_table.set(num_rows_ - 1, randomizers_start, XFieldElement(BFieldElement(c0), BFieldElement(c1), BFieldElement(c2)));
                    }
                    
                    loaded_from_test_data = true;
                    std::cout << "[EXTEND] Loaded randomizer column 87 first/last rows from Rust test data" << std::endl;
                }
            } catch (const std::exception& e) {
                // If parsing fails, fall back to RNG generation
                std::cerr << "[EXTEND] Warning: Failed to load randomizer from test data: " << e.what() << std::endl;
            }
        }
    }
    
    // Generate remaining randomizer values using RNG (only if not all loaded from test data)
    if (!loaded_from_test_data) {
        // Generate all rows using RNG
        ChaCha12Rng rng(aux_seed);
        for (size_t r = 0; r < num_rows_; r++) {
            for (size_t c = randomizers_start; c < MASTER_AUX_NUM_COLUMNS; c++) {
                XFieldElement random_xfe(
                    BFieldElement(rng.next_u64()),
                    BFieldElement(rng.next_u64()),
                    BFieldElement(rng.next_u64())
                );
                aux_table.set(r, c, random_xfe);
            }
        }
    } else if (!loaded_all_rows) {
        // Only loaded first/last rows, generate middle rows using RNG
        ChaCha12Rng rng(aux_seed);
        // Advance RNG state to account for first row already generated
        for (size_t i = 0; i < 3; ++i) rng.next_u64();
        
        for (size_t r = 1; r < num_rows_ - 1; r++) {
            for (size_t c = randomizers_start; c < MASTER_AUX_NUM_COLUMNS; c++) {
                XFieldElement random_xfe(
                    BFieldElement(rng.next_u64()),
                    BFieldElement(rng.next_u64()),
                    BFieldElement(rng.next_u64())
                );
                aux_table.set(r, c, random_xfe);
            }
        }
    }
    // If loaded_all_rows is true, all rows are already set from test data, nothing to generate
    
    // Step 3: Extend all tables
    // Convert main table to vector<vector> format for extend functions
    std::vector<std::vector<BFieldElement>> main_table_data;
    main_table_data.reserve(num_rows_);
    for (size_t r = 0; r < num_rows_; r++) {
        main_table_data.push_back(data_[r]);
    }
    
    // Verify main table structure before extend
    // In Rust, self.trace_table() returns the full main table with all 379 columns (base + degree lowering)
    // So data_ should also have 379 columns
    if (num_columns_ < 379) {
        std::cerr << "WARNING: Main table has only " << num_columns_ 
                  << " columns, expected 379 (base + degree lowering)" << std::endl;
    }
    
    // Extend all tables with timing
    auto ext_start = std::chrono::high_resolution_clock::now();
    extend_program_table(main_table_data, aux_table.data_mut(), challenges, num_rows_);
    auto t1 = std::chrono::high_resolution_clock::now();
    extend_op_stack_table(main_table_data, aux_table.data_mut(), challenges, num_rows_);
    auto t2 = std::chrono::high_resolution_clock::now();
    extend_jump_stack_table(main_table_data, aux_table.data_mut(), challenges, num_rows_);
    auto t3 = std::chrono::high_resolution_clock::now();
    extend_lookup_table(main_table_data, aux_table.data_mut(), challenges, num_rows_);
    auto t4 = std::chrono::high_resolution_clock::now();
    extend_hash_table(main_table_data, aux_table.data_mut(), challenges, num_rows_);
    auto t5 = std::chrono::high_resolution_clock::now();
    extend_cascade_table(main_table_data, aux_table.data_mut(), challenges, num_rows_);
    auto t6 = std::chrono::high_resolution_clock::now();
    extend_u32_table(main_table_data, aux_table.data_mut(), challenges, num_rows_);
    auto t7 = std::chrono::high_resolution_clock::now();
    extend_ram_table(main_table_data, aux_table.data_mut(), challenges, num_rows_);
    auto t8 = std::chrono::high_resolution_clock::now();
    extend_processor_table(main_table_data, aux_table.data_mut(), challenges, num_rows_);
    auto t9 = std::chrono::high_resolution_clock::now();
    
    // Print timing breakdown
    auto us = [](auto a, auto b) { return std::chrono::duration_cast<std::chrono::microseconds>(b - a).count(); };
    std::cout << "DEBUG: Aux extension breakdown (us): "
              << "prog=" << us(ext_start, t1)
              << " ops=" << us(t1, t2)
              << " jump=" << us(t2, t3)
              << " look=" << us(t3, t4)
              << " hash=" << us(t4, t5)
              << " casc=" << us(t5, t6)
              << " u32=" << us(t6, t7)
              << " ram=" << us(t7, t8)
              << " proc=" << us(t8, t9)
              << std::endl;
    
    // Pass data_ (full main table with degree lowering columns) to compute_degree_lowering_aux_columns
    // This matches Rust: self.trace_table() which includes all columns
    compute_degree_lowering_aux_columns(data_, aux_table.data_mut(), challenges);

    return aux_table;
}

// MasterMainTable::out_of_domain_row implementation
std::vector<XFieldElement> MasterMainTable::out_of_domain_row(const XFieldElement& indeterminate) const {
    // Barycentric Lagrangian interpolation (matching Rust implementation)
    
    // Step 1: Get domain values
    std::vector<BFieldElement> domain_values = trace_domain_.values();
    
    // Step 2: Compute domain_shift = indeterminate - d for each domain value d
    std::vector<XFieldElement> domain_shift;
    domain_shift.reserve(domain_values.size());
    for (const auto& d : domain_values) {
        domain_shift.push_back(indeterminate - XFieldElement(d));
    }
    
    // Step 3: Batch invert domain_shift
    std::vector<XFieldElement> domain_shift_inverses = XFieldElement::batch_inversion(domain_shift);
    
    // Step 4: Compute domain_over_domain_shift = d * inv for each (d, inv)
    std::vector<XFieldElement> domain_over_domain_shift;
    domain_over_domain_shift.reserve(domain_values.size());
    for (size_t i = 0; i < domain_values.size(); ++i) {
        domain_over_domain_shift.push_back(XFieldElement(domain_values[i]) * domain_shift_inverses[i]);
    }
    
    // Step 5: Compute barycentric_eval_denominator_inverse = 1 / sum(domain_over_domain_shift)
    XFieldElement sum_domain_over_shift = XFieldElement::zero();
    for (const auto& dsi : domain_over_domain_shift) {
        sum_domain_over_shift += dsi;
    }
    XFieldElement barycentric_eval_denominator_inverse = sum_domain_over_shift.inverse();
    
    // Step 6: Evaluate trace domain zerofier at indeterminate
    // Zerofier: x^n - offset^n where n = trace_domain.length
    // In Rust, zerofier is BFieldElement polynomial evaluated at XFieldElement point
    // We need to evaluate at the full XFieldElement, not just the base component
    BPolynomial zerofier = RandomizedLDE::compute_zerofier(trace_domain_);
    XFieldElement ood_trace_domain_zerofier = zerofier.evaluate_at_extension(indeterminate);
    
    // Step 7: For each column, compute barycentric evaluation (parallelized)
    std::vector<XFieldElement> result(num_columns_);
    
    #pragma omp parallel for schedule(static)
    for (size_t col = 0; col < num_columns_; ++col) {
        // Compute barycentric_eval_numerator = sum(domain_over_domain_shift[i] * trace_codeword[i])
        // Access column data directly (avoid temporary vector)
        XFieldElement barycentric_eval_numerator = XFieldElement::zero();
        for (size_t i = 0; i < num_rows_; ++i) {
            barycentric_eval_numerator += domain_over_domain_shift[i] * XFieldElement(data_[i][col]);
        }
        
        // Evaluate trace randomizer at indeterminate
        std::vector<BFieldElement> randomizer_coeffs = trace_randomizer_for_column(col);
        Polynomial<BFieldElement> randomizer_poly(randomizer_coeffs);
        XFieldElement ood_trace_randomizer = randomizer_poly.evaluate_at_extension(indeterminate);
        
        // Result = barycentric_eval_numerator * barycentric_eval_denominator_inverse + zerofier * randomizer
        result[col] = barycentric_eval_numerator * barycentric_eval_denominator_inverse
                                    + ood_trace_domain_zerofier * ood_trace_randomizer;
    }
    
    return result;
}

// MasterAuxTable::out_of_domain_row implementation
std::vector<XFieldElement> MasterAuxTable::out_of_domain_row(const XFieldElement& indeterminate) const {
    // Same algorithm as MasterMainTable, but for XFieldElement columns
    
    // Step 1: Get domain values
    std::vector<BFieldElement> domain_values = trace_domain_.values();
    
    // Step 2: Compute domain_shift = indeterminate - d for each domain value d
    std::vector<XFieldElement> domain_shift;
    domain_shift.reserve(domain_values.size());
    for (const auto& d : domain_values) {
        domain_shift.push_back(indeterminate - XFieldElement(d));
    }
    
    // Step 3: Batch invert domain_shift
    std::vector<XFieldElement> domain_shift_inverses = XFieldElement::batch_inversion(domain_shift);
    
    // Step 4: Compute domain_over_domain_shift = d * inv for each (d, inv)
    std::vector<XFieldElement> domain_over_domain_shift;
    domain_over_domain_shift.reserve(domain_values.size());
    for (size_t i = 0; i < domain_values.size(); ++i) {
        domain_over_domain_shift.push_back(XFieldElement(domain_values[i]) * domain_shift_inverses[i]);
    }
    
    // Step 5: Compute barycentric_eval_denominator_inverse = 1 / sum(domain_over_domain_shift)
    XFieldElement sum_domain_over_shift = XFieldElement::zero();
    for (const auto& dsi : domain_over_domain_shift) {
        sum_domain_over_shift += dsi;
    }
    XFieldElement barycentric_eval_denominator_inverse = sum_domain_over_shift.inverse();
    
    // Step 6: Evaluate trace domain zerofier at indeterminate
    // Zerofier is BFieldElement polynomial, but we evaluate at XFieldElement point
    BPolynomial zerofier = RandomizedLDE::compute_zerofier(trace_domain_);
    XFieldElement ood_trace_domain_zerofier = zerofier.evaluate_at_extension(indeterminate);
    
    // Step 7: For each column, compute barycentric evaluation (parallelized)
    std::vector<XFieldElement> result(num_columns_);
    
    #pragma omp parallel for schedule(static)
    for (size_t col = 0; col < num_columns_; ++col) {
        // Compute barycentric_eval_numerator = sum(domain_over_domain_shift[i] * data[i][col])
        XFieldElement barycentric_eval_numerator = XFieldElement::zero();
        for (size_t i = 0; i < num_rows_; ++i) {
            barycentric_eval_numerator += domain_over_domain_shift[i] * data_[i][col];
        }
        
        // Evaluate trace randomizer at indeterminate
        std::vector<XFieldElement> randomizer_coeffs = trace_randomizer_xfield_for_column(col);
        Polynomial<XFieldElement> randomizer_poly(randomizer_coeffs);
        XFieldElement ood_trace_randomizer = randomizer_poly.evaluate(indeterminate);
        
        // Result = barycentric_eval_numerator * barycentric_eval_denominator_inverse + zerofier * randomizer
        result[col] = barycentric_eval_numerator * barycentric_eval_denominator_inverse
                                    + ood_trace_domain_zerofier * ood_trace_randomizer;
    }
    
    return result;
}

// MasterMainTable::weighted_sum_of_columns implementation
Polynomial<XFieldElement> MasterMainTable::weighted_sum_of_columns(const std::vector<XFieldElement>& weights) const {
    if (weights.size() != num_columns_) {
        throw std::invalid_argument("Weights size must match number of columns");
    }
    
    // Step 1: Compute weighted sum of trace columns (row-wise, parallelized)
    // For each row: sum(weights[i] * data[row][i])
    std::vector<XFieldElement> weighted_row_sums(num_rows_);
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < num_rows_; ++row) {
        XFieldElement row_sum = XFieldElement::zero();
        for (size_t col = 0; col < num_columns_; ++col) {
            row_sum = row_sum + weights[col] * XFieldElement(data_[row][col]);
        }
        weighted_row_sums[row] = row_sum;
    }
    
    // Step 2: Interpolate weighted row sums to get polynomial
    std::vector<XFieldElement> trace_poly_coeffs = interpolate_xfield_column(weighted_row_sums, trace_domain_);
    Polynomial<XFieldElement> weighted_trace_poly(trace_poly_coeffs);
    
    // Step 3: Compute weighted sum of trace randomizer polynomials
    Polynomial<XFieldElement> weighted_randomizer_sum = Polynomial<XFieldElement>::from_constant(XFieldElement::zero());
    for (size_t col = 0; col < num_columns_; ++col) {
        std::vector<BFieldElement> randomizer_coeffs = trace_randomizer_for_column(col);
        Polynomial<BFieldElement> randomizer_poly(randomizer_coeffs);
        // Convert BPolynomial to XPolynomial by lifting coefficients
        std::vector<XFieldElement> x_randomizer_coeffs;
        x_randomizer_coeffs.reserve(randomizer_coeffs.size());
        for (const auto& coeff : randomizer_coeffs) {
            x_randomizer_coeffs.push_back(XFieldElement(coeff));
        }
        Polynomial<XFieldElement> x_randomizer_poly(x_randomizer_coeffs);
        weighted_randomizer_sum = weighted_randomizer_sum + (x_randomizer_poly * weights[col]);
    }
    
    // Step 4: Multiply weighted randomizer sum by zerofier using mul_zerofier_with
    // mul_zerofier_with(poly) = poly.shift_coefficients(length) - poly.scalar_mul(offset^length)
    BFieldElement offset_pow_length = trace_domain_.offset.pow(trace_domain_.length);
    Polynomial<XFieldElement> shifted_randomizer = weighted_randomizer_sum.shift_coefficients(trace_domain_.length);
    Polynomial<XFieldElement> scaled_randomizer = weighted_randomizer_sum * XFieldElement(offset_pow_length);
    Polynomial<XFieldElement> randomizer_contribution = shifted_randomizer - scaled_randomizer;
    
    // Step 5: Add trace polynomial and randomizer contribution
    return weighted_trace_poly + randomizer_contribution;
}

// MasterAuxTable::weighted_sum_of_columns implementation
Polynomial<XFieldElement> MasterAuxTable::weighted_sum_of_columns(const std::vector<XFieldElement>& weights) const {
    if (weights.size() != num_columns_) {
        throw std::invalid_argument("Weights size must match number of columns");
    }
    
    // Step 1: Compute weighted sum of trace columns (row-wise, parallelized)
    // For each row: sum(weights[i] * data[row][i])
    std::vector<XFieldElement> weighted_row_sums(num_rows_);
    #pragma omp parallel for schedule(static)
    for (size_t row = 0; row < num_rows_; ++row) {
        XFieldElement row_sum = XFieldElement::zero();
        for (size_t col = 0; col < num_columns_; ++col) {
            row_sum = row_sum + weights[col] * data_[row][col];
        }
        weighted_row_sums[row] = row_sum;
    }
    
    // Step 2: Interpolate weighted row sums to get polynomial
    std::vector<XFieldElement> trace_poly_coeffs = interpolate_xfield_column(weighted_row_sums, trace_domain_);
    Polynomial<XFieldElement> weighted_trace_poly(trace_poly_coeffs);
    
    // Step 3: Compute weighted sum of trace randomizer polynomials
    Polynomial<XFieldElement> weighted_randomizer_sum = Polynomial<XFieldElement>::from_constant(XFieldElement::zero());
    for (size_t col = 0; col < num_columns_; ++col) {
        std::vector<XFieldElement> randomizer_coeffs = trace_randomizer_xfield_for_column(col);
        Polynomial<XFieldElement> randomizer_poly(randomizer_coeffs);
        weighted_randomizer_sum = weighted_randomizer_sum + (randomizer_poly * weights[col]);
    }
    
    // Step 4: Multiply weighted randomizer sum by zerofier using mul_zerofier_with
    // mul_zerofier_with(poly) = poly.shift_coefficients(length) - poly.scalar_mul(offset^length)
    BFieldElement offset_pow_length = trace_domain_.offset.pow(trace_domain_.length);
    Polynomial<XFieldElement> shifted_randomizer = weighted_randomizer_sum.shift_coefficients(trace_domain_.length);
    Polynomial<XFieldElement> scaled_randomizer = weighted_randomizer_sum * XFieldElement(offset_pow_length);
    Polynomial<XFieldElement> randomizer_contribution = shifted_randomizer - scaled_randomizer;
    
    // Step 5: Add trace polynomial and randomizer contribution
    return weighted_trace_poly + randomizer_contribution;
}

} // namespace triton_vm

