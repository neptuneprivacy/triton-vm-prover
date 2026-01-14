#include "quotient/quotient.hpp"
#include "stark/challenges.hpp"
#include "ntt/ntt.hpp"
#include "common/debug_control.hpp"
#include <stdexcept>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <string>
#include <fstream>
#include <nlohmann/json.hpp>
#include <nlohmann/json.hpp>

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/quotient_compute_kernel.cuh"
#include "gpu/kernels/quotient_constraints_split_kernel.cuh"
#include <cuda_runtime.h>
#endif

namespace triton_vm {

std::vector<XFieldElement> interpolate_xfield_column(
    const std::vector<XFieldElement>& values,
    const ArithmeticDomain& domain
) {
    const size_t n = values.size();
    std::vector<BFieldElement> component0(n);
    std::vector<BFieldElement> component1(n);
    std::vector<BFieldElement> component2(n);
    for (size_t i = 0; i < n; ++i) {
        component0[i] = values[i].coeff(0);
        component1[i] = values[i].coeff(1);
        component2[i] = values[i].coeff(2);
    }

    auto coeff0 = NTT::interpolate(component0);
    auto coeff1 = NTT::interpolate(component1);
    auto coeff2 = NTT::interpolate(component2);

    std::vector<XFieldElement> coeffs(n);
    BFieldElement offset_inv = domain.offset.inverse();
    BFieldElement scale = BFieldElement::one();
    for (size_t i = 0; i < n; ++i) {
        coeffs[i] = XFieldElement(coeff0[i], coeff1[i], coeff2[i]) * scale;
        scale *= offset_inv;
    }
    return coeffs;
}

namespace {

XFieldElement evaluate_polynomial(
    const std::vector<XFieldElement>& coeffs,
    const XFieldElement& point
) {
    XFieldElement acc = XFieldElement::zero();
    for (auto it = coeffs.rbegin(); it != coeffs.rend(); ++it) {
        acc = acc * point + *it;
    }
    return acc;
}

} // namespace

// Proper segmentification matching Rust segmentify logic
// Takes quotient evaluations on quotient domain and rearranges them into segment polynomials
std::vector<std::vector<XFieldElement>> Quotient::segmentify_quotient_evaluations(
    const std::vector<XFieldElement>& quotient_evaluations,  // Evaluations on quotient domain
    size_t trace_length,  // Original trace length
    size_t num_segments,  // NUM_QUOTIENT_SEGMENTS = 4
    const ArithmeticDomain& fri_domain
) {
    const size_t quotient_length = quotient_evaluations.size();
    const size_t num_cosets = quotient_length / trace_length;  // Should be 4 for our case
    const size_t num_output_rows = quotient_length / num_segments;  // quotient_length / 4

    // Reshape quotient_evaluations into matrix: (trace_length, num_cosets)
    std::vector<std::vector<XFieldElement>> quotient_matrix(
        trace_length, std::vector<XFieldElement>(num_cosets));

    for (size_t row = 0; row < trace_length; ++row) {
        for (size_t col = 0; col < num_cosets; ++col) {
            size_t idx = row * num_cosets + col;
            if (idx < quotient_evaluations.size()) {
                quotient_matrix[row][col] = quotient_evaluations[idx];
            }
        }
    }

    // Create output matrix: (num_output_rows, num_segments)
    std::vector<std::vector<XFieldElement>> segment_matrix(
        num_output_rows, std::vector<XFieldElement>(num_segments, XFieldElement::zero()));

    // Apply the rearrangement logic from Rust segmentify
    // For each output position (output_row_idx, output_col_idx)
    for (size_t output_row_idx = 0; output_row_idx < num_output_rows; ++output_row_idx) {
        for (size_t output_col_idx = 0; output_col_idx < num_segments; ++output_col_idx) {
            // Calculate exponent_of_iota = output_row_idx + output_col_idx * num_output_rows
            size_t exponent_of_iota = output_row_idx + output_col_idx * num_output_rows;

            // Calculate input indices
            size_t input_row_idx = exponent_of_iota / num_cosets;
            size_t input_col_idx = exponent_of_iota % num_cosets;

            // Copy the value if indices are valid
            if (input_row_idx < trace_length && input_col_idx < num_cosets) {
                segment_matrix[output_row_idx][output_col_idx] =
                    quotient_matrix[input_row_idx][input_col_idx];
            }
        }
    }

    // Apply inverse NTT to each row to get polynomial coefficients
    // Rust uses segment_domain with offset = psi^NUM_SEGMENTS for interpolation
    // where psi = fri_domain.offset
    BFieldElement psi = fri_domain.offset;
    BFieldElement segment_domain_offset = psi.pow(num_segments);  // psi^NUM_SEGMENTS
    
    std::vector<std::vector<XFieldElement>> segment_polynomials(num_segments);

    for (size_t segment_idx = 0; segment_idx < num_segments; ++segment_idx) {
        // Extract the column for this segment
        std::vector<XFieldElement> segment_row(num_output_rows);
        for (size_t row = 0; row < num_output_rows; ++row) {
            segment_row[row] = segment_matrix[row][segment_idx];
        }

        // For coset interpolation with offset omega:
        // We have evaluations f(omega * g^i) where g is the generator
        // Define h(x) = f(omega * x), so h(g^i) = f(omega * g^i)
        // Interpolate h(g^i) to get h(x), then f(x) = h(x/omega)
        // So coefficient of x^i in f is coefficient of x^i in h divided by omega^i
        const size_t n = segment_row.size();
        std::vector<BFieldElement> component0(n);
        std::vector<BFieldElement> component1(n);
        std::vector<BFieldElement> component2(n);
        
        // Extract components (these are h(g^i) = f(omega * g^i))
        for (size_t i = 0; i < n; ++i) {
            component0[i] = segment_row[i].coeff(0);
            component1[i] = segment_row[i].coeff(1);
            component2[i] = segment_row[i].coeff(2);
        }

        // Interpolate to get h(x)
        auto coeff0 = NTT::interpolate(component0);
        auto coeff1 = NTT::interpolate(component1);
        auto coeff2 = NTT::interpolate(component2);

        // Scale coefficients by omega^-i to get f(x) = h(x/omega)
        // Coefficient of x^i in f is coefficient of x^i in h divided by omega^i
        std::vector<XFieldElement> coeffs(n);
        BFieldElement offset_inv = segment_domain_offset.inverse();
        BFieldElement offset_inv_power = BFieldElement::one();
        for (size_t i = 0; i < n; ++i) {
            coeffs[i] = XFieldElement(
                coeff0[i] * offset_inv_power,
                coeff1[i] * offset_inv_power,
                coeff2[i] * offset_inv_power
            );
            offset_inv_power = offset_inv_power * offset_inv;
        }

        // Store as polynomial coefficients
        segment_polynomials[segment_idx] = coeffs;
    }

    return segment_polynomials;
}

std::vector<std::vector<XFieldElement>> Quotient::compute_quotient(
    const MasterMainTable& main_table,
    const MasterAuxTable& aux_table,
    const Challenges& challenges,
    const std::vector<XFieldElement>& quotient_weights,
    const ArithmeticDomain& fri_domain,
    std::vector<std::vector<XFieldElement>>* out_segment_polynomials,
    std::vector<XFieldElement>* out_quotient_values
) {
    auto quotient_start = std::chrono::high_resolution_clock::now();
    std::cout << "Computing quotient polynomial..." << std::endl;

    if (main_table.num_rows() != aux_table.num_rows()) {
        throw std::invalid_argument("Main and auxiliary tables must have same number of rows");
    }
    if (quotient_weights.size() != MASTER_AUX_NUM_CONSTRAINTS) {
        throw std::invalid_argument("Quotient weight vector has incorrect length");
    }

    auto ensure_domain = [](ArithmeticDomain domain, size_t fallback) {
        if (domain.length == 0) {
            return ArithmeticDomain::of_length(fallback);
        }
        return domain;
    };

    const size_t num_trace_rows = main_table.num_rows();
    const size_t main_width = main_table.num_columns();
    const size_t aux_width = aux_table.num_columns();

    ArithmeticDomain trace_domain = ensure_domain(main_table.trace_domain(), num_trace_rows);
    // Quotient domain should be loaded from table (may be 4x or 8x trace length depending on parameters)
    // Use a reasonable fallback if not set (4x is minimum, but actual may be larger)
    ArithmeticDomain quotient_domain = ensure_domain(main_table.quotient_domain(), num_trace_rows * 4);
    ArithmeticDomain aux_trace_domain = ensure_domain(aux_table.trace_domain(), num_trace_rows);

    const size_t quotient_len = quotient_domain.length;
    const size_t trace_len = trace_domain.length;

    // DEBUG: Print domain lengths used in quotient computation
    if (const char* env = std::getenv("TVM_DEBUG_QUOTIENT")) {
        std::cout << "DEBUG Quotient::compute_quotient:" << std::endl;
        std::cout << "  main_table.num_rows(): " << num_trace_rows << std::endl;
        std::cout << "  main_table.quotient_domain().length: " << main_table.quotient_domain().length << std::endl;
        std::cout << "  main_table.trace_domain().length: " << main_table.trace_domain().length << std::endl;
        std::cout << "  quotient_domain.length (after ensure_domain): " << quotient_len << std::endl;
        std::cout << "  trace_domain.length (after ensure_domain): " << trace_len << std::endl;
    }

    if (trace_len == 0 || quotient_len == 0 || quotient_len % trace_len != 0) {
        throw std::invalid_argument("Invalid trace and quotient domain lengths");
    }

    // IMPORTANT: Match Rust's cached quotient calculation path.
    // Rust's `all_quotients_combined` expects LDE tables on the quotient domain, INCLUDING trace randomizers.
    // In our prover pipeline, `main_table.low_degree_extend(...)` and `aux_table.low_degree_extend(...)`
    // already produce randomized LDE tables. Recomputing LDE here from the trace table would omit randomizers
    // and diverge from Rust (this was the cause of the mismatch).
    if (!main_table.has_lde()) {
        throw std::runtime_error("Main table must have cached LDE before computing quotient. Call main_table.low_degree_extend(...) first.");
    }
    if (!aux_table.has_low_degree_extension()) {
        throw std::runtime_error("Aux table must have cached LDE before computing quotient. Call aux_table.low_degree_extend(...) first.");
    }

    const auto& main_lde_full = main_table.lde_table(); // length = evaluation domain length (often max(fri, quotient))
    const auto& aux_lde_full = aux_table.lde_table();

    if (main_lde_full.empty() || aux_lde_full.empty()) {
        throw std::runtime_error("Cached LDE tables are empty.");
    }

    // Downsample cached evaluation-domain table to the quotient domain exactly like Rust's `quotient_domain_table`.
    // If nrows == quotient_len, this is identity. If nrows > quotient_len, slice with step nrows/quotient_len.
    const size_t main_nrows = main_lde_full.size();
    const size_t aux_nrows = aux_lde_full.size();
    if (main_nrows != aux_nrows) {
        throw std::runtime_error("Main and aux cached LDE tables must have the same number of rows.");
    }
    if (main_nrows < quotient_len) {
        throw std::runtime_error("Cached LDE table shorter than quotient domain length.");
    }
    if (main_nrows % quotient_len != 0) {
        throw std::runtime_error("Cached LDE table length must be a multiple of quotient domain length.");
    }
    const size_t unit_distance_eval_to_quot = main_nrows / quotient_len;

    std::vector<std::vector<BFieldElement>> main_lde(quotient_len, std::vector<BFieldElement>(main_width));
    std::vector<std::vector<XFieldElement>> aux_lde(quotient_len, std::vector<XFieldElement>(aux_width, XFieldElement::zero()));
    for (size_t row = 0; row < quotient_len; ++row) {
        const size_t src_row = row * unit_distance_eval_to_quot;
        // main
        const auto& src_main_row = main_lde_full[src_row];
        for (size_t col = 0; col < main_width; ++col) {
            main_lde[row][col] = src_main_row[col];
        }
        // aux
        const auto& src_aux_row = aux_lde_full[src_row];
        for (size_t col = 0; col < aux_width; ++col) {
            aux_lde[row][col] = src_aux_row[col];
        }
    }

    // Debug: dump main_lde subsample
    std::cerr << "[DEBUG] CPU: About to dump main_lde subsample, quotient_len=" << quotient_len << ", main_width=" << main_width << std::endl;
    {
        std::ofstream debug_file("/tmp/cpu_quotient_main_lde.bin", std::ios::binary);
        for (size_t row = 0; row < quotient_len; ++row) {
            for (size_t col = 0; col < main_width; ++col) {
                uint64_t val = main_lde[row][col].value();
                debug_file.write(reinterpret_cast<char*>(&val), sizeof(uint64_t));
            }
        }
        debug_file.close();
        std::cerr << "[DEBUG] CPU: dumped main_lde subsample to /tmp/cpu_quotient_main_lde.bin" << std::endl;
    }

    auto init_inv = initial_zerofier_inverse(quotient_domain);
    auto cons_inv = consistency_zerofier_inverse(trace_domain, quotient_domain);
    auto tran_inv = transition_zerofier_inverse(trace_domain, quotient_domain);
    auto term_inv = terminal_zerofier_inverse(trace_domain, quotient_domain);

    // DEBUG: Dump zerofier inverses for comparison
    if (const char* env = std::getenv("TVM_DEBUG_QUOTIENT")) {
        std::string debug_dir = env;
        std::ofstream zf_file(debug_dir + "/quotient_zerofier_inverses.json");
        nlohmann::json zf_json;
        zf_json["quotient_len"] = quotient_len;
        zf_json["trace_len"] = trace_len;
        zf_json["init_inv"] = nlohmann::json::array();
        zf_json["cons_inv"] = nlohmann::json::array();
        zf_json["tran_inv"] = nlohmann::json::array();
        zf_json["term_inv"] = nlohmann::json::array();
        for (size_t i = 0; i < std::min(size_t(10), quotient_len); ++i) {
            zf_json["init_inv"].push_back(init_inv[i].value());
            zf_json["cons_inv"].push_back(cons_inv[i].value());
            zf_json["tran_inv"].push_back(tran_inv[i].value());
            zf_json["term_inv"].push_back(term_inv[i].value());
        }
        zf_file << zf_json.dump(2) << std::endl;
        zf_file.close();
        std::cout << "DEBUG: Dumped zerofier inverses to " << debug_dir << "/quotient_zerofier_inverses.json" << std::endl;
    }

    const size_t init_end = NUM_INITIAL_CONSTRAINTS;
    const size_t cons_end = init_end + NUM_CONSISTENCY_CONSTRAINTS;
    const size_t tran_end = cons_end + NUM_TRANSITION_CONSTRAINTS;

    const size_t unit_distance = quotient_len / trace_len;
    std::vector<XFieldElement> quotient_values(quotient_len, XFieldElement::zero());
    
    // DEBUG: Store intermediate constraint values for first row (declared outside parallel region)
    std::vector<XFieldElement> debug_first_row_initial;
    std::vector<XFieldElement> debug_first_row_consistency;
    std::vector<XFieldElement> debug_first_row_transition;
    std::vector<XFieldElement> debug_first_row_terminal;
    XFieldElement debug_first_row_quotient = XFieldElement::zero();

    auto weighted_sum = [&](const std::vector<XFieldElement>& values, size_t offset) {
        XFieldElement acc = XFieldElement::zero();
        for (size_t i = 0; i < values.size(); ++i) {
            acc += values[i] * quotient_weights[offset + i];
        }
        return acc;
    };

#ifdef TRITON_CUDA_ENABLED
    // GPU constraint evaluation path
    // Default: enabled. Set `TVM_USE_GPU_CONSTRAINTS=0` to force CPU constraints.
    bool use_gpu_constraints = true;
    if (const char* env = std::getenv("TVM_USE_GPU_CONSTRAINTS")) {
        use_gpu_constraints = std::string(env) != "0";
    }
    if (use_gpu_constraints) {
        auto gpu_start = std::chrono::high_resolution_clock::now();
        
        // Flatten main_lde to GPU format: [row0_col0, row0_col1, ..., row1_col0, ...]
        std::vector<uint64_t> h_main_lde(quotient_len * main_width);
        for (size_t row = 0; row < quotient_len; ++row) {
            for (size_t col = 0; col < main_width; ++col) {
                h_main_lde[row * main_width + col] = main_lde[row][col].value();
            }
        }
        
        // Flatten aux_lde: XFE has 3 components
        std::vector<uint64_t> h_aux_lde(quotient_len * aux_width * 3);
        for (size_t row = 0; row < quotient_len; ++row) {
            for (size_t col = 0; col < aux_width; ++col) {
                size_t base = row * aux_width * 3 + col * 3;
                h_aux_lde[base + 0] = aux_lde[row][col].coeff(0).value();
                h_aux_lde[base + 1] = aux_lde[row][col].coeff(1).value();
                h_aux_lde[base + 2] = aux_lde[row][col].coeff(2).value();
            }
        }
        
        // Flatten challenges (63 XFieldElements)
        std::vector<uint64_t> h_challenges(Challenges::COUNT * 3);
        for (size_t i = 0; i < Challenges::COUNT; ++i) {
            h_challenges[i * 3 + 0] = challenges[i].coeff(0).value();
            h_challenges[i * 3 + 1] = challenges[i].coeff(1).value();
            h_challenges[i * 3 + 2] = challenges[i].coeff(2).value();
        }
        
        // Flatten weights (596 XFieldElements)
        std::vector<uint64_t> h_weights(quotient_weights.size() * 3);
        for (size_t i = 0; i < quotient_weights.size(); ++i) {
            h_weights[i * 3 + 0] = quotient_weights[i].coeff(0).value();
            h_weights[i * 3 + 1] = quotient_weights[i].coeff(1).value();
            h_weights[i * 3 + 2] = quotient_weights[i].coeff(2).value();
        }
        
        // Flatten zerofier inverses (BFieldElements)
        std::vector<uint64_t> h_init_inv(quotient_len);
        std::vector<uint64_t> h_cons_inv(quotient_len);
        std::vector<uint64_t> h_tran_inv(quotient_len);
        std::vector<uint64_t> h_term_inv(quotient_len);
        for (size_t i = 0; i < quotient_len; ++i) {
            h_init_inv[i] = init_inv[i].value();
            h_cons_inv[i] = cons_inv[i].value();
            h_tran_inv[i] = tran_inv[i].value();
            h_term_inv[i] = term_inv[i].value();
        }
        
        // Allocate GPU memory
        uint64_t* d_main_lde;
        uint64_t* d_aux_lde;
        uint64_t* d_challenges;
        uint64_t* d_weights;
        uint64_t* d_init_inv;
        uint64_t* d_cons_inv;
        uint64_t* d_tran_inv;
        uint64_t* d_term_inv;
        uint64_t* d_out_init;
        uint64_t* d_out_cons;
        uint64_t* d_out_tran;
        uint64_t* d_out_term;
        uint64_t* d_out_tran_parts[4];
        uint64_t* d_quotient;
        
        cudaMalloc(&d_main_lde, h_main_lde.size() * sizeof(uint64_t));
        cudaMalloc(&d_aux_lde, h_aux_lde.size() * sizeof(uint64_t));
        cudaMalloc(&d_challenges, h_challenges.size() * sizeof(uint64_t));
        cudaMalloc(&d_weights, h_weights.size() * sizeof(uint64_t));
        cudaMalloc(&d_init_inv, quotient_len * sizeof(uint64_t));
        cudaMalloc(&d_cons_inv, quotient_len * sizeof(uint64_t));
        cudaMalloc(&d_tran_inv, quotient_len * sizeof(uint64_t));
        cudaMalloc(&d_term_inv, quotient_len * sizeof(uint64_t));
        cudaMalloc(&d_out_init, quotient_len * 3 * sizeof(uint64_t));
        cudaMalloc(&d_out_cons, quotient_len * 3 * sizeof(uint64_t));
        cudaMalloc(&d_out_tran, quotient_len * 3 * sizeof(uint64_t));
        cudaMalloc(&d_out_term, quotient_len * 3 * sizeof(uint64_t));
        for (int i = 0; i < 4; ++i) {
            cudaMalloc(&d_out_tran_parts[i], quotient_len * 3 * sizeof(uint64_t));
        }
        cudaMalloc(&d_quotient, quotient_len * 3 * sizeof(uint64_t));
        
        // Upload to GPU
        cudaMemcpy(d_main_lde, h_main_lde.data(), h_main_lde.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_aux_lde, h_aux_lde.data(), h_aux_lde.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_challenges, h_challenges.data(), h_challenges.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_weights, h_weights.data(), h_weights.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_init_inv, h_init_inv.data(), quotient_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_cons_inv, h_cons_inv.data(), quotient_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_tran_inv, h_tran_inv.data(), quotient_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
        cudaMemcpy(d_term_inv, h_term_inv.data(), quotient_len * sizeof(uint64_t), cudaMemcpyHostToDevice);
        
        // Launch kernels
        // Initial, Consistency, Terminal
        gpu::kernels::compute_quotient_split_partial(
            d_main_lde, main_width,
            d_aux_lde, aux_width,
            quotient_len,
            d_challenges, d_weights,
            d_init_inv, d_cons_inv, d_term_inv,
            d_out_init, d_out_cons, d_out_term,
            nullptr
        );
        
        // Transition parts 0-3
        gpu::kernels::launch_quotient_transition_part0(
            d_main_lde, main_width, d_aux_lde, aux_width,
            quotient_len, unit_distance, d_challenges, d_weights,
            d_out_tran_parts[0], nullptr
        );
        gpu::kernels::launch_quotient_transition_part1(
            d_main_lde, main_width, d_aux_lde, aux_width,
            quotient_len, unit_distance, d_challenges, d_weights,
            d_out_tran_parts[1], nullptr
        );
        gpu::kernels::launch_quotient_transition_part2(
            d_main_lde, main_width, d_aux_lde, aux_width,
            quotient_len, unit_distance, d_challenges, d_weights,
            d_out_tran_parts[2], nullptr
        );
        gpu::kernels::launch_quotient_transition_part3(
            d_main_lde, main_width, d_aux_lde, aux_width,
            quotient_len, unit_distance, d_challenges, d_weights,
            d_out_tran_parts[3], nullptr
        );
        
        // Combine transition parts: d_out_tran = sum of d_out_tran_parts[0..3]
        cudaMemcpy(d_out_tran, d_out_tran_parts[0], quotient_len * 3 * sizeof(uint64_t), cudaMemcpyDeviceToDevice);
        gpu::kernels::add_xfield_arrays(d_out_tran, d_out_tran_parts[1], quotient_len, nullptr);
        gpu::kernels::add_xfield_arrays(d_out_tran, d_out_tran_parts[2], quotient_len, nullptr);
        gpu::kernels::add_xfield_arrays(d_out_tran, d_out_tran_parts[3], quotient_len, nullptr);
        
        // Scale transition by tran_inv
        gpu::kernels::scale_xfield_by_bfield(d_out_tran, d_tran_inv, quotient_len, d_out_tran, nullptr);
        
        // Combine all: quotient = init + cons + tran + term
        gpu::kernels::combine_quotient_results(
            d_out_init, d_out_cons, d_out_tran, d_out_term,
            quotient_len, d_quotient, nullptr
        );
        
        // If requested, do quotient post-processing (interpolation + segment LDE) on GPU too.
        // Default: enabled. Set `TVM_USE_GPU_QUOTIENT_POST=0` to force CPU post-processing.
        bool use_gpu_post = true;
        if (const char* env = std::getenv("TVM_USE_GPU_QUOTIENT_POST")) {
            use_gpu_post = std::string(env) != "0";
        }

        if (use_gpu_post) {
            auto post_start = std::chrono::high_resolution_clock::now();

            const size_t num_segments = NUM_QUOTIENT_SEGMENTS;
            const size_t seg_len = quotient_len / num_segments;

            // Allocate GPU buffers for outputs
            uint64_t* d_seg_coeffs_compact = nullptr;           // num_segments*3*seg_len
            uint64_t* d_seg_codewords_colmajor = nullptr;        // num_segments*3*fri_len
            cudaMalloc(&d_seg_coeffs_compact, num_segments * 3 * seg_len * sizeof(uint64_t));
            cudaMalloc(&d_seg_codewords_colmajor, num_segments * 3 * fri_domain.length * sizeof(uint64_t));

            // quotient_offset_inv for coset interpolation
            uint64_t quotient_offset_inv = quotient_domain.offset.inverse().value();

            gpu::kernels::quotient_segmentify_and_lde_gpu(
                d_quotient,
                quotient_len,
                num_segments,
                quotient_offset_inv,
                fri_domain.offset.value(),
                fri_domain.length,
                d_seg_coeffs_compact,
                d_seg_codewords_colmajor,
                0
            );
            cudaDeviceSynchronize();

            // Download segment coefficients (compact) for Step 11 polynomial evals
            std::vector<uint64_t> h_seg_coeffs(num_segments * 3 * seg_len);
            cudaMemcpy(
                h_seg_coeffs.data(),
                d_seg_coeffs_compact,
                h_seg_coeffs.size() * sizeof(uint64_t),
                cudaMemcpyDeviceToHost
            );

            // Download segment codewords (column-major) and convert to CPU structure
            std::vector<uint64_t> h_seg_codewords(num_segments * 3 * fri_domain.length);
            cudaMemcpy(
                h_seg_codewords.data(),
                d_seg_codewords_colmajor,
                h_seg_codewords.size() * sizeof(uint64_t),
                cudaMemcpyDeviceToHost
            );

            std::vector<std::vector<XFieldElement>> segment_polynomials(num_segments);
            for (size_t seg = 0; seg < num_segments; ++seg) {
                segment_polynomials[seg].resize(seg_len);
                for (size_t i = 0; i < seg_len; ++i) {
                    uint64_t c0 = h_seg_coeffs[(seg * 3 + 0) * seg_len + i];
                    uint64_t c1 = h_seg_coeffs[(seg * 3 + 1) * seg_len + i];
                    uint64_t c2 = h_seg_coeffs[(seg * 3 + 2) * seg_len + i];
                    segment_polynomials[seg][i] = XFieldElement(BFieldElement(c0), BFieldElement(c1), BFieldElement(c2));
                }
            }

            std::vector<std::vector<XFieldElement>> segment_codewords(
                num_segments,
                std::vector<XFieldElement>(fri_domain.length, XFieldElement::zero())
            );
            for (size_t seg = 0; seg < num_segments; ++seg) {
                const uint64_t* base0 = &h_seg_codewords[(seg * 3 + 0) * fri_domain.length];
                const uint64_t* base1 = &h_seg_codewords[(seg * 3 + 1) * fri_domain.length];
                const uint64_t* base2 = &h_seg_codewords[(seg * 3 + 2) * fri_domain.length];
                for (size_t row = 0; row < fri_domain.length; ++row) {
                    segment_codewords[seg][row] = XFieldElement(
                        BFieldElement(base0[row]),
                        BFieldElement(base1[row]),
                        BFieldElement(base2[row])
                    );
                }
            }

            if (out_segment_polynomials != nullptr) {
                *out_segment_polynomials = segment_polynomials;
            }

            // Optional: output raw quotient codeword on host if requested (debug)
            if (out_quotient_values != nullptr) {
                std::vector<uint64_t> h_quotient(quotient_len * 3);
                cudaMemcpy(h_quotient.data(), d_quotient, quotient_len * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
                out_quotient_values->resize(quotient_len);
                for (size_t row = 0; row < quotient_len; ++row) {
                    (*out_quotient_values)[row] = XFieldElement(
                        BFieldElement(h_quotient[row * 3 + 0]),
                        BFieldElement(h_quotient[row * 3 + 1]),
                        BFieldElement(h_quotient[row * 3 + 2])
                    );
                }
            }

            cudaFree(d_seg_coeffs_compact);
            cudaFree(d_seg_codewords_colmajor);

            auto post_end = std::chrono::high_resolution_clock::now();
            double post_ms = std::chrono::duration_cast<std::chrono::milliseconds>(post_end - post_start).count();
            std::cout << "GPU: Quotient interpolation + segment LDE: " << post_ms << " ms" << std::endl;

            // Cleanup GPU constraint buffers before returning
            cudaFree(d_main_lde);
            cudaFree(d_aux_lde);
            cudaFree(d_challenges);
            cudaFree(d_weights);
            cudaFree(d_init_inv);
            cudaFree(d_cons_inv);
            cudaFree(d_tran_inv);
            cudaFree(d_term_inv);
            cudaFree(d_out_init);
            cudaFree(d_out_cons);
            cudaFree(d_out_tran);
            cudaFree(d_out_term);
            for (int i = 0; i < 4; ++i) {
                cudaFree(d_out_tran_parts[i]);
            }
            cudaFree(d_quotient);

            auto gpu_end = std::chrono::high_resolution_clock::now();
            auto gpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count() / 1000.0;
            std::cout << "GPU: Quotient constraint evaluation: " << gpu_ms << " ms" << std::endl;

            return segment_codewords;
        }

        cudaDeviceSynchronize();

        // Download results (CPU post-processing path)
        std::vector<uint64_t> h_quotient(quotient_len * 3);
        cudaMemcpy(h_quotient.data(), d_quotient, quotient_len * 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);

        for (size_t row = 0; row < quotient_len; ++row) {
            quotient_values[row] = XFieldElement(
                BFieldElement(h_quotient[row * 3 + 0]),
                BFieldElement(h_quotient[row * 3 + 1]),
                BFieldElement(h_quotient[row * 3 + 2])
            );
        }
        
        // Cleanup
        cudaFree(d_main_lde);
        cudaFree(d_aux_lde);
        cudaFree(d_challenges);
        cudaFree(d_weights);
        cudaFree(d_init_inv);
        cudaFree(d_cons_inv);
        cudaFree(d_tran_inv);
        cudaFree(d_term_inv);
        cudaFree(d_out_init);
        cudaFree(d_out_cons);
        cudaFree(d_out_tran);
        cudaFree(d_out_term);
        for (int i = 0; i < 4; ++i) {
            cudaFree(d_out_tran_parts[i]);
        }
        cudaFree(d_quotient);
        
        auto gpu_end = std::chrono::high_resolution_clock::now();
        auto gpu_ms = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end - gpu_start).count() / 1000.0;
        std::cout << "GPU: Quotient constraint evaluation: " << gpu_ms << " ms" << std::endl;
    } else
#endif
    {
    auto constraint_start = std::chrono::high_resolution_clock::now();
    // CPU constraint evaluation loop - using C++ constraint evaluators (parallelized, optional)
    // Control via TRITON_OMP_QUOTIENT (default: enabled if OpenMP available)
    static int omp_quotient_enabled = -1;
    if (omp_quotient_enabled == -1) {
        const char* env = std::getenv("TRITON_OMP_QUOTIENT");
        omp_quotient_enabled = (env == nullptr || (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
        if (omp_quotient_enabled) {
            std::cout << "Quotient: OpenMP parallelization enabled" << std::endl;
        } else {
            std::cout << "Quotient: OpenMP parallelization disabled" << std::endl;
        }
    }
    
#ifdef _OPENMP
    if (omp_quotient_enabled) {
        #pragma omp parallel for schedule(dynamic, 64)
        for (size_t row = 0; row < quotient_len; ++row) {
#else
    {
        for (size_t row = 0; row < quotient_len; ++row) {
#endif
        const size_t next_row = (row + unit_distance) % quotient_len;
        const auto& current_main_row = main_lde[row];
        const auto& current_aux_row = aux_lde[row];
        const auto& next_main_row = main_lde[next_row];
        const auto& next_aux_row = aux_lde[next_row];

        // Use C++ constraint evaluation
        auto initial_values = evaluate_initial_constraints(current_main_row, current_aux_row, challenges);
        XFieldElement initial_inner = weighted_sum(initial_values, 0);
        XFieldElement quotient_value = initial_inner * init_inv[row];

        auto consistency_values = evaluate_consistency_constraints(current_main_row, current_aux_row, challenges);
        XFieldElement consistency_inner = weighted_sum(consistency_values, init_end);
        quotient_value += consistency_inner * cons_inv[row];

        auto transition_values = evaluate_transition_constraints(
            current_main_row,
            current_aux_row,
            next_main_row,
            next_aux_row,
            challenges);
        XFieldElement transition_inner = weighted_sum(transition_values, cons_end);
        quotient_value += transition_inner * tran_inv[row];

        auto terminal_values = evaluate_terminal_constraints(current_main_row, current_aux_row, challenges);
        XFieldElement terminal_inner = weighted_sum(terminal_values, tran_end);
        quotient_value += terminal_inner * term_inv[row];

        quotient_values[row] = quotient_value;
    }
#ifdef _OPENMP
    }
#endif
    auto constraint_end = std::chrono::high_resolution_clock::now();
    double constraint_ms = std::chrono::duration_cast<std::chrono::milliseconds>(constraint_end - constraint_start).count();
    std::cout << "Quotient: Constraint evaluation (" << quotient_len << " rows): " << constraint_ms << " ms" << std::endl;
    }

    // Output quotient_values if requested
    if (out_quotient_values != nullptr) {
        *out_quotient_values = quotient_values;
    }
    
    // DEBUG: Dump quotient codeword (before interpolation)
    if (const char* env = std::getenv("TVM_DEBUG_QUOTIENT")) {
        std::string debug_dir = env;
        std::ofstream qc_file(debug_dir + "/quotient_codeword.json");
        nlohmann::json qc_json;
        qc_json["quotient_len"] = quotient_len;
        qc_json["quotient_values"] = nlohmann::json::array();
        for (size_t i = 0; i < std::min(size_t(20), quotient_len); ++i) {
            nlohmann::json qv;
            qv["c0"] = quotient_values[i].coeff(0).value();
            qv["c1"] = quotient_values[i].coeff(1).value();
            qv["c2"] = quotient_values[i].coeff(2).value();
            qc_json["quotient_values"].push_back(qv);
        }
        qc_file << qc_json.dump(2) << std::endl;
        qc_file.close();
        std::cout << "DEBUG: Dumped quotient codeword (first 20 rows) to " << debug_dir << "/quotient_codeword.json" << std::endl;
        
        // Dump first row constraint values (re-evaluate after parallel loop)
        const size_t first_row = 0;
        const size_t first_next_row = (first_row + unit_distance) % quotient_len;
        const auto& current_main_row = main_lde[first_row];
        const auto& current_aux_row = aux_lde[first_row];
        const auto& next_main_row = main_lde[first_next_row];
        const auto& next_aux_row = aux_lde[first_next_row];
        
        debug_first_row_initial = evaluate_initial_constraints(current_main_row, current_aux_row, challenges);
        debug_first_row_consistency = evaluate_consistency_constraints(current_main_row, current_aux_row, challenges);
        debug_first_row_transition = evaluate_transition_constraints(
            current_main_row, current_aux_row, next_main_row, next_aux_row, challenges);
        debug_first_row_terminal = evaluate_terminal_constraints(current_main_row, current_aux_row, challenges);
        debug_first_row_quotient = quotient_values[first_row];
        
        std::ofstream cr_file(debug_dir + "/quotient_first_row_constraints.json");
        nlohmann::json cr_json;
        cr_json["initial_constraints"] = nlohmann::json::array();
        cr_json["consistency_constraints"] = nlohmann::json::array();
        cr_json["transition_constraints"] = nlohmann::json::array();
        cr_json["terminal_constraints"] = nlohmann::json::array();
        for (const auto& v : debug_first_row_initial) {
            nlohmann::json xfe;
            xfe["c0"] = v.coeff(0).value();
            xfe["c1"] = v.coeff(1).value();
            xfe["c2"] = v.coeff(2).value();
            cr_json["initial_constraints"].push_back(xfe);
        }
        for (const auto& v : debug_first_row_consistency) {
            nlohmann::json xfe;
            xfe["c0"] = v.coeff(0).value();
            xfe["c1"] = v.coeff(1).value();
            xfe["c2"] = v.coeff(2).value();
            cr_json["consistency_constraints"].push_back(xfe);
        }
        for (const auto& v : debug_first_row_transition) {
            nlohmann::json xfe;
            xfe["c0"] = v.coeff(0).value();
            xfe["c1"] = v.coeff(1).value();
            xfe["c2"] = v.coeff(2).value();
            cr_json["transition_constraints"].push_back(xfe);
        }
        for (const auto& v : debug_first_row_terminal) {
            nlohmann::json xfe;
            xfe["c0"] = v.coeff(0).value();
            xfe["c1"] = v.coeff(1).value();
            xfe["c2"] = v.coeff(2).value();
            cr_json["terminal_constraints"].push_back(xfe);
        }
        nlohmann::json qv;
        qv["c0"] = debug_first_row_quotient.coeff(0).value();
        qv["c1"] = debug_first_row_quotient.coeff(1).value();
        qv["c2"] = debug_first_row_quotient.coeff(2).value();
        cr_json["quotient_value"] = qv;
        cr_file << cr_json.dump(2) << std::endl;
        cr_file.close();
        std::cout << "DEBUG: Dumped first row constraint values to " << debug_dir << "/quotient_first_row_constraints.json" << std::endl;
    }

    // Use cached path: interpolate quotient codeword, split into segments, evaluate on FRI domain
    // This matches Rust's interpolate_quotient_segments + fri_domain_segment_polynomials
    
    // Step 1: Interpolate quotient codeword on quotient domain
    // Rust: quotient_domain.interpolate(&quotient_codeword.to_vec())
    auto interp_start = std::chrono::high_resolution_clock::now();
    std::vector<XFieldElement> quotient_poly_coeffs = interpolate_xfield_column(quotient_values, quotient_domain);
    auto interp_end = std::chrono::high_resolution_clock::now();
    double interp_ms = std::chrono::duration_cast<std::chrono::milliseconds>(interp_end - interp_start).count();
    std::cout << "Quotient: Interpolation: " << interp_ms << " ms" << std::endl;
    
    // DEBUG: Dump interpolated polynomial coefficients
    if (const char* env = std::getenv("TVM_DEBUG_QUOTIENT")) {
        std::string debug_dir = env;
        std::ofstream pc_file(debug_dir + "/quotient_poly_coeffs.json");
        nlohmann::json pc_json;
        pc_json["num_coeffs"] = quotient_poly_coeffs.size();
        pc_json["coeffs"] = nlohmann::json::array();
        for (size_t i = 0; i < std::min(size_t(50), quotient_poly_coeffs.size()); ++i) {
            nlohmann::json coeff;
            coeff["c0"] = quotient_poly_coeffs[i].coeff(0).value();
            coeff["c1"] = quotient_poly_coeffs[i].coeff(1).value();
            coeff["c2"] = quotient_poly_coeffs[i].coeff(2).value();
            pc_json["coeffs"].push_back(coeff);
        }
        pc_file << pc_json.dump(2) << std::endl;
        pc_file.close();
        std::cout << "DEBUG: Dumped quotient polynomial coefficients (first 50) to " << debug_dir << "/quotient_poly_coeffs.json" << std::endl;
    }
    
    // Step 2: Split polynomial into segments (coefficient-based)
    // Rust: split_polynomial_into_segments(quotient_interpolation_poly)
    // This takes coefficients and splits them: c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7
    // into segments: [c_0, c_4], [c_1, c_5], [c_2, c_6], [c_3, c_7] for NUM_SEGMENTS=4
    std::vector<std::vector<XFieldElement>> segment_polynomials(NUM_QUOTIENT_SEGMENTS);
    for (size_t seg_idx = 0; seg_idx < NUM_QUOTIENT_SEGMENTS; ++seg_idx) {
        std::vector<XFieldElement> segment_coeffs;
        for (size_t i = seg_idx; i < quotient_poly_coeffs.size(); i += NUM_QUOTIENT_SEGMENTS) {
            segment_coeffs.push_back(quotient_poly_coeffs[i]);
        }
        segment_polynomials[seg_idx] = segment_coeffs;
    }
    
    // DEBUG: Dump segment polynomial coefficients
    if (const char* env = std::getenv("TVM_DEBUG_QUOTIENT")) {
        std::string debug_dir = env;
        std::ofstream sp_file(debug_dir + "/quotient_segment_polynomials.json");
        nlohmann::json sp_json;
        sp_json["num_segments"] = segment_polynomials.size();
        sp_json["segments"] = nlohmann::json::array();
        for (size_t seg = 0; seg < segment_polynomials.size(); ++seg) {
            nlohmann::json seg_json;
            seg_json["segment_index"] = seg;
            seg_json["num_coeffs"] = segment_polynomials[seg].size();
            seg_json["coeffs"] = nlohmann::json::array();
            for (size_t i = 0; i < std::min(size_t(20), segment_polynomials[seg].size()); ++i) {
                nlohmann::json coeff;
                coeff["c0"] = segment_polynomials[seg][i].coeff(0).value();
                coeff["c1"] = segment_polynomials[seg][i].coeff(1).value();
                coeff["c2"] = segment_polynomials[seg][i].coeff(2).value();
                seg_json["coeffs"].push_back(coeff);
            }
            sp_json["segments"].push_back(seg_json);
        }
        sp_file << sp_json.dump(2) << std::endl;
        sp_file.close();
        std::cout << "DEBUG: Dumped segment polynomial coefficients to " << debug_dir << "/quotient_segment_polynomials.json" << std::endl;
    }

    // Step 3: Evaluate each segment polynomial on FRI domain
    // Rust: fri_domain.evaluate(&segment_polynomial)
    auto lde_start = std::chrono::high_resolution_clock::now();
    std::vector<std::vector<XFieldElement>> segment_codewords(
        NUM_QUOTIENT_SEGMENTS,
        std::vector<XFieldElement>(fri_domain.length, XFieldElement::zero()));

    for (size_t segment = 0; segment < NUM_QUOTIENT_SEGMENTS; ++segment) {
        const auto& poly = segment_polynomials[segment];
        
        // Extract component coefficients
        std::vector<BFieldElement> coeff0, coeff1, coeff2;
        coeff0.reserve(poly.size());
        coeff1.reserve(poly.size());
        coeff2.reserve(poly.size());
        for (const auto& xfe : poly) {
            coeff0.push_back(xfe.coeff(0));
            coeff1.push_back(xfe.coeff(1));
            coeff2.push_back(xfe.coeff(2));
        }
        
        // Extend to FRI domain length (pad with zeros if needed)
        if (coeff0.size() < fri_domain.length) {
            coeff0.resize(fri_domain.length, BFieldElement::zero());
            coeff1.resize(fri_domain.length, BFieldElement::zero());
            coeff2.resize(fri_domain.length, BFieldElement::zero());
        } else if (coeff0.size() > fri_domain.length) {
            coeff0.resize(fri_domain.length);
            coeff1.resize(fri_domain.length);
            coeff2.resize(fri_domain.length);
        }
        
        // Evaluate using NTT (component-wise)
        std::vector<BFieldElement> eval0 = NTT::evaluate_on_coset(coeff0, fri_domain.length, fri_domain.offset);
        std::vector<BFieldElement> eval1 = NTT::evaluate_on_coset(coeff1, fri_domain.length, fri_domain.offset);
        std::vector<BFieldElement> eval2 = NTT::evaluate_on_coset(coeff2, fri_domain.length, fri_domain.offset);
        
        // Combine components back into XFieldElements
        for (size_t row = 0; row < fri_domain.length; ++row) {
            segment_codewords[segment][row] = XFieldElement(eval0[row], eval1[row], eval2[row]);
        }
    }

    auto lde_end = std::chrono::high_resolution_clock::now();
    double lde_ms = std::chrono::duration_cast<std::chrono::milliseconds>(lde_end - lde_start).count();
    std::cout << "Quotient: Segment LDE (" << NUM_QUOTIENT_SEGMENTS << " segments x " << fri_domain.length << " rows): " << lde_ms << " ms" << std::endl;

    if (out_segment_polynomials != nullptr) {
        *out_segment_polynomials = segment_polynomials;
    }

    std::cout << "✓ Quotient computation completed" << std::endl;
    std::cout << "  - Generated " << segment_codewords.size() << " segments" << std::endl;
    std::cout << "  - Total values: " << quotient_len << std::endl;

    return segment_codewords;
}

XFieldElement Quotient::evaluate_air(
    const XFieldElement& point,
    const std::vector<BFieldElement>& main_row,
    const std::vector<XFieldElement>& aux_row,
    const Challenges& challenges
) {
    // Evaluate Triton VM AIR constraints at the given point
    // This is a simplified implementation focusing on basic structure

    XFieldElement result = XFieldElement::zero();

    // TODO: Implement full AIR evaluation including:
    // - Processor table constraints (instruction execution)
    // - Memory table constraints (RAM operations)
    // - Jump stack constraints
    // - Hash table constraints (Tip5 sponge)
    // - Lookup table constraints
    // - Cascade table constraints
    // - U32 table constraints
    // - Boundary constraints (initial/final states)
    // - Transition constraints

    // Evaluate constraints for all Triton VM tables
    if (main_row.size() < 379) {  // Full master table size
        return result;  // Not enough data for full evaluation
    }

    // Extract table boundaries from main_row structure
    // Based on the table column offsets defined in extend_helpers.hpp

    // PROGRAM TABLE CONSTRAINTS (columns 0-6)
    XFieldElement program_constraints = evaluate_program_table_constraints(point, main_row, aux_row, challenges);

    // PROCESSOR TABLE CONSTRAINTS (columns 7-45)
    XFieldElement processor_constraints = evaluate_processor_table_constraints(point, main_row, aux_row, challenges);

    // OP STACK TABLE CONSTRAINTS (columns 46-49)
    XFieldElement op_stack_constraints = evaluate_op_stack_table_constraints(point, main_row, aux_row, challenges);

    // RAM TABLE CONSTRAINTS (columns 50-56)
    XFieldElement ram_constraints = evaluate_ram_table_constraints(point, main_row, aux_row, challenges);

    // JUMP STACK TABLE CONSTRAINTS (columns 57-61)
    XFieldElement jump_stack_constraints = evaluate_jump_stack_table_constraints(point, main_row, aux_row, challenges);

    // HASH TABLE CONSTRAINTS (columns 62-128)
    XFieldElement hash_constraints = evaluate_hash_table_constraints(point, main_row, aux_row, challenges);

    // CASCADE TABLE CONSTRAINTS (columns 129-134)
    XFieldElement cascade_constraints = evaluate_cascade_table_constraints(point, main_row, aux_row, challenges);

    // LOOKUP TABLE CONSTRAINTS (columns 135-138)
    XFieldElement lookup_constraints = evaluate_lookup_table_constraints(point, main_row, aux_row, challenges);

    // U32 TABLE CONSTRAINTS (columns 139-148)
    XFieldElement u32_constraints = evaluate_u32_table_constraints(point, main_row, aux_row, challenges);

    // Combine all table constraints
    result = program_constraints + processor_constraints + op_stack_constraints +
             ram_constraints + jump_stack_constraints + hash_constraints +
             cascade_constraints + lookup_constraints + u32_constraints;

    return result;
}

XFieldElement Quotient::evaluate_program_table_constraints(
    const XFieldElement& point,
    const std::vector<BFieldElement>& main_row,
    const std::vector<XFieldElement>& aux_row,
    const Challenges& challenges
) {
    // Program table constraints (columns 0-6)
    // Program table validates instruction loading and program structure

    XFieldElement result = XFieldElement::zero();

    // Check bounds before accessing
    if (main_row.size() < 7) return result;

    // Extract program table columns
    BFieldElement address = main_row[0];          // Address
    BFieldElement instruction = main_row[1];      // Instruction
    BFieldElement lookup_mult = main_row[2];      // LookupMultiplicity
    BFieldElement index_in_chunk = main_row[3];   // IndexInChunk
    BFieldElement is_padding = main_row[6];       // IsTablePadding

    // Initial constraints: First row should start at address 0
    if (point == XFieldElement::one()) {
        result = result + XFieldElement(address);  // Address should be 0
        result = result + XFieldElement(index_in_chunk);  // Index should be 0
    }

    // Consistency constraints: Padding rows should have specific properties
    if (is_padding != BFieldElement::zero()) {
        result = result + XFieldElement(instruction);  // Padding instruction should be 0
        result = result + XFieldElement(lookup_mult);  // Padding multiplicity should be 0
    }

    // Transition constraints: Address should increase properly
    // Simplified: address increases by 1 (single instruction per address)
    XFieldElement address_transition = XFieldElement(point) - XFieldElement(address) - XFieldElement::one();
    result = result + address_transition;

    return result;
}

XFieldElement Quotient::evaluate_processor_table_constraints(
    const XFieldElement& point,
    const std::vector<BFieldElement>& main_row,
    const std::vector<XFieldElement>& aux_row,
    const Challenges& challenges
) {
    // Processor table constraints (columns 7-45)
    // Processor table validates instruction execution

    XFieldElement result = XFieldElement::zero();

    // Check bounds before accessing
    const size_t PROC_OFFSET = 7;
    if (main_row.size() < PROC_OFFSET + 16) return result;

    // Extract key processor table columns (offset by program table columns)
    BFieldElement clk = main_row[PROC_OFFSET + 0];     // CLK
    BFieldElement ip = main_row[PROC_OFFSET + 2];      // IP
    BFieldElement ci = main_row[PROC_OFFSET + 3];      // CI
    BFieldElement nia = main_row[PROC_OFFSET + 4];     // NIA
    BFieldElement st0 = main_row[PROC_OFFSET + 15];    // ST0

    // Initial constraints: Execution starts with CLK=0, IP=0
    if (point == XFieldElement::one()) {
        result = result + XFieldElement(clk);      // CLK = 0
        result = result + XFieldElement(ip);       // IP = 0
        result = result + XFieldElement(st0);      // ST0 = 0 (empty stack)
    }

    // Consistency constraints: Program counter arithmetic
    XFieldElement pc_consistency = XFieldElement(nia) - XFieldElement(ip) - XFieldElement::one();
    result = result + pc_consistency;

    // Transition constraints: Clock and IP progression
    XFieldElement clk_transition = XFieldElement(point) - XFieldElement(clk) - XFieldElement::one();
    XFieldElement ip_transition = XFieldElement(point) - XFieldElement(nia);
    result = result + clk_transition + ip_transition;

    return result;
}

XFieldElement Quotient::evaluate_op_stack_table_constraints(
    const XFieldElement& point,
    const std::vector<BFieldElement>& main_row,
    const std::vector<XFieldElement>& aux_row,
    const Challenges& challenges
) {
    // Op Stack table constraints (columns 46-49)
    // Validates operand stack operations

    XFieldElement result = XFieldElement::zero();

    const size_t OP_STACK_OFFSET = 46;
    if (main_row.size() < OP_STACK_OFFSET + 4) return result;

    BFieldElement clk = main_row[OP_STACK_OFFSET + 0];     // CLK
    BFieldElement ib1 = main_row[OP_STACK_OFFSET + 1];     // IB1ShrinkStack
    BFieldElement sp = main_row[OP_STACK_OFFSET + 2];      // StackPointer
    BFieldElement top = main_row[OP_STACK_OFFSET + 3];     // FirstUnderflowElement

    // Consistency constraints: Stack pointer bounds
    // Stack pointer should be between 0 and 16
    if (sp.value() > 16) {
        result = result + XFieldElement(BFieldElement(1));  // Constraint violation
    }

    // Transition constraints: Stack operations based on IB1
    if (ib1 != BFieldElement::zero()) {
        // Stack shrink operation - stack pointer should decrease
        XFieldElement sp_transition = XFieldElement(point) - XFieldElement(sp) + XFieldElement::one();
        result = result + sp_transition;
    }

    return result;
}

XFieldElement Quotient::evaluate_ram_table_constraints(
    const XFieldElement& point,
    const std::vector<BFieldElement>& main_row,
    const std::vector<XFieldElement>& aux_row,
    const Challenges& challenges
) {
    // RAM table constraints (columns 50-56)
    // Validates memory read/write operations

    XFieldElement result = XFieldElement::zero();

    const size_t RAM_OFFSET = 50;
    if (main_row.size() < RAM_OFFSET + 4) return result;

    BFieldElement clk = main_row[RAM_OFFSET + 0];         // CLK
    BFieldElement ram_pointer = main_row[RAM_OFFSET + 2]; // RamPointer
    BFieldElement ram_value = main_row[RAM_OFFSET + 3];   // RamValue

    // Consistency constraints: RAM pointer should be valid
    // For simplified validation, ensure RAM pointer is non-negative
    if (ram_pointer.value() > 1000000) {  // Arbitrary large bound
        result = result + XFieldElement(BFieldElement(1));  // Constraint violation
    }

    // Transition constraints: Memory consistency
    // Simplified: RAM values should be consistent across operations
    XFieldElement ram_transition = XFieldElement(point) - XFieldElement(ram_value);
    result = result + ram_transition;

    return result;
}

XFieldElement Quotient::evaluate_jump_stack_table_constraints(
    const XFieldElement& point,
    const std::vector<BFieldElement>& main_row,
    const std::vector<XFieldElement>& aux_row,
    const Challenges& challenges
) {
    // Jump Stack table constraints (columns 57-61)
    // Validates jump/call stack operations

    XFieldElement result = XFieldElement::zero();

    const size_t JUMP_OFFSET = 57;
    if (main_row.size() < JUMP_OFFSET + 5) return result;

    BFieldElement clk = main_row[JUMP_OFFSET + 0];    // CLK
    BFieldElement ci = main_row[JUMP_OFFSET + 1];     // CI
    BFieldElement jsp = main_row[JUMP_OFFSET + 2];    // JSP
    BFieldElement jso = main_row[JUMP_OFFSET + 3];    // JSO
    BFieldElement jsd = main_row[JUMP_OFFSET + 4];    // JSD

    // Consistency constraints: Jump stack depth bounds
    if (jsp.value() > 16) {  // Arbitrary stack depth limit
        result = result + XFieldElement(BFieldElement(1));  // Constraint violation
    }

    // Transition constraints: Jump stack operations based on JSD
    if (jsd != BFieldElement::zero()) {
        // Jump operation - stack should change
        XFieldElement jsp_transition = XFieldElement(point) - XFieldElement(jsp) + XFieldElement::one();
        result = result + jsp_transition;
    }

    return result;
}

XFieldElement Quotient::evaluate_hash_table_constraints(
    const XFieldElement& point,
    const std::vector<BFieldElement>& main_row,
    const std::vector<XFieldElement>& aux_row,
    const Challenges& challenges
) {
    // Hash table constraints (columns 62-128)
    // Validates cryptographic hash operations

    XFieldElement result = XFieldElement::zero();

    const size_t HASH_OFFSET = 62;
    if (main_row.size() < HASH_OFFSET + 3) return result;

    BFieldElement mode = main_row[HASH_OFFSET + 0];       // Mode
    BFieldElement round = main_row[HASH_OFFSET + 2];      // RoundNumber

    // Consistency constraints: Round number bounds for Tip5
    if (round.value() > 63) {  // Tip5 has 64 rounds max
        result = result + XFieldElement(BFieldElement(1));  // Constraint violation
    }

    // Transition constraints: Round progression
    if (mode != BFieldElement::zero()) {  // Active hash operation
        XFieldElement round_transition = XFieldElement(point) - XFieldElement(round) - XFieldElement::one();
        result = result + round_transition;
    }

    return result;
}

XFieldElement Quotient::evaluate_cascade_table_constraints(
    const XFieldElement& point,
    const std::vector<BFieldElement>& main_row,
    const std::vector<XFieldElement>& aux_row,
    const Challenges& challenges
) {
    // Cascade table constraints (columns 129-134)
    // Validates hash input/output linking

    XFieldElement result = XFieldElement::zero();

    const size_t CASCADE_OFFSET = 129;
    if (main_row.size() < CASCADE_OFFSET + 5) return result;

    BFieldElement input = main_row[CASCADE_OFFSET + 0];     // Input
    BFieldElement output = main_row[CASCADE_OFFSET + 1];    // Output
    BFieldElement mult = main_row[CASCADE_OFFSET + 4];      // LookupMultiplicity

    // Consistency constraints: Input/output relationship
    // Simplified: For cascade table, input and output should be related
    XFieldElement io_relation = XFieldElement(output) - XFieldElement(input) - XFieldElement(mult);
    result = result + io_relation;

    return result;
}

XFieldElement Quotient::evaluate_lookup_table_constraints(
    const XFieldElement& point,
    const std::vector<BFieldElement>& main_row,
    const std::vector<XFieldElement>& aux_row,
    const Challenges& challenges
) {
    // Lookup table constraints (columns 135-138)
    // Validates table lookup operations

    XFieldElement result = XFieldElement::zero();

    const size_t LOOKUP_OFFSET = 135;
    if (main_row.size() < LOOKUP_OFFSET + 3) return result;

    BFieldElement look_in = main_row[LOOKUP_OFFSET + 0];    // LookIn
    BFieldElement look_out = main_row[LOOKUP_OFFSET + 1];   // LookOut
    BFieldElement is_padding = main_row[LOOKUP_OFFSET + 2]; // IsPadding

    // Initial constraints: First lookup should start with LookIn = 0
    if (point == XFieldElement::one()) {
        result = result + XFieldElement(look_in);  // LookIn = 0 initially
    }

    // Consistency constraints: Padding rows
    if (is_padding != BFieldElement::zero()) {
        result = result + XFieldElement(look_in);   // Padding LookIn = 0
        result = result + XFieldElement(look_out);  // Padding LookOut = 0
    }

    return result;
}

XFieldElement Quotient::evaluate_u32_table_constraints(
    const XFieldElement& point,
    const std::vector<BFieldElement>& main_row,
    const std::vector<XFieldElement>& aux_row,
    const Challenges& challenges
) {
    // U32 table constraints (columns 139-148)
    // Validates 32-bit arithmetic operations

    XFieldElement result = XFieldElement::zero();

    const size_t U32_OFFSET = 139;
    if (main_row.size() < U32_OFFSET + 9) return result;

    BFieldElement copy_flag = main_row[U32_OFFSET + 0];    // CopyFlag
    BFieldElement lhs = main_row[U32_OFFSET + 4];          // LHS
    BFieldElement rhs = main_row[U32_OFFSET + 5];          // RHS
    BFieldElement result_val = main_row[U32_OFFSET + 8];   // Result

    // Consistency constraints: U32 bounds (0 <= x < 2^32)
    if (lhs.value() >= (1ULL << 32) || rhs.value() >= (1ULL << 32) || result_val.value() >= (1ULL << 32)) {
        result = result + XFieldElement(BFieldElement(1));  // Constraint violation
    }

    // Transition constraints: Arithmetic correctness
    if (copy_flag != BFieldElement::zero()) {
        // For copy operations: result should equal lhs
        XFieldElement copy_correctness = XFieldElement(result_val) - XFieldElement(lhs);
        result = result + copy_correctness;
    }

    return result;
}

std::vector<std::vector<XFieldElement>> Quotient::segments_to_rows(
    const std::vector<std::vector<XFieldElement>>& segments
) {
    if (segments.empty()) {
        return {};
    }

    const size_t num_segments = segments.size();
    const size_t row_count = segments[0].size();
    std::vector<std::vector<XFieldElement>> rows(
        row_count,
        std::vector<XFieldElement>(num_segments, XFieldElement::zero()));

    for (size_t segment = 0; segment < num_segments; ++segment) {
        for (size_t row = 0; row < row_count; ++row) {
            rows[row][segment] = segments[segment][row];
        }
    }
    return rows;
}

std::vector<BFieldElement> Quotient::initial_zerofier_inverse(
    const ArithmeticDomain& quotient_domain
) {
    std::vector<BFieldElement> codeword;
    codeword.reserve(quotient_domain.length);
    const auto domain_values = quotient_domain.values();
    for (const auto& value : domain_values) {
        codeword.push_back(value - BFieldElement::one());
    }
    return BFieldElement::batch_inversion(codeword);
}

std::vector<BFieldElement> Quotient::consistency_zerofier_inverse(
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain
) {
    std::vector<BFieldElement> codeword;
    codeword.reserve(quotient_domain.length);
    const auto domain_values = quotient_domain.values();
    for (const auto& value : domain_values) {
        codeword.push_back(value.pow(static_cast<uint64_t>(trace_domain.length)) - BFieldElement::one());
        }
    return BFieldElement::batch_inversion(codeword);
}

std::vector<BFieldElement> Quotient::transition_zerofier_inverse(
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain
) {
    std::vector<BFieldElement> result;
    result.reserve(quotient_domain.length);

    const auto domain_values = quotient_domain.values();
    std::vector<BFieldElement> subgroup_zerofier;
    subgroup_zerofier.reserve(domain_values.size());
    for (const auto& value : domain_values) {
        subgroup_zerofier.push_back(value.pow(static_cast<uint64_t>(trace_domain.length)) - BFieldElement::one());
    }
    auto subgroup_inverse = BFieldElement::batch_inversion(subgroup_zerofier);

    const BFieldElement generator_inverse = trace_domain.generator.inverse();
    for (size_t i = 0; i < domain_values.size(); ++i) {
        result.push_back((domain_values[i] - generator_inverse) * subgroup_inverse[i]);
    }

    return result;
}

std::vector<BFieldElement> Quotient::terminal_zerofier_inverse(
    const ArithmeticDomain& trace_domain,
    const ArithmeticDomain& quotient_domain
) {
    std::vector<BFieldElement> codeword;
    codeword.reserve(quotient_domain.length);
    const BFieldElement generator_inverse = trace_domain.generator.inverse();
    const auto domain_values = quotient_domain.values();
    for (const auto& value : domain_values) {
        codeword.push_back(value - generator_inverse);
        }
    return BFieldElement::batch_inversion(codeword);
}

} // namespace triton_vm