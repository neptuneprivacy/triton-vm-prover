/**
 * Co-Verification Test for GPU Quotient Constraint Evaluation
 *
 * Compares GPU-computed quotient_values (constraint evaluation + zerofier inverses)
 * against the existing CPU reference implementation for a small random instance.
 */

#include <gtest/gtest.h>
#include <random>
#include <vector>

#include "quotient/quotient.hpp"
#include "stark/challenges.hpp"

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/quotient_constraints_kernel.cuh"
#include <cuda_runtime.h>
#endif

using namespace triton_vm;

class QuotientConstraintsCoVerifyTest : public ::testing::Test {
protected:
    void SetUp() override {
#ifdef TRITON_CUDA_ENABLED
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess || device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }
#else
        GTEST_SKIP() << "CUDA not enabled";
#endif
    }
};

#ifdef TRITON_CUDA_ENABLED

static inline std::vector<uint64_t> flatten_main_bfe(
    const std::vector<std::vector<BFieldElement>>& rows
) {
    if (rows.empty()) return {};
    size_t nrows = rows.size();
    size_t ncols = rows[0].size();
    std::vector<uint64_t> out(nrows * ncols);
    for (size_t r = 0; r < nrows; ++r) {
        for (size_t c = 0; c < ncols; ++c) {
            out[r * ncols + c] = rows[r][c].value();
        }
    }
    return out;
}

static inline std::vector<uint64_t> flatten_aux_xfe(
    const std::vector<std::vector<XFieldElement>>& rows
) {
    if (rows.empty()) return {};
    size_t nrows = rows.size();
    size_t ncols = rows[0].size();
    std::vector<uint64_t> out(nrows * ncols * 3);
    for (size_t r = 0; r < nrows; ++r) {
        for (size_t c = 0; c < ncols; ++c) {
            size_t idx = (r * ncols + c) * 3;
            out[idx + 0] = rows[r][c].coeff(0).value();
            out[idx + 1] = rows[r][c].coeff(1).value();
            out[idx + 2] = rows[r][c].coeff(2).value();
        }
    }
    return out;
}

static inline std::vector<uint64_t> flatten_challenges(const Challenges& ch) {
    std::vector<uint64_t> out(Challenges::COUNT * 3);
    for (size_t i = 0; i < Challenges::COUNT; ++i) {
        out[i * 3 + 0] = ch[i].coeff(0).value();
        out[i * 3 + 1] = ch[i].coeff(1).value();
        out[i * 3 + 2] = ch[i].coeff(2).value();
    }
    return out;
}

static inline std::vector<uint64_t> flatten_xfe_vec(const std::vector<XFieldElement>& v) {
    std::vector<uint64_t> out(v.size() * 3);
    for (size_t i = 0; i < v.size(); ++i) {
        out[i * 3 + 0] = v[i].coeff(0).value();
        out[i * 3 + 1] = v[i].coeff(1).value();
        out[i * 3 + 2] = v[i].coeff(2).value();
    }
    return out;
}

TEST_F(QuotientConstraintsCoVerifyTest, QuotientValues_BitForBit_SmallRandom) {
    constexpr size_t trace_len = 32;
    constexpr size_t quotient_len = 128;
    constexpr size_t unit_distance = quotient_len / trace_len; // 4
    constexpr size_t main_width = 379;
    constexpr size_t aux_width = 88;

    ArithmeticDomain trace_domain = ArithmeticDomain::of_length(trace_len);
    ArithmeticDomain quotient_domain = ArithmeticDomain::of_length(quotient_len).with_offset(trace_domain.offset);

    std::mt19937_64 rng(123);
    auto rbfe = [&]() { return BFieldElement(rng() % BFieldElement::MODULUS); };
    auto rxfe = [&]() {
        return XFieldElement(rbfe(), rbfe(), rbfe());
    };

    // Random main/aux LDE rows (already quotient-domain sized)
    std::vector<std::vector<BFieldElement>> main_lde(quotient_len, std::vector<BFieldElement>(main_width));
    std::vector<std::vector<XFieldElement>> aux_lde(quotient_len, std::vector<XFieldElement>(aux_width));
    for (size_t r = 0; r < quotient_len; ++r) {
        for (size_t c = 0; c < main_width; ++c) main_lde[r][c] = rbfe();
        for (size_t c = 0; c < aux_width; ++c) aux_lde[r][c] = rxfe();
    }

    Challenges ch;
    for (size_t i = 0; i < Challenges::COUNT; ++i) ch[i] = rxfe();

    std::vector<XFieldElement> weights(Quotient::MASTER_AUX_NUM_CONSTRAINTS);
    for (auto& w : weights) w = rxfe();

    // Zerofier inverses (BFE)
    auto init_inv = Quotient::initial_zerofier_inverse(quotient_domain);
    auto cons_inv = Quotient::consistency_zerofier_inverse(trace_domain, quotient_domain);
    auto tran_inv = Quotient::transition_zerofier_inverse(trace_domain, quotient_domain);
    auto term_inv = Quotient::terminal_zerofier_inverse(trace_domain, quotient_domain);

    auto weighted_sum = [&](const std::vector<XFieldElement>& values, size_t offset) {
        XFieldElement acc = XFieldElement::zero();
        for (size_t i = 0; i < values.size(); ++i) {
            acc += values[i] * weights[offset + i];
        }
        return acc;
    };

    const size_t init_end = Quotient::NUM_INITIAL_CONSTRAINTS;
    const size_t cons_end = init_end + Quotient::NUM_CONSISTENCY_CONSTRAINTS;
    const size_t tran_end = cons_end + Quotient::NUM_TRANSITION_CONSTRAINTS;

    // CPU reference quotient_values
    std::vector<XFieldElement> cpu_q(quotient_len, XFieldElement::zero());
    for (size_t row = 0; row < quotient_len; ++row) {
        size_t next = (row + unit_distance) % quotient_len;
        auto init_vals = Quotient::evaluate_initial_constraints(main_lde[row], aux_lde[row], ch);
        XFieldElement q = weighted_sum(init_vals, 0) * init_inv[row];
        auto cons_vals = Quotient::evaluate_consistency_constraints(main_lde[row], aux_lde[row], ch);
        q += weighted_sum(cons_vals, init_end) * cons_inv[row];
        auto tran_vals = Quotient::evaluate_transition_constraints(main_lde[row], aux_lde[row], main_lde[next], aux_lde[next], ch);
        q += weighted_sum(tran_vals, cons_end) * tran_inv[row];
        auto term_vals = Quotient::evaluate_terminal_constraints(main_lde[row], aux_lde[row], ch);
        q += weighted_sum(term_vals, tran_end) * term_inv[row];
        cpu_q[row] = q;
    }

    // GPU compute
    std::vector<uint64_t> h_main = flatten_main_bfe(main_lde);
    std::vector<uint64_t> h_aux = flatten_aux_xfe(aux_lde);
    std::vector<uint64_t> h_ch = flatten_challenges(ch);
    std::vector<uint64_t> h_weights = flatten_xfe_vec(weights);

    std::vector<uint64_t> h_init_inv(quotient_len), h_cons_inv(quotient_len), h_tran_inv(quotient_len), h_term_inv(quotient_len);
    for (size_t i = 0; i < quotient_len; ++i) {
        h_init_inv[i] = init_inv[i].value();
        h_cons_inv[i] = cons_inv[i].value();
        h_tran_inv[i] = tran_inv[i].value();
        h_term_inv[i] = term_inv[i].value();
    }

    uint64_t *d_main = nullptr, *d_aux = nullptr, *d_ch = nullptr, *d_w = nullptr;
    uint64_t *d_init = nullptr, *d_cons = nullptr, *d_tran = nullptr, *d_term = nullptr;
    uint64_t *d_out = nullptr;

    cudaMalloc(&d_main, h_main.size() * sizeof(uint64_t));
    cudaMalloc(&d_aux, h_aux.size() * sizeof(uint64_t));
    cudaMalloc(&d_ch, h_ch.size() * sizeof(uint64_t));
    cudaMalloc(&d_w, h_weights.size() * sizeof(uint64_t));
    cudaMalloc(&d_init, h_init_inv.size() * sizeof(uint64_t));
    cudaMalloc(&d_cons, h_cons_inv.size() * sizeof(uint64_t));
    cudaMalloc(&d_tran, h_tran_inv.size() * sizeof(uint64_t));
    cudaMalloc(&d_term, h_term_inv.size() * sizeof(uint64_t));
    cudaMalloc(&d_out, quotient_len * 3 * sizeof(uint64_t));

    cudaMemcpy(d_main, h_main.data(), h_main.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_aux, h_aux.data(), h_aux.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ch, h_ch.data(), h_ch.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, h_weights.data(), h_weights.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_init, h_init_inv.data(), h_init_inv.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cons, h_cons_inv.data(), h_cons_inv.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_tran, h_tran_inv.data(), h_tran_inv.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_term, h_term_inv.data(), h_term_inv.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);

    gpu::kernels::compute_quotient_values_gpu(
        d_main, main_width,
        d_aux, aux_width,
        quotient_len, unit_distance,
        d_ch, d_w,
        d_init, d_cons, d_tran, d_term,
        d_out, 0
    );
    cudaDeviceSynchronize();

    std::vector<uint64_t> h_out(quotient_len * 3);
    cudaMemcpy(h_out.data(), d_out, h_out.size() * sizeof(uint64_t), cudaMemcpyDeviceToHost);

    cudaFree(d_main); cudaFree(d_aux); cudaFree(d_ch); cudaFree(d_w);
    cudaFree(d_init); cudaFree(d_cons); cudaFree(d_tran); cudaFree(d_term);
    cudaFree(d_out);

    // Compare
    size_t mismatches = 0;
    for (size_t row = 0; row < quotient_len; ++row) {
        XFieldElement gpu_q(
            BFieldElement(h_out[row * 3 + 0]),
            BFieldElement(h_out[row * 3 + 1]),
            BFieldElement(h_out[row * 3 + 2])
        );
        if (gpu_q != cpu_q[row]) {
            if (mismatches < 5) {
                std::cerr << "Mismatch row=" << row
                          << " GPU=" << gpu_q.to_string()
                          << " CPU=" << cpu_q[row].to_string() << "\n";
            }
            ++mismatches;
        }
    }
    EXPECT_EQ(mismatches, 0u);
}

#endif // TRITON_CUDA_ENABLED



