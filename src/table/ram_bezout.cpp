// Standalone RAM Bézout computation extracted for reuse by GPU Phase1 host-prep.

#include "table/ram_bezout.hpp"

#include "ntt/ntt.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef TRITON_CUDA_ENABLED
#include "gpu/kernels/poly_mul_kernel.cuh"
#include "gpu/kernels/bezout_kernel.cuh"
#include <cuda_runtime.h>
#endif

namespace triton_vm {

std::pair<std::vector<BFieldElement>, std::vector<BFieldElement>>
compute_ram_bezout_coefficients(const std::vector<BFieldElement>& roots) {
    const size_t n = roots.size();
    if (n == 0) {
        return {{}, {}};
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

        size_t ntt_n = 1;
        while (ntt_n < result_size) ntt_n *= 2;

        std::vector<BFieldElement> a_padded(ntt_n, BFieldElement::zero());
        std::vector<BFieldElement> b_padded(ntt_n, BFieldElement::zero());
        std::copy(a.begin(), a.end(), a_padded.begin());
        std::copy(b.begin(), b.end(), b_padded.begin());

        NTT::forward(a_padded);
        NTT::forward(b_padded);
        for (size_t i = 0; i < ntt_n; ++i) a_padded[i] *= b_padded[i];
        NTT::inverse(a_padded);

        a_padded.resize(result_size);
        return a_padded;
    };

    auto poly_derivative = [](const std::vector<BFieldElement>& a) {
        if (a.size() <= 1) return std::vector<BFieldElement>{BFieldElement::zero()};
        std::vector<BFieldElement> d(a.size() - 1, BFieldElement::zero());
        for (size_t i = 1; i < a.size(); ++i) {
            d[i - 1] = a[i] * BFieldElement(static_cast<uint64_t>(i));
        }
        return d;
    };

    auto poly_eval = [](const std::vector<BFieldElement>& a, BFieldElement x) {
        BFieldElement acc = BFieldElement::zero();
        for (int i = static_cast<int>(a.size()) - 1; i >= 0; --i) {
            acc = acc * x + a[static_cast<size_t>(i)];
        }
        return acc;
    };

    // Exact division f/g where g divides f. (Matches master_table.cpp implementation.)
    auto poly_div_exact = [&](const std::vector<BFieldElement>& dividend,
                              const std::vector<BFieldElement>& divisor) {
        auto a = trim(dividend);
        auto d = trim(divisor);
        if (a.empty()) return std::vector<BFieldElement>{BFieldElement::zero()};
        if (d.empty()) return std::vector<BFieldElement>{BFieldElement::zero()};
        size_t deg_a = a.size() - 1;
        size_t deg_d = d.size() - 1;
        if (deg_a < deg_d) return std::vector<BFieldElement>{BFieldElement::zero()};

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

        size_t k = deg_a - deg_d + 1;
        std::vector<BFieldElement> f_rev(a.rbegin(), a.rend());
        std::vector<BFieldElement> g_rev(d.rbegin(), d.rend());

        std::vector<BFieldElement> inv{g_rev[0].inverse()};
        inv.reserve(k);

        size_t prec = 1;
        while (prec < k) {
            size_t next_prec = std::min(prec * 2, k);
            size_t g_len = std::min(g_rev.size(), next_prec);
            std::vector<BFieldElement> g_trunc(g_rev.begin(), g_rev.begin() + g_len);

            auto prod = poly_mul(g_trunc, inv);
            if (prod.size() > next_prec) prod.resize(next_prec);

            std::vector<BFieldElement> two_minus(next_prec, BFieldElement::zero());
            two_minus[0] = BFieldElement(2);
            for (size_t i = 0; i < prod.size(); ++i) two_minus[i] -= prod[i];

            inv = poly_mul(inv, two_minus);
            if (inv.size() > next_prec) inv.resize(next_prec);
            prec = next_prec;
        }

        size_t f_len = std::min(f_rev.size(), k);
        std::vector<BFieldElement> f_trunc(f_rev.begin(), f_rev.begin() + f_len);
        auto q_rev = poly_mul(f_trunc, inv);
        if (q_rev.size() > k) q_rev.resize(k);
        while (q_rev.size() < k) q_rev.push_back(BFieldElement::zero());
        std::vector<BFieldElement> q(q_rev.rbegin(), q_rev.rend());
        return trim(q);
    };

    // Build subproduct tree once
    size_t num_levels = 0;
    size_t sz = n;
    while (sz > 1) { ++num_levels; sz = (sz + 1) / 2; }
    ++num_levels;

    std::vector<std::vector<std::vector<BFieldElement>>> tree(num_levels);
    tree[0].resize(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        tree[0][i] = {BFieldElement::zero() - roots[i], BFieldElement::one()};
    }

    for (size_t d = 1; d < num_levels; ++d) {
        size_t prev_sz = tree[d - 1].size();
        size_t new_sz = (prev_sz + 1) / 2;
        tree[d].resize(new_sz);
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < new_sz; ++i) {
            size_t left = 2 * i;
            size_t right = 2 * i + 1;
            if (right < prev_sz) tree[d][i] = poly_mul(tree[d - 1][left], tree[d - 1][right]);
            else tree[d][i] = tree[d - 1][left];
        }
    }

    std::vector<BFieldElement> rp = trim(tree[num_levels - 1][0]);
    std::vector<BFieldElement> fd = poly_derivative(rp);

    // Evaluate fd at all roots
    std::vector<BFieldElement> fd_in_roots(n);
#ifdef TRITON_CUDA_ENABLED
    {
        uint64_t *d_coeffs = nullptr, *d_points = nullptr, *d_results = nullptr;
        cudaError_t err;
        err = cudaMalloc(&d_coeffs, fd.size() * sizeof(uint64_t));
        if (err == cudaSuccess) err = cudaMalloc(&d_points, n * sizeof(uint64_t));
        if (err == cudaSuccess) err = cudaMalloc(&d_results, n * sizeof(uint64_t));
        if (err == cudaSuccess) {
            std::vector<uint64_t> fd_raw(fd.size());
            for (size_t i = 0; i < fd.size(); ++i) fd_raw[i] = fd[i].value();
            std::vector<uint64_t> roots_raw(n);
            for (size_t i = 0; i < n; ++i) roots_raw[i] = roots[i].value();
            cudaMemcpy(d_coeffs, fd_raw.data(), fd.size() * sizeof(uint64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(d_points, roots_raw.data(), n * sizeof(uint64_t), cudaMemcpyHostToDevice);
            triton_vm::gpu::kernels::gpu_poly_eval_batch(d_coeffs, fd.size() - 1, d_points, n, d_results, nullptr);
            cudaDeviceSynchronize();
            std::vector<uint64_t> results_raw(n);
            cudaMemcpy(results_raw.data(), d_results, n * sizeof(uint64_t), cudaMemcpyDeviceToHost);
            for (size_t i = 0; i < n; ++i) fd_in_roots[i] = BFieldElement(results_raw[i]);
        } else {
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < n; ++i) fd_in_roots[i] = poly_eval(fd, roots[i]);
        }
        if (d_coeffs) cudaFree(d_coeffs);
        if (d_points) cudaFree(d_points);
        if (d_results) cudaFree(d_results);
    }
#else
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) fd_in_roots[i] = poly_eval(fd, roots[i]);
#endif

    std::vector<BFieldElement> b_in_roots = BFieldElement::batch_inversion(fd_in_roots);

    // scaled = 1/fd^2
    std::vector<BFieldElement> scaled(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) scaled[i] = b_in_roots[i] * b_in_roots[i];

    // Build weighted sums bottom-up through the tree
    std::vector<std::vector<std::vector<BFieldElement>>> weights(num_levels);
    weights[0].resize(n);
    for (size_t i = 0; i < n; ++i) weights[0][i] = {scaled[i]};

    for (size_t d = 1; d < num_levels; ++d) {
        size_t this_sz = tree[d].size();
        weights[d].resize(this_sz);
        size_t prev_sz = tree[d - 1].size();
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < this_sz; ++i) {
            size_t left = 2 * i;
            size_t right = 2 * i + 1;
            if (right < prev_sz) {
                auto left_contrib = poly_mul(weights[d - 1][left], tree[d - 1][right]);
                auto right_contrib = poly_mul(weights[d - 1][right], tree[d - 1][left]);
                weights[d][i] = poly_add(left_contrib, right_contrib);
            } else {
                weights[d][i] = weights[d - 1][left];
            }
        }
    }

    std::vector<BFieldElement> b_coeffs = trim(weights[num_levels - 1][0]);
    std::vector<BFieldElement> fd_b = poly_mul(fd, b_coeffs);
    std::vector<BFieldElement> one{BFieldElement::one()};
    std::vector<BFieldElement> one_minus_fd_b = trim(poly_sub(one, fd_b));
    std::vector<BFieldElement> a_coeffs = poly_div_exact(one_minus_fd_b, rp);

    a_coeffs.resize(n, BFieldElement::zero());
    b_coeffs.resize(n, BFieldElement::zero());
    return {a_coeffs, b_coeffs};
}

} // namespace triton_vm


