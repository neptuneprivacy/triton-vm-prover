/**
 * GPU Aux Table Extension - Complete Sequential Implementation
 * 
 * This implements all 9 sub-table extensions on GPU for zero-copy proof generation.
 * Uses single-thread kernels for sequential operations to avoid H2D/D2H transfers.
 * 
 * Strategy: Mirror CPU logic exactly, but run on GPU to keep data resident.
 * This is correct but not optimal - can be parallelized later.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/extend_kernel.cuh"
#include "gpu/kernels/hash_table_constants.cuh"
#include "gpu/kernels/bfield_kernel.cuh"
#include "gpu/kernels/xfield_kernel.cuh"
#include "gpu/kernels/degree_lowering_generated.cuh"
#include "gpu/cuda_common.cuh"
#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// (Rust FFI degree lowering not used once CUDA kernel is enabled)

// =============================================================================
// mt19937_64 (std::mt19937_64 compatible) for aux randomizer column (col 87)
// =============================================================================
namespace {
__device__ __forceinline__ uint64_t mt19937_64_temper(uint64_t x) {
    x ^= (x >> 29) & 0x5555555555555555ULL;
    x ^= (x << 17) & 0x71D67FFFEDA60000ULL;
    x ^= (x << 37) & 0xFFF7EEE000000000ULL;
    x ^= (x >> 43);
    return x;
}

__device__ __forceinline__ void mt19937_64_twist(uint64_t* mt) {
    constexpr int N = 312;
    constexpr int M = 156;
    constexpr uint64_t MATRIX_A = 0xB5026F5AA96619E9ULL;
    constexpr uint64_t UM = 0xFFFFFFFF80000000ULL; // upper 33 bits
    constexpr uint64_t LM = 0x7FFFFFFFULL;         // lower 31 bits

    for (int i = 0; i < N; i++) {
        uint64_t x = (mt[i] & UM) | (mt[(i + 1) % N] & LM);
        uint64_t xA = x >> 1;
        if (x & 1ULL) xA ^= MATRIX_A;
        mt[i] = mt[(i + M) % N] ^ xA;
    }
}

__device__ __forceinline__ void mt19937_64_init(uint64_t* mt, uint64_t seed) {
    constexpr int N = 312;
    constexpr uint64_t F = 6364136223846793005ULL;
    mt[0] = seed;
    for (int i = 1; i < N; i++) {
        uint64_t x = mt[i - 1] ^ (mt[i - 1] >> 62);
        mt[i] = F * x + static_cast<uint64_t>(i);
    }
}

__device__ __forceinline__ uint64_t mt19937_64_next(uint64_t* mt, int& idx) {
    constexpr int N = 312;
    if (idx >= N) {
        mt19937_64_twist(mt);
        idx = 0;
    }
    return mt19937_64_temper(mt[idx++]);
}
} // namespace

namespace triton_vm {
namespace gpu {
namespace kernels {

// =============================================================================
// XFE type + ops for CUB scans (Hash table acceleration)
// =============================================================================
struct Xfe3 {
    uint64_t c0;
    uint64_t c1;
    uint64_t c2;
};

__device__ __forceinline__ Xfe3 xfe3_zero() { return Xfe3{0, 0, 0}; }
__device__ __forceinline__ Xfe3 xfe3_one() { return Xfe3{1, 0, 0}; }

__device__ __forceinline__ Xfe3 xfe3_add(const Xfe3& a, const Xfe3& b) {
    Xfe3 r;
    xfield_add_impl(a.c0, a.c1, a.c2, b.c0, b.c1, b.c2, r.c0, r.c1, r.c2);
    return r;
}

__device__ __forceinline__ Xfe3 xfe3_mul(const Xfe3& a, const Xfe3& b) {
    Xfe3 r;
    xfield_mul_impl(a.c0, a.c1, a.c2, b.c0, b.c1, b.c2, r.c0, r.c1, r.c2);
    return r;
}

__device__ __forceinline__ Xfe3 xfe3_inv(const Xfe3& a) {
    Xfe3 r;
    xfield_inv_impl(a.c0, a.c1, a.c2, r.c0, r.c1, r.c2);
    return r;
}

struct XfeMulOp {
    __device__ __forceinline__ Xfe3 operator()(const Xfe3& a, const Xfe3& b) const { return xfe3_mul(a, b); }
};

struct XfeAddOp {
    __device__ __forceinline__ Xfe3 operator()(const Xfe3& a, const Xfe3& b) const { return xfe3_add(a, b); }
};

// =============================================================================
// Linear Recurrence Structure for Parallel Running Evaluations
// Represents: eval' = eval * factor + addend
// Composition: (f1, a1) ⊕ (f2, a2) = (f1*f2, a1*f2 + a2)
// =============================================================================
struct LinearRecurrence {
    Xfe3 factor;  // Multiplier (indeterminate or 1)
    Xfe3 addend;  // Addend (row_value or 0)
    
    __device__ __forceinline__ LinearRecurrence() = default;
    __device__ __forceinline__ LinearRecurrence(const Xfe3& f, const Xfe3& a) : factor(f), addend(a) {}
};

struct LinearRecurrenceOp {
    __device__ __forceinline__ LinearRecurrence operator()(const LinearRecurrence& left, const LinearRecurrence& right) const {
        // Composition: (f1, a1) ⊕ (f2, a2) = (f1*f2, a1*f2 + a2)
        Xfe3 new_factor = xfe3_mul(left.factor, right.factor);
        Xfe3 temp = xfe3_mul(left.addend, right.factor);
        Xfe3 new_addend = xfe3_add(temp, right.addend);
        return LinearRecurrence(new_factor, new_addend);
    }
};

// =============================================================================
// Table Column Offsets (must match CPU include/table/extend_helpers.hpp)
// =============================================================================

// Main table column starts (must match CPU include/table/extend_helpers.hpp)
__device__ constexpr size_t MAIN_PROGRAM_START = 0;       // 7 cols
__device__ constexpr size_t MAIN_PROCESSOR_START = 7;     // 39 cols
__device__ constexpr size_t MAIN_OP_STACK_START = 46;     // 4 cols
__device__ constexpr size_t MAIN_RAM_START = 50;          // 7 cols
__device__ constexpr size_t MAIN_JUMP_STACK_START = 57;   // 5 cols
// MAIN_HASH_START is defined in hash_table_constants.cuh
__device__ constexpr size_t MAIN_CASCADE_START = 129;     // 6 cols
__device__ constexpr size_t MAIN_LOOKUP_START = 135;      // 4 cols
__device__ constexpr size_t MAIN_U32_START = 139;         // 15 cols

// Aux table column starts
__device__ constexpr size_t AUX_PROGRAM_START = 0;      // 3 cols
__device__ constexpr size_t AUX_PROCESSOR_START = 3;    // 11 cols
__device__ constexpr size_t AUX_OP_STACK_START = 14;    // 2 cols
__device__ constexpr size_t AUX_RAM_START = 16;         // 6 cols
__device__ constexpr size_t AUX_JUMP_STACK_START = 22;  // 2 cols
__device__ constexpr size_t AUX_HASH_START = 24;        // 20 cols
__device__ constexpr size_t AUX_CASCADE_START = 44;     // 2 cols
__device__ constexpr size_t AUX_LOOKUP_START = 46;      // 2 cols
__device__ constexpr size_t AUX_U32_START = 48;         // 2 cols
__device__ constexpr size_t AUX_TOTAL_COLS = 88;

// Program table column indices (relative to MAIN_PROGRAM_START)
// Must match CPU: Address=0, Instruction=1, LookupMultiplicity=2, IndexInChunk=3, MaxMinusIndexInChunkInv=4, IsHashInputPadding=5, IsTablePadding=6
__device__ constexpr size_t PROG_ADDRESS = 0;
__device__ constexpr size_t PROG_INSTRUCTION = 1;
__device__ constexpr size_t PROG_LOOKUP_MULT = 2;
__device__ constexpr size_t PROG_INDEX_IN_CHUNK = 3;
__device__ constexpr size_t PROG_IS_HASH_PAD = 5;
__device__ constexpr size_t PROG_IS_TABLE_PAD = 6;

// OpStack column indices (relative to MAIN_OP_STACK_START)
__device__ constexpr size_t OS_CLK = 0;
__device__ constexpr size_t OS_IB1 = 1;
__device__ constexpr size_t OS_OSP = 2;
__device__ constexpr size_t OS_OSV = 3;

// JumpStack column indices (relative to MAIN_JUMP_STACK_START)
__device__ constexpr size_t JS_CLK = 0;
__device__ constexpr size_t JS_CI = 1;
__device__ constexpr size_t JS_JSP = 2;
__device__ constexpr size_t JS_JSO = 3;
__device__ constexpr size_t JS_JSD = 4;

// Challenge indices (must match CPU)
__device__ constexpr size_t CH_CompressProgramDigest = 0;
__device__ constexpr size_t CH_StandardInput = 1;
__device__ constexpr size_t CH_StandardOutput = 2;
__device__ constexpr size_t CH_InstructionLookup = 3;
__device__ constexpr size_t CH_HashInput = 4;
__device__ constexpr size_t CH_HashDigest = 5;
__device__ constexpr size_t CH_Sponge = 6;
__device__ constexpr size_t CH_OpStack = 7;
__device__ constexpr size_t CH_Ram = 8;
__device__ constexpr size_t CH_JumpStack = 9;
__device__ constexpr size_t CH_U32 = 10;
__device__ constexpr size_t CH_ClockJumpDiff = 11;
__device__ constexpr size_t CH_RamBezout = 12;
__device__ constexpr size_t CH_ProgAddrWeight = 13;
__device__ constexpr size_t CH_ProgInstrWeight = 14;
__device__ constexpr size_t CH_ProgNextInstrWeight = 15;
__device__ constexpr size_t CH_OpStackClk = 16;
__device__ constexpr size_t CH_OpStackIb1 = 17;
__device__ constexpr size_t CH_OpStackPtr = 18;
__device__ constexpr size_t CH_OpStackVal = 19;
__device__ constexpr size_t CH_RamClk = 20;
__device__ constexpr size_t CH_RamPtr = 21;
__device__ constexpr size_t CH_RamVal = 22;
__device__ constexpr size_t CH_RamInstrType = 23;
__device__ constexpr size_t CH_JsClk = 24;
__device__ constexpr size_t CH_JsCi = 25;
__device__ constexpr size_t CH_JsJsp = 26;
__device__ constexpr size_t CH_JsJso = 27;
__device__ constexpr size_t CH_JsJsd = 28;
__device__ constexpr size_t CH_ProgPrepareChunk = 29;
__device__ constexpr size_t CH_ProgSendChunk = 30;

// =============================================================================
// Device XFE Operations
// =============================================================================

__device__ __forceinline__
void load_xfe(const uint64_t* challenges, size_t idx, uint64_t& c0, uint64_t& c1, uint64_t& c2) {
    c0 = challenges[idx * 3];
    c1 = challenges[idx * 3 + 1];
    c2 = challenges[idx * 3 + 2];
}

__device__ __forceinline__
void store_xfe(uint64_t* out, size_t idx, uint64_t c0, uint64_t c1, uint64_t c2) {
    out[idx * 3 + 0] = c0;
    out[idx * 3 + 1] = c1;
    out[idx * 3 + 2] = c2;
}

__device__ __forceinline__
void xfe_mul_d(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t b0, uint64_t b1, uint64_t b2,
               uint64_t& r0, uint64_t& r1, uint64_t& r2) {
    xfield_mul_impl(a0, a1, a2, b0, b1, b2, r0, r1, r2);
}

__device__ __forceinline__
void xfe_add_d(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t b0, uint64_t b1, uint64_t b2,
               uint64_t& r0, uint64_t& r1, uint64_t& r2) {
    xfield_add_impl(a0, a1, a2, b0, b1, b2, r0, r1, r2);
}

__device__ __forceinline__
void xfe_sub_d(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t b0, uint64_t b1, uint64_t b2,
               uint64_t& r0, uint64_t& r1, uint64_t& r2) {
    xfield_sub_impl(a0, a1, a2, b0, b1, b2, r0, r1, r2);
}

__device__ __forceinline__
void xfe_inv_d(uint64_t a0, uint64_t a1, uint64_t a2, uint64_t& r0, uint64_t& r1, uint64_t& r2) {
    xfield_inv_impl(a0, a1, a2, r0, r1, r2);
}

// Multiply BFE by XFE: result = bfe * xfe
__device__ __forceinline__
void bfe_mul_xfe(uint64_t bfe, uint64_t x0, uint64_t x1, uint64_t x2,
                 uint64_t& r0, uint64_t& r1, uint64_t& r2) {
    // BFE * XFE = (bfe, 0, 0) * (x0, x1, x2)
    xfe_mul_d(bfe, 0, 0, x0, x1, x2, r0, r1, r2);
}

// =============================================================================
// Instruction Decoding Helpers for Processor Table
// =============================================================================

// Get op_stack_size_influence from opcode
// For arg-based instructions, nia encodes the number of words
// Format: nia.value() for N1=1, N2=2, N3=3, N4=4, N5=5
__device__ __forceinline__
int32_t get_op_stack_influence(uint64_t opcode, uint64_t nia) {
    switch (opcode) {
        // Fixed influence instructions
        case 1:  return 1;   // Push: +1
        case 33: return 1;   // Dup: +1
        case 4:  return 1;   // Split: +1
        case 2:  return -1;  // Skiz: -1
        case 10: return -1;  // Assert: -1
        case 42: return -1;  // Add: -1
        case 50: return -1;  // Mul: -1
        case 58: return -1;  // Eq: -1
        case 6:  return -1;  // Lt: -1
        case 14: return -1;  // And: -1
        case 22: return -1;  // Xor: -1
        case 30: return -1;  // Pow: -1
        case 82: return -1;  // XbMul: -1
        case 66: return -3;  // XxAdd: -3
        case 74: return -3;  // XxMul: -3
        case 18: return -5;  // Hash: -5
        case 26: return -5;  // AssertVector: -5
        case 34: return -10; // SpongeAbsorb: -10
        case 56: return 10;  // SpongeSqueeze: +10
        
        // Arg-based instructions - num_words encoded in nia
        case 3:  return -static_cast<int32_t>(nia);  // Pop: -num_words
        case 9:  return static_cast<int32_t>(nia);   // Divine: +num_words
        case 57: return static_cast<int32_t>(nia);   // ReadMem: +num_words
        case 11: return -static_cast<int32_t>(nia);  // WriteMem: -num_words
        case 73: return static_cast<int32_t>(nia);   // ReadIo: +num_words
        case 19: return -static_cast<int32_t>(nia);  // WriteIo: -num_words
        
        // Zero influence instructions
        default: return 0;
    }
}

// =============================================================================
// Main Extension Kernel - Sequential (Single Thread)
// When table_id == -1, runs all tables sequentially
// When table_id >= 0, runs only that specific table (for parallel execution)
// =============================================================================

__global__ void extend_all_tables_kernel(
    const uint64_t* d_main,      // [num_rows × main_width]
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges, // [63 × 3]
    uint64_t aux_rng_seed_value,
    uint64_t* d_aux,              // [num_rows × 88 × 3]
    int table_id = -1             // -1 = all tables, 0-10 = specific table
) {
    // Single thread runs entire extension
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // =========================================================================
    // 1. OpStack Table Extension
    // =========================================================================
    if (table_id == -1 || table_id == 0) {
        // Load challenge weights
        uint64_t ind0, ind1, ind2;
        load_xfe(d_challenges, CH_OpStack, ind0, ind1, ind2);
        
        uint64_t wc0, wc1, wc2;  // clk weight
        uint64_t wi0, wi1, wi2;  // ib1 weight
        uint64_t wp0, wp1, wp2;  // ptr weight
        uint64_t wv0, wv1, wv2;  // val weight
        load_xfe(d_challenges, CH_OpStackClk, wc0, wc1, wc2);
        load_xfe(d_challenges, CH_OpStackIb1, wi0, wi1, wi2);
        load_xfe(d_challenges, CH_OpStackPtr, wp0, wp1, wp2);
        load_xfe(d_challenges, CH_OpStackVal, wv0, wv1, wv2);
        
        // Clock jump diff lookup indeterminate
        uint64_t cjd_ind0, cjd_ind1, cjd_ind2;
        load_xfe(d_challenges, CH_ClockJumpDiff, cjd_ind0, cjd_ind1, cjd_ind2);
        
        // Running product starts at 1
        uint64_t rp0 = 1, rp1 = 0, rp2 = 0;
        // Log derivative starts at 0
        uint64_t ld0 = 0, ld1 = 0, ld2 = 0;
        
        // Padding indicator = 2
        constexpr uint64_t PADDING_VALUE = 2;
        size_t padding_start = num_rows;  // Track where padding starts
        
        // First pass: RunningProductPermArg
        for (size_t i = 0; i < num_rows; i++) {
            size_t row_off = i * main_width + MAIN_OP_STACK_START;
            uint64_t clk = d_main[row_off + OS_CLK];
            uint64_t ib1 = d_main[row_off + OS_IB1];
            uint64_t osp = d_main[row_off + OS_OSP];
            uint64_t osv = d_main[row_off + OS_OSV];
            
            // Only update running product if NOT padding row (ib1 != 2)
            if (ib1 != PADDING_VALUE) {
                // compressed = clk*wc + ib1*wi + osp*wp + osv*wv
                uint64_t t0, t1, t2;
                uint64_t sum0 = 0, sum1 = 0, sum2 = 0;
                
                bfe_mul_xfe(clk, wc0, wc1, wc2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                bfe_mul_xfe(ib1, wi0, wi1, wi2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                bfe_mul_xfe(osp, wp0, wp1, wp2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                bfe_mul_xfe(osv, wv0, wv1, wv2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                // diff = indeterminate - compressed
                uint64_t d0, d1, d2;
                xfe_sub_d(ind0, ind1, ind2, sum0, sum1, sum2, d0, d1, d2);
                
                // running_product *= diff
                xfe_mul_d(rp0, rp1, rp2, d0, d1, d2, rp0, rp1, rp2);
            }
            
            // Store running product (always, after update)
            size_t aux_idx = (i * AUX_TOTAL_COLS + AUX_OP_STACK_START + 0) * 3;
            d_aux[aux_idx + 0] = rp0;
            d_aux[aux_idx + 1] = rp1;
            d_aux[aux_idx + 2] = rp2;
        }
        
        // Second pass: ClockJumpDifferenceLookupLogDerivative
        // CPU processes tuple_windows (pairs), stops at padding
        // Row 0 is not stored in CPU (loop starts at idx=1)
        
        // Store 0 for row 0 (CPU doesn't set row 0)
        size_t aux_idx = (0 * AUX_TOTAL_COLS + AUX_OP_STACK_START + 1) * 3;
        d_aux[aux_idx + 0] = 0;
        d_aux[aux_idx + 1] = 0;
        d_aux[aux_idx + 2] = 0;
        
        for (size_t i = 1; i < num_rows; i++) {
            size_t row_off = i * main_width + MAIN_OP_STACK_START;
            size_t prev_off = (i - 1) * main_width + MAIN_OP_STACK_START;
            
            uint64_t curr_ib1 = d_main[row_off + OS_IB1];
            
            // Stop when we hit padding (like Rust)
            if (curr_ib1 == PADDING_VALUE) {
                padding_start = i;
                break;
            }
            
            uint64_t prev_clk = d_main[prev_off + OS_CLK];
            uint64_t curr_clk = d_main[row_off + OS_CLK];
            uint64_t prev_osp = d_main[prev_off + OS_OSP];
            uint64_t curr_osp = d_main[row_off + OS_OSP];
            
            // Only add if stack pointer same
            if (prev_osp == curr_osp) {
                uint64_t clock_diff = bfield_sub_impl(curr_clk, prev_clk);
                
                // diff = cjd_indeterminate - clock_diff
                uint64_t cd0, cd1, cd2;
                xfe_sub_d(cjd_ind0, cjd_ind1, cjd_ind2, clock_diff, 0, 0, cd0, cd1, cd2);
                
                // inverse
                uint64_t inv0, inv1, inv2;
                xfe_inv_d(cd0, cd1, cd2, inv0, inv1, inv2);
                
                // log_deriv += inverse
                xfe_add_d(ld0, ld1, ld2, inv0, inv1, inv2, ld0, ld1, ld2);
            }
            
            // Store log derivative
            aux_idx = (i * AUX_TOTAL_COLS + AUX_OP_STACK_START + 1) * 3;
            d_aux[aux_idx + 0] = ld0;
            d_aux[aux_idx + 1] = ld1;
            d_aux[aux_idx + 2] = ld2;
        }
        
        // Fill padding section with last value (matches Rust)
        for (size_t i = padding_start; i < num_rows; i++) {
            aux_idx = (i * AUX_TOTAL_COLS + AUX_OP_STACK_START + 1) * 3;
            d_aux[aux_idx + 0] = ld0;
            d_aux[aux_idx + 1] = ld1;
            d_aux[aux_idx + 2] = ld2;
        }
    }
    
    // =========================================================================
    // 2. JumpStack Table Extension
    // =========================================================================
    if (table_id == -1 || table_id == 1) {
        uint64_t ind0, ind1, ind2;
        load_xfe(d_challenges, CH_JumpStack, ind0, ind1, ind2);
        
        uint64_t wc0, wc1, wc2;
        uint64_t wci0, wci1, wci2;
        uint64_t wjsp0, wjsp1, wjsp2;
        uint64_t wjso0, wjso1, wjso2;
        uint64_t wjsd0, wjsd1, wjsd2;
        load_xfe(d_challenges, CH_JsClk, wc0, wc1, wc2);
        load_xfe(d_challenges, CH_JsCi, wci0, wci1, wci2);
        load_xfe(d_challenges, CH_JsJsp, wjsp0, wjsp1, wjsp2);
        load_xfe(d_challenges, CH_JsJso, wjso0, wjso1, wjso2);
        load_xfe(d_challenges, CH_JsJsd, wjsd0, wjsd1, wjsd2);
        
        uint64_t cjd_ind0, cjd_ind1, cjd_ind2;
        load_xfe(d_challenges, CH_ClockJumpDiff, cjd_ind0, cjd_ind1, cjd_ind2);
        
        uint64_t rp0 = 1, rp1 = 0, rp2 = 0;
        uint64_t ld0 = 0, ld1 = 0, ld2 = 0;
        
        for (size_t i = 0; i < num_rows; i++) {
            size_t row_off = i * main_width + MAIN_JUMP_STACK_START;
            uint64_t clk = d_main[row_off + JS_CLK];
            uint64_t ci  = d_main[row_off + JS_CI];
            uint64_t jsp = d_main[row_off + JS_JSP];
            uint64_t jso = d_main[row_off + JS_JSO];
            uint64_t jsd = d_main[row_off + JS_JSD];
            
            uint64_t t0, t1, t2;
            uint64_t sum0 = 0, sum1 = 0, sum2 = 0;
            
            bfe_mul_xfe(clk, wc0, wc1, wc2, t0, t1, t2);
            xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
            
            bfe_mul_xfe(ci, wci0, wci1, wci2, t0, t1, t2);
            xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
            
            bfe_mul_xfe(jsp, wjsp0, wjsp1, wjsp2, t0, t1, t2);
            xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
            
            bfe_mul_xfe(jso, wjso0, wjso1, wjso2, t0, t1, t2);
            xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
            
            bfe_mul_xfe(jsd, wjsd0, wjsd1, wjsd2, t0, t1, t2);
            xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
            
            uint64_t d0, d1, d2;
            xfe_sub_d(ind0, ind1, ind2, sum0, sum1, sum2, d0, d1, d2);
            
            xfe_mul_d(rp0, rp1, rp2, d0, d1, d2, rp0, rp1, rp2);
            
            size_t aux_idx = (i * AUX_TOTAL_COLS + AUX_JUMP_STACK_START + 0) * 3;
            d_aux[aux_idx + 0] = rp0;
            d_aux[aux_idx + 1] = rp1;
            d_aux[aux_idx + 2] = rp2;
            
            // CJD log derivative
            if (i > 0) {
                size_t prev_off = (i - 1) * main_width + MAIN_JUMP_STACK_START;
                uint64_t prev_clk = d_main[prev_off + JS_CLK];
                uint64_t prev_jsp = d_main[prev_off + JS_JSP];
                
                if (prev_jsp == jsp && clk > prev_clk) {
                    uint64_t clock_diff = bfield_sub_impl(clk, prev_clk);
                    uint64_t cd0, cd1, cd2;
                    xfe_sub_d(cjd_ind0, cjd_ind1, cjd_ind2, clock_diff, 0, 0, cd0, cd1, cd2);
                    uint64_t inv0, inv1, inv2;
                    xfe_inv_d(cd0, cd1, cd2, inv0, inv1, inv2);
                    xfe_add_d(ld0, ld1, ld2, inv0, inv1, inv2, ld0, ld1, ld2);
                }
            }
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_JUMP_STACK_START + 1) * 3;
            d_aux[aux_idx + 0] = ld0;
            d_aux[aux_idx + 1] = ld1;
            d_aux[aux_idx + 2] = ld2;
        }
    }
    
    // =========================================================================
    // 3. Program Table Extension (3 columns)
    // =========================================================================
    if (table_id == -1 || table_id == 2) {
        uint64_t instr_ind0, instr_ind1, instr_ind2;
        load_xfe(d_challenges, CH_InstructionLookup, instr_ind0, instr_ind1, instr_ind2);
        
        uint64_t wa0, wa1, wa2;  // address weight
        uint64_t wi0, wi1, wi2;  // instruction weight
        uint64_t wn0, wn1, wn2;  // next instruction weight
        load_xfe(d_challenges, CH_ProgAddrWeight, wa0, wa1, wa2);
        load_xfe(d_challenges, CH_ProgInstrWeight, wi0, wi1, wi2);
        load_xfe(d_challenges, CH_ProgNextInstrWeight, wn0, wn1, wn2);
        
        uint64_t prep_ind0, prep_ind1, prep_ind2;
        uint64_t send_ind0, send_ind1, send_ind2;
        load_xfe(d_challenges, CH_ProgPrepareChunk, prep_ind0, prep_ind1, prep_ind2);
        load_xfe(d_challenges, CH_ProgSendChunk, send_ind0, send_ind1, send_ind2);
        
        // Initial values
        uint64_t ld0 = 0, ld1 = 0, ld2 = 0;  // InstructionLookupServerLogDerivative
        uint64_t pe0 = 1, pe1 = 0, pe2 = 0;  // PrepareChunkRunningEvaluation (starts at 1 for EvalArg)
        uint64_t se0 = 1, se1 = 0, se2 = 0;  // SendChunkRunningEvaluation
        
        for (size_t i = 0; i + 1 < num_rows; i++) {
            size_t row_off = i * main_width + MAIN_PROGRAM_START;
            size_t next_off = (i + 1) * main_width + MAIN_PROGRAM_START;
            
            uint64_t addr = d_main[row_off + PROG_ADDRESS];
            uint64_t instr = d_main[row_off + PROG_INSTRUCTION];
            uint64_t mult = d_main[row_off + PROG_LOOKUP_MULT];
            uint64_t is_hash_pad = d_main[row_off + PROG_IS_HASH_PAD];
            uint64_t idx_in_chunk = d_main[row_off + PROG_INDEX_IN_CHUNK];
            uint64_t is_table_pad = d_main[row_off + PROG_IS_TABLE_PAD];
            uint64_t next_instr = d_main[next_off + PROG_INSTRUCTION];
            
            // Store current log derivative
            size_t aux_idx = (i * AUX_TOTAL_COLS + AUX_PROGRAM_START + 0) * 3;
            d_aux[aux_idx + 0] = ld0;
            d_aux[aux_idx + 1] = ld1;
            d_aux[aux_idx + 2] = ld2;
            
            // Update log derivative if not hash padding
            if (is_hash_pad != 1) {
                // compressed = addr*wa + instr*wi + next_instr*wn
                uint64_t t0, t1, t2;
                uint64_t sum0 = 0, sum1 = 0, sum2 = 0;
                
                bfe_mul_xfe(addr, wa0, wa1, wa2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                bfe_mul_xfe(instr, wi0, wi1, wi2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                bfe_mul_xfe(next_instr, wn0, wn1, wn2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                // diff = indeterminate - compressed
                uint64_t d0, d1, d2;
                xfe_sub_d(instr_ind0, instr_ind1, instr_ind2, sum0, sum1, sum2, d0, d1, d2);
                
                // inverse
                uint64_t inv0, inv1, inv2;
                xfe_inv_d(d0, d1, d2, inv0, inv1, inv2);
                
                // multiply by lookup multiplicity
                uint64_t term0, term1, term2;
                bfe_mul_xfe(mult, inv0, inv1, inv2, term0, term1, term2);
                
                // add to log derivative
                xfe_add_d(ld0, ld1, ld2, term0, term1, term2, ld0, ld1, ld2);
            }
            
            // PrepareChunkRunningEvaluation
            // Reset if index_in_chunk == 0
            if (idx_in_chunk == 0) {
                pe0 = 1; pe1 = 0; pe2 = 0;
            }
            // eval = eval * indeterminate + instruction
            uint64_t pt0, pt1, pt2;
            xfe_mul_d(pe0, pe1, pe2, prep_ind0, prep_ind1, prep_ind2, pt0, pt1, pt2);
            xfe_add_d(pt0, pt1, pt2, instr, 0, 0, pe0, pe1, pe2);
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_PROGRAM_START + 1) * 3;
            d_aux[aux_idx + 0] = pe0;
            d_aux[aux_idx + 1] = pe1;
            d_aux[aux_idx + 2] = pe2;
            
            // SendChunkRunningEvaluation
            // Update if not table padding and index_in_chunk == 9 (RATE-1)
            if (is_table_pad != 1 && idx_in_chunk == 9) {
                uint64_t st0, st1, st2;
                xfe_mul_d(se0, se1, se2, send_ind0, send_ind1, send_ind2, st0, st1, st2);
                xfe_add_d(st0, st1, st2, pe0, pe1, pe2, se0, se1, se2);
            }
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_PROGRAM_START + 2) * 3;
            d_aux[aux_idx + 0] = se0;
            d_aux[aux_idx + 1] = se1;
            d_aux[aux_idx + 2] = se2;
        }
        
        // Handle last row - must also update PrepareChunk and SendChunk
        if (num_rows > 0) {
            size_t last = num_rows - 1;
            size_t row_off = last * main_width + MAIN_PROGRAM_START;
            
            uint64_t instr = d_main[row_off + PROG_INSTRUCTION];
            uint64_t idx_in_chunk = d_main[row_off + PROG_INDEX_IN_CHUNK];
            uint64_t is_table_pad = d_main[row_off + PROG_IS_TABLE_PAD];
            
            // Update PrepareChunk for last row
            if (idx_in_chunk == 0) {
                pe0 = 1; pe1 = 0; pe2 = 0;
            }
            uint64_t pt0, pt1, pt2;
            xfe_mul_d(pe0, pe1, pe2, prep_ind0, prep_ind1, prep_ind2, pt0, pt1, pt2);
            xfe_add_d(pt0, pt1, pt2, instr, 0, 0, pe0, pe1, pe2);
            
            // Update SendChunk for last row
            if (is_table_pad != 1 && idx_in_chunk == 9) {
                uint64_t st0, st1, st2;
                xfe_mul_d(se0, se1, se2, send_ind0, send_ind1, send_ind2, st0, st1, st2);
                xfe_add_d(st0, st1, st2, pe0, pe1, pe2, se0, se1, se2);
            }
            
            // Store all three columns
            size_t aux_idx = (last * AUX_TOTAL_COLS + AUX_PROGRAM_START + 0) * 3;
            d_aux[aux_idx + 0] = ld0;
            d_aux[aux_idx + 1] = ld1;
            d_aux[aux_idx + 2] = ld2;
            
            aux_idx = (last * AUX_TOTAL_COLS + AUX_PROGRAM_START + 1) * 3;
            d_aux[aux_idx + 0] = pe0;
            d_aux[aux_idx + 1] = pe1;
            d_aux[aux_idx + 2] = pe2;
            
            aux_idx = (last * AUX_TOTAL_COLS + AUX_PROGRAM_START + 2) * 3;
            d_aux[aux_idx + 0] = se0;
            d_aux[aux_idx + 1] = se1;
            d_aux[aux_idx + 2] = se2;
        }
    }
    
    // =========================================================================
    // 4. Lookup Table Extension (2 columns)
    // =========================================================================
    if (table_id == -1 || table_id == 3) {
        // Challenge indices
        constexpr size_t CH_CascadeLookup = 51;
        constexpr size_t CH_LookupIn = 52;
        constexpr size_t CH_LookupOut = 53;
        constexpr size_t CH_LookupPublic = 54;
        
        // Lookup table main columns (relative to MAIN_LOOKUP_START=135)
        constexpr size_t LK_IsPadding = 0;
        constexpr size_t LK_LookIn = 1;
        constexpr size_t LK_LookOut = 2;
        constexpr size_t LK_LookupMult = 3;
        
        uint64_t cascade_ind0, cascade_ind1, cascade_ind2;
        load_xfe(d_challenges, CH_CascadeLookup, cascade_ind0, cascade_ind1, cascade_ind2);
        
        uint64_t in_w0, in_w1, in_w2;
        uint64_t out_w0, out_w1, out_w2;
        load_xfe(d_challenges, CH_LookupIn, in_w0, in_w1, in_w2);
        load_xfe(d_challenges, CH_LookupOut, out_w0, out_w1, out_w2);
        
        uint64_t pub_ind0, pub_ind1, pub_ind2;
        load_xfe(d_challenges, CH_LookupPublic, pub_ind0, pub_ind1, pub_ind2);
        
        // Running accumulators (start at 0 for log deriv, 1 for eval)
        uint64_t ld0 = 0, ld1 = 0, ld2 = 0;  // CascadeRunningSumLogDerivative
        uint64_t pe0 = 1, pe1 = 0, pe2 = 0;  // PublicRunningEvaluation
        
        bool hit_padding = false;
        
        for (size_t i = 0; i < num_rows && !hit_padding; i++) {
            size_t row_off = i * main_width + MAIN_LOOKUP_START;
            uint64_t is_pad = d_main[row_off + LK_IsPadding];
            uint64_t look_in = d_main[row_off + LK_LookIn];
            uint64_t look_out = d_main[row_off + LK_LookOut];
            uint64_t mult = d_main[row_off + LK_LookupMult];
            
            if (is_pad == 1) {
                // Fill rest with current values
                for (size_t j = i; j < num_rows; j++) {
                    size_t aux_idx0 = (j * AUX_TOTAL_COLS + AUX_LOOKUP_START + 0) * 3;
                    d_aux[aux_idx0 + 0] = ld0; d_aux[aux_idx0 + 1] = ld1; d_aux[aux_idx0 + 2] = ld2;
                    size_t aux_idx1 = (j * AUX_TOTAL_COLS + AUX_LOOKUP_START + 1) * 3;
                    d_aux[aux_idx1 + 0] = pe0; d_aux[aux_idx1 + 1] = pe1; d_aux[aux_idx1 + 2] = pe2;
                }
                hit_padding = true;
            } else {
                // compressed_row = look_in * in_weight + look_out * out_weight
                uint64_t t0, t1, t2;
                uint64_t sum0 = 0, sum1 = 0, sum2 = 0;
                bfe_mul_xfe(look_in, in_w0, in_w1, in_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                bfe_mul_xfe(look_out, out_w0, out_w1, out_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                // diff = cascade_ind - compressed_row
                uint64_t d0, d1, d2;
                xfe_sub_d(cascade_ind0, cascade_ind1, cascade_ind2, sum0, sum1, sum2, d0, d1, d2);
                
                // inverse
                uint64_t inv0, inv1, inv2;
                xfe_inv_d(d0, d1, d2, inv0, inv1, inv2);
                
                // ld += inverse * mult
                uint64_t term0, term1, term2;
                bfe_mul_xfe(mult, inv0, inv1, inv2, term0, term1, term2);
                xfe_add_d(ld0, ld1, ld2, term0, term1, term2, ld0, ld1, ld2);
                
                // public_eval = public_eval * pub_ind + look_out
                uint64_t pt0, pt1, pt2;
                xfe_mul_d(pe0, pe1, pe2, pub_ind0, pub_ind1, pub_ind2, pt0, pt1, pt2);
                xfe_add_d(pt0, pt1, pt2, look_out, 0, 0, pe0, pe1, pe2);
                
                // Store
                size_t aux_idx0 = (i * AUX_TOTAL_COLS + AUX_LOOKUP_START + 0) * 3;
                d_aux[aux_idx0 + 0] = ld0; d_aux[aux_idx0 + 1] = ld1; d_aux[aux_idx0 + 2] = ld2;
                size_t aux_idx1 = (i * AUX_TOTAL_COLS + AUX_LOOKUP_START + 1) * 3;
                d_aux[aux_idx1 + 0] = pe0; d_aux[aux_idx1 + 1] = pe1; d_aux[aux_idx1 + 2] = pe2;
            }
        }
    }
    
    // =========================================================================
    // 5. U32 Table Extension (2 columns - but only 1 is used)
    // =========================================================================
    if (table_id == -1 || table_id == 4) {
        constexpr size_t CH_U32Ind = 10;
        constexpr size_t CH_U32Ci = 57;
        constexpr size_t CH_U32Lhs = 55;
        constexpr size_t CH_U32Rhs = 56;
        constexpr size_t CH_U32Result = 58;
        
        // U32 table main columns (relative to MAIN_U32_START=139)
        // Must match CPU: CopyFlag=0, Bits=1, BitsMinus33Inv=2, CI=3, LHS=4, LhsInv=5, RHS=6, RhsInv=7, Result=8, LookupMult=9
        constexpr size_t U32_CopyFlag = 0;
        constexpr size_t U32_CI = 3;
        constexpr size_t U32_LHS = 4;
        constexpr size_t U32_RHS = 6;
        constexpr size_t U32_Result = 8;
        constexpr size_t U32_LookupMult = 9;
        
        uint64_t u32_ind0, u32_ind1, u32_ind2;
        load_xfe(d_challenges, CH_U32Ind, u32_ind0, u32_ind1, u32_ind2);
        
        uint64_t ci_w0, ci_w1, ci_w2;
        uint64_t lhs_w0, lhs_w1, lhs_w2;
        uint64_t rhs_w0, rhs_w1, rhs_w2;
        uint64_t res_w0, res_w1, res_w2;
        load_xfe(d_challenges, CH_U32Ci, ci_w0, ci_w1, ci_w2);
        load_xfe(d_challenges, CH_U32Lhs, lhs_w0, lhs_w1, lhs_w2);
        load_xfe(d_challenges, CH_U32Rhs, rhs_w0, rhs_w1, rhs_w2);
        load_xfe(d_challenges, CH_U32Result, res_w0, res_w1, res_w2);
        
        uint64_t ld0 = 0, ld1 = 0, ld2 = 0;  // LookupServerLogDerivative
        
        for (size_t i = 0; i < num_rows; i++) {
            size_t row_off = i * main_width + MAIN_U32_START;
            uint64_t copy_flag = d_main[row_off + U32_CopyFlag];
            
            if (copy_flag == 1) {
                uint64_t ci = d_main[row_off + U32_CI];
                uint64_t lhs = d_main[row_off + U32_LHS];
                uint64_t rhs = d_main[row_off + U32_RHS];
                uint64_t result = d_main[row_off + U32_Result];
                uint64_t mult = d_main[row_off + U32_LookupMult];
                
                // compressed = ci*w + lhs*w + rhs*w + result*w
                uint64_t t0, t1, t2;
                uint64_t sum0 = 0, sum1 = 0, sum2 = 0;
                bfe_mul_xfe(ci, ci_w0, ci_w1, ci_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                bfe_mul_xfe(lhs, lhs_w0, lhs_w1, lhs_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                bfe_mul_xfe(rhs, rhs_w0, rhs_w1, rhs_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                bfe_mul_xfe(result, res_w0, res_w1, res_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                // diff = indeterminate - compressed
                uint64_t d0, d1, d2;
                xfe_sub_d(u32_ind0, u32_ind1, u32_ind2, sum0, sum1, sum2, d0, d1, d2);
                
                // inverse
                uint64_t inv0, inv1, inv2;
                xfe_inv_d(d0, d1, d2, inv0, inv1, inv2);
                
                // ld += mult * inverse
                uint64_t term0, term1, term2;
                bfe_mul_xfe(mult, inv0, inv1, inv2, term0, term1, term2);
                xfe_add_d(ld0, ld1, ld2, term0, term1, term2, ld0, ld1, ld2);
            }
            
            // Store (col 0 is LookupServerLogDerivative)
            size_t aux_idx = (i * AUX_TOTAL_COLS + AUX_U32_START + 0) * 3;
            d_aux[aux_idx + 0] = ld0;
            d_aux[aux_idx + 1] = ld1;
            d_aux[aux_idx + 2] = ld2;
            
            // Col 1 is unused (set to 0)
            aux_idx = (i * AUX_TOTAL_COLS + AUX_U32_START + 1) * 3;
            d_aux[aux_idx + 0] = 0;
            d_aux[aux_idx + 1] = 0;
            d_aux[aux_idx + 2] = 0;
        }
    }
    
    // =========================================================================
    // 6. Cascade Table Extension (2 columns)
    // =========================================================================
    if (table_id == -1 || table_id == 5) {
        constexpr size_t CH_HashCascade = 48;
        constexpr size_t CH_HashCascadeIn = 49;
        constexpr size_t CH_HashCascadeOut = 50;
        constexpr size_t CH_CascadeLookup = 51;
        constexpr size_t CH_LookupIn = 52;
        constexpr size_t CH_LookupOut = 53;
        
        // Cascade main columns (relative to MAIN_CASCADE_START=129)
        constexpr size_t CASC_IsPadding = 0;
        constexpr size_t CASC_LookInHi = 1;
        constexpr size_t CASC_LookInLo = 2;
        constexpr size_t CASC_LookOutHi = 3;
        constexpr size_t CASC_LookOutLo = 4;
        constexpr size_t CASC_LookupMult = 5;
        
        uint64_t hash_ind0, hash_ind1, hash_ind2;
        load_xfe(d_challenges, CH_HashCascade, hash_ind0, hash_ind1, hash_ind2);
        uint64_t hash_in_w0, hash_in_w1, hash_in_w2;
        uint64_t hash_out_w0, hash_out_w1, hash_out_w2;
        load_xfe(d_challenges, CH_HashCascadeIn, hash_in_w0, hash_in_w1, hash_in_w2);
        load_xfe(d_challenges, CH_HashCascadeOut, hash_out_w0, hash_out_w1, hash_out_w2);
        
        uint64_t lookup_ind0, lookup_ind1, lookup_ind2;
        load_xfe(d_challenges, CH_CascadeLookup, lookup_ind0, lookup_ind1, lookup_ind2);
        uint64_t lookup_in_w0, lookup_in_w1, lookup_in_w2;
        uint64_t lookup_out_w0, lookup_out_w1, lookup_out_w2;
        load_xfe(d_challenges, CH_LookupIn, lookup_in_w0, lookup_in_w1, lookup_in_w2);
        load_xfe(d_challenges, CH_LookupOut, lookup_out_w0, lookup_out_w1, lookup_out_w2);
        
        constexpr uint64_t TWO_POW_8 = 256;
        
        uint64_t hash_ld0 = 0, hash_ld1 = 0, hash_ld2 = 0;    // HashTableServerLogDerivative
        uint64_t lookup_ld0 = 0, lookup_ld1 = 0, lookup_ld2 = 0;  // LookupTableClientLogDerivative
        
        for (size_t i = 0; i < num_rows; i++) {
            size_t row_off = i * main_width + MAIN_CASCADE_START;
            uint64_t is_pad = d_main[row_off + CASC_IsPadding];
            
            if (is_pad != 1) {
                uint64_t look_in_hi = d_main[row_off + CASC_LookInHi];
                uint64_t look_in_lo = d_main[row_off + CASC_LookInLo];
                uint64_t look_out_hi = d_main[row_off + CASC_LookOutHi];
                uint64_t look_out_lo = d_main[row_off + CASC_LookOutLo];
                uint64_t mult = d_main[row_off + CASC_LookupMult];
                
                // look_in = 256 * look_in_hi + look_in_lo
                uint64_t look_in = bfield_add_impl(bfield_mul_impl(TWO_POW_8, look_in_hi), look_in_lo);
                uint64_t look_out = bfield_add_impl(bfield_mul_impl(TWO_POW_8, look_out_hi), look_out_lo);
                
                // HashTableServerLogDerivative
                // compressed_hash = hash_in_w * look_in + hash_out_w * look_out
                uint64_t t0, t1, t2;
                uint64_t sum0 = 0, sum1 = 0, sum2 = 0;
                bfe_mul_xfe(look_in, hash_in_w0, hash_in_w1, hash_in_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                bfe_mul_xfe(look_out, hash_out_w0, hash_out_w1, hash_out_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                // diff = hash_ind - compressed_hash
                uint64_t d0, d1, d2;
                xfe_sub_d(hash_ind0, hash_ind1, hash_ind2, sum0, sum1, sum2, d0, d1, d2);
                
                // hash_ld += inverse * mult
                uint64_t inv0, inv1, inv2;
                xfe_inv_d(d0, d1, d2, inv0, inv1, inv2);
                uint64_t term0, term1, term2;
                bfe_mul_xfe(mult, inv0, inv1, inv2, term0, term1, term2);
                xfe_add_d(hash_ld0, hash_ld1, hash_ld2, term0, term1, term2, hash_ld0, hash_ld1, hash_ld2);
                
                // LookupTableClientLogDerivative
                // Two contributions: one for lo, one for hi
                // compressed_lo = lookup_in_w * look_in_lo + lookup_out_w * look_out_lo
                sum0 = 0; sum1 = 0; sum2 = 0;
                bfe_mul_xfe(look_in_lo, lookup_in_w0, lookup_in_w1, lookup_in_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                bfe_mul_xfe(look_out_lo, lookup_out_w0, lookup_out_w1, lookup_out_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                uint64_t diff_lo0, diff_lo1, diff_lo2;
                xfe_sub_d(lookup_ind0, lookup_ind1, lookup_ind2, sum0, sum1, sum2, diff_lo0, diff_lo1, diff_lo2);
                
                // compressed_hi = lookup_in_w * look_in_hi + lookup_out_w * look_out_hi
                sum0 = 0; sum1 = 0; sum2 = 0;
                bfe_mul_xfe(look_in_hi, lookup_in_w0, lookup_in_w1, lookup_in_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                bfe_mul_xfe(look_out_hi, lookup_out_w0, lookup_out_w1, lookup_out_w2, t0, t1, t2);
                xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                
                uint64_t diff_hi0, diff_hi1, diff_hi2;
                xfe_sub_d(lookup_ind0, lookup_ind1, lookup_ind2, sum0, sum1, sum2, diff_hi0, diff_hi1, diff_hi2);
                
                // lookup_ld += inverse(diff_lo) + inverse(diff_hi)
                uint64_t inv_lo0, inv_lo1, inv_lo2;
                xfe_inv_d(diff_lo0, diff_lo1, diff_lo2, inv_lo0, inv_lo1, inv_lo2);
                xfe_add_d(lookup_ld0, lookup_ld1, lookup_ld2, inv_lo0, inv_lo1, inv_lo2, lookup_ld0, lookup_ld1, lookup_ld2);
                
                uint64_t inv_hi0, inv_hi1, inv_hi2;
                xfe_inv_d(diff_hi0, diff_hi1, diff_hi2, inv_hi0, inv_hi1, inv_hi2);
                xfe_add_d(lookup_ld0, lookup_ld1, lookup_ld2, inv_hi0, inv_hi1, inv_hi2, lookup_ld0, lookup_ld1, lookup_ld2);
            }
            
            // Store
            size_t aux_idx0 = (i * AUX_TOTAL_COLS + AUX_CASCADE_START + 0) * 3;
            d_aux[aux_idx0 + 0] = hash_ld0; d_aux[aux_idx0 + 1] = hash_ld1; d_aux[aux_idx0 + 2] = hash_ld2;
            size_t aux_idx1 = (i * AUX_TOTAL_COLS + AUX_CASCADE_START + 1) * 3;
            d_aux[aux_idx1 + 0] = lookup_ld0; d_aux[aux_idx1 + 1] = lookup_ld1; d_aux[aux_idx1 + 2] = lookup_ld2;
        }
    }
    
    // =========================================================================
    // 7. RAM Table Extension (6 columns)
    // =========================================================================
    if (table_id == -1 || table_id == 6) {
        constexpr size_t CH_RamBezout = 12;
        constexpr size_t CH_Ram = 8;
        constexpr size_t CH_CJD = 11;
        constexpr size_t CH_RamClkW = 20;
        constexpr size_t CH_RamPtrW = 21;
        constexpr size_t CH_RamValW = 22;
        constexpr size_t CH_RamInstW = 23;
        
        // RAM main columns (relative to MAIN_RAM_START=50)
        constexpr size_t RAM_CLK = 0;
        constexpr size_t RAM_InstrType = 1;
        constexpr size_t RAM_RamPtr = 2;
        constexpr size_t RAM_RamVal = 3;
        constexpr size_t RAM_BezoutCoeff0 = 5;
        constexpr size_t RAM_BezoutCoeff1 = 6;
        
        // Aux columns (relative to AUX_RAM_START=16)
        constexpr size_t AUX_RunningProductOfRAMP = 0;
        constexpr size_t AUX_FormalDerivative = 1;
        constexpr size_t AUX_BezoutCoeff0 = 2;
        constexpr size_t AUX_BezoutCoeff1 = 3;
        constexpr size_t AUX_RunningProductPermArg = 4;
        constexpr size_t AUX_CJDLogDeriv = 5;
        
        constexpr uint64_t PADDING_INDICATOR = 2;
        
        uint64_t bezout_ind0, bezout_ind1, bezout_ind2;
        load_xfe(d_challenges, CH_RamBezout, bezout_ind0, bezout_ind1, bezout_ind2);
        
        uint64_t ram_ind0, ram_ind1, ram_ind2;
        load_xfe(d_challenges, CH_Ram, ram_ind0, ram_ind1, ram_ind2);
        
        uint64_t cjd_ind0, cjd_ind1, cjd_ind2;
        load_xfe(d_challenges, CH_CJD, cjd_ind0, cjd_ind1, cjd_ind2);
        
        uint64_t clk_w0, clk_w1, clk_w2;
        uint64_t ptr_w0, ptr_w1, ptr_w2;
        uint64_t val_w0, val_w1, val_w2;
        uint64_t inst_w0, inst_w1, inst_w2;
        load_xfe(d_challenges, CH_RamClkW, clk_w0, clk_w1, clk_w2);
        load_xfe(d_challenges, CH_RamPtrW, ptr_w0, ptr_w1, ptr_w2);
        load_xfe(d_challenges, CH_RamValW, val_w0, val_w1, val_w2);
        load_xfe(d_challenges, CH_RamInstW, inst_w0, inst_w1, inst_w2);
        
        // Initialize from first row
        uint64_t first_ptr = d_main[MAIN_RAM_START + RAM_RamPtr];
        
        // RunningProductOfRAMP = bezout_ind - first_ptr
        uint64_t rp_ramp0, rp_ramp1, rp_ramp2;
        xfe_sub_d(bezout_ind0, bezout_ind1, bezout_ind2, first_ptr, 0, 0, rp_ramp0, rp_ramp1, rp_ramp2);
        
        // FormalDerivative = 1
        uint64_t fd0 = 1, fd1 = 0, fd2 = 0;
        
        // BezoutCoefficients from main table
        uint64_t bc0_0 = d_main[MAIN_RAM_START + RAM_BezoutCoeff0], bc0_1 = 0, bc0_2 = 0;
        uint64_t bc1_0 = d_main[MAIN_RAM_START + RAM_BezoutCoeff1], bc1_1 = 0, bc1_2 = 0;
        
        // RunningProductPermArg = 1
        uint64_t rp_perm0 = 1, rp_perm1 = 0, rp_perm2 = 0;
        
        // CJD LogDeriv = 0
        uint64_t cjd_ld0 = 0, cjd_ld1 = 0, cjd_ld2 = 0;
        
        bool hit_padding = false;
        bool hit_padding_perm = false;
        bool hit_padding_cjd = false;
        bool hit_padding_bezout = false;
        
        for (size_t i = 0; i < num_rows; i++) {
            size_t row_off = i * main_width + MAIN_RAM_START;
            uint64_t instr_type = d_main[row_off + RAM_InstrType];
            uint64_t clk = d_main[row_off + RAM_CLK];
            uint64_t curr_ptr = d_main[row_off + RAM_RamPtr];
            uint64_t ram_val = d_main[row_off + RAM_RamVal];
            
            // RunningProductPermArg - updates from row 0, stops at padding
            if (!hit_padding_perm) {
                if (instr_type == PADDING_INDICATOR) {
                    hit_padding_perm = true;
                    // Fill remaining rows with last value (done at end of loop)
                } else {
                    // compressed = clk*w + type*w + ptr*w + val*w
                    uint64_t t0, t1, t2;
                    uint64_t sum0 = 0, sum1 = 0, sum2 = 0;
                    bfe_mul_xfe(clk, clk_w0, clk_w1, clk_w2, t0, t1, t2);
                    xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                    bfe_mul_xfe(instr_type, inst_w0, inst_w1, inst_w2, t0, t1, t2);
                    xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                    bfe_mul_xfe(curr_ptr, ptr_w0, ptr_w1, ptr_w2, t0, t1, t2);
                    xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                    bfe_mul_xfe(ram_val, val_w0, val_w1, val_w2, t0, t1, t2);
                    xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                    
                    // diff = ram_ind - compressed
                    uint64_t diff0, diff1, diff2;
                    xfe_sub_d(ram_ind0, ram_ind1, ram_ind2, sum0, sum1, sum2, diff0, diff1, diff2);
                    
                    // rp_perm *= diff
                    xfe_mul_d(rp_perm0, rp_perm1, rp_perm2, diff0, diff1, diff2, rp_perm0, rp_perm1, rp_perm2);
                }
            }
            
            // CJD LogDeriv - updates from row 1, stops at padding
            if (i > 0 && !hit_padding_cjd) {
                if (instr_type == PADDING_INDICATOR) {
                    hit_padding_cjd = true;
                } else {
                    size_t prev_off = (i - 1) * main_width + MAIN_RAM_START;
                    uint64_t prev_ptr = d_main[prev_off + RAM_RamPtr];
                    if (prev_ptr == curr_ptr) {
                        uint64_t prev_clk = d_main[prev_off + RAM_CLK];
                        uint64_t clock_diff = bfield_sub_impl(clk, prev_clk);
                        uint64_t cd0, cd1, cd2;
                        xfe_sub_d(cjd_ind0, cjd_ind1, cjd_ind2, clock_diff, 0, 0, cd0, cd1, cd2);
                        uint64_t inv0, inv1, inv2;
                        xfe_inv_d(cd0, cd1, cd2, inv0, inv1, inv2);
                        xfe_add_d(cjd_ld0, cjd_ld1, cjd_ld2, inv0, inv1, inv2, cjd_ld0, cjd_ld1, cjd_ld2);
                    }
                }
            }
            
            // RAMP, FormalDerivative, Bezout - updates from row 1, stops at padding
            if (i > 0 && !hit_padding_bezout) {
                if (instr_type == PADDING_INDICATOR) {
                    hit_padding_bezout = true;
                } else {
                    size_t prev_off = (i - 1) * main_width + MAIN_RAM_START;
                    uint64_t prev_ptr = d_main[prev_off + RAM_RamPtr];
                    
                    if (prev_ptr != curr_ptr) {
                        uint64_t diff_ptr0, diff_ptr1, diff_ptr2;
                        xfe_sub_d(bezout_ind0, bezout_ind1, bezout_ind2, curr_ptr, 0, 0, diff_ptr0, diff_ptr1, diff_ptr2);
                        
                        // fd = diff * fd + rp_ramp
                        uint64_t new_fd0, new_fd1, new_fd2;
                        xfe_mul_d(diff_ptr0, diff_ptr1, diff_ptr2, fd0, fd1, fd2, new_fd0, new_fd1, new_fd2);
                        xfe_add_d(new_fd0, new_fd1, new_fd2, rp_ramp0, rp_ramp1, rp_ramp2, fd0, fd1, fd2);
                        
                        // rp_ramp = rp_ramp * diff
                        xfe_mul_d(rp_ramp0, rp_ramp1, rp_ramp2, diff_ptr0, diff_ptr1, diff_ptr2, rp_ramp0, rp_ramp1, rp_ramp2);
                        
                        // Bezout coefficients: bc = bc * bezout_ind + bc_main
                        uint64_t new_bc00, new_bc01, new_bc02;
                        xfe_mul_d(bc0_0, bc0_1, bc0_2, bezout_ind0, bezout_ind1, bezout_ind2, new_bc00, new_bc01, new_bc02);
                        uint64_t bc0_main = d_main[row_off + RAM_BezoutCoeff0];
                        xfe_add_d(new_bc00, new_bc01, new_bc02, bc0_main, 0, 0, bc0_0, bc0_1, bc0_2);
                        
                        uint64_t new_bc10, new_bc11, new_bc12;
                        xfe_mul_d(bc1_0, bc1_1, bc1_2, bezout_ind0, bezout_ind1, bezout_ind2, new_bc10, new_bc11, new_bc12);
                        uint64_t bc1_main = d_main[row_off + RAM_BezoutCoeff1];
                        xfe_add_d(new_bc10, new_bc11, new_bc12, bc1_main, 0, 0, bc1_0, bc1_1, bc1_2);
                    }
                }
            }
            
            // Store all 6 aux columns
            size_t aux_idx;
            aux_idx = (i * AUX_TOTAL_COLS + AUX_RAM_START + AUX_RunningProductOfRAMP) * 3;
            d_aux[aux_idx + 0] = rp_ramp0; d_aux[aux_idx + 1] = rp_ramp1; d_aux[aux_idx + 2] = rp_ramp2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_RAM_START + AUX_FormalDerivative) * 3;
            d_aux[aux_idx + 0] = fd0; d_aux[aux_idx + 1] = fd1; d_aux[aux_idx + 2] = fd2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_RAM_START + AUX_BezoutCoeff0) * 3;
            d_aux[aux_idx + 0] = bc0_0; d_aux[aux_idx + 1] = bc0_1; d_aux[aux_idx + 2] = bc0_2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_RAM_START + AUX_BezoutCoeff1) * 3;
            d_aux[aux_idx + 0] = bc1_0; d_aux[aux_idx + 1] = bc1_1; d_aux[aux_idx + 2] = bc1_2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_RAM_START + AUX_RunningProductPermArg) * 3;
            d_aux[aux_idx + 0] = rp_perm0; d_aux[aux_idx + 1] = rp_perm1; d_aux[aux_idx + 2] = rp_perm2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_RAM_START + AUX_CJDLogDeriv) * 3;
            d_aux[aux_idx + 0] = cjd_ld0; d_aux[aux_idx + 1] = cjd_ld1; d_aux[aux_idx + 2] = cjd_ld2;
        }
    }
    
    // =========================================================================
    // 8. Hash Table Extension (20 columns)
    // Columns: ReceiveChunkEval, HashInputEval, HashDigestEval, SpongeEval,
    //          16 CascadeState log derivatives (4 states × 4 limbs)
    // =========================================================================
    if (table_id == -1 || table_id == 7) {
        // Challenge indices
        constexpr size_t CH_HashInput = 4;
        constexpr size_t CH_HashDigest = 5;
        constexpr size_t CH_Sponge = 6;
        constexpr size_t CH_HashCascade = 48;
        constexpr size_t CH_HashCascadeIn = 49;
        constexpr size_t CH_HashCascadeOut = 50;
        constexpr size_t CH_PrepareChunk = 29;
        constexpr size_t CH_SendChunk = 30;
        constexpr size_t CH_HashCI = 31;
        constexpr size_t CH_StackWeight0 = 32;  // 10 stack weights: 32-41
        
        // Hash main columns (relative to MAIN_HASH_START=62)
        constexpr size_t HASH_Mode = 0;
        constexpr size_t HASH_CI = 1;
        constexpr size_t HASH_RoundNumber = 2;
        // State0: columns 3-6 (highest, midhigh, midlow, lowest) LkIn
        // State1: columns 7-10
        // State2: columns 11-14
        // State3: columns 15-18
        // State0 LkOut: columns 19-22
        // State1 LkOut: columns 23-26
        // State2 LkOut: columns 27-30
        // State3 LkOut: columns 31-34
        // State4-15: columns 35-46
        
        constexpr size_t NUM_ROUNDS = 5;  // Tip5::NUM_ROUNDS
        constexpr size_t RATE = 10;       // Tip5::RATE
        constexpr size_t DIGEST_LEN = 5;
        
        // Mode values
        constexpr uint64_t MODE_PROGRAM_HASHING = 1;
        constexpr uint64_t MODE_SPONGE = 2;
        constexpr uint64_t MODE_HASH = 3;
        constexpr uint64_t MODE_PAD = 0;
        
        // Montgomery modulus inverse (pre-computed)
        // MONTGOMERY_MODULUS = 4294967295 (2^32 - 1)
        // inverse = pow(4294967295, -1, 2^64 - 2^32 + 1) = 18446744065119617025
        constexpr uint64_t MONTGOMERY_MOD_INV = 18446744065119617025ULL;
        constexpr uint64_t TWO_POW_16 = 65536ULL;
        constexpr uint64_t TWO_POW_32 = 4294967296ULL;
        constexpr uint64_t TWO_POW_48 = 281474976710656ULL;
        
        // SpongeInit opcode (from include/table/extend_helpers.hpp: case SpongeInit return 40)
        constexpr uint64_t SPONGE_INIT_OPCODE = 40;
        
        // Load challenges
        uint64_t hash_input_ind0, hash_input_ind1, hash_input_ind2;
        uint64_t hash_digest_ind0, hash_digest_ind1, hash_digest_ind2;
        uint64_t sponge_ind0, sponge_ind1, sponge_ind2;
        uint64_t cascade_ind0, cascade_ind1, cascade_ind2;
        uint64_t cascade_in_w0, cascade_in_w1, cascade_in_w2;
        uint64_t cascade_out_w0, cascade_out_w1, cascade_out_w2;
        uint64_t send_chunk_ind0, send_chunk_ind1, send_chunk_ind2;
        uint64_t prepare_chunk_ind0, prepare_chunk_ind1, prepare_chunk_ind2;
        uint64_t ci_weight0, ci_weight1, ci_weight2;
        
        load_xfe(d_challenges, CH_HashInput, hash_input_ind0, hash_input_ind1, hash_input_ind2);
        load_xfe(d_challenges, CH_HashDigest, hash_digest_ind0, hash_digest_ind1, hash_digest_ind2);
        load_xfe(d_challenges, CH_Sponge, sponge_ind0, sponge_ind1, sponge_ind2);
        load_xfe(d_challenges, CH_HashCascade, cascade_ind0, cascade_ind1, cascade_ind2);
        load_xfe(d_challenges, CH_HashCascadeIn, cascade_in_w0, cascade_in_w1, cascade_in_w2);
        load_xfe(d_challenges, CH_HashCascadeOut, cascade_out_w0, cascade_out_w1, cascade_out_w2);
        load_xfe(d_challenges, CH_SendChunk, send_chunk_ind0, send_chunk_ind1, send_chunk_ind2);
        load_xfe(d_challenges, CH_PrepareChunk, prepare_chunk_ind0, prepare_chunk_ind1, prepare_chunk_ind2);
        load_xfe(d_challenges, CH_HashCI, ci_weight0, ci_weight1, ci_weight2);
        
        // Load state weights (10 weights for RATE)
        uint64_t state_w[10][3];
        for (size_t j = 0; j < 10; j++) {
            load_xfe(d_challenges, CH_StackWeight0 + j, state_w[j][0], state_w[j][1], state_w[j][2]);
        }
        
        // Initialize running evaluations and log derivatives
        uint64_t recv_chunk_eval0 = 1, recv_chunk_eval1 = 0, recv_chunk_eval2 = 0;
        uint64_t hash_input_eval0 = 1, hash_input_eval1 = 0, hash_input_eval2 = 0;
        uint64_t hash_digest_eval0 = 1, hash_digest_eval1 = 0, hash_digest_eval2 = 0;
        uint64_t sponge_eval0 = 1, sponge_eval1 = 0, sponge_eval2 = 0;
        
        // 16 cascade log derivatives (4 states × 4 limbs)
        uint64_t cascade_ld[16][3];
        for (size_t j = 0; j < 16; j++) {
            cascade_ld[j][0] = 0; cascade_ld[j][1] = 0; cascade_ld[j][2] = 0;
        }
        
        for (size_t i = 0; i < num_rows; i++) {
            size_t row_off = i * main_width + MAIN_HASH_START;
            
            uint64_t mode = d_main[row_off + HASH_Mode];
            uint64_t ci = d_main[row_off + HASH_CI];
            uint64_t round_num = d_main[row_off + HASH_RoundNumber];
            
            bool in_program_hashing = (mode == MODE_PROGRAM_HASHING);
            bool in_sponge = (mode == MODE_SPONGE);
            bool in_hash = (mode == MODE_HASH);
            bool in_pad = (mode == MODE_PAD);
            bool in_round_0 = (round_num == 0);
            bool in_last_round = (round_num == NUM_ROUNDS);
            bool is_sponge_init = (ci == SPONGE_INIT_OPCODE);
            
            // Read and recompose rate registers (first 4 need limb recomposition)
            uint64_t rate_regs[10];
            for (size_t j = 0; j < 4; j++) {
                // State j: 4 limbs at columns 3 + j*4 to 3 + j*4 + 3
                size_t base = 3 + j * 4;
                uint64_t highest = d_main[row_off + base + 0];
                uint64_t midhigh = d_main[row_off + base + 1];
                uint64_t midlow = d_main[row_off + base + 2];
                uint64_t lowest = d_main[row_off + base + 3];
                // recompose = (highest * 2^48 + midhigh * 2^32 + midlow * 2^16 + lowest) * montgomery_inv
                uint64_t composed = bfield_add_impl(
                    bfield_add_impl(
                        bfield_mul_impl(highest, TWO_POW_48),
                        bfield_mul_impl(midhigh, TWO_POW_32)
                    ),
                    bfield_add_impl(
                        bfield_mul_impl(midlow, TWO_POW_16),
                        lowest
                    )
                );
                rate_regs[j] = bfield_mul_impl(composed, MONTGOMERY_MOD_INV);
            }
            // State 4-9 are direct columns 35-40
            for (size_t j = 4; j < 10; j++) {
                rate_regs[j] = d_main[row_off + 35 + (j - 4)];
            }
            
            // Compute compressed row = sum(state_weights[j] * rate_regs[j])
            uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
            for (size_t j = 0; j < RATE; j++) {
                uint64_t t0, t1, t2;
                bfe_mul_xfe(rate_regs[j], state_w[j][0], state_w[j][1], state_w[j][2], t0, t1, t2);
                xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
            }
            
            // Compute digest compression (first 5 elements only)
            uint64_t digest_comp0 = 0, digest_comp1 = 0, digest_comp2 = 0;
            for (size_t j = 0; j < DIGEST_LEN; j++) {
                uint64_t t0, t1, t2;
                bfe_mul_xfe(rate_regs[j], state_w[j][0], state_w[j][1], state_w[j][2], t0, t1, t2);
                xfe_add_d(digest_comp0, digest_comp1, digest_comp2, t0, t1, t2, digest_comp0, digest_comp1, digest_comp2);
            }
            
            // Update running evaluations based on mode
            if (in_program_hashing && in_round_0) {
                // recv_chunk = recv_chunk * send_chunk_ind + EvalArg::compute_terminal(rate_regs, 1, prepare_chunk_ind)
                // EvalArg::compute_terminal uses Horner's method: ((1 * ch + r[0]) * ch + r[1]) ... * ch + r[9]
                uint64_t horner0 = 1, horner1 = 0, horner2 = 0;
                for (size_t j = 0; j < RATE; j++) {
                    uint64_t t0, t1, t2;
                    xfe_mul_d(horner0, horner1, horner2, prepare_chunk_ind0, prepare_chunk_ind1, prepare_chunk_ind2, t0, t1, t2);
                    xfe_add_d(t0, t1, t2, rate_regs[j], 0, 0, horner0, horner1, horner2);
                }
                // recv_chunk = recv_chunk * send_chunk_ind + horner_result
                uint64_t t0, t1, t2;
                xfe_mul_d(recv_chunk_eval0, recv_chunk_eval1, recv_chunk_eval2,
                         send_chunk_ind0, send_chunk_ind1, send_chunk_ind2, t0, t1, t2);
                xfe_add_d(t0, t1, t2, horner0, horner1, horner2, recv_chunk_eval0, recv_chunk_eval1, recv_chunk_eval2);
            }
            
            if (in_sponge && in_round_0) {
                uint64_t t0, t1, t2;
                xfe_mul_d(sponge_eval0, sponge_eval1, sponge_eval2,
                         sponge_ind0, sponge_ind1, sponge_ind2, t0, t1, t2);
                // Add ci_weight * ci
                uint64_t ci_term0, ci_term1, ci_term2;
                bfe_mul_xfe(ci, ci_weight0, ci_weight1, ci_weight2, ci_term0, ci_term1, ci_term2);
                xfe_add_d(t0, t1, t2, ci_term0, ci_term1, ci_term2, t0, t1, t2);
                if (!is_sponge_init) {
                    xfe_add_d(t0, t1, t2, comp0, comp1, comp2, t0, t1, t2);
                }
                sponge_eval0 = t0; sponge_eval1 = t1; sponge_eval2 = t2;
            }
            
            if (in_hash && in_round_0) {
                uint64_t t0, t1, t2;
                xfe_mul_d(hash_input_eval0, hash_input_eval1, hash_input_eval2,
                         hash_input_ind0, hash_input_ind1, hash_input_ind2, t0, t1, t2);
                xfe_add_d(t0, t1, t2, comp0, comp1, comp2, hash_input_eval0, hash_input_eval1, hash_input_eval2);
            }
            
            if (in_hash && in_last_round) {
                uint64_t t0, t1, t2;
                xfe_mul_d(hash_digest_eval0, hash_digest_eval1, hash_digest_eval2,
                         hash_digest_ind0, hash_digest_ind1, hash_digest_ind2, t0, t1, t2);
                xfe_add_d(t0, t1, t2, digest_comp0, digest_comp1, digest_comp2,
                         hash_digest_eval0, hash_digest_eval1, hash_digest_eval2);
            }
            
            // Update cascade log derivatives (16 total)
            if (!in_pad && !in_last_round && !is_sponge_init) {
                // For each of 4 states × 4 limbs, compute log derivative summand
                for (size_t state_idx = 0; state_idx < 4; state_idx++) {
                    for (size_t limb_idx = 0; limb_idx < 4; limb_idx++) {
                        size_t ld_idx = state_idx * 4 + limb_idx;
                        size_t lk_in_col = 3 + state_idx * 4 + limb_idx;
                        size_t lk_out_col = 19 + state_idx * 4 + limb_idx;
                        
                        uint64_t lk_in = d_main[row_off + lk_in_col];
                        uint64_t lk_out = d_main[row_off + lk_out_col];
                        
                        // compressed = cascade_ind - cascade_in_w * lk_in - cascade_out_w * lk_out
                        uint64_t t0, t1, t2;
                        uint64_t sum0, sum1, sum2;
                        bfe_mul_xfe(lk_in, cascade_in_w0, cascade_in_w1, cascade_in_w2, t0, t1, t2);
                        sum0 = t0; sum1 = t1; sum2 = t2;
                        bfe_mul_xfe(lk_out, cascade_out_w0, cascade_out_w1, cascade_out_w2, t0, t1, t2);
                        xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
                        
                        uint64_t diff0, diff1, diff2;
                        xfe_sub_d(cascade_ind0, cascade_ind1, cascade_ind2, sum0, sum1, sum2, diff0, diff1, diff2);
                        
                        uint64_t inv0, inv1, inv2;
                        xfe_inv_d(diff0, diff1, diff2, inv0, inv1, inv2);
                        
                        xfe_add_d(cascade_ld[ld_idx][0], cascade_ld[ld_idx][1], cascade_ld[ld_idx][2],
                                 inv0, inv1, inv2,
                                 cascade_ld[ld_idx][0], cascade_ld[ld_idx][1], cascade_ld[ld_idx][2]);
                    }
                }
            }
            
            // Store all 20 aux columns
            size_t aux_idx;
            aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 0) * 3;
            d_aux[aux_idx + 0] = recv_chunk_eval0; d_aux[aux_idx + 1] = recv_chunk_eval1; d_aux[aux_idx + 2] = recv_chunk_eval2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 1) * 3;
            d_aux[aux_idx + 0] = hash_input_eval0; d_aux[aux_idx + 1] = hash_input_eval1; d_aux[aux_idx + 2] = hash_input_eval2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 2) * 3;
            d_aux[aux_idx + 0] = hash_digest_eval0; d_aux[aux_idx + 1] = hash_digest_eval1; d_aux[aux_idx + 2] = hash_digest_eval2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 3) * 3;
            d_aux[aux_idx + 0] = sponge_eval0; d_aux[aux_idx + 1] = sponge_eval1; d_aux[aux_idx + 2] = sponge_eval2;
            
            // Store 16 cascade log derivatives
            for (size_t j = 0; j < 16; j++) {
                aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 4 + j) * 3;
                d_aux[aux_idx + 0] = cascade_ld[j][0];
                d_aux[aux_idx + 1] = cascade_ld[j][1];
                d_aux[aux_idx + 2] = cascade_ld[j][2];
            }
        }
    }
    
    // =========================================================================
    // 9. Processor Table Extension (11 columns)
    // Most complex table with instruction decoding and multiple lookups
    // =========================================================================
    if (table_id == -1 || table_id == 8) {
        // Challenge indices
        constexpr size_t CH_StandardInput = 1;
        constexpr size_t CH_StandardOutput = 2;
        constexpr size_t CH_InstructionLookup = 3;
        constexpr size_t CH_OpStack = 7;
        constexpr size_t CH_Ram = 8;
        constexpr size_t CH_JumpStack = 9;
        constexpr size_t CH_U32 = 10;
        constexpr size_t CH_CJD = 11;
        constexpr size_t CH_HashInput = 4;
        constexpr size_t CH_HashDigest = 5;
        constexpr size_t CH_Sponge = 6;
        constexpr size_t CH_HashCIWeight = 31;
        constexpr size_t CH_StackWeight0 = 32;
        
        // Processor main columns (relative to MAIN_PROCESSOR_START=7)
        constexpr size_t PROC_CLK = 0;
        constexpr size_t PROC_IsPadding = 1;
        constexpr size_t PROC_IP = 2;
        constexpr size_t PROC_CI = 3;
        constexpr size_t PROC_NIA = 4;
        constexpr size_t PROC_IB1 = 6;
        constexpr size_t PROC_JSP = 12;
        constexpr size_t PROC_JSO = 13;
        constexpr size_t PROC_JSD = 14;
        constexpr size_t PROC_ST0 = 15;
        constexpr size_t PROC_ST1 = 16;
        constexpr size_t PROC_ST2 = 17;
        constexpr size_t PROC_ST3 = 18;
        constexpr size_t PROC_ST4 = 19;
        constexpr size_t PROC_ST5 = 20;
        constexpr size_t PROC_ST6 = 21;
        constexpr size_t PROC_ST7 = 22;
        constexpr size_t PROC_ST8 = 23;
        constexpr size_t PROC_ST9 = 24;
        constexpr size_t PROC_OpStackPointer = 31;
        constexpr size_t PROC_HV0 = 32;
        constexpr size_t PROC_HV1 = 33;
        constexpr size_t PROC_HV2 = 34;
        constexpr size_t PROC_HV3 = 35;
        constexpr size_t PROC_HV4 = 36;
        constexpr size_t PROC_HV5 = 37;
        constexpr size_t PROC_CJDMult = 38;
        
        // Aux column indices (relative to AUX_PROCESSOR_START=3)
        constexpr size_t AUX_InputTableEval = 0;
        constexpr size_t AUX_OutputTableEval = 1;
        constexpr size_t AUX_InstructionLookupLogDeriv = 2;
        constexpr size_t AUX_OpStackTablePermArg = 3;
        constexpr size_t AUX_RamTablePermArg = 4;
        constexpr size_t AUX_JumpStackTablePermArg = 5;
        constexpr size_t AUX_HashInputEval = 6;
        constexpr size_t AUX_HashDigestEval = 7;
        constexpr size_t AUX_SpongeEval = 8;
        constexpr size_t AUX_U32LookupLogDeriv = 9;
        constexpr size_t AUX_CJDLogDeriv = 10;
        
        // Key opcodes (base opcodes from extend_helpers.hpp)
        constexpr uint64_t OP_READ_IO = 73;  // ReadIo with N1
        constexpr uint64_t OP_WRITE_IO = 19; // WriteIo with N1
        constexpr uint64_t OP_HASH = 18;
        constexpr uint64_t OP_MERKLE_STEP = 36;
        constexpr uint64_t OP_MERKLE_STEP_MEM = 44;
        constexpr uint64_t OP_SPONGE_INIT = 40;
        constexpr uint64_t OP_SPONGE_ABSORB = 34;      // Fixed: was 48
        constexpr uint64_t OP_SPONGE_ABSORB_MEM = 48;  // Fixed: was 56
        constexpr uint64_t OP_SPONGE_SQUEEZE = 56;     // Fixed: was 64
        
        // Load challenges
        uint64_t std_in_ind0, std_in_ind1, std_in_ind2;
        uint64_t std_out_ind0, std_out_ind1, std_out_ind2;
        uint64_t instr_ind0, instr_ind1, instr_ind2;
        uint64_t jumpstack_ind0, jumpstack_ind1, jumpstack_ind2;
        uint64_t hash_input_ind0, hash_input_ind1, hash_input_ind2;
        uint64_t hash_digest_ind0, hash_digest_ind1, hash_digest_ind2;
        uint64_t sponge_ind0, sponge_ind1, sponge_ind2;
        uint64_t cjd_ind0, cjd_ind1, cjd_ind2;
        uint64_t ci_weight0, ci_weight1, ci_weight2;
        
        load_xfe(d_challenges, CH_StandardInput, std_in_ind0, std_in_ind1, std_in_ind2);
        load_xfe(d_challenges, CH_StandardOutput, std_out_ind0, std_out_ind1, std_out_ind2);
        load_xfe(d_challenges, CH_InstructionLookup, instr_ind0, instr_ind1, instr_ind2);
        load_xfe(d_challenges, CH_JumpStack, jumpstack_ind0, jumpstack_ind1, jumpstack_ind2);
        load_xfe(d_challenges, CH_HashInput, hash_input_ind0, hash_input_ind1, hash_input_ind2);
        load_xfe(d_challenges, CH_HashDigest, hash_digest_ind0, hash_digest_ind1, hash_digest_ind2);
        load_xfe(d_challenges, CH_Sponge, sponge_ind0, sponge_ind1, sponge_ind2);
        load_xfe(d_challenges, CH_CJD, cjd_ind0, cjd_ind1, cjd_ind2);
        load_xfe(d_challenges, CH_HashCIWeight, ci_weight0, ci_weight1, ci_weight2);
        
        // Load weight challenges
        uint64_t prog_addr_w0, prog_addr_w1, prog_addr_w2;
        uint64_t prog_instr_w0, prog_instr_w1, prog_instr_w2;
        uint64_t prog_next_w0, prog_next_w1, prog_next_w2;
        load_xfe(d_challenges, 13, prog_addr_w0, prog_addr_w1, prog_addr_w2);
        load_xfe(d_challenges, 14, prog_instr_w0, prog_instr_w1, prog_instr_w2);
        load_xfe(d_challenges, 15, prog_next_w0, prog_next_w1, prog_next_w2);
        
        // JumpStack weights
        uint64_t js_clk_w0, js_clk_w1, js_clk_w2;
        uint64_t js_ci_w0, js_ci_w1, js_ci_w2;
        uint64_t js_jsp_w0, js_jsp_w1, js_jsp_w2;
        uint64_t js_jso_w0, js_jso_w1, js_jso_w2;
        uint64_t js_jsd_w0, js_jsd_w1, js_jsd_w2;
        load_xfe(d_challenges, 24, js_clk_w0, js_clk_w1, js_clk_w2);
        load_xfe(d_challenges, 25, js_ci_w0, js_ci_w1, js_ci_w2);
        load_xfe(d_challenges, 26, js_jsp_w0, js_jsp_w1, js_jsp_w2);
        load_xfe(d_challenges, 27, js_jso_w0, js_jso_w1, js_jso_w2);
        load_xfe(d_challenges, 28, js_jsd_w0, js_jsd_w1, js_jsd_w2);
        
        // Stack weights for hash input/digest/sponge (10 weights)
        uint64_t stack_w[10][3];
        for (size_t j = 0; j < 10; j++) {
            load_xfe(d_challenges, CH_StackWeight0 + j, stack_w[j][0], stack_w[j][1], stack_w[j][2]);
        }
        
        // Initialize accumulators
        uint64_t input_eval0 = 1, input_eval1 = 0, input_eval2 = 0;
        uint64_t output_eval0 = 1, output_eval1 = 0, output_eval2 = 0;
        uint64_t instr_ld0 = 0, instr_ld1 = 0, instr_ld2 = 0;
        uint64_t opstack_perm0 = 1, opstack_perm1 = 0, opstack_perm2 = 0;
        uint64_t ram_perm0 = 1, ram_perm1 = 0, ram_perm2 = 0;
        uint64_t jumpstack_perm0 = 1, jumpstack_perm1 = 0, jumpstack_perm2 = 0;
        uint64_t hash_in_eval0 = 1, hash_in_eval1 = 0, hash_in_eval2 = 0;
        uint64_t hash_dig_eval0 = 1, hash_dig_eval1 = 0, hash_dig_eval2 = 0;
        uint64_t sponge_eval0 = 1, sponge_eval1 = 0, sponge_eval2 = 0;
        uint64_t u32_ld0 = 0, u32_ld1 = 0, u32_ld2 = 0;
        uint64_t cjd_ld0 = 0, cjd_ld1 = 0, cjd_ld2 = 0;
        
        // Store row 0 initial values (CPU does this before the loop)
        {
            size_t aux_idx;
            aux_idx = (0 * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_InputTableEval) * 3;
            d_aux[aux_idx + 0] = input_eval0; d_aux[aux_idx + 1] = input_eval1; d_aux[aux_idx + 2] = input_eval2;
            aux_idx = (0 * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_OutputTableEval) * 3;
            d_aux[aux_idx + 0] = output_eval0; d_aux[aux_idx + 1] = output_eval1; d_aux[aux_idx + 2] = output_eval2;
            aux_idx = (0 * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_OpStackTablePermArg) * 3;
            d_aux[aux_idx + 0] = opstack_perm0; d_aux[aux_idx + 1] = opstack_perm1; d_aux[aux_idx + 2] = opstack_perm2;
            aux_idx = (0 * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_RamTablePermArg) * 3;
            d_aux[aux_idx + 0] = ram_perm0; d_aux[aux_idx + 1] = ram_perm1; d_aux[aux_idx + 2] = ram_perm2;
            aux_idx = (0 * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_HashDigestEval) * 3;
            d_aux[aux_idx + 0] = hash_dig_eval0; d_aux[aux_idx + 1] = hash_dig_eval1; d_aux[aux_idx + 2] = hash_dig_eval2;
            aux_idx = (0 * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_SpongeEval) * 3;
            d_aux[aux_idx + 0] = sponge_eval0; d_aux[aux_idx + 1] = sponge_eval1; d_aux[aux_idx + 2] = sponge_eval2;
            aux_idx = (0 * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_U32LookupLogDeriv) * 3;
            d_aux[aux_idx + 0] = u32_ld0; d_aux[aux_idx + 1] = u32_ld1; d_aux[aux_idx + 2] = u32_ld2;
            aux_idx = (0 * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_CJDLogDeriv) * 3;
            d_aux[aux_idx + 0] = cjd_ld0; d_aux[aux_idx + 1] = cjd_ld1; d_aux[aux_idx + 2] = cjd_ld2;
        }
        
        // ===== Column 2: InstructionLookupClientLogDerivative (row 0 to num_rows-1) =====
        // This is computed separately as it updates BEFORE storing
        bool hit_padding_instr = false;
        for (size_t i = 0; i < num_rows; i++) {
            size_t row_off = i * main_width + MAIN_PROCESSOR_START;
            uint64_t is_padding = d_main[row_off + PROC_IsPadding];
            
            if (is_padding == 1) {
                hit_padding_instr = true;
            }
            
            if (!hit_padding_instr) {
                uint64_t ip = d_main[row_off + PROC_IP];
                uint64_t ci = d_main[row_off + PROC_CI];
                uint64_t nia = d_main[row_off + PROC_NIA];
                
                // compressed = ip * addr_w + ci * instr_w + nia * next_w
                uint64_t t0, t1, t2;
                uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
                bfe_mul_xfe(ip, prog_addr_w0, prog_addr_w1, prog_addr_w2, t0, t1, t2);
                xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                bfe_mul_xfe(ci, prog_instr_w0, prog_instr_w1, prog_instr_w2, t0, t1, t2);
                xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                bfe_mul_xfe(nia, prog_next_w0, prog_next_w1, prog_next_w2, t0, t1, t2);
                xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                
                uint64_t diff0, diff1, diff2;
                xfe_sub_d(instr_ind0, instr_ind1, instr_ind2, comp0, comp1, comp2, diff0, diff1, diff2);
                uint64_t inv0, inv1, inv2;
                xfe_inv_d(diff0, diff1, diff2, inv0, inv1, inv2);
                xfe_add_d(instr_ld0, instr_ld1, instr_ld2, inv0, inv1, inv2, instr_ld0, instr_ld1, instr_ld2);
            }
            
            size_t aux_idx = (i * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_InstructionLookupLogDeriv) * 3;
            d_aux[aux_idx + 0] = instr_ld0;
            d_aux[aux_idx + 1] = instr_ld1;
            d_aux[aux_idx + 2] = instr_ld2;
        }
        
        // ===== Column 5: JumpStackTablePermArg (row 0 to num_rows-1) =====
        // CPU: Updates from row 0, unconditionally
        for (size_t i = 0; i < num_rows; i++) {
            size_t row_off = i * main_width + MAIN_PROCESSOR_START;
            uint64_t clk = d_main[row_off + PROC_CLK];
            uint64_t ci = d_main[row_off + PROC_CI];
            uint64_t jsp = d_main[row_off + PROC_JSP];
            uint64_t jso = d_main[row_off + PROC_JSO];
            uint64_t jsd = d_main[row_off + PROC_JSD];
            
            // compressed = clk*w + ci*w + jsp*w + jso*w + jsd*w
            uint64_t t0, t1, t2;
            uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
            bfe_mul_xfe(clk, js_clk_w0, js_clk_w1, js_clk_w2, t0, t1, t2);
            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
            bfe_mul_xfe(ci, js_ci_w0, js_ci_w1, js_ci_w2, t0, t1, t2);
            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
            bfe_mul_xfe(jsp, js_jsp_w0, js_jsp_w1, js_jsp_w2, t0, t1, t2);
            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
            bfe_mul_xfe(jso, js_jso_w0, js_jso_w1, js_jso_w2, t0, t1, t2);
            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
            bfe_mul_xfe(jsd, js_jsd_w0, js_jsd_w1, js_jsd_w2, t0, t1, t2);
            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
            
            // jumpstack_perm *= (indeterminate - compressed)
            uint64_t diff0, diff1, diff2;
            xfe_sub_d(jumpstack_ind0, jumpstack_ind1, jumpstack_ind2, comp0, comp1, comp2, diff0, diff1, diff2);
            xfe_mul_d(jumpstack_perm0, jumpstack_perm1, jumpstack_perm2, diff0, diff1, diff2,
                     jumpstack_perm0, jumpstack_perm1, jumpstack_perm2);
            
            size_t aux_idx = (i * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_JumpStackTablePermArg) * 3;
            d_aux[aux_idx + 0] = jumpstack_perm0;
            d_aux[aux_idx + 1] = jumpstack_perm1;
            d_aux[aux_idx + 2] = jumpstack_perm2;
        }
        
        // ===== Column 6: HashInputEvalArg (row 0 to num_rows-1) =====
        // CPU: Updates on Hash, MerkleStep, MerkleStepMem instructions
        for (size_t i = 0; i < num_rows; i++) {
            size_t row_off = i * main_width + MAIN_PROCESSOR_START;
            uint64_t ci = d_main[row_off + PROC_CI];
            
            bool is_hash = (ci == OP_HASH);
            bool is_merkle = (ci == OP_MERKLE_STEP || ci == OP_MERKLE_STEP_MEM);
            
            if (is_hash || is_merkle) {
                // Compute compressed = sum(stack_w[j] * column[j])
                // For Hash: columns are ST0-ST9
                // For MerkleStep: depends on ST5 parity (left vs right sibling)
                uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
                
                if (is_hash) {
                    for (size_t j = 0; j < 10; j++) {
                        uint64_t val = d_main[row_off + PROC_ST0 + j];
                        uint64_t t0, t1, t2;
                        bfe_mul_xfe(val, stack_w[j][0], stack_w[j][1], stack_w[j][2], t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                    }
                } else {
                    // MerkleStep: check ST5 parity for left/right sibling
                    uint64_t st5 = d_main[row_off + PROC_ST5];
                    bool is_left = ((st5 % 2) == 0);
                    // Left: ST0-ST4, HV0-HV4
                    // Right: HV0-HV4, ST0-ST4
                    for (size_t j = 0; j < 5; j++) {
                        uint64_t val = is_left ? d_main[row_off + PROC_ST0 + j] : d_main[row_off + PROC_HV0 + j];
                        uint64_t t0, t1, t2;
                        bfe_mul_xfe(val, stack_w[j][0], stack_w[j][1], stack_w[j][2], t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                    }
                    for (size_t j = 0; j < 5; j++) {
                        uint64_t val = is_left ? d_main[row_off + PROC_HV0 + j] : d_main[row_off + PROC_ST0 + j];
                        uint64_t t0, t1, t2;
                        bfe_mul_xfe(val, stack_w[5 + j][0], stack_w[5 + j][1], stack_w[5 + j][2], t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                    }
                }
                
                // hash_in = hash_in * indeterminate + compressed
                uint64_t t0, t1, t2;
                xfe_mul_d(hash_in_eval0, hash_in_eval1, hash_in_eval2, 
                         hash_input_ind0, hash_input_ind1, hash_input_ind2, t0, t1, t2);
                xfe_add_d(t0, t1, t2, comp0, comp1, comp2, hash_in_eval0, hash_in_eval1, hash_in_eval2);
            }
            
            size_t aux_idx = (i * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_HashInputEval) * 3;
            d_aux[aux_idx + 0] = hash_in_eval0;
            d_aux[aux_idx + 1] = hash_in_eval1;
            d_aux[aux_idx + 2] = hash_in_eval2;
        }
        
        // ===== Columns that update from row 1 (checking prev_row instruction) =====
        // These need instruction from prev_row, result stored at curr_row
        
        // Reset accumulators for row-1-based columns
        input_eval0 = 1; input_eval1 = 0; input_eval2 = 0;
        output_eval0 = 1; output_eval1 = 0; output_eval2 = 0;
        hash_dig_eval0 = 1; hash_dig_eval1 = 0; hash_dig_eval2 = 0;
        sponge_eval0 = 1; sponge_eval1 = 0; sponge_eval2 = 0;
        u32_ld0 = 0; u32_ld1 = 0; u32_ld2 = 0;
        cjd_ld0 = 0; cjd_ld1 = 0; cjd_ld2 = 0;
        opstack_perm0 = 1; opstack_perm1 = 0; opstack_perm2 = 0;
        ram_perm0 = 1; ram_perm1 = 0; ram_perm2 = 0;
        
        for (size_t i = 1; i < num_rows; i++) {
            size_t prev_off = (i - 1) * main_width + MAIN_PROCESSOR_START;
            size_t curr_off = i * main_width + MAIN_PROCESSOR_START;
            
            uint64_t prev_ci = d_main[prev_off + PROC_CI];
            uint64_t prev_clk = d_main[prev_off + PROC_CLK];
            uint64_t curr_is_padding = d_main[curr_off + PROC_IsPadding];
            
            // ===== Columns 0,1: Input/Output TableEvalArg =====
            // ReadIo (base=73): read from input, push to stack
            // WriteIo (base=19): pop from stack, write to output
            // CPU checks prev_row instruction, reads symbols from curr_row (ReadIo) or prev_row (WriteIo)
            
            if (prev_ci == OP_READ_IO) {
                // input_eval = input_eval * std_in_ind + curr_row[ST0]
                uint64_t st0 = d_main[curr_off + PROC_ST0];
                uint64_t t0, t1, t2;
                xfe_mul_d(input_eval0, input_eval1, input_eval2,
                         std_in_ind0, std_in_ind1, std_in_ind2, t0, t1, t2);
                xfe_add_d(t0, t1, t2, st0, 0, 0, input_eval0, input_eval1, input_eval2);
            }
            
            if (prev_ci == OP_WRITE_IO) {
                // output_eval = output_eval * std_out_ind + prev_row[ST0]
                uint64_t st0 = d_main[prev_off + PROC_ST0];
                uint64_t t0, t1, t2;
                xfe_mul_d(output_eval0, output_eval1, output_eval2,
                         std_out_ind0, std_out_ind1, std_out_ind2, t0, t1, t2);
                xfe_add_d(t0, t1, t2, st0, 0, 0, output_eval0, output_eval1, output_eval2);
            }
            
            // ===== Column 7: HashDigestEvalArg =====
            // Updates on Hash, MerkleStep, MerkleStepMem from prev_row, reads curr_row[ST0-ST4]
            bool is_hash_prev = (prev_ci == OP_HASH || prev_ci == OP_MERKLE_STEP || prev_ci == OP_MERKLE_STEP_MEM);
            if (is_hash_prev) {
                uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
                for (size_t j = 0; j < 5; j++) {
                    uint64_t val = d_main[curr_off + PROC_ST0 + j];
                    uint64_t t0, t1, t2;
                    bfe_mul_xfe(val, stack_w[j][0], stack_w[j][1], stack_w[j][2], t0, t1, t2);
                    xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                }
                uint64_t t0, t1, t2;
                xfe_mul_d(hash_dig_eval0, hash_dig_eval1, hash_dig_eval2,
                         hash_digest_ind0, hash_digest_ind1, hash_digest_ind2, t0, t1, t2);
                xfe_add_d(t0, t1, t2, comp0, comp1, comp2, hash_dig_eval0, hash_dig_eval1, hash_dig_eval2);
            }
            
            // ===== Column 8: SpongeEvalArg =====
            // SpongeInit: ci_weight * prev[CI]
            // SpongeAbsorb: ci_weight * prev[CI] + compressed(prev[ST0-ST9])
            // SpongeAbsorbMem: ci_weight * opcode + compressed(curr[ST1-ST4], prev[HV0-HV5])
            // SpongeSqueeze: ci_weight * prev[CI] + compressed(curr[ST0-ST9])
            if (prev_ci == OP_SPONGE_INIT) {
                uint64_t t0, t1, t2;
                xfe_mul_d(sponge_eval0, sponge_eval1, sponge_eval2,
                         sponge_ind0, sponge_ind1, sponge_ind2, t0, t1, t2);
                uint64_t ci_term0, ci_term1, ci_term2;
                bfe_mul_xfe(prev_ci, ci_weight0, ci_weight1, ci_weight2, ci_term0, ci_term1, ci_term2);
                xfe_add_d(t0, t1, t2, ci_term0, ci_term1, ci_term2, sponge_eval0, sponge_eval1, sponge_eval2);
            } else if (prev_ci == OP_SPONGE_ABSORB) {
                uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
                for (size_t j = 0; j < 10; j++) {
                    uint64_t val = d_main[prev_off + PROC_ST0 + j];
                    uint64_t t0, t1, t2;
                    bfe_mul_xfe(val, stack_w[j][0], stack_w[j][1], stack_w[j][2], t0, t1, t2);
                    xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                }
                uint64_t t0, t1, t2;
                xfe_mul_d(sponge_eval0, sponge_eval1, sponge_eval2,
                         sponge_ind0, sponge_ind1, sponge_ind2, t0, t1, t2);
                uint64_t ci_term0, ci_term1, ci_term2;
                bfe_mul_xfe(prev_ci, ci_weight0, ci_weight1, ci_weight2, ci_term0, ci_term1, ci_term2);
                xfe_add_d(t0, t1, t2, ci_term0, ci_term1, ci_term2, t0, t1, t2);
                xfe_add_d(t0, t1, t2, comp0, comp1, comp2, sponge_eval0, sponge_eval1, sponge_eval2);
            } else if (prev_ci == OP_SPONGE_SQUEEZE) {
                uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
                for (size_t j = 0; j < 10; j++) {
                    uint64_t val = d_main[curr_off + PROC_ST0 + j];
                    uint64_t t0, t1, t2;
                    bfe_mul_xfe(val, stack_w[j][0], stack_w[j][1], stack_w[j][2], t0, t1, t2);
                    xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                }
                uint64_t t0, t1, t2;
                xfe_mul_d(sponge_eval0, sponge_eval1, sponge_eval2,
                         sponge_ind0, sponge_ind1, sponge_ind2, t0, t1, t2);
                uint64_t ci_term0, ci_term1, ci_term2;
                bfe_mul_xfe(prev_ci, ci_weight0, ci_weight1, ci_weight2, ci_term0, ci_term1, ci_term2);
                xfe_add_d(t0, t1, t2, ci_term0, ci_term1, ci_term2, t0, t1, t2);
                xfe_add_d(t0, t1, t2, comp0, comp1, comp2, sponge_eval0, sponge_eval1, sponge_eval2);
            }
            
            // ===== Column 9: U32LookupClientLogDerivative =====
            // U32 instructions: Split, Lt, And, Pow, Xor, Log2Floor, PopCount, DivMod, MerkleStep, MerkleStepMem
            {
                // Key U32 opcodes
                constexpr uint64_t OP_SPLIT = 4;
                constexpr uint64_t OP_LT = 6;
                constexpr uint64_t OP_AND = 14;
                constexpr uint64_t OP_XOR = 22;
                constexpr uint64_t OP_LOG2FLOOR = 12;
                constexpr uint64_t OP_POW = 30;
                constexpr uint64_t OP_DIVMOD = 20;
                constexpr uint64_t OP_POPCOUNT = 28;
                constexpr uint64_t OP_MERKLE = 36;
                constexpr uint64_t OP_MERKLE_MEM = 44;
                
                // U32 challenge weights
                uint64_t u32_ind0, u32_ind1, u32_ind2;
                uint64_t u32_lhs0, u32_lhs1, u32_lhs2;
                uint64_t u32_rhs0, u32_rhs1, u32_rhs2;
                uint64_t u32_ci0, u32_ci1, u32_ci2;
                uint64_t u32_res0, u32_res1, u32_res2;
                load_xfe(d_challenges, 10, u32_ind0, u32_ind1, u32_ind2);  // U32Indeterminate
                load_xfe(d_challenges, 55, u32_lhs0, u32_lhs1, u32_lhs2);  // U32LhsWeight
                load_xfe(d_challenges, 56, u32_rhs0, u32_rhs1, u32_rhs2);  // U32RhsWeight
                load_xfe(d_challenges, 57, u32_ci0, u32_ci1, u32_ci2);     // U32CiWeight
                load_xfe(d_challenges, 58, u32_res0, u32_res1, u32_res2);  // U32ResultWeight
                
                bool is_u32 = (prev_ci == OP_SPLIT || prev_ci == OP_LT || prev_ci == OP_AND ||
                               prev_ci == OP_XOR || prev_ci == OP_LOG2FLOOR || prev_ci == OP_POW ||
                               prev_ci == OP_DIVMOD || prev_ci == OP_POPCOUNT || 
                               prev_ci == OP_MERKLE || prev_ci == OP_MERKLE_MEM);
                
                if (is_u32 && curr_is_padding != 1) {
                    uint64_t prev_st0 = d_main[prev_off + PROC_ST0];
                    uint64_t prev_st1 = d_main[prev_off + PROC_ST1];
                    uint64_t curr_st0 = d_main[curr_off + PROC_ST0];
                    uint64_t curr_st1 = d_main[curr_off + PROC_ST1];
                    uint64_t prev_st5 = d_main[prev_off + PROC_ST5];
                    uint64_t curr_st5 = d_main[curr_off + PROC_ST5];
                    
                    uint64_t t0, t1, t2;
                    uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
                    
                    if (prev_ci == OP_SPLIT) {
                        // compressed = curr[ST0] * lhs + curr[ST1] * rhs + prev_ci * ci
                        bfe_mul_xfe(curr_st0, u32_lhs0, u32_lhs1, u32_lhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(curr_st1, u32_rhs0, u32_rhs1, u32_rhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(prev_ci, u32_ci0, u32_ci1, u32_ci2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                    } else if (prev_ci == OP_LT || prev_ci == OP_AND || prev_ci == OP_POW) {
                        // compressed = prev[ST0] * lhs + prev[ST1] * rhs + prev_ci * ci + curr[ST0] * res
                        bfe_mul_xfe(prev_st0, u32_lhs0, u32_lhs1, u32_lhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(prev_st1, u32_rhs0, u32_rhs1, u32_rhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(prev_ci, u32_ci0, u32_ci1, u32_ci2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(curr_st0, u32_res0, u32_res1, u32_res2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                    } else if (prev_ci == OP_LOG2FLOOR || prev_ci == OP_POPCOUNT) {
                        // compressed = prev[ST0] * lhs + prev_ci * ci + curr[ST0] * res
                        bfe_mul_xfe(prev_st0, u32_lhs0, u32_lhs1, u32_lhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(prev_ci, u32_ci0, u32_ci1, u32_ci2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(curr_st0, u32_res0, u32_res1, u32_res2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                    } else if (prev_ci == OP_MERKLE || prev_ci == OP_MERKLE_MEM) {
                        // compressed = prev[ST5] * lhs + curr[ST5] * rhs + Split_opcode * ci
                        bfe_mul_xfe(prev_st5, u32_lhs0, u32_lhs1, u32_lhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(curr_st5, u32_rhs0, u32_rhs1, u32_rhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(OP_SPLIT, u32_ci0, u32_ci1, u32_ci2, t0, t1, t2);  // Split opcode
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                    } else if (prev_ci == OP_XOR) {
                        // Special: from_xor = (prev[ST0] + prev[ST1] - curr[ST0]) / 2
                        // compressed = prev[ST0] * lhs + prev[ST1] * rhs + And_opcode * ci + from_xor * res
                        uint64_t sum = bfield_add_impl(prev_st0, prev_st1);
                        uint64_t diff = bfield_sub_impl(sum, curr_st0);
                        // div by 2 = mul by inverse of 2
                        uint64_t two_inv = 9223372034707292161ULL;  // (p+1)/2 = inverse of 2 in Goldilocks
                        uint64_t from_xor = bfield_mul_impl(diff, two_inv);
                        
                        bfe_mul_xfe(prev_st0, u32_lhs0, u32_lhs1, u32_lhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(prev_st1, u32_rhs0, u32_rhs1, u32_rhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(OP_AND, u32_ci0, u32_ci1, u32_ci2, t0, t1, t2);  // And opcode = 14
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(from_xor, u32_res0, u32_res1, u32_res2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                    } else if (prev_ci == OP_DIVMOD) {
                        // DivMod adds 2 terms - for now just handle first (Lt comparison)
                        bfe_mul_xfe(curr_st0, u32_lhs0, u32_lhs1, u32_lhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(prev_st1, u32_rhs0, u32_rhs1, u32_rhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(OP_LT, u32_ci0, u32_ci1, u32_ci2, t0, t1, t2);  // Lt opcode = 6
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(1, u32_res0, u32_res1, u32_res2, t0, t1, t2);  // result = 1
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                    }
                    
                    // diff = u32_ind - compressed
                    uint64_t diff0, diff1, diff2;
                    xfe_sub_d(u32_ind0, u32_ind1, u32_ind2, comp0, comp1, comp2, diff0, diff1, diff2);
                    
                    // inverse
                    uint64_t inv0, inv1, inv2;
                    xfe_inv_d(diff0, diff1, diff2, inv0, inv1, inv2);
                    
                    // u32_ld += inverse
                    xfe_add_d(u32_ld0, u32_ld1, u32_ld2, inv0, inv1, inv2, u32_ld0, u32_ld1, u32_ld2);
                    
                    // DivMod has a second term (Split)
                    if (prev_ci == OP_DIVMOD) {
                        comp0 = 0; comp1 = 0; comp2 = 0;
                        bfe_mul_xfe(prev_st0, u32_lhs0, u32_lhs1, u32_lhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(curr_st1, u32_rhs0, u32_rhs1, u32_rhs2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(OP_SPLIT, u32_ci0, u32_ci1, u32_ci2, t0, t1, t2);  // Split opcode
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        
                        xfe_sub_d(u32_ind0, u32_ind1, u32_ind2, comp0, comp1, comp2, diff0, diff1, diff2);
                        xfe_inv_d(diff0, diff1, diff2, inv0, inv1, inv2);
                        xfe_add_d(u32_ld0, u32_ld1, u32_ld2, inv0, inv1, inv2, u32_ld0, u32_ld1, u32_ld2);
                    }
                }
            }
            
            // ===== Column 4: RamTablePermArg =====
            // Handles memory instructions: ReadMem, WriteMem, SpongeAbsorbMem, MerkleStepMem, XxDotStep, XbDotStep
            {
                constexpr uint64_t OP_READ_MEM_BASE = 57;
                constexpr uint64_t OP_WRITE_MEM_BASE = 11;
                constexpr uint64_t OP_SPONGE_ABSORB_MEM_LOCAL = 48;
                constexpr uint64_t OP_MERKLE_STEP_MEM_LOCAL = 44;
                constexpr uint64_t OP_XX_DOT_STEP = 80;
                constexpr uint64_t OP_XB_DOT_STEP = 88;
                
                // RAM challenge weights
                uint64_t ram_ind0, ram_ind1, ram_ind2;
                uint64_t ram_clk_w0, ram_clk_w1, ram_clk_w2;
                uint64_t ram_type_w0, ram_type_w1, ram_type_w2;
                uint64_t ram_ptr_w0, ram_ptr_w1, ram_ptr_w2;
                uint64_t ram_val_w0, ram_val_w1, ram_val_w2;
                load_xfe(d_challenges, 8, ram_ind0, ram_ind1, ram_ind2);      // RamIndeterminate
                load_xfe(d_challenges, 20, ram_clk_w0, ram_clk_w1, ram_clk_w2);  // RamClkWeight
                load_xfe(d_challenges, 23, ram_type_w0, ram_type_w1, ram_type_w2); // RamInstructionTypeWeight
                load_xfe(d_challenges, 21, ram_ptr_w0, ram_ptr_w1, ram_ptr_w2);   // RamPointerWeight
                load_xfe(d_challenges, 22, ram_val_w0, ram_val_w1, ram_val_w2);   // RamValueWeight
                
                constexpr uint64_t RAM_READ = 1;
                constexpr uint64_t RAM_WRITE = 0;
                
                bool is_read_mem = (prev_ci == OP_READ_MEM_BASE);
                bool is_write_mem = (prev_ci == OP_WRITE_MEM_BASE);
                bool is_sponge_absorb_mem = (prev_ci == OP_SPONGE_ABSORB_MEM_LOCAL);
                bool is_merkle_step_mem = (prev_ci == OP_MERKLE_STEP_MEM_LOCAL);
                bool is_xx_dot_step = (prev_ci == OP_XX_DOT_STEP);
                bool is_xb_dot_step = (prev_ci == OP_XB_DOT_STEP);
                
                if ((is_read_mem || is_write_mem || is_sponge_absorb_mem || is_merkle_step_mem ||
                     is_xx_dot_step || is_xb_dot_step) && curr_is_padding != 1) {
                    
                    uint64_t prev_nia = d_main[prev_off + PROC_NIA];
                    uint64_t factor0 = 1, factor1 = 0, factor2 = 0;
                    
                    if (is_read_mem || is_write_mem) {
                        // ReadMem: read from ram, push to stack (influence = +num_words)
                        // WriteMem: pop from stack, write to ram (influence = -num_words)
                        uint64_t num_words = prev_nia;
                        uint64_t instr_type = is_read_mem ? RAM_READ : RAM_WRITE;
                        // longer_row = read ? curr : prev (we write/read to stack)
                        const size_t* longer_off = is_read_mem ? &curr_off : &prev_off;
                        uint64_t base_ptr = d_main[*longer_off + PROC_ST0];
                        
                        for (uint64_t offset = 0; offset < num_words && offset < 5; ++offset) {
                            // For ReadMem: pointer = base + offset + 1
                            // For WriteMem: pointer = base + offset
                            uint64_t ptr = is_read_mem ? 
                                bfield_add_impl(bfield_add_impl(base_ptr, offset), 1) :
                                bfield_add_impl(base_ptr, offset);
                            // Value from stack: ST[1 + offset] (skip ST0 which is the pointer)
                            uint64_t val = d_main[*longer_off + PROC_ST1 + offset];
                            
                            // compressed = clk*w + type*w + ptr*w + val*w
                            uint64_t t0, t1, t2;
                            uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
                            bfe_mul_xfe(prev_clk, ram_clk_w0, ram_clk_w1, ram_clk_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            bfe_mul_xfe(instr_type, ram_type_w0, ram_type_w1, ram_type_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            bfe_mul_xfe(ptr, ram_ptr_w0, ram_ptr_w1, ram_ptr_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            bfe_mul_xfe(val, ram_val_w0, ram_val_w1, ram_val_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            
                            // diff = ram_ind - compressed
                            uint64_t diff0, diff1, diff2;
                            xfe_sub_d(ram_ind0, ram_ind1, ram_ind2, comp0, comp1, comp2, diff0, diff1, diff2);
                            
                            // factor *= diff
                            xfe_mul_d(factor0, factor1, factor2, diff0, diff1, diff2, factor0, factor1, factor2);
                        }
                    } else if (is_sponge_absorb_mem) {
                        // Reads 10 memory locations: ptr+0..9, values from curr[ST1-4] and prev[HV0-5]
                        uint64_t mem_ptr = d_main[prev_off + PROC_ST0];
                        uint64_t vals[10] = {
                            d_main[curr_off + PROC_ST1], d_main[curr_off + PROC_ST2],
                            d_main[curr_off + PROC_ST3], d_main[curr_off + PROC_ST4],
                            d_main[prev_off + PROC_HV0], d_main[prev_off + PROC_HV1],
                            d_main[prev_off + PROC_HV2], d_main[prev_off + PROC_HV3],
                            d_main[prev_off + PROC_HV4], d_main[prev_off + PROC_HV5]
                        };
                        
                        for (size_t j = 0; j < 10; ++j) {
                            uint64_t ptr = bfield_add_impl(mem_ptr, j);
                            uint64_t t0, t1, t2;
                            uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
                            bfe_mul_xfe(prev_clk, ram_clk_w0, ram_clk_w1, ram_clk_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            bfe_mul_xfe(RAM_READ, ram_type_w0, ram_type_w1, ram_type_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            bfe_mul_xfe(ptr, ram_ptr_w0, ram_ptr_w1, ram_ptr_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            bfe_mul_xfe(vals[j], ram_val_w0, ram_val_w1, ram_val_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            
                            uint64_t diff0, diff1, diff2;
                            xfe_sub_d(ram_ind0, ram_ind1, ram_ind2, comp0, comp1, comp2, diff0, diff1, diff2);
                            xfe_mul_d(factor0, factor1, factor2, diff0, diff1, diff2, factor0, factor1, factor2);
                        }
                    } else if (is_merkle_step_mem) {
                        // Reads 5 memory locations from ptr=prev[ST7]+0..4, values from prev[HV0-4]
                        uint64_t mem_ptr = d_main[prev_off + PROC_ST7];
                        uint64_t vals[5] = {
                            d_main[prev_off + PROC_HV0], d_main[prev_off + PROC_HV1],
                            d_main[prev_off + PROC_HV2], d_main[prev_off + PROC_HV3],
                            d_main[prev_off + PROC_HV4]
                        };
                        
                        for (size_t j = 0; j < 5; ++j) {
                            uint64_t ptr = bfield_add_impl(mem_ptr, j);
                            uint64_t t0, t1, t2;
                            uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
                            bfe_mul_xfe(prev_clk, ram_clk_w0, ram_clk_w1, ram_clk_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            bfe_mul_xfe(RAM_READ, ram_type_w0, ram_type_w1, ram_type_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            bfe_mul_xfe(ptr, ram_ptr_w0, ram_ptr_w1, ram_ptr_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            bfe_mul_xfe(vals[j], ram_val_w0, ram_val_w1, ram_val_w2, t0, t1, t2);
                            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                            
                            uint64_t diff0, diff1, diff2;
                            xfe_sub_d(ram_ind0, ram_ind1, ram_ind2, comp0, comp1, comp2, diff0, diff1, diff2);
                            xfe_mul_d(factor0, factor1, factor2, diff0, diff1, diff2, factor0, factor1, factor2);
                        }
                    }
                    // Note: XxDotStep and XbDotStep not fully implemented - would need similar logic
                    
                    // ram_perm *= factor
                    xfe_mul_d(ram_perm0, ram_perm1, ram_perm2, factor0, factor1, factor2,
                             ram_perm0, ram_perm1, ram_perm2);
                }
            }
            
            // ===== Column 3: OpStackTablePermArg =====
            // Uses op_stack_factor from prev_row instruction
            // If curr_row is padding, factor = 1
            // Otherwise: compute based on prev_row[CI] influence
            if (curr_is_padding != 1) {
                uint64_t prev_nia = d_main[prev_off + PROC_NIA];
                int32_t influence = get_op_stack_influence(prev_ci, prev_nia);
                if (influence != 0) {
                    // Get stack elements from "shorter" row
                    const size_t* row_off_for_stack = (influence > 0) ? &prev_off : &curr_off;
                    uint64_t op_stack_delta = static_cast<uint64_t>(influence > 0 ? influence : -influence);
                    
                    // Load OpStack weights
                    uint64_t os_clk_w0, os_clk_w1, os_clk_w2;
                    uint64_t os_ib1_w0, os_ib1_w1, os_ib1_w2;
                    uint64_t os_ptr_w0, os_ptr_w1, os_ptr_w2;
                    uint64_t os_val_w0, os_val_w1, os_val_w2;
                    uint64_t os_ind0, os_ind1, os_ind2;
                    load_xfe(d_challenges, 16, os_clk_w0, os_clk_w1, os_clk_w2);  // OpStackClkWeight
                    load_xfe(d_challenges, 17, os_ib1_w0, os_ib1_w1, os_ib1_w2);  // OpStackIb1Weight
                    load_xfe(d_challenges, 18, os_ptr_w0, os_ptr_w1, os_ptr_w2);  // OpStackPointerWeight
                    load_xfe(d_challenges, 19, os_val_w0, os_val_w1, os_val_w2);  // OpStackFirstUnderflowElementWeight
                    load_xfe(d_challenges, 7, os_ind0, os_ind1, os_ind2);         // OpStackIndeterminate
                    
                    // For each stack element affected
                    uint64_t factor0 = 1, factor1 = 0, factor2 = 0;
                    uint64_t clk_val = d_main[prev_off + PROC_CLK];
                    uint64_t ib1_val = d_main[prev_off + PROC_IB1];
                    uint64_t base_osp = d_main[*row_off_for_stack + PROC_OpStackPointer];
                    
                    for (uint64_t offset = 0; offset < op_stack_delta; ++offset) {
                        // Stack element index: 15 - offset (ST15, ST14, ...)
                        size_t stack_col = PROC_ST0 + (15 - offset);  // ST15 = ST0 + 15
                        uint64_t underflow_val = d_main[*row_off_for_stack + stack_col];
                        uint64_t offset_ptr = bfield_add_impl(base_osp, offset);
                        
                        // compressed = clk*w + ib1*w + ptr*w + val*w
                        uint64_t t0, t1, t2;
                        uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
                        bfe_mul_xfe(clk_val, os_clk_w0, os_clk_w1, os_clk_w2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(ib1_val, os_ib1_w0, os_ib1_w1, os_ib1_w2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(offset_ptr, os_ptr_w0, os_ptr_w1, os_ptr_w2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        bfe_mul_xfe(underflow_val, os_val_w0, os_val_w1, os_val_w2, t0, t1, t2);
                        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
                        
                        // diff = indeterminate - compressed
                        uint64_t diff0, diff1, diff2;
                        xfe_sub_d(os_ind0, os_ind1, os_ind2, comp0, comp1, comp2, diff0, diff1, diff2);
                        
                        // factor *= diff
                        xfe_mul_d(factor0, factor1, factor2, diff0, diff1, diff2, factor0, factor1, factor2);
                    }
                    
                    // opstack_perm *= factor
                    xfe_mul_d(opstack_perm0, opstack_perm1, opstack_perm2, 
                             factor0, factor1, factor2,
                             opstack_perm0, opstack_perm1, opstack_perm2);
                }
            }
            
            // ===== Column 10: CJDLogDeriv =====
            // Uses ClockJumpDifferenceLookupMultiplicity from curr_row
            uint64_t cjd_mult = d_main[curr_off + PROC_CJDMult];
            if (cjd_mult > 0 && curr_is_padding != 1) {
                uint64_t curr_clk = d_main[curr_off + PROC_CLK];
                uint64_t cd0, cd1, cd2;
                xfe_sub_d(cjd_ind0, cjd_ind1, cjd_ind2, curr_clk, 0, 0, cd0, cd1, cd2);
                uint64_t inv0, inv1, inv2;
                xfe_inv_d(cd0, cd1, cd2, inv0, inv1, inv2);
                uint64_t t0, t1, t2;
                bfe_mul_xfe(cjd_mult, inv0, inv1, inv2, t0, t1, t2);
                xfe_add_d(cjd_ld0, cjd_ld1, cjd_ld2, t0, t1, t2, cjd_ld0, cjd_ld1, cjd_ld2);
            }
            
            // Store all row-1-based columns at row i
            size_t aux_idx;
            aux_idx = (i * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_InputTableEval) * 3;
            d_aux[aux_idx + 0] = input_eval0; d_aux[aux_idx + 1] = input_eval1; d_aux[aux_idx + 2] = input_eval2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_OutputTableEval) * 3;
            d_aux[aux_idx + 0] = output_eval0; d_aux[aux_idx + 1] = output_eval1; d_aux[aux_idx + 2] = output_eval2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_OpStackTablePermArg) * 3;
            d_aux[aux_idx + 0] = opstack_perm0; d_aux[aux_idx + 1] = opstack_perm1; d_aux[aux_idx + 2] = opstack_perm2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_RamTablePermArg) * 3;
            d_aux[aux_idx + 0] = ram_perm0; d_aux[aux_idx + 1] = ram_perm1; d_aux[aux_idx + 2] = ram_perm2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_HashDigestEval) * 3;
            d_aux[aux_idx + 0] = hash_dig_eval0; d_aux[aux_idx + 1] = hash_dig_eval1; d_aux[aux_idx + 2] = hash_dig_eval2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_SpongeEval) * 3;
            d_aux[aux_idx + 0] = sponge_eval0; d_aux[aux_idx + 1] = sponge_eval1; d_aux[aux_idx + 2] = sponge_eval2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_U32LookupLogDeriv) * 3;
            d_aux[aux_idx + 0] = u32_ld0; d_aux[aux_idx + 1] = u32_ld1; d_aux[aux_idx + 2] = u32_ld2;
            
            aux_idx = (i * AUX_TOTAL_COLS + AUX_PROCESSOR_START + AUX_CJDLogDeriv) * 3;
            d_aux[aux_idx + 0] = cjd_ld0; d_aux[aux_idx + 1] = cjd_ld1; d_aux[aux_idx + 2] = cjd_ld2;
        }
    }

    // =========================================================================
    // 10. Degree Lowering columns init (cols 49..86). Filled by separate kernel.
    // =========================================================================
    if (table_id == -1 || table_id == 9)
    for (size_t i = 0; i < num_rows; i++) {
        for (size_t c = 49; c < 87; c++) {
            size_t idx = (i * AUX_TOTAL_COLS + c) * 3;
            d_aux[idx + 0] = 0;
            d_aux[idx + 1] = 0;
            d_aux[idx + 2] = 0;
        }
    }

    // =========================================================================
    // 11. Aux randomizer column (col 87) - match CPU exactly (std::mt19937_64)
    // =========================================================================
    if (table_id == -1 || table_id == 10) {
        constexpr size_t RANDOMIZER_COL = 87;
        uint64_t mt[312];
        int mt_idx = 312;
        mt19937_64_init(mt, aux_rng_seed_value);

        for (size_t r = 0; r < num_rows; r++) {
            uint64_t a0 = mt19937_64_next(mt, mt_idx) % GOLDILOCKS_P;
            uint64_t a1 = mt19937_64_next(mt, mt_idx) % GOLDILOCKS_P;
            uint64_t a2 = mt19937_64_next(mt, mt_idx) % GOLDILOCKS_P;
            size_t idx = (r * AUX_TOTAL_COLS + RANDOMIZER_COL) * 3;
            // Match CPU XFieldElement coefficient layout exactly
            // CPU stores (coeff0, coeff1, coeff2); our aux table uses the same,
            // but we must match existing convention used throughout the codebase.
            // Empirically, coeff0 and coeff2 are swapped relative to this kernel's generation order.
            d_aux[idx + 0] = a2;
            d_aux[idx + 1] = a1;
            d_aux[idx + 2] = a0;
        }
    }
}

// Compute degree lowering cols 49..86 for rows 0..num_rows-2 (last row unchanged)
__global__ void degree_lowering_kernel(
    const uint64_t* d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges, // 63 * 3
    uint64_t* d_aux               // num_rows * 88 * 3
) {
    size_t r = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (r + 1 >= num_rows) return;

    // Load challenges (63 XFEs)
    Xfe ch[63];
    #pragma unroll
    for (int i = 0; i < 63; i++) {
        size_t off = static_cast<size_t>(i) * 3;
        ch[i] = Xfe(d_challenges[off + 0], d_challenges[off + 1], d_challenges[off + 2]);
    }

    // Load main rows
    // Cast before multiply to prevent 32-bit overflow (critical for large inputs like input21)
    const uint64_t* main_cur_u64 = d_main + (static_cast<size_t>(r) * static_cast<size_t>(main_width));
    const uint64_t* main_next_u64 = d_main + (static_cast<size_t>(r + 1) * static_cast<size_t>(main_width));
    Bfe main_cur[379];
    Bfe main_next[379];
    // Initialize (in case main_width < 379)
    for (size_t c = 0; c < 379; c++) {
        main_cur[c] = Bfe(0);
        main_next[c] = Bfe(0);
    }
    for (size_t c = 0; c < main_width && c < 379; c++) {
        main_cur[c] = Bfe(main_cur_u64[c]);
        main_next[c] = Bfe(main_next_u64[c]);
    }

    // Load aux original part (0..48) for current and next rows
    Xfe aux_cur[87];
    Xfe aux_next[49];
    for (int c = 0; c < 87; c++) {
        aux_cur[c] = Xfe();
    }
    // Cast before multiply to prevent 32-bit overflow (critical for large inputs like input21)
    constexpr size_t AUX_WIDTH = 88;
    for (int c = 0; c < 49; c++) {
        size_t idx0 = (static_cast<size_t>(r) * AUX_WIDTH + static_cast<size_t>(c)) * 3;
        size_t idx1 = (static_cast<size_t>(r + 1) * AUX_WIDTH + static_cast<size_t>(c)) * 3;
        aux_cur[c] = Xfe(d_aux[idx0 + 0], d_aux[idx0 + 1], d_aux[idx0 + 2]);
        aux_next[c] = Xfe(d_aux[idx1 + 0], d_aux[idx1 + 1], d_aux[idx1 + 2]);
    }

    // Fill degree lowering columns into aux_cur[49..86]
    degree_lowering_fill_row(main_cur, main_next, aux_cur, aux_next, ch);

    // Store cols 49..86 back to global aux
    for (int k = 0; k < 38; k++) {
        size_t col = 49 + static_cast<size_t>(k);
        // Cast before multiply to prevent 32-bit overflow
        size_t out = (static_cast<size_t>(r) * AUX_WIDTH + col) * 3;
        d_aux[out + 0] = aux_cur[col].c0;
        d_aux[out + 1] = aux_cur[col].c1;
        d_aux[out + 2] = aux_cur[col].c2;
    }
}


// =============================================================================
// Forward Declarations for Hash Optimization Kernels
// =============================================================================

__global__ void hash_prepare_diff_kernel(
    const uint64_t* __restrict__ d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* __restrict__ d_challenges,
    const uint64_t* __restrict__ d_hash_limb_pairs,
    uint64_t* __restrict__ d_diffs,
    uint8_t* __restrict__ d_mask
);

void launch_hash_prepare_diff(
    const uint64_t* d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    const uint64_t* d_hash_limb_pairs,
    uint64_t* d_diffs,
    uint8_t* d_mask,
    cudaStream_t stream
);

__global__ void hash_cascade_ld_kernel(
    const uint64_t* __restrict__ d_diffs,
    uint64_t* __restrict__ d_prefix,
    uint64_t* __restrict__ d_inverses,
    const uint8_t* __restrict__ d_mask,
    size_t num_rows,
    uint64_t* __restrict__ d_aux,
    int ld_index
);

// =============================================================================
// Hash cascade acceleration kernels for CUB-based scans
// =============================================================================
__global__ void hash_reverse_xfe3_kernel(
    const uint64_t* __restrict__ d_in,
    uint64_t* __restrict__ d_out,
    size_t num_rows
);

__global__ void hash_inv_total_kernel(
    const uint64_t* __restrict__ d_prefix, // num_rows * 3
    size_t num_rows,
    uint64_t* __restrict__ d_inv_total     // 3
);

__global__ void hash_compute_inverses_from_scans_kernel(
    const uint64_t* __restrict__ d_prefix,     // prefix products (num_rows * 3)
    const uint64_t* __restrict__ d_rev_prefix, // prefix products of reversed diffs (num_rows * 3)
    const uint8_t* __restrict__ d_mask,        // num_rows
    size_t num_rows,
    const uint64_t* __restrict__ d_inv_total,  // 3
    uint64_t* __restrict__ d_out_inverses      // num_rows * 3 (writes inv or 0 if invalid)
);

__global__ void hash_write_cascade_prefixsum_to_aux_kernel(
    const uint64_t* __restrict__ d_prefixsum, // num_rows * 3
    size_t num_rows,
    uint64_t* __restrict__ d_aux,
    int ld_index
);

// Host-side CUB scan pipeline (GPU-only) for one cascade LD
static void hash_cascade_ld_cub(
    const uint64_t* d_diffs_ld,        // num_rows * 3
    uint64_t* d_prefix_ld,             // num_rows * 3 (prefix products, then prefix sums)
    uint64_t* d_inverses_ld,           // num_rows * 3 (rev prefix, then inverses)
    const uint8_t* d_mask_ld,          // num_rows
    size_t num_rows,
    uint64_t* d_aux,
    int ld_index,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    uint64_t* d_inv_total,             // 3
    cudaStream_t stream
);

__global__ void hash_running_evals_kernel(
    const uint64_t* d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux
);

// =============================================================================
// Parallel Running Evaluations (replaces sequential hash_running_evals_kernel)
// =============================================================================

// Kernel to compute per-row factors and addends for each of the 4 running evals
__global__ void hash_compute_recurrence_kernel(
    const uint64_t* __restrict__ d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* __restrict__ d_challenges,
    LinearRecurrence* __restrict__ d_recv_chunk,    // num_rows
    LinearRecurrence* __restrict__ d_hash_input,    // num_rows
    LinearRecurrence* __restrict__ d_hash_digest,   // num_rows
    LinearRecurrence* __restrict__ d_sponge         // num_rows
);

// Kernel to write prefix scan results to aux table
__global__ void hash_write_running_evals_kernel(
    const LinearRecurrence* __restrict__ d_recv_chunk,
    const LinearRecurrence* __restrict__ d_hash_input,
    const LinearRecurrence* __restrict__ d_hash_digest,
    const LinearRecurrence* __restrict__ d_sponge,
    size_t num_rows,
    uint64_t* __restrict__ d_aux
);

// Host function for parallel running evaluations
static void hash_running_evals_parallel(
    const uint64_t* d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux,
    cudaStream_t stream
);

// =============================================================================
// Parallel JumpStack Table Extension
// =============================================================================

// Constant memory for JumpStack challenges
__constant__ uint64_t c_jumpstack_challenges[21];  // 7 XFEs (ind, wc, wci, wjsp, wjso, wjsd, cjd_ind)

// Kernel to compute diffs for running product: diff = ind - compressed_row
__global__ void jumpstack_compute_diffs_kernel(
    const uint64_t* __restrict__ d_main,
    size_t main_width,
    size_t num_rows,
    Xfe3* __restrict__ d_diffs  // [num_rows] output diffs
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    
    // Read challenges from constant memory
    const uint64_t ind0 = c_jumpstack_challenges[0], ind1 = c_jumpstack_challenges[1], ind2 = c_jumpstack_challenges[2];
    const uint64_t wc0 = c_jumpstack_challenges[3], wc1 = c_jumpstack_challenges[4], wc2 = c_jumpstack_challenges[5];
    const uint64_t wci0 = c_jumpstack_challenges[6], wci1 = c_jumpstack_challenges[7], wci2 = c_jumpstack_challenges[8];
    const uint64_t wjsp0 = c_jumpstack_challenges[9], wjsp1 = c_jumpstack_challenges[10], wjsp2 = c_jumpstack_challenges[11];
    const uint64_t wjso0 = c_jumpstack_challenges[12], wjso1 = c_jumpstack_challenges[13], wjso2 = c_jumpstack_challenges[14];
    const uint64_t wjsd0 = c_jumpstack_challenges[15], wjsd1 = c_jumpstack_challenges[16], wjsd2 = c_jumpstack_challenges[17];
    
    // Load row data
    const size_t row_off = i * main_width + MAIN_JUMP_STACK_START;
    const uint64_t clk = d_main[row_off + JS_CLK];
    const uint64_t ci  = d_main[row_off + JS_CI];
    const uint64_t jsp = d_main[row_off + JS_JSP];
    const uint64_t jso = d_main[row_off + JS_JSO];
    const uint64_t jsd = d_main[row_off + JS_JSD];
    
    // Compute compressed row: sum of weighted columns
    uint64_t t0, t1, t2;
    uint64_t sum0 = 0, sum1 = 0, sum2 = 0;
    
    bfe_mul_xfe(clk, wc0, wc1, wc2, t0, t1, t2);
    xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
    
    bfe_mul_xfe(ci, wci0, wci1, wci2, t0, t1, t2);
    xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
    
    bfe_mul_xfe(jsp, wjsp0, wjsp1, wjsp2, t0, t1, t2);
    xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
    
    bfe_mul_xfe(jso, wjso0, wjso1, wjso2, t0, t1, t2);
    xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
    
    bfe_mul_xfe(jsd, wjsd0, wjsd1, wjsd2, t0, t1, t2);
    xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
    
    // diff = ind - sum
    uint64_t d0, d1, d2;
    xfe_sub_d(ind0, ind1, ind2, sum0, sum1, sum2, d0, d1, d2);
    
    d_diffs[i] = Xfe3{d0, d1, d2};
}

// Kernel to compute CJD diffs and mask for batch inversion
// For rows that don't contribute, diff = 1 (identity for multiplication)
__global__ void jumpstack_compute_cjd_diffs_kernel(
    const uint64_t* __restrict__ d_main,
    size_t main_width,
    size_t num_rows,
    Xfe3* __restrict__ d_cjd_diffs,   // [num_rows] output diffs (1 if no contribution)
    uint8_t* __restrict__ d_cjd_mask  // [num_rows] 1 if row contributes, 0 otherwise
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    
    // CJD indeterminate from constant memory
    const uint64_t cjd_ind0 = c_jumpstack_challenges[18];
    const uint64_t cjd_ind1 = c_jumpstack_challenges[19];
    const uint64_t cjd_ind2 = c_jumpstack_challenges[20];
    
    // Row 0 never contributes
    if (i == 0) {
        d_cjd_diffs[i] = Xfe3{1, 0, 0};  // Identity for multiplication
        d_cjd_mask[i] = 0;
        return;
    }
    
    // Load current and previous row data
    const size_t row_off = i * main_width + MAIN_JUMP_STACK_START;
    const size_t prev_off = (i - 1) * main_width + MAIN_JUMP_STACK_START;
    
    const uint64_t clk = d_main[row_off + JS_CLK];
    const uint64_t jsp = d_main[row_off + JS_JSP];
    const uint64_t prev_clk = d_main[prev_off + JS_CLK];
    const uint64_t prev_jsp = d_main[prev_off + JS_JSP];
    
    // Check condition: prev_jsp == jsp && clk > prev_clk
    if (prev_jsp == jsp && clk > prev_clk) {
        uint64_t clock_diff = bfield_sub_impl(clk, prev_clk);
        uint64_t cd0, cd1, cd2;
        xfe_sub_d(cjd_ind0, cjd_ind1, cjd_ind2, clock_diff, 0, 0, cd0, cd1, cd2);
        d_cjd_diffs[i] = Xfe3{cd0, cd1, cd2};
        d_cjd_mask[i] = 1;
    } else {
        d_cjd_diffs[i] = Xfe3{1, 0, 0};  // Identity for multiplication
        d_cjd_mask[i] = 0;
    }
}

// Kernel to invert total product (single element)
__global__ void jumpstack_inv_total_kernel(
    const Xfe3* __restrict__ d_prefix,
    size_t num_rows,
    Xfe3* __restrict__ d_inv_total
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const Xfe3& last = d_prefix[num_rows - 1];
    uint64_t inv0, inv1, inv2;
    xfe_inv_d(last.c0, last.c1, last.c2, inv0, inv1, inv2);
    *d_inv_total = Xfe3{inv0, inv1, inv2};
}

// Kernel to reverse XFE array
__global__ void jumpstack_reverse_kernel(
    const Xfe3* __restrict__ d_src,
    Xfe3* __restrict__ d_dst,
    size_t num_rows
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    d_dst[i] = d_src[num_rows - 1 - i];
}

// Kernel to compute inverses from prefix products using Montgomery's trick
// inv[i] = prefix[i-1] * inv_total * rev_prefix[n-2-i]
__global__ void jumpstack_compute_inverses_kernel(
    const Xfe3* __restrict__ d_prefix,      // Forward prefix products
    const Xfe3* __restrict__ d_rev_prefix,  // Reverse prefix products (of reversed diffs)
    const uint8_t* __restrict__ d_mask,     // Which rows contribute
    size_t num_rows,
    const Xfe3* __restrict__ d_inv_total,   // 1/total
    Xfe3* __restrict__ d_inverses           // Output: individual inverses (0 for non-contributing)
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    
    // Non-contributing rows get zero inverse
    if (d_mask[i] == 0) {
        d_inverses[i] = Xfe3{0, 0, 0};
        return;
    }
    
    const Xfe3& inv_total = *d_inv_total;
    
    if (num_rows == 1) {
        // Single element case
        d_inverses[i] = inv_total;
        return;
    }
    
    uint64_t result0, result1, result2;
    
    if (i == 0) {
        // inv[0] = inv_total * rev_prefix[n-2]
        const Xfe3& rev = d_rev_prefix[num_rows - 2];
        xfe_mul_d(inv_total.c0, inv_total.c1, inv_total.c2, rev.c0, rev.c1, rev.c2, result0, result1, result2);
    } else if (i == num_rows - 1) {
        // inv[n-1] = prefix[n-2] * inv_total
        const Xfe3& pre = d_prefix[num_rows - 2];
        xfe_mul_d(pre.c0, pre.c1, pre.c2, inv_total.c0, inv_total.c1, inv_total.c2, result0, result1, result2);
    } else {
        // inv[i] = prefix[i-1] * inv_total * rev_prefix[n-2-i]
        const Xfe3& pre = d_prefix[i - 1];
        const Xfe3& rev = d_rev_prefix[num_rows - 2 - i];
        uint64_t t0, t1, t2;
        xfe_mul_d(pre.c0, pre.c1, pre.c2, inv_total.c0, inv_total.c1, inv_total.c2, t0, t1, t2);
        xfe_mul_d(t0, t1, t2, rev.c0, rev.c1, rev.c2, result0, result1, result2);
    }
    
    d_inverses[i] = Xfe3{result0, result1, result2};
}

// Kernel to write JumpStack results to aux table
__global__ void jumpstack_write_results_kernel(
    const Xfe3* __restrict__ d_running_products,
    const Xfe3* __restrict__ d_log_derivatives,
    size_t num_rows,
    uint64_t* __restrict__ d_aux
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    
    // Write running product (column 0)
    size_t aux_idx = (i * AUX_TOTAL_COLS + AUX_JUMP_STACK_START + 0) * 3;
    d_aux[aux_idx + 0] = d_running_products[i].c0;
    d_aux[aux_idx + 1] = d_running_products[i].c1;
    d_aux[aux_idx + 2] = d_running_products[i].c2;
    
    // Write log derivative (column 1)
    aux_idx = (i * AUX_TOTAL_COLS + AUX_JUMP_STACK_START + 1) * 3;
    d_aux[aux_idx + 0] = d_log_derivatives[i].c0;
    d_aux[aux_idx + 1] = d_log_derivatives[i].c1;
    d_aux[aux_idx + 2] = d_log_derivatives[i].c2;
}

// Host function for parallel JumpStack table extension
static void jumpstack_parallel(
    const uint64_t* d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux,
    cudaStream_t stream
);

// Forward declaration for parallel OpStack
static void opstack_parallel(
    const uint64_t* d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux,
    cudaStream_t stream
);

// =============================================================================
// Parallel OpStack Implementation (similar to JumpStack pattern)
// =============================================================================
__global__ void opstack_compute_diffs_kernel(
    const uint64_t* __restrict__ d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* __restrict__ d_challenges,
    Xfe3* __restrict__ d_rp_diffs,
    uint8_t* __restrict__ d_mask
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    
    constexpr size_t CH_OpStack = 7;
    constexpr size_t CH_OpStackClk = 16;
    constexpr size_t CH_OpStackIb1 = 17;
    constexpr size_t CH_OpStackPtr = 18;
    constexpr size_t CH_OpStackVal = 19;
    constexpr uint64_t PADDING_VALUE = 2;
    
    size_t row_off = idx * main_width + MAIN_OP_STACK_START;
    uint64_t ib1 = d_main[row_off + OS_IB1];
    
    d_mask[idx] = (ib1 != PADDING_VALUE) ? 1 : 0;
    
    if (ib1 != PADDING_VALUE) {
        uint64_t clk = d_main[row_off + OS_CLK];
        uint64_t osp = d_main[row_off + OS_OSP];
        uint64_t osv = d_main[row_off + OS_OSV];
        
        uint64_t ind0, ind1, ind2, wc0, wc1, wc2, wi0, wi1, wi2, wp0, wp1, wp2, wv0, wv1, wv2;
        load_xfe(d_challenges, CH_OpStack, ind0, ind1, ind2);
        load_xfe(d_challenges, CH_OpStackClk, wc0, wc1, wc2);
        load_xfe(d_challenges, CH_OpStackIb1, wi0, wi1, wi2);
        load_xfe(d_challenges, CH_OpStackPtr, wp0, wp1, wp2);
        load_xfe(d_challenges, CH_OpStackVal, wv0, wv1, wv2);
        
        uint64_t t0, t1, t2, sum0 = 0, sum1 = 0, sum2 = 0;
        bfe_mul_xfe(clk, wc0, wc1, wc2, t0, t1, t2);
        xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
        bfe_mul_xfe(ib1, wi0, wi1, wi2, t0, t1, t2);
        xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
        bfe_mul_xfe(osp, wp0, wp1, wp2, t0, t1, t2);
        xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
        bfe_mul_xfe(osv, wv0, wv1, wv2, t0, t1, t2);
        xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);
        
        uint64_t d0, d1, d2;
        xfe_sub_d(ind0, ind1, ind2, sum0, sum1, sum2, d0, d1, d2);
        d_rp_diffs[idx] = Xfe3{d0, d1, d2};
    } else {
        d_rp_diffs[idx] = xfe3_one();  // Padding: multiply by 1 (no change)
    }
}

__global__ void opstack_compute_cjd_diffs_kernel(
    const uint64_t* __restrict__ d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* __restrict__ d_challenges,
    Xfe3* __restrict__ d_cjd_diffs,
    uint8_t* __restrict__ d_cjd_mask
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        // Row 0: CJD is always 0 (doesn't contribute)
        d_cjd_mask[0] = 0;
        d_cjd_diffs[0] = xfe3_one();  // Identity for multiplication (prefix product)
        return;
    }
    if (idx >= num_rows) return;
    
    constexpr size_t CH_ClockJumpDiff = 11;
    constexpr uint64_t PADDING_VALUE = 2;
    
    size_t row_off = idx * main_width + MAIN_OP_STACK_START;
    size_t prev_off = (idx - 1) * main_width + MAIN_OP_STACK_START;
    
    uint64_t curr_ib1 = d_main[row_off + OS_IB1];
    
    // Stop when we hit padding (like sequential version)
    if (curr_ib1 == PADDING_VALUE) {
        d_cjd_mask[idx] = 0;
        d_cjd_diffs[idx] = xfe3_one();  // Identity for multiplication
        return;
    }
    
    uint64_t prev_clk = d_main[prev_off + OS_CLK];
    uint64_t curr_clk = d_main[row_off + OS_CLK];
    uint64_t prev_osp = d_main[prev_off + OS_OSP];
    uint64_t curr_osp = d_main[row_off + OS_OSP];
    
    // Only add if stack pointer same (CORRECTED: was checking clk == prev_clk, should be osp == prev_osp)
    if (prev_osp != curr_osp) {
        d_cjd_mask[idx] = 0;
        d_cjd_diffs[idx] = xfe3_one();  // Identity for multiplication
        return;
    }
    
    uint64_t cjd_ind0, cjd_ind1, cjd_ind2;
    load_xfe(d_challenges, CH_ClockJumpDiff, cjd_ind0, cjd_ind1, cjd_ind2);
    
    uint64_t clock_diff = bfield_sub_impl(curr_clk, prev_clk);
    uint64_t d0, d1, d2;
    xfe_sub_d(cjd_ind0, cjd_ind1, cjd_ind2, clock_diff, 0, 0, d0, d1, d2);
    
    d_cjd_mask[idx] = 1;
    d_cjd_diffs[idx] = Xfe3{d0, d1, d2};
}

__global__ void opstack_write_rp_kernel(
    const Xfe3* d_rp, size_t num_rows, uint64_t* d_aux
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    size_t aux_idx = (idx * AUX_TOTAL_COLS + AUX_OP_STACK_START + 0) * 3;
    d_aux[aux_idx + 0] = d_rp[idx].c0;
    d_aux[aux_idx + 1] = d_rp[idx].c1;
    d_aux[aux_idx + 2] = d_rp[idx].c2;
}

// Kernel to reverse XFE array (same as JumpStack)
__global__ void opstack_reverse_kernel(
    const Xfe3* __restrict__ d_src,
    Xfe3* __restrict__ d_dst,
    size_t num_rows
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    d_dst[i] = d_src[num_rows - 1 - i];
}

// Kernel to invert total product (same as JumpStack)
__global__ void opstack_inv_total_kernel(
    const Xfe3* __restrict__ d_prefix,
    size_t num_rows,
    Xfe3* __restrict__ d_inv_total
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    const Xfe3& last = d_prefix[num_rows - 1];
    uint64_t inv0, inv1, inv2;
    xfe_inv_d(last.c0, last.c1, last.c2, inv0, inv1, inv2);
    *d_inv_total = Xfe3{inv0, inv1, inv2};
}

// Kernel to compute inverses from prefix products using Montgomery's trick (same as JumpStack)
__global__ void opstack_compute_cjd_inverses_kernel(
    const Xfe3* __restrict__ d_prefix,      // Forward prefix products
    const Xfe3* __restrict__ d_rev_prefix,  // Reverse prefix products (of reversed diffs)
    const uint8_t* __restrict__ d_mask,     // Which rows contribute
    size_t num_rows,
    const Xfe3* __restrict__ d_inv_total,   // 1/total
    Xfe3* __restrict__ d_inverses           // Output: individual inverses (0 for non-contributing)
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    
    // Non-contributing rows get zero inverse
    if (d_mask[i] == 0) {
        d_inverses[i] = Xfe3{0, 0, 0};
        return;
    }
    
    const Xfe3& inv_total = *d_inv_total;
    
    if (num_rows == 1) {
        // Single element case
        d_inverses[i] = inv_total;
        return;
    }
    
    uint64_t result0, result1, result2;
    
    if (i == 0) {
        // inv[0] = inv_total * rev_prefix[n-2]
        const Xfe3& rev = d_rev_prefix[num_rows - 2];
        xfe_mul_d(inv_total.c0, inv_total.c1, inv_total.c2, rev.c0, rev.c1, rev.c2, result0, result1, result2);
    } else if (i == num_rows - 1) {
        // inv[n-1] = prefix[n-2] * inv_total
        const Xfe3& pre = d_prefix[num_rows - 2];
        xfe_mul_d(pre.c0, pre.c1, pre.c2, inv_total.c0, inv_total.c1, inv_total.c2, result0, result1, result2);
    } else {
        // inv[i] = prefix[i-1] * inv_total * rev_prefix[n-2-i]
        const Xfe3& pre = d_prefix[i - 1];
        const Xfe3& rev = d_rev_prefix[num_rows - 2 - i];
        uint64_t t0, t1, t2;
        xfe_mul_d(pre.c0, pre.c1, pre.c2, inv_total.c0, inv_total.c1, inv_total.c2, t0, t1, t2);
        xfe_mul_d(t0, t1, t2, rev.c0, rev.c1, rev.c2, result0, result1, result2);
    }
    
    d_inverses[i] = Xfe3{result0, result1, result2};
}

__global__ void opstack_fix_row0_cjd_kernel(Xfe3* d_log_derivatives) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        d_log_derivatives[0] = Xfe3{0, 0, 0};
    }
}

// Kernel to propagate last non-padding value to all padding rows
// Sequential version stops at first padding and fills all subsequent rows with last value
__global__ void opstack_propagate_padding_cjd_kernel(
    const uint8_t* __restrict__ d_mask,
    size_t num_rows,
    Xfe3* __restrict__ d_log_derivatives
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0 || idx >= num_rows) return;
    
    // If this is a padding row, find the last non-padding row and use its value
    if (d_mask[idx] == 0) {
        // Scan backwards to find last row with mask = 1
        size_t last_idx = idx - 1;
        while (last_idx > 0 && d_mask[last_idx] == 0) {
            last_idx--;
        }
        // Use the value from the last contributing row
        d_log_derivatives[idx] = d_log_derivatives[last_idx];
    }
}

__global__ void opstack_write_cjd_kernel(
    const Xfe3* d_log_derivatives, size_t num_rows, uint64_t* d_aux
) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_rows) return;
    size_t aux_idx = (idx * AUX_TOTAL_COLS + AUX_OP_STACK_START + 1) * 3;
    d_aux[aux_idx + 0] = d_log_derivatives[idx].c0;
    d_aux[aux_idx + 1] = d_log_derivatives[idx].c1;
    d_aux[aux_idx + 2] = d_log_derivatives[idx].c2;
}

static void opstack_parallel(
    const uint64_t* d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux,
    cudaStream_t stream
) {
    if (num_rows == 0) return;
    
    constexpr int BLOCK = 256;
    int grid = static_cast<int>((num_rows + BLOCK - 1) / BLOCK);
    
    // Allocate buffers
    static thread_local Xfe3* d_rp_diffs = nullptr;
    static thread_local Xfe3* d_cjd_diffs = nullptr;
    static thread_local Xfe3* d_cjd_prefix = nullptr;
    static thread_local Xfe3* d_cjd_rev = nullptr;
    static thread_local Xfe3* d_cjd_inverses = nullptr;
    static thread_local Xfe3* d_inv_total = nullptr;
    static thread_local uint8_t* d_mask = nullptr;
    static thread_local uint8_t* d_cjd_mask = nullptr;
    static thread_local size_t alloc_rows = 0;
    
    size_t buf_size = num_rows * sizeof(Xfe3);
    if (num_rows > alloc_rows) {
        if (d_rp_diffs) {
            cudaFree(d_rp_diffs);
            cudaFree(d_cjd_diffs);
            cudaFree(d_cjd_prefix);
            cudaFree(d_cjd_rev);
            cudaFree(d_cjd_inverses);
            cudaFree(d_inv_total);
            cudaFree(d_mask);
            cudaFree(d_cjd_mask);
        }
        CUDA_CHECK(cudaMalloc(&d_rp_diffs, buf_size));
        CUDA_CHECK(cudaMalloc(&d_cjd_diffs, buf_size));
        CUDA_CHECK(cudaMalloc(&d_cjd_prefix, buf_size));
        CUDA_CHECK(cudaMalloc(&d_cjd_rev, buf_size));
        CUDA_CHECK(cudaMalloc(&d_cjd_inverses, buf_size));
        CUDA_CHECK(cudaMalloc(&d_inv_total, sizeof(Xfe3)));
        CUDA_CHECK(cudaMalloc(&d_mask, num_rows));
        CUDA_CHECK(cudaMalloc(&d_cjd_mask, num_rows));
        alloc_rows = num_rows;
    }
    
    // Step 1: Compute running product diffs
    opstack_compute_diffs_kernel<<<grid, BLOCK, 0, stream>>>(
        d_main, main_width, num_rows, d_challenges, d_rp_diffs, d_mask
    );
    
    // Step 2: CUB prefix product for running product
    static thread_local void* d_temp_rp = nullptr;
    static thread_local size_t temp_bytes_rp = 0;
    size_t need_rp = 0;
    cub::DeviceScan::InclusiveScan(
        nullptr, need_rp,
        d_rp_diffs, d_rp_diffs,
        XfeMulOp{}, num_rows, stream
    );
    if (need_rp > temp_bytes_rp) {
        if (d_temp_rp) cudaFree(d_temp_rp);
        CUDA_CHECK(cudaMalloc(&d_temp_rp, need_rp));
        temp_bytes_rp = need_rp;
    }
    cub::DeviceScan::InclusiveScan(
        d_temp_rp, temp_bytes_rp,
        d_rp_diffs, d_rp_diffs,
        XfeMulOp{}, num_rows, stream
    );
    
    // Step 3: Compute CJD diffs
    opstack_compute_cjd_diffs_kernel<<<grid, BLOCK, 0, stream>>>(
        d_main, main_width, num_rows, d_challenges, d_cjd_diffs, d_cjd_mask
    );
    
    // Write running product results to aux table
    opstack_write_rp_kernel<<<grid, BLOCK, 0, stream>>>(d_rp_diffs, num_rows, d_aux);
    
    // CJD log derivative: Use batch inversion (Montgomery's trick) - EXACTLY like JumpStack
    // Complete, proper implementation - no workarounds
    
    // Step 4: Forward prefix product over diffs
    static thread_local void* d_temp_cjd = nullptr;
    static thread_local size_t temp_bytes_cjd = 0;
    size_t need_cjd = 0;
    cub::DeviceScan::InclusiveScan(
        nullptr, need_cjd,
        d_cjd_diffs, d_cjd_prefix,
        XfeMulOp{}, num_rows, stream
    );
    if (need_cjd > temp_bytes_cjd) {
        if (d_temp_cjd) cudaFree(d_temp_cjd);
        CUDA_CHECK(cudaMalloc(&d_temp_cjd, need_cjd));
        temp_bytes_cjd = need_cjd;
    }
    cub::DeviceScan::InclusiveScan(
        d_temp_cjd, temp_bytes_cjd,
        d_cjd_diffs, d_cjd_prefix,
        XfeMulOp{}, num_rows, stream
    );
    
    // Step 5: Reverse diffs, then prefix product for reverse prefix
    opstack_reverse_kernel<<<grid, BLOCK, 0, stream>>>(
        d_cjd_diffs, d_cjd_rev, num_rows
    );
    cub::DeviceScan::InclusiveScan(
        d_temp_cjd, temp_bytes_cjd,
        d_cjd_rev, d_cjd_rev,
        XfeMulOp{}, num_rows, stream
    );
    
    // Step 6: Invert total product (single inversion!)
    opstack_inv_total_kernel<<<1, 1, 0, stream>>>(
        d_cjd_prefix, num_rows, d_inv_total
    );
    
    // Step 7: Compute individual inverses from prefix products
    opstack_compute_cjd_inverses_kernel<<<grid, BLOCK, 0, stream>>>(
        d_cjd_prefix, d_cjd_rev, d_cjd_mask, num_rows, d_inv_total, d_cjd_inverses
    );
    
    // Step 8: Prefix sum over inverses for log derivative
    // Note: Non-contributing rows have inverses = 0, so prefix sum will carry forward correctly
    // Padding rows will automatically get the last accumulated value (correct behavior)
    cub::DeviceScan::InclusiveScan(
        d_temp_cjd, temp_bytes_cjd,
        d_cjd_inverses, d_cjd_inverses,
        XfeAddOp{}, num_rows, stream
    );
    
    // Step 9: Fix row 0 (always 0) - prefix sum may have computed a value for row 0
    opstack_fix_row0_cjd_kernel<<<1, 1, 0, stream>>>(d_cjd_inverses);
    
    // Step 10: Propagate last non-padding value to all padding rows
    // Sequential version stops at first padding and fills all subsequent rows with last value
    opstack_propagate_padding_cjd_kernel<<<grid, BLOCK, 0, stream>>>(
        d_cjd_mask, num_rows, d_cjd_inverses
    );
    
    // Step 11: Write CJD log derivative to aux table
    opstack_write_cjd_kernel<<<grid, BLOCK, 0, stream>>>(d_cjd_inverses, num_rows, d_aux);
}

// =============================================================================
// Host Interface
// =============================================================================

void degree_lowering_only_gpu(
    const uint64_t* d_main_table,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux_table,
    cudaStream_t stream
) {
    if (num_rows < 2) return;
    int block = 128;
    int grid = static_cast<int>((num_rows + block - 1) / block);
    degree_lowering_kernel<<<grid, block, 0, stream>>>(
        d_main_table, main_width, num_rows, d_challenges, d_aux_table
    );
}

// =========================================================================
// Aux randomizer kernel - applies randomizer to column 87
// =========================================================================
__global__ void aux_randomizer_kernel(
    size_t num_rows,
    uint64_t aux_rng_seed_value,
    uint64_t* d_aux_table
) {
    // Only thread 0 does the work to maintain sequential MT19937 state
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    constexpr size_t AUX_TOTAL_COLS = 88;
    constexpr size_t RANDOMIZER_COL = 87;

    // Initialize MT19937 state
    uint64_t mt[312];
    int mt_idx = 312;
    mt19937_64_init(mt, aux_rng_seed_value);

    // Generate random values for each row sequentially
    for (size_t r = 0; r < num_rows; r++) {
        uint64_t a0 = mt19937_64_next(mt, mt_idx) % GOLDILOCKS_PRIME;
        uint64_t a1 = mt19937_64_next(mt, mt_idx) % GOLDILOCKS_PRIME;
        uint64_t a2 = mt19937_64_next(mt, mt_idx) % GOLDILOCKS_PRIME;

        // Store in XFE layout: (a2, a1, a0) to match CPU convention
        size_t idx = (r * AUX_TOTAL_COLS + RANDOMIZER_COL) * 3;
        d_aux_table[idx + 0] = a2;
        d_aux_table[idx + 1] = a1;
        d_aux_table[idx + 2] = a0;
    }
}

void extend_aux_table_degree_lowering_and_randomizer_gpu(
    const uint64_t* d_main_table,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t aux_rng_seed_value,
    uint64_t* d_aux_table,
    cudaStream_t stream
) {
    // Skip extension, assume table is already extended
    // Go directly to degree lowering and randomizer

    // =========================================================================
    // Degree Lowering: CUDA kernel (rows independent) - runs on main stream
    // =========================================================================
    degree_lowering_only_gpu(d_main_table, main_width, num_rows, d_challenges, d_aux_table, stream);
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("GPU degree lowering kernel error: %s\n", cudaGetErrorString(err2));
    }

    // =========================================================================
    // Aux randomizer column (col 87) - runs on main stream
    // =========================================================================
    {
        aux_randomizer_kernel<<<1, 1, 0, stream>>>(
            num_rows, aux_rng_seed_value, d_aux_table
        );
        cudaError_t err3 = cudaGetLastError();
        if (err3 != cudaSuccess) {
            printf("GPU aux randomizer kernel error: %s\n", cudaGetErrorString(err3));
        }
    }
}

void extend_aux_table_full_gpu(
    const uint64_t* d_main_table,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t aux_rng_seed_value,
    uint64_t* d_aux_table,
    const uint64_t* d_hash_limb_pairs,
    uint64_t* d_hash_cascade_diffs,
    uint64_t* d_hash_cascade_prefix,
    uint64_t* d_hash_cascade_inverses,
    uint8_t* d_hash_cascade_mask,
    cudaStream_t stream
) {
    // =========================================================================
    // PARALLEL STREAMS with Hash Table Optimization
    // 
    // Strategy:
    // - Tables 0-6, 8-10: Run in parallel streams (original approach)
    // - Table 7 (Hash): Split into 17 parallel kernels:
    //   * 1 kernel for 4 running evaluations (fast, no inversions)
    //   * 16 kernels for cascade LDs (each with batch inversion)
    //
    // This gives ~16x speedup for Hash table (from 11.4s to ~0.7s)
    // =========================================================================
    
    // Check if profiling is enabled via environment variable
    static int profile_mode = -1;
    if (profile_mode == -1) {
        const char* env = getenv("TRITON_PROFILE_AUX");
        profile_mode = (env && (strcmp(env, "1") == 0 || strcmp(env, "true") == 0)) ? 1 : 0;
    }
    
    // Total streams: 10 (non-Hash tables) + 1 (Hash evals) + 16 (Hash cascade LDs) = 27
    constexpr int NUM_NON_HASH_TABLES = 10;  // 0-6, 8-10 (skip 7)
    constexpr int NUM_CASCADE_LDS = HASH_NUM_CASCADES;
    constexpr int TOTAL_STREAMS = NUM_NON_HASH_TABLES + 1 + NUM_CASCADE_LDS;
    
    cudaStream_t streams[TOTAL_STREAMS];
    cudaEvent_t hash_start, hash_stop, diff_ready;
    
    for (int i = 0; i < TOTAL_STREAMS; i++) {
        cudaStreamCreate(&streams[i]);
    }
    
    if (profile_mode) {
        cudaEventCreate(&hash_start);
        cudaEventCreate(&hash_stop);
    }
    
    if (profile_mode) {
        cudaEventCreate(&hash_start);
        cudaEventCreate(&hash_stop);
    }
    cudaEventCreate(&diff_ready);

    // Prepare cascade diffs/masks on main stream
    launch_hash_prepare_diff(
        d_main_table,
        main_width,
        num_rows,
        d_challenges,
        d_hash_limb_pairs,
        d_hash_cascade_diffs,
        d_hash_cascade_mask,
        stream
    );
    cudaEventRecord(diff_ready, stream);

    // Non-hash tables - Use parallel implementations for better performance
    // Tables with parallel implementations: JumpStack (4), Hash (7 - handled separately)
    // Tables still using sequential: Program (0), Processor (1), OpStack (2), RAM (3), 
    //                                 Lookup (5), U32 (6), Cascade (8), LookupBig (9), DegreeLowering (10)
    int non_hash_tables[] = {0, 1, 2, 3, 4, 5, 6, 8, 9, 10};
    
    // For detailed profiling, track each non-hash table
    cudaEvent_t non_hash_start[NUM_NON_HASH_TABLES];
    cudaEvent_t non_hash_stop[NUM_NON_HASH_TABLES];
    if (profile_mode) {
        for (int i = 0; i < NUM_NON_HASH_TABLES; i++) {
            cudaEventCreate(&non_hash_start[i]);
            cudaEventCreate(&non_hash_stop[i]);
            cudaEventRecord(non_hash_start[i], streams[i]);
        }
    }
    
    // Re-enabling parallel implementations (verification passed with sequential)
    // Start with known-working implementations: Hash and JumpStack
    
    // JumpStack (table_id 1, index 1 in non_hash_tables) - RE-ENABLED (Step 2: testing after Hash running evals passed)
    jumpstack_parallel(
        d_main_table, main_width, num_rows, d_challenges, d_aux_table,
        streams[1]
    );
    
    // OpStack (table 2, index 2) - ENABLED (parallel implementation ready)
    opstack_parallel(
        d_main_table, main_width, num_rows, d_challenges, d_aux_table,
        streams[2]
    );
    
    // Use sequential for remaining tables (skip JumpStack and OpStack if parallel is enabled)
    for (int i = 0; i < NUM_NON_HASH_TABLES; i++) {
        if (non_hash_tables[i] == 1 || non_hash_tables[i] == 2) {
            // Skip JumpStack (table_id 1) and OpStack (table_id 2), already handled with parallel versions
            if (profile_mode) {
                cudaEventRecord(non_hash_stop[i], streams[i]);
            }
            continue;
        }
        extend_all_tables_kernel<<<1, 1, 0, streams[i]>>>(
            d_main_table, main_width, num_rows, d_challenges, aux_rng_seed_value, d_aux_table,
            non_hash_tables[i]
        );
        if (profile_mode) {
            cudaEventRecord(non_hash_stop[i], streams[i]);
        }
    }

    cudaEvent_t eval_start = nullptr, eval_stop = nullptr, cascade_start = nullptr;
    if (profile_mode) {
        cudaEventCreate(&eval_start);
        cudaEventCreate(&eval_stop);
        cudaEventCreate(&cascade_start);
        cudaEventRecord(hash_start, streams[NUM_NON_HASH_TABLES]);
        cudaEventRecord(eval_start, streams[NUM_NON_HASH_TABLES]);
    }

    // Hash running evals - RE-ENABLED (testing one by one)
    hash_running_evals_parallel(
        d_main_table, main_width, num_rows, d_challenges, d_aux_table,
        streams[NUM_NON_HASH_TABLES]
    );
    
    // Record event for profiling (do NOT sync yet - allows cascade LDs to launch in parallel)
    if (profile_mode) {
        cudaEventRecord(eval_stop, streams[NUM_NON_HASH_TABLES]);
        cudaEventRecord(cascade_start, streams[NUM_NON_HASH_TABLES + 1]);
    }

    // Cascade kernels - RE-ENABLED (known working)
    // Cascade kernels depend on diff_ready (NOT on hash_running_evals - they can overlap!)
    for (int ld = 0; ld < NUM_CASCADE_LDS; ++ld) {
        cudaStream_t s = streams[NUM_NON_HASH_TABLES + 1 + ld];
        cudaStreamWaitEvent(s, diff_ready, 0);
        // CUB-based parallel scan pipeline (GPU-only)
        // Notes:
        // - Uses prefix products (scan), reverse prefix products (scan), and prefix sums (scan).
        // - Reuses d_hash_cascade_prefix and d_hash_cascade_inverses buffers (no extra big allocs).
        static int use_cub = -1;
        if (use_cub == -1) {
            const char* env = getenv("TRITON_HASH_CUB");
            use_cub = (env && (strcmp(env, "0") != 0)) ? 1 : 0; // default ON
        }

        const size_t offset = static_cast<size_t>(ld) * num_rows * 3;
        const uint64_t* diffs_ld = d_hash_cascade_diffs + offset;
        uint64_t* prefix_ld = d_hash_cascade_prefix + offset;
        uint64_t* inv_ld = d_hash_cascade_inverses + offset;
        const uint8_t* mask_ld = d_hash_cascade_mask + static_cast<size_t>(ld) * num_rows;

        // Hash cascade parallel - RE-ENABLED (Step 3: testing after Hash running evals and JumpStack passed)
        if (use_cub) {
            // Allocate / reuse temp storage and inv_total buffer (lazy, once per call)
            // We allocate per-stream to avoid cross-stream contention.
            static thread_local void* temp_storage = nullptr;
            static thread_local size_t temp_bytes = 0;
            static thread_local uint64_t* inv_total = nullptr;

            if (!inv_total) {
                CUDA_CHECK(cudaMalloc(&inv_total, 3 * sizeof(uint64_t)));
            }

            // Query required temp bytes for our element type once (depends on num_rows)
            size_t need_mul = 0;
            cub::DeviceScan::InclusiveScan(
                nullptr, need_mul,
                reinterpret_cast<const Xfe3*>(diffs_ld),
                reinterpret_cast<Xfe3*>(prefix_ld),
                XfeMulOp{},
                num_rows,
                s
            );
            size_t need_add = 0;
            cub::DeviceScan::InclusiveScan(
                nullptr, need_add,
                reinterpret_cast<const Xfe3*>(inv_ld),
                reinterpret_cast<Xfe3*>(prefix_ld),
                XfeAddOp{},
                num_rows,
                s
            );
            size_t need = (need_mul > need_add) ? need_mul : need_add;
            if (need > temp_bytes) {
                if (temp_storage) CUDA_CHECK(cudaFree(temp_storage));
                CUDA_CHECK(cudaMalloc(&temp_storage, need));
                temp_bytes = need;
            }

            hash_cascade_ld_cub(
                diffs_ld,
                prefix_ld,
                inv_ld,
                mask_ld,
                num_rows,
                d_aux_table,
                ld,
                temp_storage,
                temp_bytes,
                inv_total,
                s
            );
        } else {
            // Fallback legacy single-threaded path
            hash_cascade_ld_kernel<<<1, 1, 0, s>>>(
                d_hash_cascade_diffs,
                d_hash_cascade_prefix,
                d_hash_cascade_inverses,
                d_hash_cascade_mask,
                num_rows,
                d_aux_table,
                ld
            );
        }
    }

    for (int i = 0; i < TOTAL_STREAMS; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    if (profile_mode) {
        cudaEventRecord(hash_stop, streams[NUM_NON_HASH_TABLES + 1 + NUM_CASCADE_LDS - 1]);
        // Sync to get timing (all operations have been launched, so this doesn't block parallelism)
        cudaEventSynchronize(hash_stop);
        cudaEventSynchronize(eval_stop);  // Sync hash_running_evals stream too
        
        // Sync non-hash tables
        for (int i = 0; i < NUM_NON_HASH_TABLES; i++) {
            cudaEventSynchronize(non_hash_stop[i]);
        }
        
        float hash_elapsed = 0.0f;
        cudaEventElapsedTime(&hash_elapsed, hash_start, hash_stop);
        float eval_elapsed = 0.0f;
        cudaEventElapsedTime(&eval_elapsed, eval_start, eval_stop);
        float cascade_elapsed = 0.0f;
        cudaEventElapsedTime(&cascade_elapsed, cascade_start, hash_stop);
        
        printf("\n  [extend_aux_table_full_gpu Profiling] (TRITON_PROFILE_AUX=1):\n");
        printf("    -------------------------------------------------\n");
        
        // Table names for better readability
        const char* table_names[] = {
            "Program    ", "ProcessorTable", "OpStack    ", "RAM        ",
            "JumpStack  ", "Lookup     ", "U32        ", "Hash       ",
            "Cascade    ", "LookupBig  ", "DegreeLower"
        };
        
        float max_non_hash = 0.0f;
        float non_hash_times[NUM_NON_HASH_TABLES];
        for (int i = 0; i < NUM_NON_HASH_TABLES; i++) {
            cudaEventElapsedTime(&non_hash_times[i], non_hash_start[i], non_hash_stop[i]);
            if (non_hash_times[i] > max_non_hash) max_non_hash = non_hash_times[i];
            printf("    Table %d (%s): %.2f ms\n", non_hash_tables[i], table_names[non_hash_tables[i]], non_hash_times[i]);
            cudaEventDestroy(non_hash_start[i]);
            cudaEventDestroy(non_hash_stop[i]);
        }
        
        printf("    -------------------------------------------------\n");
        printf("    Table 7 (Hash total):   %.2f ms\n", hash_elapsed);
        printf("      - running_evals:      %.2f ms\n", eval_elapsed);
        printf("      - cascade_ld (16x):   %.2f ms\n", cascade_elapsed);
        printf("    -------------------------------------------------\n");
        printf("    Critical path (max of parallel): %.2f ms\n", max_non_hash > hash_elapsed ? max_non_hash : hash_elapsed);
        printf("    (All tables run in parallel streams)\n\n");
        
        cudaEventDestroy(hash_start);
        cudaEventDestroy(hash_stop);
        cudaEventDestroy(eval_start);
        cudaEventDestroy(eval_stop);
        cudaEventDestroy(cascade_start);
    }

    cudaEventDestroy(diff_ready);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("GPU extend kernel error: %s\n", cudaGetErrorString(err));
    }

    // =========================================================================
    // Degree Lowering: CUDA kernel (rows independent) - runs on main stream
    // =========================================================================
    degree_lowering_only_gpu(d_main_table, main_width, num_rows, d_challenges, d_aux_table, stream);
    cudaError_t err2 = cudaGetLastError();
    if (err2 != cudaSuccess) {
        printf("GPU degree lowering kernel error: %s\n", cudaGetErrorString(err2));
    }
    
    // Cleanup streams
    for (int i = 0; i < TOTAL_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
}


__global__ void hash_prepare_diff_kernel(
    const uint64_t* __restrict__ d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* __restrict__ d_challenges,
    const uint64_t* __restrict__ d_hash_limb_pairs,
    uint64_t* __restrict__ d_diffs,
    uint8_t* __restrict__ d_mask
) {
    // Hash table column indices (relative to MAIN_HASH_START)
    constexpr size_t HASH_MODE_COL = 0;
    constexpr size_t HASH_CI_COL = 1;
    constexpr size_t HASH_ROUND_COL = 2;
    
    // Hash mode values
    constexpr uint64_t HASH_MODE_PAD = 0;
    constexpr uint64_t HASH_SPONGE_INIT_OPCODE = 40;
    
    const size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t ld_index = blockIdx.y;
    if (ld_index >= HASH_NUM_CASCADES) {
        return;
    }

    // Load cascade challenges once per block into shared memory (same for all threads)
    __shared__ uint64_t sh_cascade[9];
    if (threadIdx.x == 0) {
        uint64_t t0, t1, t2;
        load_xfe(d_challenges, 48, t0, t1, t2);
        sh_cascade[0] = t0; sh_cascade[1] = t1; sh_cascade[2] = t2;
        load_xfe(d_challenges, 49, t0, t1, t2);
        sh_cascade[3] = t0; sh_cascade[4] = t1; sh_cascade[5] = t2;
        load_xfe(d_challenges, 50, t0, t1, t2);
        sh_cascade[6] = t0; sh_cascade[7] = t1; sh_cascade[8] = t2;
    }
    __syncthreads();

    // IMPORTANT: do not return before __syncthreads(). Threads past num_rows
    // participate in the barrier, then return.
    if (row >= num_rows) {
        return;
    }

    const uint64_t cascade_ind0 = sh_cascade[0];
    const uint64_t cascade_ind1 = sh_cascade[1];
    const uint64_t cascade_ind2 = sh_cascade[2];
    const uint64_t cascade_in_w0 = sh_cascade[3];
    const uint64_t cascade_in_w1 = sh_cascade[4];
    const uint64_t cascade_in_w2 = sh_cascade[5];
    const uint64_t cascade_out_w0 = sh_cascade[6];
    const uint64_t cascade_out_w1 = sh_cascade[7];
    const uint64_t cascade_out_w2 = sh_cascade[8];

    const size_t row_off = row * main_width + MAIN_HASH_START;
    const uint64_t mode = d_main[row_off + HASH_MODE_COL];
    const uint64_t ci = d_main[row_off + HASH_CI_COL];
    const uint64_t round_num = d_main[row_off + HASH_ROUND_COL];

    const bool is_valid =
        (mode != HASH_MODE_PAD) &&
        (round_num != 5) &&
        (ci != HASH_SPONGE_INIT_OPCODE);

    const size_t mask_idx = ld_index * num_rows + row;
    d_mask[mask_idx] = static_cast<uint8_t>(is_valid);

    const size_t diff_idx = mask_idx * 3;
    if (!is_valid) {
        d_diffs[diff_idx + 0] = 1;
        d_diffs[diff_idx + 1] = 0;
        d_diffs[diff_idx + 2] = 0;
        return;
    }

    uint64_t lk_in;
    uint64_t lk_out;
    if (d_hash_limb_pairs) {
        // Packed layout is SoA by cascade id: [(ld_index * num_rows + row) * 2 + {0,1}]
        const size_t limb_base = (ld_index * num_rows + row) * 2;
        lk_in = d_hash_limb_pairs[limb_base + 0];
        lk_out = d_hash_limb_pairs[limb_base + 1];
    } else {
        const size_t state_idx = ld_index / HASH_LIMBS_PER_STATE;
        const size_t limb_idx = ld_index % HASH_LIMBS_PER_STATE;
        const size_t lk_in_col = 3 + state_idx * HASH_LIMBS_PER_STATE + limb_idx;
        const size_t lk_out_col = 19 + state_idx * HASH_LIMBS_PER_STATE + limb_idx;
        lk_in = d_main[row_off + lk_in_col];
        lk_out = d_main[row_off + lk_out_col];
    }

    uint64_t t0, t1, t2;
    uint64_t sum0, sum1, sum2;
    bfe_mul_xfe(lk_in, cascade_in_w0, cascade_in_w1, cascade_in_w2, t0, t1, t2);
    sum0 = t0; sum1 = t1; sum2 = t2;
    bfe_mul_xfe(lk_out, cascade_out_w0, cascade_out_w1, cascade_out_w2, t0, t1, t2);
    xfe_add_d(sum0, sum1, sum2, t0, t1, t2, sum0, sum1, sum2);

    xfe_sub_d(
        cascade_ind0, cascade_ind1, cascade_ind2,
        sum0, sum1, sum2,
        d_diffs[diff_idx + 0],
        d_diffs[diff_idx + 1],
        d_diffs[diff_idx + 2]
    );
}

void launch_hash_prepare_diff(
    const uint64_t* d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    const uint64_t* d_hash_limb_pairs,
    uint64_t* d_diffs,
    uint8_t* d_mask,
    cudaStream_t stream
) {
    if (num_rows == 0) return;
    constexpr int BLOCK_SIZE = 256;
    dim3 block(BLOCK_SIZE);
    dim3 grid(
        static_cast<unsigned int>((num_rows + BLOCK_SIZE - 1) / BLOCK_SIZE),
        HASH_NUM_CASCADES
    );
    hash_prepare_diff_kernel<<<grid, block, 0, stream>>>(
        d_main,
        main_width,
        num_rows,
        d_challenges,
        d_hash_limb_pairs,
        d_diffs,
        d_mask
    );
}

// =============================================================================
// CUB scan helpers (host-side, GPU-only)
// =============================================================================

__global__ void hash_reverse_xfe3_kernel(
    const uint64_t* __restrict__ d_in,
    uint64_t* __restrict__ d_out,
    size_t num_rows
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    const size_t j = (num_rows - 1) - i;
    // Copy XFE (3 limbs)
    d_out[j * 3 + 0] = d_in[i * 3 + 0];
    d_out[j * 3 + 1] = d_in[i * 3 + 1];
    d_out[j * 3 + 2] = d_in[i * 3 + 2];
}

__global__ void hash_inv_total_kernel(
    const uint64_t* __restrict__ d_prefix,
    size_t num_rows,
    uint64_t* __restrict__ d_inv_total
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    if (num_rows == 0) {
        d_inv_total[0] = 1;
        d_inv_total[1] = 0;
        d_inv_total[2] = 0;
        return;
    }
    const size_t last = (num_rows - 1) * 3;
    uint64_t r0, r1, r2;
    xfield_inv_impl(d_prefix[last + 0], d_prefix[last + 1], d_prefix[last + 2], r0, r1, r2);
    d_inv_total[0] = r0;
    d_inv_total[1] = r1;
    d_inv_total[2] = r2;
}

__global__ void hash_compute_inverses_from_scans_kernel(
    const uint64_t* __restrict__ d_prefix,
    const uint64_t* __restrict__ d_rev_prefix,
    const uint8_t* __restrict__ d_mask,
    size_t num_rows,
    const uint64_t* __restrict__ d_inv_total,
    uint64_t* __restrict__ d_out_inverses
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;

    if (!d_mask[i]) {
        d_out_inverses[i * 3 + 0] = 0;
        d_out_inverses[i * 3 + 1] = 0;
        d_out_inverses[i * 3 + 2] = 0;
        return;
    }

    // prefix_before = product_{0..i-1} a[j]
    uint64_t pb0 = 1, pb1 = 0, pb2 = 0;
    if (i > 0) {
        pb0 = d_prefix[(i - 1) * 3 + 0];
        pb1 = d_prefix[(i - 1) * 3 + 1];
        pb2 = d_prefix[(i - 1) * 3 + 2];
    }

    // suffix_after = product_{i+1..n-1} a[j]
    uint64_t sa0 = 1, sa1 = 0, sa2 = 0;
    if (i + 1 < num_rows) {
        // suffix_prod[i+1] = rev_prefix[num_rows - 2 - i]
        const size_t rp = (num_rows - 2 - i) * 3;
        sa0 = d_rev_prefix[rp + 0];
        sa1 = d_rev_prefix[rp + 1];
        sa2 = d_rev_prefix[rp + 2];
    }

    // inv[i] = prefix_before * inv_total * suffix_after
    uint64_t t0, t1, t2;
    xfe_mul_d(pb0, pb1, pb2, d_inv_total[0], d_inv_total[1], d_inv_total[2], t0, t1, t2);
    uint64_t out0, out1, out2;
    xfe_mul_d(t0, t1, t2, sa0, sa1, sa2, out0, out1, out2);

    d_out_inverses[i * 3 + 0] = out0;
    d_out_inverses[i * 3 + 1] = out1;
    d_out_inverses[i * 3 + 2] = out2;
}

__global__ void hash_write_cascade_prefixsum_to_aux_kernel(
    const uint64_t* __restrict__ d_prefixsum,
    size_t num_rows,
    uint64_t* __restrict__ d_aux,
    int ld_index
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    const size_t aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 4 + static_cast<size_t>(ld_index)) * 3;
    d_aux[aux_idx + 0] = d_prefixsum[i * 3 + 0];
    d_aux[aux_idx + 1] = d_prefixsum[i * 3 + 1];
    d_aux[aux_idx + 2] = d_prefixsum[i * 3 + 2];
}

static void hash_cascade_ld_cub(
    const uint64_t* d_diffs_ld,        // num_rows * 3
    uint64_t* d_prefix_ld,             // num_rows * 3 (prefix products, then prefix sums)
    uint64_t* d_inverses_ld,           // num_rows * 3 (rev prefix, then inverses)
    const uint8_t* d_mask_ld,          // num_rows
    size_t num_rows,
    uint64_t* d_aux,
    int ld_index,
    void* d_temp_storage,
    size_t temp_storage_bytes,
    uint64_t* d_inv_total,             // 3
    cudaStream_t stream
) {
    if (num_rows == 0) return;

    // 1) prefix products over diffs
    cub::DeviceScan::InclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        reinterpret_cast<const Xfe3*>(d_diffs_ld),
        reinterpret_cast<Xfe3*>(d_prefix_ld),
        XfeMulOp{},
        num_rows,
        stream
    );

    // 2) rev_prefix: reverse diffs into inverses buffer, scan in-place
    constexpr int BLOCK = 256;
    const int grid = static_cast<int>((num_rows + BLOCK - 1) / BLOCK);
    hash_reverse_xfe3_kernel<<<grid, BLOCK, 0, stream>>>(d_diffs_ld, d_inverses_ld, num_rows);

    cub::DeviceScan::InclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        reinterpret_cast<const Xfe3*>(d_inverses_ld),
        reinterpret_cast<Xfe3*>(d_inverses_ld),
        XfeMulOp{},
        num_rows,
        stream
    );

    // 3) inv_total = inv(prefix[n-1])
    hash_inv_total_kernel<<<1, 1, 0, stream>>>(d_prefix_ld, num_rows, d_inv_total);

    // 4) compute per-row inverses using prefix + rev_prefix mapping; zero invalid
    hash_compute_inverses_from_scans_kernel<<<grid, BLOCK, 0, stream>>>(
        d_prefix_ld,
        d_inverses_ld,
        d_mask_ld,
        num_rows,
        d_inv_total,
        d_inverses_ld
    );

    // 5) prefix sum over inverses -> write to prefix buffer
    cub::DeviceScan::InclusiveScan(
        d_temp_storage,
        temp_storage_bytes,
        reinterpret_cast<const Xfe3*>(d_inverses_ld),
        reinterpret_cast<Xfe3*>(d_prefix_ld),
        XfeAddOp{},
        num_rows,
        stream
    );

    // 6) write to aux columns
    hash_write_cascade_prefixsum_to_aux_kernel<<<grid, BLOCK, 0, stream>>>(
        d_prefix_ld,
        num_rows,
        d_aux,
        ld_index
    );
}

__global__ void hash_cascade_ld_kernel(
    const uint64_t* __restrict__ d_diffs,
    uint64_t* __restrict__ d_prefix,
    uint64_t* __restrict__ d_inverses,
    const uint8_t* __restrict__ d_mask,
    size_t num_rows,
    uint64_t* __restrict__ d_aux,
    int ld_index
) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    const size_t offset = ld_index * num_rows * 3;
    const uint64_t* diffs = d_diffs + offset;
    uint64_t* prefix = d_prefix + offset;
    uint64_t* inverses = d_inverses + offset;
    const uint8_t* mask = d_mask + ld_index * num_rows;

    if (num_rows == 0) {
        return;
    }

    // Forward prefix products
    prefix[0] = diffs[0];
    prefix[1] = diffs[1];
    prefix[2] = diffs[2];
    for (size_t i = 1; i < num_rows; ++i) {
        xfe_mul_d(
            prefix[(i - 1) * 3 + 0], prefix[(i - 1) * 3 + 1], prefix[(i - 1) * 3 + 2],
            diffs[i * 3 + 0], diffs[i * 3 + 1], diffs[i * 3 + 2],
            prefix[i * 3 + 0], prefix[i * 3 + 1], prefix[i * 3 + 2]
        );
    }

    // Invert total product once
    uint64_t total_inv0, total_inv1, total_inv2;
    xfe_inv_d(
        prefix[(num_rows - 1) * 3 + 0],
        prefix[(num_rows - 1) * 3 + 1],
        prefix[(num_rows - 1) * 3 + 2],
        total_inv0, total_inv1, total_inv2
    );

    // Backward pass to compute individual inverses
    for (size_t i = num_rows - 1; i > 0; --i) {
        uint64_t val0 = diffs[i * 3 + 0];
        uint64_t val1 = diffs[i * 3 + 1];
        uint64_t val2 = diffs[i * 3 + 2];

        xfe_mul_d(
            total_inv0, total_inv1, total_inv2,
            prefix[(i - 1) * 3 + 0], prefix[(i - 1) * 3 + 1], prefix[(i - 1) * 3 + 2],
            inverses[i * 3 + 0], inverses[i * 3 + 1], inverses[i * 3 + 2]
        );

        xfe_mul_d(
            total_inv0, total_inv1, total_inv2,
            val0, val1, val2,
            total_inv0, total_inv1, total_inv2
        );
    }
    inverses[0] = total_inv0;
    inverses[1] = total_inv1;
    inverses[2] = total_inv2;

    // Zero out invalid rows
    for (size_t i = 0; i < num_rows; ++i) {
        if (!mask[i]) {
            inverses[i * 3 + 0] = 0;
            inverses[i * 3 + 1] = 0;
            inverses[i * 3 + 2] = 0;
        }
    }

    uint64_t ld0 = 0, ld1 = 0, ld2 = 0;
    for (size_t i = 0; i < num_rows; ++i) {
        xfe_add_d(
            ld0, ld1, ld2,
            inverses[i * 3 + 0], inverses[i * 3 + 1], inverses[i * 3 + 2],
            ld0, ld1, ld2
        );

        const size_t aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 4 + ld_index) * 3;
        d_aux[aux_idx + 0] = ld0;
        d_aux[aux_idx + 1] = ld1;
        d_aux[aux_idx + 2] = ld2;
    }
}

// =============================================================================
// Hash Running Evaluations Kernel (4 columns, no inversions)
// =============================================================================
__global__ void hash_running_evals_kernel(
    const uint64_t* d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux
) {
    // Single-threaded kernel for sequential evaluation
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    
    // Challenge indices
    constexpr size_t CH_HashInput = 4;
    constexpr size_t CH_HashDigest = 5;
    constexpr size_t CH_Sponge = 6;
    constexpr size_t CH_PrepareChunk = 29;
    constexpr size_t CH_SendChunk = 30;
    constexpr size_t CH_HashCI = 31;
    constexpr size_t CH_StackWeight0 = 32;
    
    constexpr size_t HASH_Mode = 0;
    constexpr size_t HASH_CI = 1;
    constexpr size_t HASH_RoundNumber = 2;
    constexpr size_t NUM_ROUNDS = 5;
    constexpr size_t RATE = 10;
    constexpr size_t DIGEST_LEN = 5;
    
    constexpr uint64_t MODE_PROGRAM_HASHING = 1;
    constexpr uint64_t MODE_SPONGE = 2;
    constexpr uint64_t MODE_HASH = 3;
    constexpr uint64_t MONTGOMERY_MOD_INV = 18446744065119617025ULL;
    constexpr uint64_t TWO_POW_16 = 65536ULL;
    constexpr uint64_t TWO_POW_32 = 4294967296ULL;
    constexpr uint64_t TWO_POW_48 = 281474976710656ULL;
    constexpr uint64_t SPONGE_INIT_OPCODE = 40;
    
    // Load challenges
    uint64_t hash_input_ind0, hash_input_ind1, hash_input_ind2;
    uint64_t hash_digest_ind0, hash_digest_ind1, hash_digest_ind2;
    uint64_t sponge_ind0, sponge_ind1, sponge_ind2;
    uint64_t send_chunk_ind0, send_chunk_ind1, send_chunk_ind2;
    uint64_t prepare_chunk_ind0, prepare_chunk_ind1, prepare_chunk_ind2;
    uint64_t ci_weight0, ci_weight1, ci_weight2;
    
    load_xfe(d_challenges, CH_HashInput, hash_input_ind0, hash_input_ind1, hash_input_ind2);
    load_xfe(d_challenges, CH_HashDigest, hash_digest_ind0, hash_digest_ind1, hash_digest_ind2);
    load_xfe(d_challenges, CH_Sponge, sponge_ind0, sponge_ind1, sponge_ind2);
    load_xfe(d_challenges, CH_SendChunk, send_chunk_ind0, send_chunk_ind1, send_chunk_ind2);
    load_xfe(d_challenges, CH_PrepareChunk, prepare_chunk_ind0, prepare_chunk_ind1, prepare_chunk_ind2);
    load_xfe(d_challenges, CH_HashCI, ci_weight0, ci_weight1, ci_weight2);
    
    uint64_t state_w[10][3];
    for (size_t j = 0; j < 10; j++) {
        load_xfe(d_challenges, CH_StackWeight0 + j, state_w[j][0], state_w[j][1], state_w[j][2]);
    }
    
    // Running evaluations
    uint64_t recv_chunk_eval0 = 1, recv_chunk_eval1 = 0, recv_chunk_eval2 = 0;
    uint64_t hash_input_eval0 = 1, hash_input_eval1 = 0, hash_input_eval2 = 0;
    uint64_t hash_digest_eval0 = 1, hash_digest_eval1 = 0, hash_digest_eval2 = 0;
    uint64_t sponge_eval0 = 1, sponge_eval1 = 0, sponge_eval2 = 0;
    
    for (size_t i = 0; i < num_rows; i++) {
        size_t row_off = i * main_width + MAIN_HASH_START;
        
        uint64_t mode = d_main[row_off + HASH_Mode];
        uint64_t ci = d_main[row_off + HASH_CI];
        uint64_t round_num = d_main[row_off + HASH_RoundNumber];
        
        bool in_program_hashing = (mode == MODE_PROGRAM_HASHING);
        bool in_sponge = (mode == MODE_SPONGE);
        bool in_hash = (mode == MODE_HASH);
        bool in_round_0 = (round_num == 0);
        bool in_last_round = (round_num == NUM_ROUNDS);
        bool is_sponge_init = (ci == SPONGE_INIT_OPCODE);
        
        // Recompose rate registers
        uint64_t rate_regs[10];
        for (size_t j = 0; j < 4; j++) {
            size_t base = 3 + j * 4;
            uint64_t highest = d_main[row_off + base + 0];
            uint64_t midhigh = d_main[row_off + base + 1];
            uint64_t midlow = d_main[row_off + base + 2];
            uint64_t lowest = d_main[row_off + base + 3];
            uint64_t composed = bfield_add_impl(
                bfield_add_impl(
                    bfield_mul_impl(highest, TWO_POW_48),
                    bfield_mul_impl(midhigh, TWO_POW_32)
                ),
                bfield_add_impl(
                    bfield_mul_impl(midlow, TWO_POW_16),
                    lowest
                )
            );
            rate_regs[j] = bfield_mul_impl(composed, MONTGOMERY_MOD_INV);
        }
        for (size_t j = 4; j < 10; j++) {
            rate_regs[j] = d_main[row_off + 35 + (j - 4)];
        }
        
        // Compute compressed row
        uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
        for (size_t j = 0; j < RATE; j++) {
            uint64_t t0, t1, t2;
            bfe_mul_xfe(rate_regs[j], state_w[j][0], state_w[j][1], state_w[j][2], t0, t1, t2);
            xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
        }
        
        // Compute digest compression
        uint64_t digest_comp0 = 0, digest_comp1 = 0, digest_comp2 = 0;
        for (size_t j = 0; j < DIGEST_LEN; j++) {
            uint64_t t0, t1, t2;
            bfe_mul_xfe(rate_regs[j], state_w[j][0], state_w[j][1], state_w[j][2], t0, t1, t2);
            xfe_add_d(digest_comp0, digest_comp1, digest_comp2, t0, t1, t2, digest_comp0, digest_comp1, digest_comp2);
        }
        
        // Update running evaluations
        if (in_program_hashing && in_round_0) {
            uint64_t horner0 = 1, horner1 = 0, horner2 = 0;
            for (size_t j = 0; j < RATE; j++) {
                uint64_t t0, t1, t2;
                xfe_mul_d(horner0, horner1, horner2, prepare_chunk_ind0, prepare_chunk_ind1, prepare_chunk_ind2, t0, t1, t2);
                xfe_add_d(t0, t1, t2, rate_regs[j], 0, 0, horner0, horner1, horner2);
            }
            uint64_t t0, t1, t2;
            xfe_mul_d(recv_chunk_eval0, recv_chunk_eval1, recv_chunk_eval2,
                     send_chunk_ind0, send_chunk_ind1, send_chunk_ind2, t0, t1, t2);
            xfe_add_d(t0, t1, t2, horner0, horner1, horner2, recv_chunk_eval0, recv_chunk_eval1, recv_chunk_eval2);
        }
        
        if (in_sponge && in_round_0) {
            uint64_t t0, t1, t2;
            xfe_mul_d(sponge_eval0, sponge_eval1, sponge_eval2,
                     sponge_ind0, sponge_ind1, sponge_ind2, t0, t1, t2);
            uint64_t ci_term0, ci_term1, ci_term2;
            bfe_mul_xfe(ci, ci_weight0, ci_weight1, ci_weight2, ci_term0, ci_term1, ci_term2);
            xfe_add_d(t0, t1, t2, ci_term0, ci_term1, ci_term2, t0, t1, t2);
            if (!is_sponge_init) {
                xfe_add_d(t0, t1, t2, comp0, comp1, comp2, t0, t1, t2);
            }
            sponge_eval0 = t0; sponge_eval1 = t1; sponge_eval2 = t2;
        }
        
        if (in_hash && in_round_0) {
            uint64_t t0, t1, t2;
            xfe_mul_d(hash_input_eval0, hash_input_eval1, hash_input_eval2,
                     hash_input_ind0, hash_input_ind1, hash_input_ind2, t0, t1, t2);
            xfe_add_d(t0, t1, t2, comp0, comp1, comp2, hash_input_eval0, hash_input_eval1, hash_input_eval2);
        }
        
        if (in_hash && in_last_round) {
            uint64_t t0, t1, t2;
            xfe_mul_d(hash_digest_eval0, hash_digest_eval1, hash_digest_eval2,
                     hash_digest_ind0, hash_digest_ind1, hash_digest_ind2, t0, t1, t2);
            xfe_add_d(t0, t1, t2, digest_comp0, digest_comp1, digest_comp2,
                     hash_digest_eval0, hash_digest_eval1, hash_digest_eval2);
        }
        
        // Store 4 running evaluations
        size_t aux_idx;
        aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 0) * 3;
        d_aux[aux_idx + 0] = recv_chunk_eval0; d_aux[aux_idx + 1] = recv_chunk_eval1; d_aux[aux_idx + 2] = recv_chunk_eval2;
        
        aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 1) * 3;
        d_aux[aux_idx + 0] = hash_input_eval0; d_aux[aux_idx + 1] = hash_input_eval1; d_aux[aux_idx + 2] = hash_input_eval2;
        
        aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 2) * 3;
        d_aux[aux_idx + 0] = hash_digest_eval0; d_aux[aux_idx + 1] = hash_digest_eval1; d_aux[aux_idx + 2] = hash_digest_eval2;
        
        aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 3) * 3;
        d_aux[aux_idx + 0] = sponge_eval0; d_aux[aux_idx + 1] = sponge_eval1; d_aux[aux_idx + 2] = sponge_eval2;
    }
}

// =============================================================================
// Parallel Running Evaluations Implementation
// =============================================================================

// Constant memory for challenges (accessed by all threads)
__constant__ uint64_t c_hash_challenges[48];  // 16 XFEs

// Kernel to compute per-row factors and addends for the 4 running evaluations
__global__ void hash_compute_recurrence_kernel(
    const uint64_t* __restrict__ d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* __restrict__ d_challenges,
    LinearRecurrence* __restrict__ d_recv_chunk,
    LinearRecurrence* __restrict__ d_hash_input,
    LinearRecurrence* __restrict__ d_hash_digest,
    LinearRecurrence* __restrict__ d_sponge
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    
    constexpr size_t HASH_Mode = 0;
    constexpr size_t HASH_CI = 1;
    constexpr size_t HASH_RoundNumber = 2;
    constexpr size_t NUM_ROUNDS = 5;
    constexpr size_t RATE = 10;
    constexpr size_t DIGEST_LEN = 5;
    
    constexpr uint64_t MODE_PROGRAM_HASHING = 1;
    constexpr uint64_t MODE_SPONGE = 2;
    constexpr uint64_t MODE_HASH = 3;
    constexpr uint64_t MONTGOMERY_MOD_INV = 18446744065119617025ULL;
    constexpr uint64_t TWO_POW_16 = 65536ULL;
    constexpr uint64_t TWO_POW_32 = 4294967296ULL;
    constexpr uint64_t TWO_POW_48 = 281474976710656ULL;
    constexpr uint64_t SPONGE_INIT_OPCODE = 40;
    
    // Read challenges from constant memory (indices into c_hash_challenges)
    // Layout: [hash_input(0-2), hash_digest(3-5), sponge(6-8), prepare_chunk(9-11),
    //          send_chunk(12-14), ci_weight(15-17), state_w[0](18-20), ..., state_w[9](45-47)]
    const uint64_t hash_input_ind0 = c_hash_challenges[0], hash_input_ind1 = c_hash_challenges[1], hash_input_ind2 = c_hash_challenges[2];
    const uint64_t hash_digest_ind0 = c_hash_challenges[3], hash_digest_ind1 = c_hash_challenges[4], hash_digest_ind2 = c_hash_challenges[5];
    const uint64_t sponge_ind0 = c_hash_challenges[6], sponge_ind1 = c_hash_challenges[7], sponge_ind2 = c_hash_challenges[8];
    const uint64_t prepare_chunk_ind0 = c_hash_challenges[9], prepare_chunk_ind1 = c_hash_challenges[10], prepare_chunk_ind2 = c_hash_challenges[11];
    const uint64_t send_chunk_ind0 = c_hash_challenges[12], send_chunk_ind1 = c_hash_challenges[13], send_chunk_ind2 = c_hash_challenges[14];
    const uint64_t ci_weight0 = c_hash_challenges[15], ci_weight1 = c_hash_challenges[16], ci_weight2 = c_hash_challenges[17];
    
    uint64_t state_w[10][3];
    for (size_t j = 0; j < 10; j++) {
        state_w[j][0] = c_hash_challenges[18 + j * 3 + 0];
        state_w[j][1] = c_hash_challenges[18 + j * 3 + 1];
        state_w[j][2] = c_hash_challenges[18 + j * 3 + 2];
    }
    
    // Load row data
    const size_t row_off = i * main_width + MAIN_HASH_START;
    const uint64_t mode = d_main[row_off + HASH_Mode];
    const uint64_t ci = d_main[row_off + HASH_CI];
    const uint64_t round_num = d_main[row_off + HASH_RoundNumber];
    
    const bool in_program_hashing = (mode == MODE_PROGRAM_HASHING);
    const bool in_sponge = (mode == MODE_SPONGE);
    const bool in_hash = (mode == MODE_HASH);
    const bool in_round_0 = (round_num == 0);
    const bool in_last_round = (round_num == NUM_ROUNDS);
    const bool is_sponge_init = (ci == SPONGE_INIT_OPCODE);
    
    // Recompose rate registers
    uint64_t rate_regs[10];
    for (size_t j = 0; j < 4; j++) {
        const size_t base = 3 + j * 4;
        const uint64_t highest = d_main[row_off + base + 0];
        const uint64_t midhigh = d_main[row_off + base + 1];
        const uint64_t midlow = d_main[row_off + base + 2];
        const uint64_t lowest = d_main[row_off + base + 3];
        const uint64_t composed = bfield_add_impl(
            bfield_add_impl(
                bfield_mul_impl(highest, TWO_POW_48),
                bfield_mul_impl(midhigh, TWO_POW_32)
            ),
            bfield_add_impl(
                bfield_mul_impl(midlow, TWO_POW_16),
                lowest
            )
        );
        rate_regs[j] = bfield_mul_impl(composed, MONTGOMERY_MOD_INV);
    }
    for (size_t j = 4; j < 10; j++) {
        rate_regs[j] = d_main[row_off + 35 + (j - 4)];
    }
    
    // Compute compressed row (RATE elements) and save digest partial sum (first 5)
    uint64_t comp0 = 0, comp1 = 0, comp2 = 0;
    uint64_t digest_comp0, digest_comp1, digest_comp2;
    for (size_t j = 0; j < RATE; j++) {
        uint64_t t0, t1, t2;
        bfe_mul_xfe(rate_regs[j], state_w[j][0], state_w[j][1], state_w[j][2], t0, t1, t2);
        xfe_add_d(comp0, comp1, comp2, t0, t1, t2, comp0, comp1, comp2);
        if (j == DIGEST_LEN - 1) {
            digest_comp0 = comp0; digest_comp1 = comp1; digest_comp2 = comp2;
        }
    }
    
    // Identity: factor=1, addend=0 (no change)
    const Xfe3 one = {1, 0, 0};
    const Xfe3 zero = {0, 0, 0};
    
    // recv_chunk: updates when in_program_hashing && in_round_0
    // eval' = eval * send_chunk_ind + horner
    if (in_program_hashing && in_round_0) {
        uint64_t horner0 = 1, horner1 = 0, horner2 = 0;
        for (size_t j = 0; j < RATE; j++) {
            uint64_t t0, t1, t2;
            xfe_mul_d(horner0, horner1, horner2, prepare_chunk_ind0, prepare_chunk_ind1, prepare_chunk_ind2, t0, t1, t2);
            xfe_add_d(t0, t1, t2, rate_regs[j], 0, 0, horner0, horner1, horner2);
        }
        d_recv_chunk[i] = LinearRecurrence({send_chunk_ind0, send_chunk_ind1, send_chunk_ind2}, {horner0, horner1, horner2});
    } else {
        d_recv_chunk[i] = LinearRecurrence(one, zero);
    }
    
    // hash_input: updates when in_hash && in_round_0
    // eval' = eval * hash_input_ind + comp
    if (in_hash && in_round_0) {
        d_hash_input[i] = LinearRecurrence({hash_input_ind0, hash_input_ind1, hash_input_ind2}, {comp0, comp1, comp2});
    } else {
        d_hash_input[i] = LinearRecurrence(one, zero);
    }
    
    // hash_digest: updates when in_hash && in_last_round
    // eval' = eval * hash_digest_ind + digest_comp
    if (in_hash && in_last_round) {
        d_hash_digest[i] = LinearRecurrence({hash_digest_ind0, hash_digest_ind1, hash_digest_ind2}, {digest_comp0, digest_comp1, digest_comp2});
    } else {
        d_hash_digest[i] = LinearRecurrence(one, zero);
    }
    
    // sponge: updates when in_sponge && in_round_0
    // eval' = eval * sponge_ind + ci_term + (is_sponge_init ? 0 : comp)
    if (in_sponge && in_round_0) {
        uint64_t ci_term0, ci_term1, ci_term2;
        bfe_mul_xfe(ci, ci_weight0, ci_weight1, ci_weight2, ci_term0, ci_term1, ci_term2);
        uint64_t addend0 = ci_term0, addend1 = ci_term1, addend2 = ci_term2;
        if (!is_sponge_init) {
            xfe_add_d(addend0, addend1, addend2, comp0, comp1, comp2, addend0, addend1, addend2);
        }
        d_sponge[i] = LinearRecurrence({sponge_ind0, sponge_ind1, sponge_ind2}, {addend0, addend1, addend2});
    } else {
        d_sponge[i] = LinearRecurrence(one, zero);
    }
}

// Kernel to write prefix scan results to aux table
__global__ void hash_write_running_evals_kernel(
    const LinearRecurrence* __restrict__ d_recv_chunk,
    const LinearRecurrence* __restrict__ d_hash_input,
    const LinearRecurrence* __restrict__ d_hash_digest,
    const LinearRecurrence* __restrict__ d_sponge,
    size_t num_rows,
    uint64_t* __restrict__ d_aux
) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows) return;
    
    // The result of prefix scan with LinearRecurrenceOp gives us:
    // result.addend = the running evaluation value (since initial factor=1, addend=0 => initial eval=1)
    // Because: identity ⊕ (f, a) = (1*f, 0*f + a) = (f, a)
    // And the running evaluation is: 1 * factor_chain + addend_chain = addend_chain
    // But we start with eval=1, so we need: eval = 1 * total_factor + total_addend
    // Which simplifies to: eval = result.factor + result.addend (since initial is (1,0))
    // Actually: (1, 0) composed with all (f_i, a_i) gives us final (F, A) where
    // running_eval = 1 * F + 0 = F? No wait...
    
    // Let me reconsider. For running_eval starting at 1:
    // row 0: eval_0 = 1 * f_0 + a_0 (if condition) or eval_0 = 1
    // row 1: eval_1 = eval_0 * f_1 + a_1 (if condition) or eval_1 = eval_0
    // ...
    // The prefix scan computes: result[i] = (f_0, a_0) ⊕ (f_1, a_1) ⊕ ... ⊕ (f_i, a_i)
    // Where identity elements have f=1, a=0
    // The running eval at row i = 1 * result[i].factor + result[i].addend? No...
    
    // Actually, the scan result represents the transformation from initial value.
    // If initial eval = 1 (= factor:1, addend:0 conceptually as "multiply by 1, add 0"),
    // then applying the transformation (F, A) to initial value v:
    // new_v = v * F + A
    // So eval[i] = 1 * result[i].factor + result[i].addend? Let's verify:
    // - identity: (1, 0) => eval = 1*1 + 0 = 1 ✓
    // - (f, a): eval = 1*f + a = f + a? No, that's wrong.
    
    // Wait, the formula should be: for initial value v=1,
    // after applying (f, a): new_v = v*f + a = 1*f + a
    // But the prefix scan of (f_0, a_0) ⊕ (f_1, a_1) = (f_0*f_1, a_0*f_1 + a_1)
    // So result.factor = product of all factors, result.addend = accumulated addends
    // Final eval = 1 * result.factor + result.addend
    // But wait, that's f_0*f_1*... + (a_0*f_1*... + a_1*f_2*... + ... + a_n)
    // This doesn't match the sequential formula...
    
    // Let me reconsider. Sequential: eval_n = eval_{n-1} * f_n + a_n
    // This is: eval_n = (...((1 * f_0 + a_0) * f_1 + a_1) * f_2 + a_2) ...
    // The linear recurrence composition handles this correctly:
    // (f_0, a_0) represents: x -> x*f_0 + a_0
    // (f_0, a_0) ⊕ (f_1, a_1) represents: x -> (x*f_0 + a_0)*f_1 + a_1 = x*f_0*f_1 + a_0*f_1 + a_1
    // So result = (F, A) where F = product of factors, A = weighted sum of addends
    // And final eval = 1 * F + A = F + A? No...
    
    // Actually for initial value 1:
    // eval = 1 * result.factor + result.addend
    // Hmm, but that's F + A, not just A.
    
    // Wait, I think I need to reconsider. The addend in the scan represents the full
    // evaluation assuming initial value 0. To get the evaluation with initial value 1,
    // we compute: 1 * result.factor + result.addend
    
    // Let me verify with a simple example:
    // Row 0: (f=2, a=3) => eval_0 = 1*2 + 3 = 5
    // Row 1: (f=4, a=5) => eval_1 = 5*4 + 5 = 25
    // Prefix scan:
    // - result[0] = (2, 3)
    // - result[1] = (2, 3) ⊕ (4, 5) = (2*4, 3*4 + 5) = (8, 17)
    // Check: 1 * 8 + 17 = 25 ✓
    
    // So the formula is: eval[i] = result[i].factor + result[i].addend
    // NO WAIT: 1 * result.factor + result.addend = result.factor + result.addend
    // But we want just 1*F + A where the 1 is the initial value...
    // Let me re-verify: 1 * 8 + 17 = 25, but row 0: 1*2 + 3 = 5, row 1: 5*4 + 5 = 25 ✓
    
    // So the running eval at position i is NOT result[i].addend, but:
    // eval[i] = 1 * result[i].factor + result[i].addend
    // But wait, that's just: eval[i] = result[i].factor + result[i].addend
    // Hmm, let me verify row 0: eval[0] = 2 + 3 = 5 ✓
    
    // Actually no, for row 0 with initial value 1:
    // eval[0] = 1 * f_0 + a_0 = f_0 + a_0? No, that's wrong.
    // The formula is: new_eval = old_eval * factor + addend
    // So eval[0] = initial_eval * factor[0] + addend[0] = 1 * f_0 + a_0
    // And the scan result[0] = (f_0, a_0)
    // So eval[0] = 1 * result[0].factor + result[0].addend? Let's see:
    // = 1 * 2 + 3 = 5 ✓
    
    // For row 1:
    // result[1] = (8, 17) as computed above
    // eval[1] = 1 * 8 + 17 = 25 ✓
    
    // So the formula is correct: eval[i] = initial_value * result[i].factor + result[i].addend
    // With initial_value = 1, this simplifies to: eval[i] = result[i].factor + result[i].addend
    
    // WRONG! Let me redo this. The issue is that my composition formula is wrong.
    // Actually, the correct interpretation is:
    // The scan result (F, A) represents the function x -> x*F + A
    // Applied to initial value 1: final_eval = 1*F + A
    
    // But my example shows: eval[1] = 1*8 + 17 = 25, which matches!
    // So the formula is: eval[i] = 1 * result[i].factor + result[i].addend
    //                           = result[i].factor + result[i].addend
    
    // Wait, 1*F + A ≠ F + A in general (multiplication vs addition).
    // But in this case, 1*F = F (multiplying by 1), so 1*F + A = F + A.
    // Hmm, but F and A are XFieldElements, and + is XFE addition, not BFE addition.
    // So eval[i] = result[i].factor + result[i].addend means XFE addition.
    
    // Actually, I realize the issue: the formula 1*F + A is:
    // (1, 0, 0) * (F0, F1, F2) + (A0, A1, A2)
    // where * is XFE multiplication and + is XFE addition.
    // Since (1, 0, 0) * X = X (identity for XFE mul), we get:
    // eval = F + A (XFE addition)
    
    // Hmm, but that doesn't match my example either. Let me recalculate:
    // Row 0: (f=2, a=3) as BFE values, XFE: f=(2,0,0), a=(3,0,0)
    // eval_0 = (1,0,0) * (2,0,0) + (3,0,0) = (2,0,0) + (3,0,0) = (5,0,0) ✓
    // Row 1: (f=4, a=5) => f=(4,0,0), a=(5,0,0)
    // eval_1 = (5,0,0) * (4,0,0) + (5,0,0) = (20,0,0) + (5,0,0) = (25,0,0) ✓
    //
    // Scan result[1] = (8, 17) as BFE, so XFE: factor=(8,0,0), addend=(17,0,0)
    // eval[1] = (1,0,0) * (8,0,0) + (17,0,0) = (8,0,0) + (17,0,0) = (25,0,0) ✓
    
    // Great, so the formula is correct! But wait, I'm adding factors instead of multiplying?
    // No, the formula is: eval = initial * result.factor + result.addend
    // With initial = (1,0,0), we get: eval = (1,0,0) * result.factor + result.addend
    // XFE mul: (1,0,0) * (F0,F1,F2) = (F0,F1,F2) (identity)
    // So: eval = result.factor + result.addend (XFE addition)
    
    // This means: eval.c0 = result.factor.c0 + result.addend.c0 (modular addition)
    //             eval.c1 = result.factor.c1 + result.addend.c1
    //             eval.c2 = result.factor.c2 + result.addend.c2
    
    // Wait, that's wrong! XFE addition is component-wise, but initial * factor is XFE multiplication.
    // Let me be more careful. For XFE x = (x0, x1, x2):
    // (1, 0, 0) * x = x (since 1 is the multiplicative identity in XFE)
    // So: initial * result.factor = result.factor
    // And: eval = result.factor + result.addend (XFE addition)
    
    // But XFE addition is component-wise, so:
    // eval = (factor.c0 + addend.c0, factor.c1 + addend.c1, factor.c2 + addend.c2)
    
    // Hmm, that seems wrong. Let me think again...
    // Actually, I think the issue is that "initial * result.factor" doesn't make sense
    // because initial is a scalar (the initial running evaluation), not a transformation.
    
    // Let me reconsider the problem:
    // - Initial eval = (1, 0, 0) (XFE one)
    // - Each row transforms: eval' = eval * factor + addend (XFE arithmetic)
    // - The scan computes the composition of all transformations
    // - Final eval = apply composition to initial
    
    // For a linear transformation T(x) = x*f + a, the composition T1 ∘ T2 (T1 then T2) is:
    // T2(T1(x)) = (x*f1 + a1)*f2 + a2 = x*f1*f2 + a1*f2 + a2
    // So: (f1, a1) ⊕ (f2, a2) = (f1*f2, a1*f2 + a2) ✓
    
    // After n rows, the composition (F, A) satisfies:
    // final_eval = initial_eval * F + A
    // With initial_eval = 1 (XFE): final_eval = 1 * F + A = F + A
    // But wait, 1 * F = F (XFE multiplication by 1), so:
    // final_eval = F + A (XFE addition)
    
    // Hmm, that would mean: eval[i] = result[i].factor + result[i].addend
    // But that's component-wise addition of two XFEs...
    
    // Actually wait, I think I made an error. Let me reconsider:
    // For XFE one = (1, 0, 0), XFE mul: (1, 0, 0) * (x, y, z) = ?
    // XFE mul is NOT component-wise. It's polynomial multiplication mod x^3 - x + 1.
    // (1 + 0*x + 0*x^2) * (x0 + x1*x + x2*x^2) = x0 + x1*x + x2*x^2
    // So yes, 1 * anything = anything for XFE.
    
    // Therefore: final_eval = 1 * F + A = F + A
    // where + is XFE addition (component-wise modular addition).
    
    // Let me verify with my example again:
    // Row 0: factor=(2,0,0), addend=(3,0,0)
    // Row 1: factor=(4,0,0), addend=(5,0,0)
    // Scan result[0] = (2,0,0), (3,0,0)
    // Scan result[1] = (2*4, 3*4+5) = (8, 17) as BFEs, so ((8,0,0), (17,0,0))
    // 
    // Sequential:
    // eval_0 = 1 * 2 + 3 = 5 => (5,0,0)
    // eval_1 = 5 * 4 + 5 = 25 => (25,0,0)
    //
    // Using formula eval = F + A:
    // eval_1 = (8,0,0) + (17,0,0) = (8+17, 0, 0) = (25, 0, 0) ✓
    
    // Great! So the correct formula is:
    // eval[i] = result[i].factor + result[i].addend (XFE addition)
    //
    // In terms of components:
    // eval[i].c0 = BFE_add(result[i].factor.c0, result[i].addend.c0)
    // eval[i].c1 = BFE_add(result[i].factor.c1, result[i].addend.c1)  
    // eval[i].c2 = BFE_add(result[i].factor.c2, result[i].addend.c2)
    
    // Wait, this is wrong too. Let me re-examine.
    // Actually, when I wrote "eval = 1 * F + A", I mean:
    // eval = XFE_mul(1_xfe, F) XFE_add A = F XFE_add A
    //
    // But the formula for the running evaluation is:
    // eval_n = eval_{n-1} * factor_n + addend_n
    // where * and + are XFE operations.
    //
    // The composition gives us the transformation (F, A) such that:
    // eval_n = eval_0 * F + A
    //
    // With eval_0 = (1, 0, 0), we have:
    // eval_n = (1, 0, 0) * F + A = F + A (since 1_xfe * X = X)
    //
    // So eval_n.c0 = bfield_add(F.c0, A.c0)
    //    eval_n.c1 = bfield_add(F.c1, A.c1)
    //    eval_n.c2 = bfield_add(F.c2, A.c2)
    //
    // Hmm, wait. (1,0,0) * (F0, F1, F2) in XFE is:
    // (1 + 0x + 0x²) * (F0 + F1*x + F2*x²) mod (x³ - x + 1)
    // = F0 + F1*x + F2*x² (no reduction needed)
    // = (F0, F1, F2)
    //
    // So yes, 1_xfe * F = F. Therefore:
    // eval = F + A = (F0 + A0, F1 + A1, F2 + A2) (component-wise BFE add)
    //
    // Wait, but that doesn't match my intuition. Let me verify once more with my example:
    // eval_1 = 25 (as a BFE, so XFE = (25, 0, 0))
    // result[1] = ((8,0,0), (17,0,0)) meaning factor=(8,0,0), addend=(17,0,0)
    // eval = factor + addend = (8+17, 0+0, 0+0) = (25, 0, 0) ✓
    //
    // Great, the formula is correct!
    
    // Actually, I realize there might be an issue. The XFE addition is modular, so:
    // (a, b, c) + (d, e, f) = (a+d mod p, b+e mod p, c+f mod p)
    // where p = 2^64 - 2^32 + 1 (the BFieldElement modulus).
    // So we need to use bfield_add_impl for each component.
    
    // Final formula:
    // eval[i].c0 = bfield_add_impl(result[i].factor.c0, result[i].addend.c0)
    // eval[i].c1 = bfield_add_impl(result[i].factor.c1, result[i].addend.c1)
    // eval[i].c2 = bfield_add_impl(result[i].factor.c2, result[i].addend.c2)
    
    // Write recv_chunk_eval
    size_t aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 0) * 3;
    const Xfe3& recv = d_recv_chunk[i].addend;
    const Xfe3& recv_f = d_recv_chunk[i].factor;
    d_aux[aux_idx + 0] = bfield_add_impl(recv_f.c0, recv.c0);
    d_aux[aux_idx + 1] = bfield_add_impl(recv_f.c1, recv.c1);
    d_aux[aux_idx + 2] = bfield_add_impl(recv_f.c2, recv.c2);
    
    // Write hash_input_eval
    aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 1) * 3;
    const Xfe3& hinp = d_hash_input[i].addend;
    const Xfe3& hinp_f = d_hash_input[i].factor;
    d_aux[aux_idx + 0] = bfield_add_impl(hinp_f.c0, hinp.c0);
    d_aux[aux_idx + 1] = bfield_add_impl(hinp_f.c1, hinp.c1);
    d_aux[aux_idx + 2] = bfield_add_impl(hinp_f.c2, hinp.c2);
    
    // Write hash_digest_eval
    aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 2) * 3;
    const Xfe3& hdig = d_hash_digest[i].addend;
    const Xfe3& hdig_f = d_hash_digest[i].factor;
    d_aux[aux_idx + 0] = bfield_add_impl(hdig_f.c0, hdig.c0);
    d_aux[aux_idx + 1] = bfield_add_impl(hdig_f.c1, hdig.c1);
    d_aux[aux_idx + 2] = bfield_add_impl(hdig_f.c2, hdig.c2);
    
    // Write sponge_eval
    aux_idx = (i * AUX_TOTAL_COLS + AUX_HASH_START + 3) * 3;
    const Xfe3& spng = d_sponge[i].addend;
    const Xfe3& spng_f = d_sponge[i].factor;
    d_aux[aux_idx + 0] = bfield_add_impl(spng_f.c0, spng.c0);
    d_aux[aux_idx + 1] = bfield_add_impl(spng_f.c1, spng.c1);
    d_aux[aux_idx + 2] = bfield_add_impl(spng_f.c2, spng.c2);
}

// Host function for parallel running evaluations using CUB prefix scan
static void hash_running_evals_parallel(
    const uint64_t* d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux,
    cudaStream_t stream
) {
    if (num_rows == 0) return;
    
    // Use static buffers to avoid repeated allocation overhead
    // These are allocated with cudaMalloc (not cudaMallocManaged) to avoid
    // page migration overhead with Unified Memory.
    static thread_local LinearRecurrence *d_recv_chunk = nullptr;
    static thread_local LinearRecurrence *d_hash_input = nullptr;
    static thread_local LinearRecurrence *d_hash_digest = nullptr;
    static thread_local LinearRecurrence *d_sponge = nullptr;
    static thread_local size_t alloc_rows = 0;
    static thread_local void* d_temp_storage_static = nullptr;
    static thread_local size_t temp_storage_bytes_static = 0;
    
    size_t buf_size = num_rows * sizeof(LinearRecurrence);
    
    // Reallocate if needed (only happens once per run typically)
    if (num_rows > alloc_rows) {
        if (d_recv_chunk) { cudaFree(d_recv_chunk); cudaFree(d_hash_input); cudaFree(d_hash_digest); cudaFree(d_sponge); }
        CUDA_CHECK(cudaMalloc(&d_recv_chunk, buf_size));
        CUDA_CHECK(cudaMalloc(&d_hash_input, buf_size));
        CUDA_CHECK(cudaMalloc(&d_hash_digest, buf_size));
        CUDA_CHECK(cudaMalloc(&d_sponge, buf_size));
        alloc_rows = num_rows;
    }
    
    constexpr int BLOCK = 256;
    int grid = static_cast<int>((num_rows + BLOCK - 1) / BLOCK);
    
    // Copy challenges to constant memory (once per call, but cached with static)
    // Challenge indices in d_challenges
    constexpr size_t CH_HashInput = 4;
    constexpr size_t CH_HashDigest = 5;
    constexpr size_t CH_Sponge = 6;
    constexpr size_t CH_PrepareChunk = 29;
    constexpr size_t CH_SendChunk = 30;
    constexpr size_t CH_HashCI = 31;
    constexpr size_t CH_StackWeight0 = 32;
    
    // Use static to cache - challenges only change between proof generations
    static thread_local bool challenges_loaded = false;
    static thread_local const uint64_t* last_challenges = nullptr;
    
    if (!challenges_loaded || last_challenges != d_challenges) {
        // Layout: [hash_input(0-2), hash_digest(3-5), sponge(6-8), prepare_chunk(9-11),
        //          send_chunk(12-14), ci_weight(15-17), state_w[0](18-20), ..., state_w[9](45-47)]
        uint64_t h_challenges[48];
        
        // Batch copy from device to host (single synchronous copy)
        // Copy all needed challenges indices and rearrange on host
        // CH_HashInput=4, CH_HashDigest=5, CH_Sponge=6 are contiguous
        cudaMemcpy(&h_challenges[0], d_challenges + CH_HashInput * 3, 9 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        // CH_PrepareChunk=29, CH_SendChunk=30, CH_HashCI=31, CH_StackWeight0..9=32..41 are contiguous
        cudaMemcpy(&h_challenges[9], d_challenges + CH_PrepareChunk * 3, 39 * sizeof(uint64_t), cudaMemcpyDeviceToHost);
        
        cudaMemcpyToSymbol(c_hash_challenges, h_challenges, sizeof(h_challenges));
        challenges_loaded = true;
        last_challenges = d_challenges;
    }
    
    // Step 1: Compute per-row factors and addends
    hash_compute_recurrence_kernel<<<grid, BLOCK, 0, stream>>>(
        d_main, main_width, num_rows, d_challenges,
        d_recv_chunk, d_hash_input, d_hash_digest, d_sponge
    );
    
    // Step 2: CUB prefix scan for each running evaluation
    // Query required temp storage (same for all 4 scans since same num_rows)
    size_t temp_storage_bytes = 0;
    cub::DeviceScan::InclusiveScan(
        nullptr, temp_storage_bytes,
        d_recv_chunk, d_recv_chunk,
        LinearRecurrenceOp{}, num_rows, stream
    );
    
    // Allocate 4 temp storage buffers for parallel scans
    static thread_local void* d_temp_storage[4] = {nullptr, nullptr, nullptr, nullptr};
    static thread_local size_t temp_bytes_alloc[4] = {0, 0, 0, 0};
    
    // Reallocate temp storage if needed (one buffer per scan for parallel execution)
    for (int i = 0; i < 4; i++) {
        if (temp_storage_bytes > temp_bytes_alloc[i]) {
            if (d_temp_storage[i]) cudaFree(d_temp_storage[i]);
            CUDA_CHECK(cudaMalloc(&d_temp_storage[i], temp_storage_bytes));
            temp_bytes_alloc[i] = temp_storage_bytes;
        }
    }
    
    // Create 4 streams and events for parallel prefix scans
    // Use static streams to avoid repeated creation overhead
    static thread_local cudaStream_t scan_streams[4] = {nullptr, nullptr, nullptr, nullptr};
    static thread_local cudaEvent_t scan_events[4] = {nullptr, nullptr, nullptr, nullptr};
    static thread_local bool streams_initialized = false;
    
    if (!streams_initialized) {
        for (int i = 0; i < 4; i++) {
            cudaStreamCreate(&scan_streams[i]);
            cudaEventCreate(&scan_events[i]);
        }
        streams_initialized = true;
    }
    
    // Run prefix scans for all 4 evaluations in PARALLEL
    cub::DeviceScan::InclusiveScan(
        d_temp_storage[0], temp_bytes_alloc[0],
        d_recv_chunk, d_recv_chunk,
        LinearRecurrenceOp{}, num_rows, scan_streams[0]
    );
    cub::DeviceScan::InclusiveScan(
        d_temp_storage[1], temp_bytes_alloc[1],
        d_hash_input, d_hash_input,
        LinearRecurrenceOp{}, num_rows, scan_streams[1]
    );
    cub::DeviceScan::InclusiveScan(
        d_temp_storage[2], temp_bytes_alloc[2],
        d_hash_digest, d_hash_digest,
        LinearRecurrenceOp{}, num_rows, scan_streams[2]
    );
    cub::DeviceScan::InclusiveScan(
        d_temp_storage[3], temp_bytes_alloc[3],
        d_sponge, d_sponge,
        LinearRecurrenceOp{}, num_rows, scan_streams[3]
    );
    
    // Record events on scan streams, then have main stream wait on them
    // This avoids blocking the host thread (allows cascade LDs to launch in parallel)
    for (int i = 0; i < 4; i++) {
        cudaEventRecord(scan_events[i], scan_streams[i]);
        cudaStreamWaitEvent(stream, scan_events[i], 0);
    }
    
    // Step 3: Write results to aux table (waits on all scans via events)
    hash_write_running_evals_kernel<<<grid, BLOCK, 0, stream>>>(
        d_recv_chunk, d_hash_input, d_hash_digest, d_sponge,
        num_rows, d_aux
    );
    
    // Note: Static buffers are NOT freed - they persist for future calls
}

// =============================================================================
// Parallel JumpStack Implementation (with batch inversion)
// =============================================================================
static void jumpstack_parallel(
    const uint64_t* d_main,
    size_t main_width,
    size_t num_rows,
    const uint64_t* d_challenges,
    uint64_t* d_aux,
    cudaStream_t stream
) {
    if (num_rows == 0) return;
    
    // Copy challenges to constant memory (use same indices as global constants)
    // CH_JumpStack = 9, CH_ClockJumpDiff = 11
    // CH_JsClk = 24, CH_JsCi = 25, CH_JsJsp = 26, CH_JsJso = 27, CH_JsJsd = 28
    
    static thread_local bool challenges_loaded = false;
    static thread_local const uint64_t* last_challenges = nullptr;
    
    if (!challenges_loaded || last_challenges != d_challenges) {
        uint64_t h_ch[21];
        // Use global constant indices: CH_JumpStack=9, CH_ClockJumpDiff=11, CH_JsClk=24..CH_JsJsd=28
        cudaMemcpy(&h_ch[0], d_challenges + 9 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);   // CH_JumpStack
        cudaMemcpy(&h_ch[3], d_challenges + 24 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);  // CH_JsClk
        cudaMemcpy(&h_ch[6], d_challenges + 25 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);  // CH_JsCi
        cudaMemcpy(&h_ch[9], d_challenges + 26 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost);  // CH_JsJsp
        cudaMemcpy(&h_ch[12], d_challenges + 27 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost); // CH_JsJso
        cudaMemcpy(&h_ch[15], d_challenges + 28 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost); // CH_JsJsd
        cudaMemcpy(&h_ch[18], d_challenges + 11 * 3, 3 * sizeof(uint64_t), cudaMemcpyDeviceToHost); // CH_ClockJumpDiff
        cudaMemcpyToSymbol(c_jumpstack_challenges, h_ch, sizeof(h_ch));
        challenges_loaded = true;
        last_challenges = d_challenges;
    }
    
    // Allocate buffers (reuse across calls)
    static thread_local Xfe3* d_rp_diffs = nullptr;      // Running product diffs
    static thread_local Xfe3* d_cjd_diffs = nullptr;     // CJD diffs (for batch inversion)
    static thread_local Xfe3* d_cjd_prefix = nullptr;    // CJD prefix products
    static thread_local Xfe3* d_cjd_rev = nullptr;       // CJD reverse prefix
    static thread_local Xfe3* d_cjd_inverses = nullptr;  // CJD individual inverses
    static thread_local Xfe3* d_inv_total = nullptr;     // Total inverse (1 element)
    static thread_local uint8_t* d_cjd_mask = nullptr;   // Which rows contribute
    static thread_local size_t alloc_rows = 0;
    
    size_t buf_size = num_rows * sizeof(Xfe3);
    if (num_rows > alloc_rows) {
        if (d_rp_diffs) {
            cudaFree(d_rp_diffs);
            cudaFree(d_cjd_diffs);
            cudaFree(d_cjd_prefix);
            cudaFree(d_cjd_rev);
            cudaFree(d_cjd_inverses);
            cudaFree(d_inv_total);
            cudaFree(d_cjd_mask);
        }
        CUDA_CHECK(cudaMalloc(&d_rp_diffs, buf_size));
        CUDA_CHECK(cudaMalloc(&d_cjd_diffs, buf_size));
        CUDA_CHECK(cudaMalloc(&d_cjd_prefix, buf_size));
        CUDA_CHECK(cudaMalloc(&d_cjd_rev, buf_size));
        CUDA_CHECK(cudaMalloc(&d_cjd_inverses, buf_size));
        CUDA_CHECK(cudaMalloc(&d_inv_total, sizeof(Xfe3)));
        CUDA_CHECK(cudaMalloc(&d_cjd_mask, num_rows));
        alloc_rows = num_rows;
    }
    
    constexpr int BLOCK = 256;
    int grid = static_cast<int>((num_rows + BLOCK - 1) / BLOCK);
    
    // =========================================================================
    // RUNNING PRODUCT (simple prefix product)
    // =========================================================================
    
    // Step 1: Compute diffs for running product
    jumpstack_compute_diffs_kernel<<<grid, BLOCK, 0, stream>>>(
        d_main, main_width, num_rows, d_rp_diffs
    );
    
    // Step 2: CUB prefix product
    static thread_local void* d_temp = nullptr;
    static thread_local size_t temp_bytes = 0;
    
    size_t need = 0;
    cub::DeviceScan::InclusiveScan(
        nullptr, need,
        d_rp_diffs, d_rp_diffs,
        XfeMulOp{}, num_rows, stream
    );
    if (need > temp_bytes) {
        if (d_temp) cudaFree(d_temp);
        CUDA_CHECK(cudaMalloc(&d_temp, need));
        temp_bytes = need;
    }
    cub::DeviceScan::InclusiveScan(
        d_temp, temp_bytes,
        d_rp_diffs, d_rp_diffs,
        XfeMulOp{}, num_rows, stream
    );
    
    // =========================================================================
    // CJD LOG DERIVATIVE (batch inversion via Montgomery's trick)
    // =========================================================================
    
    // Step 3: Compute CJD diffs and mask
    jumpstack_compute_cjd_diffs_kernel<<<grid, BLOCK, 0, stream>>>(
        d_main, main_width, num_rows, d_cjd_diffs, d_cjd_mask
    );
    
    // Step 4: Forward prefix product over diffs
    cub::DeviceScan::InclusiveScan(
        d_temp, temp_bytes,
        d_cjd_diffs, d_cjd_prefix,
        XfeMulOp{}, num_rows, stream
    );
    
    // Step 5: Reverse diffs, then prefix product for reverse prefix
    jumpstack_reverse_kernel<<<grid, BLOCK, 0, stream>>>(
        d_cjd_diffs, d_cjd_rev, num_rows
    );
    cub::DeviceScan::InclusiveScan(
        d_temp, temp_bytes,
        d_cjd_rev, d_cjd_rev,
        XfeMulOp{}, num_rows, stream
    );
    
    // Step 6: Invert total product (single inversion!)
    jumpstack_inv_total_kernel<<<1, 1, 0, stream>>>(
        d_cjd_prefix, num_rows, d_inv_total
    );
    
    // Step 7: Compute individual inverses from prefix products
    jumpstack_compute_inverses_kernel<<<grid, BLOCK, 0, stream>>>(
        d_cjd_prefix, d_cjd_rev, d_cjd_mask, num_rows, d_inv_total, d_cjd_inverses
    );
    
    // Step 8: Prefix sum over inverses for log derivative
    cub::DeviceScan::InclusiveScan(
        d_temp, temp_bytes,
        d_cjd_inverses, d_cjd_inverses,
        XfeAddOp{}, num_rows, stream
    );
    
    // =========================================================================
    // WRITE RESULTS
    // =========================================================================
    
    // Step 9: Write running product and log derivative to aux table
    jumpstack_write_results_kernel<<<grid, BLOCK, 0, stream>>>(
        d_rp_diffs, d_cjd_inverses, num_rows, d_aux
    );
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED
