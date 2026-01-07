/**
 * GPU kernels for table filling and padding
 * 
 * These kernels parallelize the padding of main table rows across GPU threads.
 */

#ifdef TRITON_CUDA_ENABLED

#include "gpu/kernels/table_fill_kernel.cuh"
#include "gpu/cuda_common.cuh"
#include <iostream>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Table column offsets (must match C++ definitions)
namespace TableOffsets {
    constexpr size_t PROGRAM_TABLE_START = 0;
    constexpr size_t PROGRAM_TABLE_COLS = 7;
    constexpr size_t PROCESSOR_TABLE_START = 7;
    constexpr size_t PROCESSOR_TABLE_COLS = 39;
    constexpr size_t OP_STACK_TABLE_START = 46;
    constexpr size_t OP_STACK_TABLE_COLS = 4;
    constexpr size_t RAM_TABLE_START = 50;
    constexpr size_t RAM_TABLE_COLS = 7;
    constexpr size_t JUMP_STACK_TABLE_START = 57;
    constexpr size_t JUMP_STACK_TABLE_COLS = 5;
    constexpr size_t HASH_TABLE_START = 62;
    constexpr size_t HASH_TABLE_COLS = 67;
    constexpr size_t CASCADE_TABLE_START = 129;
    constexpr size_t CASCADE_TABLE_COLS = 6;
    constexpr size_t LOOKUP_TABLE_START = 135;
    constexpr size_t LOOKUP_TABLE_COLS = 4;
    constexpr size_t U32_TABLE_START = 139;
    constexpr size_t U32_TABLE_COLS = 10;
}

// Processor table column indices (within processor table)
namespace ProcessorCol {
    constexpr size_t CLK = 0;
    constexpr size_t IsPadding = 1;
    constexpr size_t ClockJumpDifferenceLookupMultiplicity = 38;
}

// Program table column indices
namespace ProgramCol {
    constexpr size_t Address = 0;
    // Match `ProgramMainColumn` layout from `table/extend_helpers.hpp`:
    // 0 Address
    // 1 Instruction
    // 2 LookupMultiplicity
    // 3 IndexInChunk
    // 4 MaxMinusIndexInChunkInv
    // 5 IsHashInputPadding
    // 6 IsTablePadding
    constexpr size_t Instruction = 1;
    constexpr size_t LookupMultiplicity = 2;
    constexpr size_t IndexInChunk = 3;
    constexpr size_t MaxMinusIndexInChunkInv = 4;
    constexpr size_t IsHashInputPadding = 5;
    constexpr size_t IsTablePadding = 6;
}

constexpr size_t TIP5_RATE = 10;

// Field arithmetic for BFieldElement
__device__ __forceinline__ uint64_t bfe_inverse(uint64_t val) {
    // Prime field inverse using Fermat's little theorem: a^(-1) = a^(p-2) mod p
    // p = 2^64 - 2^32 + 1
    // This is expensive but correct
    constexpr uint64_t P = 0xFFFFFFFF00000001ULL;
    if (val == 0) return 0;
    
    // Extended Euclidean algorithm for inverse
    int64_t t = 0, newt = 1;
    uint64_t r = P, newr = val;
    
    while (newr != 0) {
        uint64_t q = r / newr;
        int64_t tmp_t = t - (int64_t)(q) * newt;
        t = newt;
        newt = tmp_t;
        uint64_t tmp_r = r - q * newr;
        r = newr;
        newr = tmp_r;
    }
    
    if (t < 0) t += P;
    return (uint64_t)t;
}

// Host-side version of the same inversion (used to precompute padding constants once per call)
static inline uint64_t bfe_inverse_host(uint64_t val) {
    constexpr uint64_t P = 0xFFFFFFFF00000001ULL;
    if (val == 0) return 0;
    int64_t t = 0, newt = 1;
    uint64_t r = P, newr = val;
    while (newr != 0) {
        uint64_t q = r / newr;
        int64_t tmp_t = t - (int64_t)(q) * newt;
        t = newt;
        newt = tmp_t;
        uint64_t tmp_r = r - q * newr;
        r = newr;
        newr = tmp_r;
    }
    if (t < 0) t += (int64_t)P;
    return (uint64_t)t;
}

/**
 * Pad processor table - each thread handles one padding row
 */
__global__ void pad_processor_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t processor_table_start,
    size_t processor_table_cols,
    size_t table_len,
    size_t padded_height,
    size_t clk_col,
    size_t is_padding_col,
    size_t cjd_col
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t padding_row = table_len + row;
    
    if (padding_row >= padded_height) return;
    if (table_len == 0) return;
    
    size_t last_row = table_len - 1;
    
    // Copy all processor columns from last row
    for (size_t c = 0; c < processor_table_cols; ++c) {
        size_t col = processor_table_start + c;
        d_table[padding_row * num_cols + col] = d_table[last_row * num_cols + col];
    }
    
    // Override specific columns for padding
    d_table[padding_row * num_cols + processor_table_start + is_padding_col] = 1;  // IsPadding = 1
    d_table[padding_row * num_cols + processor_table_start + cjd_col] = 0;  // CLK jump diff = 0
    d_table[padding_row * num_cols + processor_table_start + clk_col] = padding_row;  // CLK = row index
}

/**
 * Pad program table
 */
__global__ void pad_program_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t program_table_start,
    size_t program_len,
    size_t padded_height,
    size_t rate
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t padding_row = program_len + row;
    
    if (padding_row >= padded_height) return;
    
    // Address column
    d_table[padding_row * num_cols + program_table_start + ProgramCol::Address] = padding_row;

    // Instruction and LookupMultiplicity: leave as zero for table padding rows (matches CPU padding behavior)
    d_table[padding_row * num_cols + program_table_start + ProgramCol::Instruction] = 0;
    d_table[padding_row * num_cols + program_table_start + ProgramCol::LookupMultiplicity] = 0;
    
    // IndexInChunk = row % rate
    size_t index_in_chunk = padding_row % rate;
    d_table[padding_row * num_cols + program_table_start + ProgramCol::IndexInChunk] = index_in_chunk;
    
    // MaxMinusIndexInChunkInv = inverse_or_zero(rate - 1 - index_in_chunk)
    size_t max_minus_index = rate - 1 - index_in_chunk;
    d_table[padding_row * num_cols + program_table_start + ProgramCol::MaxMinusIndexInChunkInv] = 
        (max_minus_index == 0) ? 0 : bfe_inverse(max_minus_index);
    
    // IsHashInputPadding = 1
    d_table[padding_row * num_cols + program_table_start + ProgramCol::IsHashInputPadding] = 1;
    
    // IsTablePadding = 1
    d_table[padding_row * num_cols + program_table_start + ProgramCol::IsTablePadding] = 1;
}

/**
 * Pad op stack table - copy last row to padding rows
 */
__global__ void pad_op_stack_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t op_stack_table_start,
    size_t table_len,
    size_t padded_height
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t padding_row = table_len + row;
    
    if (padding_row >= padded_height) return;
    if (table_len == 0) return;
    
    size_t last_row = table_len - 1;
    
    // Copy all 4 columns from last row
    for (size_t c = 0; c < TableOffsets::OP_STACK_TABLE_COLS; ++c) {
        size_t col = op_stack_table_start + c;
        d_table[padding_row * num_cols + col] = d_table[last_row * num_cols + col];
    }

    // Rust special-case: set IB1ShrinkStack to PADDING_VALUE (2) for all padding rows.
    // OpStack columns: CLK=0, IB1ShrinkStack=1, StackPointer=2, FirstUnderflowElement=3
    d_table[padding_row * num_cols + op_stack_table_start + 1] = 2;
}

/**
 * Pad RAM table
 */
__global__ void pad_ram_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t ram_table_start,
    size_t table_len,
    size_t padded_height
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t padding_row = table_len + row;
    
    if (padding_row >= padded_height) return;

    // RAM columns: CLK=0, InstructionType=1, ..., BezoutCoeff0=5, BezoutCoeff1=6
    constexpr size_t RAM_INSTRUCTION_TYPE_COL = 1;
    constexpr size_t RAM_BEZOUT_COEFF1_COL = 6;
    constexpr uint64_t RAM_PADDING_INDICATOR = 2;

    if (table_len == 0) {
        // Rust: if table is empty, set BezoutCoefficientPolynomialCoefficient1 = 1
        d_table[padding_row * num_cols + ram_table_start + RAM_BEZOUT_COEFF1_COL] = 1;
        // And set InstructionType to PADDING_INDICATOR
        d_table[padding_row * num_cols + ram_table_start + RAM_INSTRUCTION_TYPE_COL] = RAM_PADDING_INDICATOR;
        return;
    }

    size_t last_row = table_len - 1;
    for (size_t c = 0; c < TableOffsets::RAM_TABLE_COLS; ++c) {
        size_t col = ram_table_start + c;
        d_table[padding_row * num_cols + col] = d_table[last_row * num_cols + col];
    }

    // Rust: set InstructionType to PADDING_INDICATOR (2) for all padding rows
    d_table[padding_row * num_cols + ram_table_start + RAM_INSTRUCTION_TYPE_COL] = RAM_PADDING_INDICATOR;
}

/**
 * Pad jump stack table
 */
__global__ void pad_jump_stack_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t jump_stack_table_start,
    size_t table_len,
    size_t padded_height
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t padding_row = table_len + row;
    
    if (padding_row >= padded_height) return;
    if (table_len == 0) return;
    
    size_t last_row = table_len - 1;
    
    for (size_t c = 0; c < TableOffsets::JUMP_STACK_TABLE_COLS; ++c) {
        size_t col = jump_stack_table_start + c;
        d_table[padding_row * num_cols + col] = d_table[last_row * num_cols + col];
    }
}

/**
 * JumpStack padding (matches `pad_jump_stack_table` in `src/table/table_padding.cpp`).
 *
 * - Find the first row `max_clk_row_idx` where CLK == table_len - 1
 * - Move rows below it (max_clk_row_idx+1 .. table_len-1) down by num_padding_rows
 * - Fill the created gap (padding section) with copies of the max-clk row, but with CLK increasing:
 *     CLK = table_len + k   for k in [0..num_padding_rows)
 */
__global__ void find_jump_stack_max_clk_row_idx_kernel(
    const uint64_t* d_table,
    size_t num_cols,
    size_t jump_stack_table_start,
    size_t table_len,
    size_t* d_out_idx
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (table_len == 0) { *d_out_idx = 0; return; }
    const uint64_t target_clk = (uint64_t)(table_len - 1);
    size_t found = 0;
    for (size_t i = 0; i < table_len; ++i) {
        uint64_t clk = d_table[i * num_cols + jump_stack_table_start + 0];
        if (clk == target_clk) { found = i; break; }
    }
    *d_out_idx = found;
}

__global__ void move_jump_stack_rows_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t jump_stack_table_start,
    size_t table_len,
    size_t padded_height,
    const size_t* d_max_clk_row_idx
) {
    (void)padded_height;
    const size_t max_idx = *d_max_clk_row_idx;
    const size_t num_padding_rows = padded_height - table_len;
    const size_t rows_to_move_start = max_idx + 1;
    if (rows_to_move_start >= table_len) return;
    const size_t num_rows_to_move = table_len - rows_to_move_start;

    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_rows_to_move) return;

    const size_t src_row = rows_to_move_start + i;
    const size_t dst_row = rows_to_move_start + num_padding_rows + i;
    for (size_t c = 0; c < TableOffsets::JUMP_STACK_TABLE_COLS; ++c) {
        const size_t col = jump_stack_table_start + c;
        d_table[dst_row * num_cols + col] = d_table[src_row * num_cols + col];
    }
}

__global__ void pad_jump_stack_section_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t jump_stack_table_start,
    size_t table_len,
    size_t padded_height,
    const size_t* d_max_clk_row_idx
) {
    const size_t num_padding_rows = padded_height - table_len;
    const size_t max_idx = *d_max_clk_row_idx;
    const size_t padding_section_start = max_idx + 1;
    if (padding_section_start >= padded_height) return;

    size_t k = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= num_padding_rows) return;

    const size_t row_idx = padding_section_start + k;
    if (row_idx >= padded_height) return;

    // Copy template row (max clk row) into padding row
    for (size_t c = 0; c < TableOffsets::JUMP_STACK_TABLE_COLS; ++c) {
        const size_t col = jump_stack_table_start + c;
        d_table[row_idx * num_cols + col] = d_table[max_idx * num_cols + col];
    }
    // Rust: CLK keeps increasing by 1 in the padding section.
    d_table[row_idx * num_cols + jump_stack_table_start + 0] = (uint64_t)(table_len + k);
}

/**
 * Pad hash table
 */
__global__ void pad_hash_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t hash_table_start,
    size_t hash_table_cols,
    size_t table_len,
    size_t padded_height,
    uint64_t inv_2p32_minus_1
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t padding_row = table_len + row;
    
    if (padding_row >= padded_height) return;
    (void)hash_table_cols;

    // Match `pad_hash_table` in `src/table/table_padding.cpp`:
    // - State0Inv..State3Inv = inv_2p32_minus_1
    // - Constant0..Constant15 = Tip5 round-0 constants
    // - Mode = Pad (0)
    // - CI = Hash opcode (18)
    constexpr size_t MODE_COL = 0;
    constexpr size_t CI_COL = 1;
    constexpr size_t STATE0_INV_COL = 47;
    constexpr size_t STATE1_INV_COL = 48;
    constexpr size_t STATE2_INV_COL = 49;
    constexpr size_t STATE3_INV_COL = 50;
    constexpr size_t CONST0_COL = 51; // 16 constants follow

    // Tip5 round-0 constants (16)
    constexpr uint64_t ROUND0[16] = {
        0xBD2A3DEB61AB60DEULL, 0xEA7DF21AD9547ED2ULL, 0x900B3677A1DE063FULL, 0x1B46887E876C8677ULL,
        0xD364D977889CFB97ULL, 0xDC8DFAC843699F02ULL, 0x375C405D7190DB58ULL, 0x27924006D2B0D4B1ULL,
        0x78DD1172D483CD38ULL, 0x3346C66244882A56ULL, 0xB0249B279F498AA5ULL, 0x94CD51BE79338D4DULL,
        0xB0E0DC7052C5B218ULL, 0xF8DCC4D248ADAD95ULL, 0x68E3C635FEC868B7ULL, 0xD7D06B3FFB6B0D8CULL,
    };

    // State inverses
    d_table[padding_row * num_cols + hash_table_start + STATE0_INV_COL] = inv_2p32_minus_1;
    d_table[padding_row * num_cols + hash_table_start + STATE1_INV_COL] = inv_2p32_minus_1;
    d_table[padding_row * num_cols + hash_table_start + STATE2_INV_COL] = inv_2p32_minus_1;
    d_table[padding_row * num_cols + hash_table_start + STATE3_INV_COL] = inv_2p32_minus_1;

    // Round-0 constants
    #pragma unroll
    for (int k = 0; k < 16; ++k) {
        d_table[padding_row * num_cols + hash_table_start + CONST0_COL + (size_t)k] = ROUND0[k];
    }

    // Mode + CI
    d_table[padding_row * num_cols + hash_table_start + MODE_COL] = 0;  // Pad
    d_table[padding_row * num_cols + hash_table_start + CI_COL] = 18;   // Hash opcode
}

/**
 * Pad cascade table
 */
__global__ void pad_cascade_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t cascade_table_start,
    size_t table_len,
    size_t padded_height
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t padding_row = table_len + row;
    
    if (padding_row >= padded_height) return;

    // Match `pad_cascade_table` in `src/table/table_padding.cpp`:
    // Set IsPadding = 1 for all padding rows; leave other columns as-is (they are zeroed).
    constexpr size_t IS_PADDING_COL = 0;
    d_table[padding_row * num_cols + cascade_table_start + IS_PADDING_COL] = 1;
}

/**
 * Pad lookup table
 */
__global__ void pad_lookup_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t lookup_table_start,
    size_t table_len,
    size_t padded_height
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t padding_row = table_len + row;
    
    if (padding_row >= padded_height) return;

    // Match `pad_lookup_table` in `src/table/table_padding.cpp`:
    // Set IsPadding = 1 for all padding rows; leave other columns as-is (they are zeroed).
    constexpr size_t IS_PADDING_COL = 0;
    d_table[padding_row * num_cols + lookup_table_start + IS_PADDING_COL] = 1;
}

/**
 * Pad U32 table
 */
__global__ void pad_u32_table_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t u32_table_start,
    size_t table_len,
    size_t padded_height,
    uint64_t inv_neg33
) {
    size_t row = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t padding_row = table_len + row;
    
    if (padding_row >= padded_height) return;

    // Match `pad_u32_table` in `src/table/table_padding.cpp`:
    // Build a padding row with most fields 0, BitsMinus33Inv = (-33)^(-1),
    // and copy CI/LHS/LhsInv/Result from last row when table_len>0.
    // Column order: CopyFlag, Bits, BitsMinus33Inv, CI, LHS, LhsInv, RHS, RhsInv, Result, LookupMultiplicity
    constexpr size_t COL_COPYFLAG = 0;
    constexpr size_t COL_BITS = 1;
    constexpr size_t COL_BM33INV = 2;
    constexpr size_t COL_CI = 3;
    constexpr size_t COL_LHS = 4;
    constexpr size_t COL_LHSINV = 5;
    constexpr size_t COL_RHS = 6;
    constexpr size_t COL_RHSINV = 7;
    constexpr size_t COL_RES = 8;
    constexpr size_t COL_MULT = 9;

    uint64_t ci = 0, lhs = 0, lhsinv = 0, res = 0;
    if (table_len > 0) {
        const size_t last = table_len - 1;
        ci = d_table[last * num_cols + u32_table_start + COL_CI];
        lhs = d_table[last * num_cols + u32_table_start + COL_LHS];
        lhsinv = d_table[last * num_cols + u32_table_start + COL_LHSINV];
        res = d_table[last * num_cols + u32_table_start + COL_RES];
    }

    d_table[padding_row * num_cols + u32_table_start + COL_COPYFLAG] = 0;
    d_table[padding_row * num_cols + u32_table_start + COL_BITS] = 0;
    d_table[padding_row * num_cols + u32_table_start + COL_BM33INV] = inv_neg33;
    d_table[padding_row * num_cols + u32_table_start + COL_CI] = ci;
    d_table[padding_row * num_cols + u32_table_start + COL_LHS] = lhs;
    d_table[padding_row * num_cols + u32_table_start + COL_LHSINV] = lhsinv;
    d_table[padding_row * num_cols + u32_table_start + COL_RHS] = 0;
    d_table[padding_row * num_cols + u32_table_start + COL_RHSINV] = 0;
    d_table[padding_row * num_cols + u32_table_start + COL_RES] = res;
    d_table[padding_row * num_cols + u32_table_start + COL_MULT] = 0;
}

/**
 * Rust special-case: Jump Stack Table has no padding indicator; clock jump diffs are looked up
 * in its padding section and are always 1. Increase processor ClockJumpDifferenceLookupMultiplicity
 * at clock value 1 by the number of JumpStack padding rows.
 *
 * This matches `pad_processor_table` in `src/table/table_padding.cpp`.
 */
__global__ void add_jump_stack_padding_to_processor_cjd_kernel(
    uint64_t* d_table,
    size_t num_cols,
    size_t processor_table_start,
    size_t cjd_col,
    uint64_t add_value
) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (add_value == 0) return;
    // Row 1, processor column `ClockJumpDifferenceLookupMultiplicity`
    d_table[1 * num_cols + processor_table_start + cjd_col] += add_value;
}

/**
 * Host function to pad entire main table on GPU
 */
void gpu_pad_main_table(
    uint64_t* d_table,
    size_t num_cols,
    size_t padded_height,
    const size_t table_lengths[9],
    cudaStream_t stream
) {
    constexpr size_t BLOCK_SIZE = 256;
    
    // Table lengths in order: Program, Processor, OpStack, RAM, JumpStack, Hash, Cascade, Lookup, U32
    size_t program_len = table_lengths[0];
    size_t processor_len = table_lengths[1];
    size_t op_stack_len = table_lengths[2];
    size_t ram_len = table_lengths[3];
    size_t jump_stack_len = table_lengths[4];
    size_t hash_len = table_lengths[5];
    size_t cascade_len = table_lengths[6];
    size_t lookup_len = table_lengths[7];
    size_t u32_len = table_lengths[8];
    
    // Launch padding kernels for each table
    
    // Program table
    if (program_len < padded_height) {
        size_t num_padding = padded_height - program_len;
        size_t blocks = (num_padding + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pad_program_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_table, num_cols, TableOffsets::PROGRAM_TABLE_START, 
            program_len, padded_height, TIP5_RATE);
    }
    
    // Processor table
    if (processor_len < padded_height) {
        size_t num_padding = padded_height - processor_len;
        size_t blocks = (num_padding + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pad_processor_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_table, num_cols, TableOffsets::PROCESSOR_TABLE_START,
            TableOffsets::PROCESSOR_TABLE_COLS, processor_len, padded_height,
            ProcessorCol::CLK, ProcessorCol::IsPadding,
            ProcessorCol::ClockJumpDifferenceLookupMultiplicity);
    }
    
    // OpStack table
    if (op_stack_len < padded_height) {
        size_t num_padding = padded_height - op_stack_len;
        size_t blocks = (num_padding + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pad_op_stack_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_table, num_cols, TableOffsets::OP_STACK_TABLE_START,
            op_stack_len, padded_height);
    }
    
    // RAM table
    if (ram_len < padded_height) {
        size_t num_padding = padded_height - ram_len;
        size_t blocks = (num_padding + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pad_ram_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_table, num_cols, TableOffsets::RAM_TABLE_START,
            ram_len, padded_height);
    }
    
    // JumpStack table
    if (jump_stack_len < padded_height) {
        // Implement the Rust row-movement padding logic.
        size_t* d_max_idx = nullptr;
        CUDA_CHECK(cudaMallocAsync(&d_max_idx, sizeof(size_t), stream));
        find_jump_stack_max_clk_row_idx_kernel<<<1, 1, 0, stream>>>(
            d_table,
            num_cols,
            TableOffsets::JUMP_STACK_TABLE_START,
            jump_stack_len,
            d_max_idx
        );

        // Move rows below max-clk row down by num_padding_rows
        {
            size_t num_rows_to_move_max = jump_stack_len; // safe upper bound
            size_t blocks = (num_rows_to_move_max + BLOCK_SIZE - 1) / BLOCK_SIZE;
            move_jump_stack_rows_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                d_table,
                num_cols,
                TableOffsets::JUMP_STACK_TABLE_START,
                jump_stack_len,
                padded_height,
                d_max_idx
            );
        }

        // Fill padding section (gap) with template row and increasing CLK
        {
            size_t num_padding = padded_height - jump_stack_len;
            size_t blocks = (num_padding + BLOCK_SIZE - 1) / BLOCK_SIZE;
            pad_jump_stack_section_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
                d_table,
                num_cols,
                TableOffsets::JUMP_STACK_TABLE_START,
                jump_stack_len,
                padded_height,
                d_max_idx
            );
        }

        CUDA_CHECK(cudaFreeAsync(d_max_idx, stream));
    }

    // Rust special-case: JumpStack padding contributes clock-diff=1 lookups.
    if (padded_height > 1 && jump_stack_len < padded_height) {
        uint64_t num_padding_rows = static_cast<uint64_t>(padded_height - jump_stack_len);
        add_jump_stack_padding_to_processor_cjd_kernel<<<1, 1, 0, stream>>>(
            d_table,
            num_cols,
            TableOffsets::PROCESSOR_TABLE_START,
            ProcessorCol::ClockJumpDifferenceLookupMultiplicity,
            num_padding_rows
        );
    }
    
    // Hash table
    if (hash_len < padded_height) {
        size_t num_padding = padded_height - hash_len;
        size_t blocks = (num_padding + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const uint64_t inv_2p32_minus_1 = bfe_inverse_host((1ULL << 32) - 1);
        pad_hash_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_table, num_cols, TableOffsets::HASH_TABLE_START,
            TableOffsets::HASH_TABLE_COLS, hash_len, padded_height, inv_2p32_minus_1);
    }
    
    // Cascade table
    if (cascade_len < padded_height) {
        size_t num_padding = padded_height - cascade_len;
        size_t blocks = (num_padding + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pad_cascade_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_table, num_cols, TableOffsets::CASCADE_TABLE_START,
            cascade_len, padded_height);
    }
    
    // Lookup table
    if (lookup_len < padded_height) {
        size_t num_padding = padded_height - lookup_len;
        size_t blocks = (num_padding + BLOCK_SIZE - 1) / BLOCK_SIZE;
        pad_lookup_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_table, num_cols, TableOffsets::LOOKUP_TABLE_START,
            lookup_len, padded_height);
    }
    
    // U32 table
    if (u32_len < padded_height) {
        size_t num_padding = padded_height - u32_len;
        size_t blocks = (num_padding + BLOCK_SIZE - 1) / BLOCK_SIZE;
        const uint64_t P = 0xFFFFFFFF00000001ULL;
        const uint64_t inv_neg33 = bfe_inverse_host(P - 33);
        pad_u32_table_kernel<<<blocks, BLOCK_SIZE, 0, stream>>>(
            d_table, num_cols, TableOffsets::U32_TABLE_START,
            u32_len, padded_height, inv_neg33);
    }
    
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

/**
 * Upload unpadded table to GPU and pad it there
 */
uint64_t* gpu_upload_and_pad_table(
    const uint64_t* h_unpadded_table,
    size_t unpadded_rows,
    size_t num_cols,
    size_t padded_height,
    const size_t table_lengths[9],
    cudaStream_t stream
) {
    // Allocate GPU memory for padded table
    size_t padded_size = padded_height * num_cols * sizeof(uint64_t);
    uint64_t* d_table = nullptr;
    CUDA_CHECK(cudaMalloc(&d_table, padded_size));
    
    // Zero out the entire table first (for padding rows)
    CUDA_CHECK(cudaMemsetAsync(d_table, 0, padded_size, stream));
    
    // Upload unpadded data to first unpadded_rows
    size_t unpadded_size = unpadded_rows * num_cols * sizeof(uint64_t);
    CUDA_CHECK(cudaMemcpyAsync(d_table, h_unpadded_table, unpadded_size, 
                               cudaMemcpyHostToDevice, stream));
    
    // Pad on GPU
    gpu_pad_main_table(d_table, num_cols, padded_height, table_lengths, stream);
    
    return d_table;
}

} // namespace kernels
} // namespace gpu
} // namespace triton_vm

#endif // TRITON_CUDA_ENABLED

