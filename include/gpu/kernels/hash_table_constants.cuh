#pragma once

#include <cstddef>
#include <cstdint>

namespace triton_vm {
namespace gpu {
namespace kernels {

// Main table hash section offset (matches extend_aux_table_full_gpu definitions)
constexpr size_t MAIN_HASH_START = 62;

// Column offsets inside the hash section
constexpr size_t HASH_MODE_COL = 0;
constexpr size_t HASH_CI_COL = 1;
constexpr size_t HASH_ROUND_COL = 2;

// Tip5 mode identifiers
constexpr uint64_t HASH_MODE_PROGRAM_HASHING = 1;
constexpr uint64_t HASH_MODE_SPONGE = 2;
constexpr uint64_t HASH_MODE_HASH = 3;
constexpr uint64_t HASH_MODE_PAD = 0;

constexpr uint64_t HASH_SPONGE_INIT_OPCODE = 40;

// Hash table structure
constexpr int HASH_NUM_STATES = 4;
constexpr int HASH_LIMBS_PER_STATE = 4;
constexpr int HASH_NUM_CASCADES = HASH_NUM_STATES * HASH_LIMBS_PER_STATE; // 16

// Packed limb buffer layout (SoA by cascade id for coalesced row access):
//   packed[(cascade_idx * num_rows + row) * 2 + 0] = LkIn
//   packed[(cascade_idx * num_rows + row) * 2 + 1] = LkOut
constexpr int HASH_LIMB_PAIR_STRIDE = HASH_NUM_CASCADES * 2;

} // namespace kernels
} // namespace gpu
} // namespace triton_vm


