#pragma once
// Combined transition constraint evaluator that calls all parts
// DO NOT EDIT MANUALLY. Regenerate with: python3 tools/gen_gpu_quotient_constraints_split.py

#include "quotient_transition_part0_generated.cuh"
#include "quotient_transition_part1_generated.cuh"
#include "quotient_transition_part2_generated.cuh"
#include "quotient_transition_part3_generated.cuh"

namespace triton_vm { namespace gpu { namespace quotient_gen {

__device__ __forceinline__ Xfe eval_transition_weighted(
    const Bfe* current_main_row,
    const Xfe* current_aux_row,
    const Bfe* next_main_row,
    const Xfe* next_aux_row,
    const Xfe* challenges,
    const Xfe* weights
) {
    Xfe acc = Xfe::zero();
    acc = acc + eval_transition_part0_weighted(current_main_row, current_aux_row, next_main_row, next_aux_row, challenges, weights);
    acc = acc + eval_transition_part1_weighted(current_main_row, current_aux_row, next_main_row, next_aux_row, challenges, weights);
    acc = acc + eval_transition_part2_weighted(current_main_row, current_aux_row, next_main_row, next_aux_row, challenges, weights);
    acc = acc + eval_transition_part3_weighted(current_main_row, current_aux_row, next_main_row, next_aux_row, challenges, weights);
    return acc;
}

}}} // namespace triton_vm::gpu::quotient_gen
