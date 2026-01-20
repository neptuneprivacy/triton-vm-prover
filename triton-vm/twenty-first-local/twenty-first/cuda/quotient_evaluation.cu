// CUDA kernel for GPU-accelerated quotient evaluation
// This kernel evaluates AIR constraints and computes the quotient codeword

#include "field_arithmetic.cuh"

// Include the generated constraint evaluation functions
// This file is generated at build time by CudaBackend
#include "quotient_constraints.cuh"

// Optimized dot product using loop unrolling for better ILP
__device__ XFieldElement dot_product(
    const XFieldElement* values,
    const XFieldElement* weights,
    u32 count
) {
    XFieldElement result = xfield_zero();

    // Unroll by 4 for better instruction-level parallelism
    u32 unrolled_count = (count / 4) * 4;
    for (u32 i = 0; i < unrolled_count; i += 4) {
        XFieldElement p0 = xfield_mul(values[i], weights[i]);
        XFieldElement p1 = xfield_mul(values[i+1], weights[i+1]);
        XFieldElement p2 = xfield_mul(values[i+2], weights[i+2]);
        XFieldElement p3 = xfield_mul(values[i+3], weights[i+3]);

        result = xfield_add(result, p0);
        result = xfield_add(result, p1);
        result = xfield_add(result, p2);
        result = xfield_add(result, p3);
    }

    // Handle remaining elements
    for (u32 i = unrolled_count; i < count; i++) {
        result = xfield_add(result, xfield_mul(values[i], weights[i]));
    }

    return result;
}

// Main quotient evaluation kernel
// Each thread processes one row of the quotient domain
extern "C" __global__ void evaluate_quotient_kernel(
    // Input tables (quotient domain)
    const u64* quotient_domain_main_table,  // BFieldElement array, flattened
    const u64* quotient_domain_aux_table,   // XFieldElement array, flattened (3 u64s per element)

    // Challenges
    const u64* challenges,  // XFieldElement array, flattened
    u32 num_challenges,

    // Zerofier inverses (BFieldElement - 1 u64 per element, lifted to XFieldElement when used)
    const u64* initial_zerofier_inv,      // BFieldElement array
    const u64* consistency_zerofier_inv,  // BFieldElement array
    const u64* transition_zerofier_inv,   // BFieldElement array
    const u64* terminal_zerofier_inv,     // BFieldElement array

    // Quotient weights
    const u64* quotient_weights,  // XFieldElement array

    // Weight section boundaries
    u32 init_section_end,
    u32 cons_section_end,
    u32 tran_section_end,
    u32 num_total_constraints,

    // Table dimensions
    u32 num_rows_local,      // Number of rows this GPU processes
    u32 num_rows_total,      // Total rows in quotient domain (for wraparound)
    u32 row_offset,          // Starting row offset for this GPU
    u32 num_main_cols,
    u32 num_aux_cols,
    u32 unit_distance,

    // Output
    u64* quotient_output  // XFieldElement array
) {
    // Calculate local row index for this thread (within this GPU's range)
    u32 local_row_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (local_row_idx >= num_rows_local) return;

    // Calculate global row index in the quotient domain
    u32 row_idx = local_row_idx + row_offset;

    //__shared__ __align__(32) u8 programBuffer[6976];

    // Calculate next row index for transition constraints (wraps around full domain)
    u32 next_row_idx = (row_idx + unit_distance) % num_rows_total;

    // Load current and next rows from main table (BFieldElement)
    const u64* main_table_base = quotient_domain_main_table;
    const BFieldElement* current_main_row = (const BFieldElement*)(main_table_base + ((u64)row_idx * (u64)num_main_cols));
    const BFieldElement* next_main_row = (const BFieldElement*)(main_table_base + ((u64)next_row_idx * (u64)num_main_cols));

    // Load current and next rows from aux table (XFieldElement - 3 u64s per element)
    const u64* aux_table_base = quotient_domain_aux_table;
    const XFieldElement* current_aux_row = (const XFieldElement*)(aux_table_base + ((u64)row_idx * (u64)num_aux_cols * 3ULL));
    const XFieldElement* next_aux_row = (const XFieldElement*)(aux_table_base + ((u64)next_row_idx * (u64)num_aux_cols * 3ULL));

    // Load challenges
    const XFieldElement* challenge_array = (const XFieldElement*)challenges;

    // Streaming approach: evaluate and accumulate each constraint type immediately
    // This reduces register pressure by not storing all constraint values at once

    // Initialize quotient accumulator
    XFieldElement quotient_value = xfield_zero();

    // Process initial constraints
    {
        // Allocate array for initial constraints only (reduced register usage)
        XFieldElement initial_values[NUM_INITIAL_CONSTRAINTS > 0 ? NUM_INITIAL_CONSTRAINTS : 1];

        /*
        for(int i = threadIdx.x;i<2556;i+=blockDim.x){
            programBuffer[i] = init_bytecode[i];
        }
        __syncthreads();
        */


        evaluate_initial_constraints(
           // init_bytecode,
            current_main_row,
            current_aux_row,
            challenge_array,
            initial_values
        );
        

        u32 num_initial = init_section_end;
        const XFieldElement* init_weights = (const XFieldElement*)quotient_weights;
        XFieldElement init_inner_product = dot_product(initial_values, init_weights, num_initial);

        // Load zerofier as BFieldElement and lift to XFieldElement (c0=zerofier, c1=0, c2=0)
        const BFieldElement* init_zerofier_bf = (const BFieldElement*)(initial_zerofier_inv + row_idx);
        XFieldElement init_zerofier = XFieldElement(*init_zerofier_bf, bfield_zero(), bfield_zero());
        quotient_value = xfield_mul(init_inner_product, init_zerofier);
    }

    // Process consistency constraints
    {
        XFieldElement consistency_values[NUM_CONSISTENCY_CONSTRAINTS > 0 ? NUM_CONSISTENCY_CONSTRAINTS : 1];


        evaluate_consistency_constraints(
            //cons_bytecode,
            current_main_row,
            current_aux_row,
            challenge_array,
            consistency_values
        );

        u32 num_consistency = cons_section_end - init_section_end;
        const XFieldElement* cons_weights = (const XFieldElement*)(quotient_weights + ((u64)init_section_end * 3ULL));
        XFieldElement cons_inner_product = dot_product(consistency_values, cons_weights, num_consistency);

        // Load zerofier as BFieldElement and lift to XFieldElement
        const BFieldElement* cons_zerofier_bf = (const BFieldElement*)(consistency_zerofier_inv + row_idx);
        XFieldElement cons_zerofier = XFieldElement(*cons_zerofier_bf, bfield_zero(), bfield_zero());
        quotient_value = xfield_add(quotient_value, xfield_mul(cons_inner_product, cons_zerofier));
    }

    // Process transition constraints
    {
        XFieldElement transition_values[NUM_TRANSITION_CONSTRAINTS > 0 ? NUM_TRANSITION_CONSTRAINTS : 1];

        /*
        for(int i = threadIdx.x;i<78786;i+=blockDim.x){
            programBuffer[i] = tran_bytecode[i];
        }
        __syncthreads();
        */

        evaluate_transition_constraints(
            //tran_bytecode,
            current_main_row,
            current_aux_row,
            next_main_row,
            next_aux_row,
            challenge_array,
            transition_values
        );
        

        u32 num_transition = tran_section_end - cons_section_end;
        const XFieldElement* tran_weights = (const XFieldElement*)(quotient_weights + ((u64)cons_section_end * 3ULL));
        XFieldElement tran_inner_product = dot_product(transition_values, tran_weights, num_transition);

        // Load zerofier as BFieldElement and lift to XFieldElement
        const BFieldElement* tran_zerofier_bf = (const BFieldElement*)(transition_zerofier_inv + row_idx);
        XFieldElement tran_zerofier = XFieldElement(*tran_zerofier_bf, bfield_zero(), bfield_zero());
        quotient_value = xfield_add(quotient_value, xfield_mul(tran_inner_product, tran_zerofier));
    }

    // Process terminal constraints
    {
        XFieldElement terminal_values[NUM_TERMINAL_CONSTRAINTS > 0 ? NUM_TERMINAL_CONSTRAINTS : 1];

        /*
        for(int i = threadIdx.x;i<974;i+=blockDim.x){
            programBuffer[i] = term_bytecode[i];
        }
        __syncthreads();
        */

        evaluate_terminal_constraints(
            //term_bytecode,
            current_main_row,
            current_aux_row,
            challenge_array,
            terminal_values
        );

        u32 num_terminal = num_total_constraints - tran_section_end;
        const XFieldElement* term_weights = (const XFieldElement*)(quotient_weights + ((u64)tran_section_end * 3ULL));
        XFieldElement term_inner_product = dot_product(terminal_values, term_weights, num_terminal);

        // Load zerofier as BFieldElement and lift to XFieldElement
        const BFieldElement* term_zerofier_bf = (const BFieldElement*)(terminal_zerofier_inv + row_idx);
        XFieldElement term_zerofier = XFieldElement(*term_zerofier_bf, bfield_zero(), bfield_zero());
        quotient_value = xfield_add(quotient_value, xfield_mul(term_inner_product, term_zerofier));
    }

    // Write output (use local_row_idx for output array indexing)
    XFieldElement* output_ptr = (XFieldElement*)(quotient_output + ((u64)local_row_idx * 3ULL));
    *output_ptr = quotient_value;
}
