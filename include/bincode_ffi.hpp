#pragma once

#include <cstdint>
#include <cstddef>

extern "C" {
    /// Serialize a Vec<u64> to bincode format and write to file
    /// 
    /// @param data Pointer to array of u64 values
    /// @param len Number of elements in the array
    /// @param file_path Null-terminated C string path to output file
    /// @return 0 on success, -1 on error
    int bincode_serialize_vec_u64_to_file(
        const uint64_t* data,
        size_t len,
        const char* file_path
    );
    
    /// Deserialize a Vec<u64> from bincode file
    /// 
    /// @param file_path Null-terminated C string path to input file
    /// @param out_data Pointer to pointer that will be set to allocated data (caller must free)
    /// @param out_len Pointer to size_t that will be set to the length
    /// @return 0 on success, -1 on error
    int bincode_deserialize_vec_u64_from_file(
        const char* file_path,
        uint64_t** out_data,
        size_t* out_len
    );
    
    /// Free memory allocated by bincode_deserialize_vec_u64_from_file
    /// 
    /// @param ptr Pointer to memory allocated by bincode_deserialize_vec_u64_from_file
    /// @param len Length of the array (needed for proper deallocation)
    void bincode_free_vec_u64(uint64_t* ptr, size_t len);
    
    /// Encode FriResponse ProofItem using Rust's BFieldCodec
    /// 
    /// @param auth_data Pointer to array of u64 values representing Digests (num_auth * 5 elements)
    /// @param num_auth Number of Digests in auth_structure
    /// @param leaves_data Pointer to array of u64 values representing XFieldElements (num_leaves * 3 elements)
    /// @param num_leaves Number of XFieldElements in revealed_leaves
    /// @param out_data Pointer to pointer that will be set to allocated encoded data (caller must free)
    /// @param out_len Pointer to size_t that will be set to the encoded length
    /// @return 0 on success, -1 on error
    /// 
    /// Note: The caller is responsible for freeing the memory using proof_item_free_encoding
    int proof_item_encode_fri_response(
        const uint64_t* auth_data,
        size_t num_auth,
        const uint64_t* leaves_data,
        size_t num_leaves,
        uint64_t** out_data,
        size_t* out_len
    );
    
    /// Encode FriCodeword ProofItem using Rust's BFieldCodec
    /// 
    /// @param codeword_data Pointer to array of u64 values representing XFieldElements (num_elements * 3 elements)
    /// @param num_elements Number of XFieldElements in the codeword
    /// @param out_data Pointer to pointer that will be set to allocated encoded data (caller must free)
    /// @param out_len Pointer to size_t that will be set to the encoded length
    /// @return 0 on success, -1 on error
    /// 
    /// Note: The caller is responsible for freeing the memory using proof_item_free_encoding
    int proof_item_encode_fri_codeword(
        const uint64_t* codeword_data,
        size_t num_elements,
        uint64_t** out_data,
        size_t* out_len
    );
    
    /// General function to encode any ProofItem using Rust's BFieldCodec
    /// 
    /// @param discriminant ProofItem variant discriminant (0-11)
    /// @param bfield_data Pointer to array of u64 values for BFieldElement data (or nullptr if not needed)
    /// @param bfield_count Number of BFieldElements (or 0 if not needed)
    /// @param xfield_data Pointer to array of u64 values for XFieldElement data (num_elements * 3) (or nullptr if not needed)
    /// @param xfield_count Number of XFieldElements (or 0 if not needed)
    /// @param digest_data Pointer to array of u64 values for Digest data (num_digests * 5) (or nullptr if not needed)
    /// @param digest_count Number of Digests (or 0 if not needed)
    /// @param u32_value u32 value for Log2PaddedHeight (or 0 if not needed)
    /// @param out_data Pointer to pointer that will be set to allocated encoded data (caller must free)
    /// @param out_len Pointer to size_t that will be set to the encoded length
    /// @return 0 on success, -1 on error
    /// 
    /// Note: The caller is responsible for freeing the memory using proof_item_free_encoding
    int proof_item_encode_general(
        uint32_t discriminant,
        const uint64_t* bfield_data,
        size_t bfield_count,
        const uint64_t* xfield_data,
        size_t xfield_count,
        const uint64_t* digest_data,
        size_t digest_count,
        uint32_t u32_value,
        uint64_t** out_data,
        size_t* out_len
    );
    
    /// Free memory allocated by proof_item_encode_* functions
    /// 
    /// @param ptr Pointer to memory allocated by proof_item_encode_* function
    /// @param len Length of the array (needed for proper deallocation)
    void proof_item_free_encoding(uint64_t* ptr, size_t len);
    
    /// Encode proof stream and serialize to file entirely in Rust
    /// 
    /// This function handles steps 12 and 13 entirely in Rust:
    ///   Step 12: Proof Stream Construction (ProofStream::encode())
    ///   Step 13: Bincode Serialization (bincode::serialize_into)
    /// 
    /// @param discriminants Array of u32 discriminants (one per item)
    /// @param num_items Number of proof items
    /// @param bfield_data_array Array of pointers to bfield data (one per item, or nullptr if not needed)
    /// @param bfield_count_array Array of bfield counts (one per item)
    /// @param xfield_data_array Array of pointers to xfield data (one per item, or nullptr if not needed)
    /// @param xfield_count_array Array of xfield counts (one per item, in number of XFieldElements)
    /// @param digest_data_array Array of pointers to digest data (one per item, or nullptr if not needed)
    /// @param digest_count_array Array of digest counts (one per item, in number of Digests)
    /// @param u32_value_array Array of u32 values (one per item, for Log2PaddedHeight)
    /// @param file_path Null-terminated C string path to output file
    /// @return 0 on success, -1 on error
    int proof_stream_encode_and_serialize(
        const uint32_t* discriminants,
        size_t num_items,
        const uint64_t* const* bfield_data_array,
        const size_t* bfield_count_array,
        const uint64_t* const* xfield_data_array,
        const size_t* xfield_count_array,
        const uint64_t* const* digest_data_array,
        const size_t* digest_count_array,
        const uint32_t* u32_value_array,
        const char* file_path
    );

    /// Evaluate initial constraints using Rust implementation via FFI
    /// This is a temporary workaround for C++ constraint evaluation bugs
    ///
    /// @param main_row Array of 379 u64 values (BFieldElement coefficients)
    /// @param aux_row Array of 88*3 = 264 u64 values (XFieldElement coefficients)
    /// @param challenges Array of 59*3 = 177 u64 values (XFieldElement coefficients for challenges)
    /// @param out_constraints Pointer to pointer that will receive the allocated constraint data
    /// @param out_len Pointer to size_t that will receive the number of u64 values
    /// @return 0 on success, -1 on error
    ///
    /// The returned array contains XFieldElement coefficients flattened:
    /// [c0_0, c1_0, c2_0, c0_1, c1_1, c2_1, ...]
    /// where each constraint is represented as 3 u64 values.
    ///
    /// Caller must free the memory using constraint_evaluation_free.
    int evaluate_initial_constraints_rust(
        const uint64_t* main_row,
        const uint64_t* aux_row,
        const uint64_t* challenges,
        uint64_t** out_constraints,
        size_t* out_len
    );

    /// Evaluate consistency constraints using Rust implementation via FFI
    int evaluate_consistency_constraints_rust(
        const uint64_t* main_row,
        const uint64_t* aux_row,
        const uint64_t* challenges,
        uint64_t** out_constraints,
        size_t* out_len
    );

    /// Evaluate transition constraints using Rust implementation via FFI
    int evaluate_transition_constraints_rust(
        const uint64_t* current_main_row,
        const uint64_t* current_aux_row,
        const uint64_t* next_main_row,
        const uint64_t* next_aux_row,
        const uint64_t* challenges,
        uint64_t** out_constraints,
        size_t* out_len
    );

    /// Evaluate terminal constraints using Rust implementation via FFI
    int evaluate_terminal_constraints_rust(
        const uint64_t* main_row,
        const uint64_t* aux_row,
        const uint64_t* challenges,
        uint64_t** out_constraints,
        size_t* out_len
    );

    /// Compute out-of-domain quotient value using Rust verifier logic via FFI
    int compute_out_of_domain_quotient_rust(
        const uint64_t* main_row_curr,
        const uint64_t* aux_row_curr,
        const uint64_t* main_row_next,
        const uint64_t* aux_row_next,
        const uint64_t* challenges,
        const uint64_t* weights,
        size_t num_weights,
        uint64_t trace_domain_length,
        uint64_t trace_domain_generator_inverse,
        uint64_t out_of_domain_point_c0,
        uint64_t out_of_domain_point_c1,
        uint64_t out_of_domain_point_c2,
        uint64_t** out_quotient_value,
        size_t* out_len
    );

    /// Compute out-of-domain quotient value using Rust verifier logic with:
    /// - main rows as XFieldElement[379] (flattened as 379*3 u64)
    /// - aux rows as XFieldElement[88]  (flattened as 88*3 u64)
    /// - challenges as full 63 XFieldElements (flattened as 63*3 u64)
    int compute_out_of_domain_quotient_xfe_main_challenges63_rust(
        const uint64_t* main_row_curr_xfe,
        const uint64_t* aux_row_curr_xfe,
        const uint64_t* main_row_next_xfe,
        const uint64_t* aux_row_next_xfe,
        const uint64_t* challenges_63_xfe,
        const uint64_t* weights,
        size_t num_weights,
        uint64_t trace_domain_length,
        uint64_t trace_domain_generator_inverse,
        uint64_t out_of_domain_point_c0,
        uint64_t out_of_domain_point_c1,
        uint64_t out_of_domain_point_c2,
        uint64_t** out_quotient_value,
        size_t* out_len
    );

    /// Evaluate randomized BFE trace column at a point x (BFE) using Rust's logic.
    int eval_randomized_bfe_column_at_point_rust(
        const uint64_t* trace_values,
        size_t trace_len,
        uint64_t trace_offset,
        const uint64_t* randomizer_coeffs,
        size_t rand_len,
        uint64_t x,
        uint64_t* out_value
    );

    /// Evaluate randomized XFE aux trace column at a point x (BFE) using Rust's logic.
    int eval_randomized_xfe_column_at_point_rust(
        const uint64_t* trace_values_xfe,
        size_t trace_len,
        uint64_t trace_offset,
        const uint64_t* randomizer_coeffs_bfe,
        size_t rand_len,
        uint64_t x,
        uint64_t* out_xfe3
    );

    /// Compute OOD quotient segments from a quotient codeword using Rust interpolation + segment split.
    int compute_ood_quot_segments_from_quotient_codeword_rust(
        const uint64_t* quotient_codeword_xfe,
        size_t quotient_len,
        uint64_t quotient_offset,
        uint64_t z0,
        uint64_t z1,
        uint64_t z2,
        uint64_t** out_segments_xfe,
        size_t* out_len
    );

    /// Interpolate a coset-domain XFE codeword and evaluate at XFE point z.
    int eval_xfe_coset_codeword_at_xfe_point_rust(
        const uint64_t* codeword_xfe,
        size_t len,
        uint64_t domain_offset,
        uint64_t z0,
        uint64_t z1,
        uint64_t z2,
        uint64_t* out_xfe3
    );

    /// Compute quotient value at a base-field point x using Rust constraint evaluation.
    int compute_quotient_value_at_bfe_point_rust(
        uint64_t x,
        uint64_t trace_domain_length,
        uint64_t trace_domain_generator_inverse,
        const uint64_t* main_row_curr,
        const uint64_t* aux_row_curr,
        const uint64_t* main_row_next,
        const uint64_t* aux_row_next,
        const uint64_t* challenges_63_xfe,
        const uint64_t* weights,
        size_t num_weights,
        uint64_t* out_xfe3
    );

    /// Evaluate randomized main trace column at an XFE point z using Rust interpolation logic.
    int eval_randomized_main_column_at_xfe_point_rust(
        const uint64_t* trace_values,
        size_t trace_len,
        uint64_t trace_offset,
        const uint64_t* randomizer_coeffs,
        size_t rand_len,
        uint64_t z0,
        uint64_t z1,
        uint64_t z2,
        uint64_t* out_xfe3
    );

    /// Evaluate a randomized aux trace column (XFE codeword + XFE randomizer polynomial)
    /// at an XFE point `z` using Rust's interpolation + zerofier construction.
    ///
    /// - trace_values_xfe: [trace_len * 3] (row-major XFE codeword)
    /// - randomizer_coeffs_xfe: [rand_len * 3] (XFE polynomial coefficients, low->high)
    int eval_randomized_aux_column_at_xfe_point_rust(
        const uint64_t* trace_values_xfe,
        size_t trace_len,
        uint64_t trace_offset,
        const uint64_t* randomizer_coeffs_xfe,
        size_t rand_len,
        uint64_t z0,
        uint64_t z1,
        uint64_t z2,
        uint64_t* out_xfe3
    );

    /// Evaluate BFE codeword interpolant at XFE point z (no randomizer).
    int eval_bfe_interpolant_at_xfe_point_rust(
        const uint64_t* trace_values,
        size_t len,
        uint64_t domain_offset,
        uint64_t z0,
        uint64_t z1,
        uint64_t z2,
        uint64_t* out_xfe3
    );

    /// Debug: compute first 4 domain_over_domain_shift values and denom_inv like Rust out_of_domain_row.
    int debug_barycentric_weights_rust(
        size_t trace_len,
        uint64_t trace_offset,
        uint64_t z0,
        uint64_t z1,
        uint64_t z2,
        uint64_t** out_u64,
        size_t* out_len
    );

    /// Free memory allocated by constraint evaluation functions
    void constraint_evaluation_free(uint64_t* ptr, size_t len);
    
    /// Encode a Claim using Rust's BFieldCodec (ensures exact compatibility)
    /// 
    /// @param program_digest Pointer to 5 u64 values representing the Digest
    /// @param version Claim version (u32)
    /// @param input Pointer to array of u64 values for input (or nullptr if empty)
    /// @param input_len Number of input elements
    /// @param output Pointer to array of u64 values for output (or nullptr if empty)
    /// @param output_len Number of output elements
    /// @param out_encoded Pointer to pointer that will be set to allocated encoded data (caller must free)
    /// @param out_len Pointer to size_t that will be set to the encoded length
    /// @return 0 on success, -1 on error
    /// 
    /// Note: The caller is responsible for freeing the memory using claim_encode_free
    int claim_encode_rust(
        const uint64_t* program_digest,
        uint32_t version,
        const uint64_t* input,
        size_t input_len,
        const uint64_t* output,
        size_t output_len,
        uint64_t** out_encoded,
        size_t* out_len
    );
    
    /// Free memory allocated by claim_encode_rust
    /// 
    /// @param ptr Pointer to memory allocated by claim_encode_rust
    /// @param len Length of the array (needed for proper deallocation)
    void claim_encode_free(uint64_t* ptr, size_t len);

// Interpolate FRI last polynomial exactly like Rust (returns flattened u64 triplets).
int fri_interpolate_last_polynomial_rust(
    const uint64_t* xfield_data,
    size_t xfield_count,
    uint64_t** output_ptr,
    size_t* output_len
);
void fri_interpolate_last_polynomial_free(uint64_t* ptr, size_t len);

    /// Run Rust trace execution + build Claim + derive domains + create+pad master main table.
    /// Returns flattened padded main table (row-major), claim digest+version+output, and domain parameters.
    int tvm_trace_and_pad_main_table_from_tasm_file(
        const char* program_path,
        const uint64_t* public_input_data,
        size_t public_input_len,
        uint64_t** out_main_table_data,
        size_t* out_main_table_len,
        size_t* out_main_table_num_rows,
        size_t* out_main_table_num_cols,
        uint64_t* out_claim_program_digest_5,
        uint32_t* out_claim_version,
        uint64_t** out_claim_output_data,
        size_t* out_claim_output_len,
        uint64_t* out_trace_domain_3,
        uint64_t* out_quotient_domain_3,
        uint64_t* out_fri_domain_3,
        uint8_t* out_randomness_seed_32
    );

    /// Free buffers allocated by tvm_trace_and_pad_main_table_from_tasm_file
    void tvm_main_table_free(uint64_t* data, size_t len);
    void tvm_claim_output_free(uint64_t* data, size_t len);

    /// Verify main table creation by comparing C++ output against Rust reference implementation
    ///
    /// @param program_path Null-terminated C string path to TASM file
    /// @param public_input_data Array of u64 public input values (or nullptr if empty)
    /// @param public_input_len Number of public input elements
    /// @param randomness_seed_32 Array of 32 u8 values for randomness seed
    /// @param cpp_main_table_data C++ main table data (row-major flattened)
    /// @param num_rows Number of rows in main table
    /// @param num_cols Number of columns in main table (should be 379)
    /// @return 0 on success (verification passed), -1 on error or mismatch
    int verify_main_table_creation_rust_ffi(
        const char* program_path,
        const uint64_t* public_input_data,
        size_t public_input_len,
        const uint8_t* randomness_seed_32,
        const uint64_t* cpp_main_table_data,
        size_t num_rows,
        size_t num_cols
    );

    /// Run Rust trace execution only (faster than C++ for large programs).
    /// Returns all traces (processor + co-processor), program bwords, instruction multiplicities, and output.
    /// 
    /// @param program_path Path to TASM file
    /// @param public_input_data Array of u64 public input values
    /// @param public_input_len Number of public input elements
    /// @param out_processor_trace_data Output: flat processor trace (row-major)
    /// @param out_processor_trace_rows Output: number of rows
    /// @param out_processor_trace_cols Output: number of columns (39)
    /// @param out_program_bwords_data Output: program bwords array
    /// @param out_program_bwords_len Output: program bwords length
    /// @param out_instruction_multiplicities_data Output: instruction multiplicities array
    /// @param out_instruction_multiplicities_len Output: multiplicities length
    /// @param out_public_output_data Output: public output array
    /// @param out_public_output_len Output: public output length
    /// @param out_op_stack_trace_data Output: flat op_stack trace (row-major)
    /// @param out_op_stack_trace_rows Output: op_stack rows
    /// @param out_op_stack_trace_cols Output: op_stack cols (4)
    /// @param out_ram_trace_data Output: flat ram trace (row-major)
    /// @param out_ram_trace_rows Output: ram rows
    /// @param out_ram_trace_cols Output: ram cols (7)
    /// @param out_program_hash_trace_data Output: flat program_hash trace (row-major)
    /// @param out_program_hash_trace_rows Output: program_hash rows
    /// @param out_program_hash_trace_cols Output: program_hash cols (67)
    /// @param out_hash_trace_data Output: flat hash trace (row-major)
    /// @param out_hash_trace_rows Output: hash rows
    /// @param out_hash_trace_cols Output: hash cols (67)
    /// @param out_sponge_trace_data Output: flat sponge trace (row-major)
    /// @param out_sponge_trace_rows Output: sponge rows
    /// @param out_sponge_trace_cols Output: sponge cols (67)
    /// @param out_u32_entries_data Output: flat u32 entries [instruction, op1, op2, mult] per entry
    /// @param out_u32_entries_len Output: number of u32 entries
    /// @param out_cascade_multiplicities_data Output: flat cascade multiplicities [limb, mult] pairs
    /// @param out_cascade_multiplicities_len Output: number of cascade entries
    /// @param out_lookup_multiplicities_256 Output: lookup multiplicities array (256 u64)
    /// @param out_table_lengths_9 Output: table lengths [program, processor, op_stack, ram, jump_stack, hash, cascade, lookup, u32]
    /// @return 0 on success, -1 on error
    int tvm_trace_execution_rust_ffi(
        const char* program_path,
        const uint64_t* public_input_data,
        size_t public_input_len,
        uint64_t** out_processor_trace_data,
        size_t* out_processor_trace_rows,
        size_t* out_processor_trace_cols,
        uint64_t** out_program_bwords_data,
        size_t* out_program_bwords_len,
        uint32_t** out_instruction_multiplicities_data,
        size_t* out_instruction_multiplicities_len,
        uint64_t** out_public_output_data,
        size_t* out_public_output_len,
        uint64_t** out_op_stack_trace_data,
        size_t* out_op_stack_trace_rows,
        size_t* out_op_stack_trace_cols,
        uint64_t** out_ram_trace_data,
        size_t* out_ram_trace_rows,
        size_t* out_ram_trace_cols,
        uint64_t** out_program_hash_trace_data,
        size_t* out_program_hash_trace_rows,
        size_t* out_program_hash_trace_cols,
        uint64_t** out_hash_trace_data,
        size_t* out_hash_trace_rows,
        size_t* out_hash_trace_cols,
        uint64_t** out_sponge_trace_data,
        size_t* out_sponge_trace_rows,
        size_t* out_sponge_trace_cols,
        uint64_t** out_u32_entries_data,
        size_t* out_u32_entries_len,
        uint64_t** out_cascade_multiplicities_data,
        size_t* out_cascade_multiplicities_len,
        uint64_t* out_lookup_multiplicities_256,
        size_t* out_table_lengths_9
    );
    
    /// Execute VM trace with NonDeterminism JSON support (for Neptune programs that need RAM/secret input)
    /// Same outputs as tvm_trace_execution_rust_ffi but takes program JSON and NonDeterminism JSON
    /// @param program_json Neptune Program JSON string (null-terminated)
    /// @param nondet_json NonDeterminism JSON string (null-terminated)
    /// @param public_input_data Array of u64 public input values
    /// @param public_input_len Length of public input array
    /// @return 0 on success, -1 on error
    int tvm_trace_execution_with_nondet(
        const char* program_json,
        const char* nondet_json,
        const uint64_t* public_input_data,
        size_t public_input_len,
        uint64_t** out_processor_trace_data,
        size_t* out_processor_trace_rows,
        size_t* out_processor_trace_cols,
        uint64_t** out_program_bwords_data,
        size_t* out_program_bwords_len,
        uint32_t** out_instruction_multiplicities_data,
        size_t* out_instruction_multiplicities_len,
        uint64_t** out_public_output_data,
        size_t* out_public_output_len,
        uint64_t** out_op_stack_trace_data,
        size_t* out_op_stack_trace_rows,
        size_t* out_op_stack_trace_cols,
        uint64_t** out_ram_trace_data,
        size_t* out_ram_trace_rows,
        size_t* out_ram_trace_cols,
        uint64_t** out_program_hash_trace_data,
        size_t* out_program_hash_trace_rows,
        size_t* out_program_hash_trace_cols,
        uint64_t** out_hash_trace_data,
        size_t* out_hash_trace_rows,
        size_t* out_hash_trace_cols,
        uint64_t** out_sponge_trace_data,
        size_t* out_sponge_trace_rows,
        size_t* out_sponge_trace_cols,
        uint64_t** out_u32_entries_data,
        size_t* out_u32_entries_len,
        uint64_t** out_cascade_multiplicities_data,
        size_t* out_cascade_multiplicities_len,
        uint64_t* out_lookup_multiplicities_256,
        size_t* out_table_lengths_9
    );

    /// Free buffers allocated by tvm_trace_execution_rust_ffi
    void tvm_trace_execution_rust_ffi_free(
        uint64_t* processor_trace_data,
        size_t processor_trace_rows,
        size_t processor_trace_cols,
        uint64_t* program_bwords_data,
        size_t program_bwords_len,
        uint32_t* instruction_multiplicities_data,
        size_t instruction_multiplicities_len,
        uint64_t* public_output_data,
        size_t public_output_len,
        uint64_t* op_stack_trace_data,
        size_t op_stack_trace_rows,
        size_t op_stack_trace_cols,
        uint64_t* ram_trace_data,
        size_t ram_trace_rows,
        size_t ram_trace_cols,
        uint64_t* program_hash_trace_data,
        size_t program_hash_trace_rows,
        size_t program_hash_trace_cols,
        uint64_t* hash_trace_data,
        size_t hash_trace_rows,
        size_t hash_trace_cols,
        uint64_t* sponge_trace_data,
        size_t sponge_trace_rows,
        size_t sponge_trace_cols,
        uint64_t* u32_entries_data,
        size_t u32_entries_len,
        uint64_t* cascade_multiplicities_data,
        size_t cascade_multiplicities_len
    );
    
    // ========================================================================
    // Neptune Integration: Prove from JSON inputs
    // ========================================================================
    
    /// Prove from Neptune's JSON format and return bincode-serialized proof
    /// 
    /// Takes the same JSON inputs that Neptune sends to triton-vm-prover.
    /// 
    /// @param claim_json Null-terminated JSON string for Claim
    /// @param program_json Null-terminated JSON string for Program
    /// @param nondet_json Null-terminated JSON string for NonDeterminism
    /// @param max_log2_json Null-terminated JSON string for max log2 padded height (may be "null")
    /// @param out_proof_bincode Output: pointer to allocated proof bytes (caller must free)
    /// @param out_proof_len Output: length of proof bytes
    /// @param out_observed_log2 Output: observed log2 padded height
    /// @return 0 on success, 1 if padded height exceeds limit, -1 on error
    int tvm_prove_from_json(
        const char* claim_json,
        const char* program_json,
        const char* nondet_json,
        const char* max_log2_json,
        uint8_t** out_proof_bincode,
        size_t* out_proof_len,
        uint8_t* out_observed_log2
    );
    
    /// Free proof buffer allocated by tvm_prove_from_json
    void tvm_prove_from_json_free(
        uint8_t* proof_bincode,
        size_t proof_len
    );
}

