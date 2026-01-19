use std::os::raw::{c_char, c_int, c_ulonglong};
use std::alloc::{alloc, dealloc, Layout};
use triton_vm::prelude::*;
use triton_vm::arithmetic_domain::ArithmeticDomain;
use triton_vm::proof_item::{ProofItem, FriResponse};
use triton_vm::table::{MainRow, AuxiliaryRow, QuotientSegments};
use triton_vm::table::auxiliary_table::Evaluable;
use triton_vm::table::master_table::MasterAuxTable;
use triton_vm::table::master_table::MasterMainTable;
use triton_vm::stark::ProverDomains;
use twenty_first::prelude::{Polynomial, ModPowU32};

// FFI function for degree lowering (from degree_lowering_ffi crate)
extern "C" {
    fn degree_lowering_fill_main_columns(table_ptr: *mut u64, num_rows: usize, num_cols: usize);
}

/// FFI function to serialize a Vec<u64> to bincode format and write to file
/// 
/// This ensures the bincode format exactly matches Rust's bincode::serialize_into
/// 
/// # Safety
/// 
/// The caller must ensure:
/// - `data` points to a valid array of `len` u64 values
/// - `file_path` is a valid null-terminated C string
/// - The file path is writable
/// 
/// # Returns
/// 
/// - 0 on success
/// - -1 on error (check errno for details)
#[no_mangle]
pub unsafe extern "C" fn bincode_serialize_vec_u64_to_file(
    data: *const c_ulonglong,
    len: usize,
    file_path: *const c_char,
) -> c_int {
    if data.is_null() || file_path.is_null() {
        return -1;
    }

    // Convert C string to Rust string
    let path_cstr = match std::ffi::CStr::from_ptr(file_path).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    // Convert C array to Rust Vec<u64>
    let slice = std::slice::from_raw_parts(data, len);
    let vec_u64: Vec<u64> = slice.to_vec();
    
    // When bincode serializes Proof using serde, it serializes the tuple struct
    // as just the inner value (Vec<BFieldElement>). So we serialize Vec<u64> directly.
    // The verifier will deserialize this as Proof, then decode ProofStream from proof.0
    // using BFieldCodec, which expects the proof stream encoding (not Proof's BFieldCodec encoding).
    
    // So we just serialize the Vec<u64> directly - no BFieldCodec encoding needed!
    let proof_u64 = vec_u64;

    // Serialize to bincode and write to file
    match std::fs::File::create(path_cstr) {
        Ok(file) => {
            match bincode::serialize_into(file, &proof_u64) {
                Ok(_) => 0,
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}

/// FFI function to deserialize a Vec<u64> from bincode file
/// 
/// # Safety
/// 
/// The caller must ensure:
/// - `file_path` is a valid null-terminated C string
/// - `out_data` points to a buffer large enough to hold the deserialized data
/// - `out_len` points to a valid usize that will be set to the length
/// 
/// # Returns
/// 
/// - 0 on success
/// - -1 on error
/// 
/// Note: The caller is responsible for freeing the memory allocated for `out_data`
#[no_mangle]
pub unsafe extern "C" fn bincode_deserialize_vec_u64_from_file(
    file_path: *const c_char,
    out_data: *mut *mut c_ulonglong,
    out_len: *mut usize,
) -> c_int {
    if file_path.is_null() || out_data.is_null() || out_len.is_null() {
        return -1;
    }

    // Convert C string to Rust string
    let path_cstr = match std::ffi::CStr::from_ptr(file_path).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    // Read and deserialize from file
    match std::fs::File::open(path_cstr) {
        Ok(file) => {
            match bincode::deserialize_from::<_, Vec<u64>>(file) {
                Ok(vec) => {
                    let len = vec.len();
                    // Allocate memory for the data (caller must free this)
                    let layout = Layout::from_size_align(len * std::mem::size_of::<c_ulonglong>(), std::mem::align_of::<c_ulonglong>())
                        .unwrap_or_else(|_| Layout::new::<c_ulonglong>());
                    let data_ptr = unsafe { alloc(layout) as *mut c_ulonglong };
                    if data_ptr.is_null() {
                        return -1;
                    }
                    // Copy data
                    std::ptr::copy_nonoverlapping(vec.as_ptr() as *const c_ulonglong, data_ptr, len);
                    *out_data = data_ptr;
                    *out_len = len;
                    0
                }
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}

/// FFI function to encode FriResponse ProofItem
/// 
/// Takes auth_structure (Vec<Digest>) and revealed_leaves (Vec<XFieldElement>)
/// and returns the encoded ProofItem as Vec<u64> (BFieldElement values)
/// 
/// # Safety
/// 
/// The caller must ensure:
/// - `auth_data` points to a valid array of `num_auth * 5` u64 values (each Digest is 5 BFieldElements)
/// - `leaves_data` points to a valid array of `num_leaves * 3` u64 values (each XFieldElement is 3 BFieldElements)
/// - `out_data` points to a pointer that will be set to allocated memory (caller must free)
/// - `out_len` points to a valid usize that will be set to the length
/// 
/// # Returns
/// 
/// - 0 on success
/// - -1 on error
/// 
/// Note: The caller is responsible for freeing the memory allocated for `out_data` using proof_item_free_encoding
#[no_mangle]
pub unsafe extern "C" fn proof_item_encode_fri_response(
    auth_data: *const c_ulonglong,
    num_auth: usize,
    leaves_data: *const c_ulonglong,
    num_leaves: usize,
    out_data: *mut *mut c_ulonglong,
    out_len: *mut usize,
) -> c_int {
    if auth_data.is_null() || leaves_data.is_null() || out_data.is_null() || out_len.is_null() {
        return -1;
    }

    // Convert auth_data to Vec<Digest>
    let auth_slice = std::slice::from_raw_parts(auth_data, num_auth * 5);
    let mut auth_structure = Vec::with_capacity(num_auth);
    for i in 0..num_auth {
        let start = i * 5;
        let digest = Digest::new([
            BFieldElement::new(auth_slice[start]),
            BFieldElement::new(auth_slice[start + 1]),
            BFieldElement::new(auth_slice[start + 2]),
            BFieldElement::new(auth_slice[start + 3]),
            BFieldElement::new(auth_slice[start + 4]),
        ]);
        auth_structure.push(digest);
    }

    // Convert leaves_data to Vec<XFieldElement>
    let leaves_slice = std::slice::from_raw_parts(leaves_data, num_leaves * 3);
    let mut revealed_leaves = Vec::with_capacity(num_leaves);
    for i in 0..num_leaves {
        let start = i * 3;
        let xfe = XFieldElement::new([
            BFieldElement::new(leaves_slice[start]),
            BFieldElement::new(leaves_slice[start + 1]),
            BFieldElement::new(leaves_slice[start + 2]),
        ]);
        revealed_leaves.push(xfe);
    }

    // Create FriResponse and encode
    let fri_response = FriResponse {
        auth_structure,
        revealed_leaves,
    };
    let proof_item = ProofItem::FriResponse(fri_response);
    let encoding = proof_item.encode();
    
    // Debug: Print encoding info
    if std::env::var("TVM_DEBUG_FFI_ENCODE").is_ok() {
        eprintln!("[FFI DEBUG] FriResponse encoding:");
        eprintln!("  Length: {} elements", encoding.len());
        if !encoding.is_empty() {
            eprintln!("  [0] Discriminant: {}", encoding[0].value());
        }
        if encoding.len() > 1 {
            eprintln!("  [1] First field length: {}", encoding[1].value());
        }
        if encoding.len() > 2 {
            eprintln!("  [2] Vec length: {}", encoding[2].value());
        }
    }

    // Convert Vec<BFieldElement> to Vec<u64> and allocate
    let len = encoding.len();
    let layout = Layout::from_size_align(len * std::mem::size_of::<c_ulonglong>(), std::mem::align_of::<c_ulonglong>())
        .unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let data_ptr = unsafe { alloc(layout) as *mut c_ulonglong };
    if data_ptr.is_null() {
        return -1;
    }

    // Copy encoded data
    for (i, elem) in encoding.iter().enumerate() {
        unsafe {
            *data_ptr.add(i) = elem.value();
        }
    }

    *out_data = data_ptr;
    *out_len = len;
    0
}

/// Run Triton VM trace execution (Rust), build Claim (Rust), derive prover domains (Rust),
/// and create+pad the master main table (Rust). Return flattened padded main table + claim
/// pieces + domain parameters for use by the C++ prover.
///
/// This is intended to replace the “export JSON test data” pre-step. C++ remains the driver.
///
/// # Safety
///
/// - `program_path` must be a valid null-terminated C string pointing to a file containing tasm source.
/// - `public_input_data` must point to `public_input_len` u64s (BFieldElement values).
/// - All out pointers must be non-null.
/// - Returned buffers must be freed via the corresponding `tvm_*_free` functions.
#[no_mangle]
pub unsafe extern "C" fn tvm_trace_and_pad_main_table_from_tasm_file(
    program_path: *const c_char,
    public_input_data: *const c_ulonglong,
    public_input_len: usize,
    // main table out
    out_main_table_data: *mut *mut c_ulonglong,
    out_main_table_len: *mut usize,
    out_main_table_num_rows: *mut usize,
    out_main_table_num_cols: *mut usize,
    // claim out
    out_claim_program_digest_5: *mut c_ulonglong, // caller provides array len 5
    out_claim_version: *mut u32,
    out_claim_output_data: *mut *mut c_ulonglong,
    out_claim_output_len: *mut usize,
    // domains out: each is (length, offset, generator)
    out_trace_domain_3: *mut c_ulonglong,    // caller provides array len 3
    out_quotient_domain_3: *mut c_ulonglong, // caller provides array len 3
    out_fri_domain_3: *mut c_ulonglong,      // caller provides array len 3
    out_randomness_seed_32: *mut u8,         // caller provides array len 32
) -> c_int {
    if program_path.is_null()
        || public_input_data.is_null()
        || out_main_table_data.is_null()
        || out_main_table_len.is_null()
        || out_main_table_num_rows.is_null()
        || out_main_table_num_cols.is_null()
        || out_claim_program_digest_5.is_null()
        || out_claim_version.is_null()
        || out_claim_output_data.is_null()
        || out_claim_output_len.is_null()
        || out_trace_domain_3.is_null()
        || out_quotient_domain_3.is_null()
        || out_fri_domain_3.is_null()
        || out_randomness_seed_32.is_null()
    {
        return -1;
    }

    let path_str = match std::ffi::CStr::from_ptr(program_path).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let code = match std::fs::read_to_string(path_str) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let program = match Program::from_code(&code) {
        Ok(p) => p,
        Err(_) => return -1,
    };

    let input_slice = std::slice::from_raw_parts(public_input_data, public_input_len);
    let input_vec: Vec<BFieldElement> = input_slice
        .iter()
        .copied()
        .map(BFieldElement::new)
        .collect();
    let public_input = PublicInput::new(input_vec.clone());

    // Run VM trace execution.
    let (aet, output) = match VM::trace_execution(program, public_input, NonDeterminism::default())
    {
        Ok(res) => res,
        Err(_) => return -1,
    };

    // Build claim.
    let claim = Claim::about_program(&aet.program)
        .with_input(input_vec)
        .with_output(output.clone());

    // Derive domains. We follow the same path as `Stark::prove` does internally.
    let stark = Stark::default();
    let padded_height = aet.padded_height();
    let fri = match stark.fri(padded_height) {
        Ok(f) => f,
        Err(_) => return -1,
    };
    let domains = ProverDomains::derive(
        padded_height,
        stark.num_trace_randomizers,
        fri.domain,
        stark.max_degree(padded_height),
    );

    // Create and pad main table.
    let mut randomness_seed = [0u8; 32];
    if getrandom::getrandom(&mut randomness_seed).is_err() {
        return -1;
    }

    let mut master_main_table = MasterMainTable::new(
        &aet,
        domains,
        stark.num_trace_randomizers,
        randomness_seed,
    );
    master_main_table.pad();

    let trace_table = master_main_table.trace_table();
    let num_rows = trace_table.nrows();
    let num_cols = trace_table.ncols();
    let flat_len = num_rows * num_cols;

    // Allocate and copy flattened main table (row-major).
    let layout = Layout::from_size_align(
        flat_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    )
    .unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let data_ptr = alloc(layout) as *mut c_ulonglong;
    if data_ptr.is_null() {
        return -1;
    }

    for r in 0..num_rows {
        for c in 0..num_cols {
            let idx = r * num_cols + c;
            *data_ptr.add(idx) = trace_table[(r, c)].value();
        }
    }

    *out_main_table_data = data_ptr;
    *out_main_table_len = flat_len;
    *out_main_table_num_rows = num_rows;
    *out_main_table_num_cols = num_cols;

    // Claim: digest (5) + version + output vec
    for (i, bfe) in claim.program_digest.values().into_iter().enumerate() {
        *out_claim_program_digest_5.add(i) = bfe.value();
    }
    *out_claim_version = claim.version;

    let out_len = claim.output.len();
    let out_layout = Layout::from_size_align(
        out_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    )
    .unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let out_ptr = alloc(out_layout) as *mut c_ulonglong;
    if out_ptr.is_null() {
        dealloc(data_ptr as *mut u8, layout);
        return -1;
    }
    for (i, bfe) in claim.output.iter().enumerate() {
        *out_ptr.add(i) = bfe.value();
    }
    *out_claim_output_data = out_ptr;
    *out_claim_output_len = out_len;

    // Domains
    let d = master_main_table.domains();
    // trace
    *out_trace_domain_3.add(0) = d.trace.length as u64;
    *out_trace_domain_3.add(1) = d.trace.offset.value();
    *out_trace_domain_3.add(2) = d.trace.generator.value();
    // quotient
    *out_quotient_domain_3.add(0) = d.quotient.length as u64;
    *out_quotient_domain_3.add(1) = d.quotient.offset.value();
    *out_quotient_domain_3.add(2) = d.quotient.generator.value();
    // fri
    *out_fri_domain_3.add(0) = d.fri.length as u64;
    *out_fri_domain_3.add(1) = d.fri.offset.value();
    *out_fri_domain_3.add(2) = d.fri.generator.value();

    // randomness seed
    std::ptr::copy_nonoverlapping(randomness_seed.as_ptr(), out_randomness_seed_32, 32);

    0
}

#[no_mangle]
pub unsafe extern "C" fn tvm_main_table_free(data: *mut c_ulonglong, len: usize) {
    if data.is_null() {
        return;
    }
    let layout = Layout::from_size_align(
        len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    )
    .unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    dealloc(data as *mut u8, layout);
}

#[no_mangle]
pub unsafe extern "C" fn tvm_claim_output_free(data: *mut c_ulonglong, len: usize) {
    tvm_main_table_free(data, len);
}

/// Verify main table creation by comparing C++ output against Rust reference implementation.
/// This function replicates the exact same C++ pipeline in Rust to ensure 100% identical results.
///
/// C++ pipeline:
/// 1. Load program and run trace execution (Rust FFI)
/// 2. Derive domains using Stark::derive_domains()
/// 3. Create MasterMainTable::from_aet() equivalent
/// 4. Pad table with table_lengths
/// 5. Compute degree lowering columns (using Rust implementation when TVM_USE_RUST_DEGREE_LOWERING=1)
///
/// # Safety
///
/// The caller must ensure:
/// - `cpp_main_table_data` points to a valid array of `num_rows * num_cols` u64 values
/// - `program_path` points to a valid null-terminated C string
/// - `public_input_data` points to a valid array of `public_input_len` u64 values (or nullptr if empty)
/// - `randomness_seed_32` points to a valid array of 32 u8 values
/// - `table_lengths_9` points to a valid array of 9 usize values (table lengths)
///
/// # Returns
///
/// - 0 on success (tables match)
/// - -1 on error or mismatch
///
#[no_mangle]
pub unsafe extern "C" fn verify_main_table_creation_rust_ffi(
    program_path: *const c_char,
    public_input_data: *const c_ulonglong,
    public_input_len: usize,
    randomness_seed_32: *const u8,
    cpp_main_table_data: *const c_ulonglong,
    num_rows: usize,
    num_cols: usize,
) -> i32 {
    // All types should be available through triton_vm::prelude::*

    if program_path.is_null()
        || randomness_seed_32.is_null()
        || cpp_main_table_data.is_null()
    {
        eprintln!("[FFI ERROR] verify_main_table_creation_rust_ffi: null pointer arguments");
        return -1;
    }

    if num_rows == 0 || num_cols != 379 {
        eprintln!("[FFI ERROR] verify_main_table_creation_rust_ffi: invalid dimensions (rows={}, cols={})", num_rows, num_cols);
        return -1;
    }

    // Convert C string to Rust string
    let program_path_str = match std::ffi::CStr::from_ptr(program_path).to_str() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("[FFI ERROR] verify_main_table_creation_rust_ffi: invalid program path: {}", e);
            return -1;
        }
    };

    // Convert randomness seed (it's already a [u8; 32] array)
    let mut seed_array = [0u8; 32];
    std::ptr::copy_nonoverlapping(randomness_seed_32, seed_array.as_mut_ptr(), 32);

    // Convert public input
    let input_slice = if !public_input_data.is_null() && public_input_len > 0 {
        unsafe { std::slice::from_raw_parts(public_input_data, public_input_len) }
    } else {
        &[]
    };
    let input_vec: Vec<BFieldElement> = input_slice
        .iter()
        .copied()
        .map(BFieldElement::new)
        .collect();
    let public_input = PublicInput::new(input_vec.clone());

    // Replicate the exact C++ pipeline in Rust
    let rust_result = std::panic::catch_unwind(|| {
        // Step 1: Load program and run trace execution (same as C++)
        let code = match std::fs::read_to_string(program_path_str) {
            Ok(c) => c,
            Err(_) => return Err(Box::new(std::io::Error::new(std::io::ErrorKind::NotFound, "program file not found"))),
        };
        let program = match Program::from_code(&code) {
            Ok(p) => p,
            Err(_) => return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "program parsing failed"))),
        };

        let (aet, _output) = match VM::trace_execution(program, public_input, NonDeterminism::default()) {
            Ok(res) => res,
            Err(_) => return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "trace execution failed"))),
        };

        // Step 2: Derive domains (same as C++)
        let stark = Stark::default();
        let padded_height = aet.padded_height();
        let fri = match stark.fri(padded_height) {
            Ok(f) => f,
            Err(_) => return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "fri setup failed"))),
        };
        let domains = ProverDomains::derive(
            padded_height,
            stark.num_trace_randomizers,
            fri.domain,
            stark.max_degree(padded_height),
        );

        // Step 3: Create MasterMainTable (equivalent to C++ from_aet)
        let mut master_main_table = MasterMainTable::new(
            &aet,
            domains,
            stark.num_trace_randomizers,
            seed_array,
        );

        // Step 4: Pad table (same as C++)
        master_main_table.pad();

        // Step 5: Apply degree lowering using the same FFI that C++ uses when TVM_USE_RUST_DEGREE_LOWERING=1
        // This ensures exact match with C++ output
        let trace_table = master_main_table.trace_table();
        let (rows, cols) = trace_table.dim();

        // Convert to flat u64 array for FFI
        let mut flat_data = vec![0u64; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                flat_data[r * cols + c] = trace_table[(r, c)].value();
            }
        }

        // Call the FFI degree lowering function (same as C++)
        unsafe {
            degree_lowering_fill_main_columns(flat_data.as_mut_ptr(), rows, cols);
        }

        let trace_table = master_main_table.trace_table();
        let rust_rows = trace_table.nrows();
        let rust_cols = trace_table.ncols();

        // Check dimensions match
        if rust_rows != num_rows || rust_cols != num_cols {
            eprintln!("[FFI ERROR] verify_main_table_creation_rust_ffi: dimension mismatch!");
            eprintln!("  Expected: {} x {}", num_rows, num_cols);
            eprintln!("  Got: {} x {}", rust_rows, rust_cols);
            return Err(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, "dimension mismatch")));
        }

        // Use the flat_data after FFI processing as the final table data
        let rust_table_data = flat_data;

        Ok(rust_table_data)
    });

    let rust_table_data = match rust_result {
        Ok(Ok(data)) => data,
        Ok(Err(e)) => {
            eprintln!("[FFI ERROR] verify_main_table_creation_rust_ffi: Rust pipeline failed: {:?}", e);
            return -1;
        }
        Err(panic) => {
            eprintln!("[FFI ERROR] verify_main_table_creation_rust_ffi: Rust panicked: {:?}", panic);
            return -1;
        }
    };

    // Compare table contents element by element - require 100% exact match
    // Both C++ and Rust now use the exact same degree lowering implementation
    let mut mismatches = 0;
    let max_mismatches_to_report = 10;

    for row in 0..num_rows {
        let cpp_row_start = unsafe { cpp_main_table_data.add(row * num_cols) };

        for col in 0..num_cols {
            let idx = row * num_cols + col;
            let cpp_value = unsafe { *cpp_row_start.add(col) };
            let rust_value = rust_table_data[idx];

            if cpp_value != rust_value {
                mismatches += 1;
                if mismatches <= max_mismatches_to_report {
                    eprintln!("[FFI ERROR] verify_main_table_creation_rust_ffi: value mismatch at [{},{}]:",
                            row, col);
                    eprintln!("  C++:  0x{:016x}", cpp_value);
                    eprintln!("  Rust: 0x{:016x}", rust_value);
                }
            }
        }
    }

    if mismatches > 0 {
        eprintln!("[FFI ERROR] verify_main_table_creation_rust_ffi: {} total mismatches found!", mismatches);
        return -1;
    }

    eprintln!("[FFI SUCCESS] verify_main_table_creation_rust_ffi: main table verification passed ({} x {} elements)", num_rows, num_cols);
    0
}

/// FFI function to encode FriCodeword ProofItem
/// 
/// Takes Vec<XFieldElement> and returns the encoded ProofItem as Vec<u64> (BFieldElement values)
/// 
/// # Safety
/// 
/// The caller must ensure:
/// - `codeword_data` points to a valid array of `num_elements * 3` u64 values (each XFieldElement is 3 BFieldElements)
/// - `out_data` points to a pointer that will be set to allocated memory (caller must free)
/// - `out_len` points to a valid usize that will be set to the length
/// 
/// # Returns
/// 
/// - 0 on success
/// - -1 on error
/// 
/// Note: The caller is responsible for freeing the memory allocated for `out_data` using proof_item_free_encoding
#[no_mangle]
pub unsafe extern "C" fn proof_item_encode_fri_codeword(
    codeword_data: *const c_ulonglong,
    num_elements: usize,
    out_data: *mut *mut c_ulonglong,
    out_len: *mut usize,
) -> c_int {
    if codeword_data.is_null() || out_data.is_null() || out_len.is_null() {
        return -1;
    }

    // Convert codeword_data to Vec<XFieldElement>
    let codeword_slice = std::slice::from_raw_parts(codeword_data, num_elements * 3);
    let mut codeword = Vec::with_capacity(num_elements);
    for i in 0..num_elements {
        let start = i * 3;
        let xfe = XFieldElement::new([
            BFieldElement::new(codeword_slice[start]),
            BFieldElement::new(codeword_slice[start + 1]),
            BFieldElement::new(codeword_slice[start + 2]),
        ]);
        codeword.push(xfe);
    }

    // Create ProofItem and encode
    let proof_item = ProofItem::FriCodeword(codeword.clone());
    let encoding = proof_item.encode();
    
    // CRITICAL: Verify encoding length matches expected format
    // Format: [discriminant, vec_encoding_length, vec_encoding]
    // vec_encoding = [vec_length, ...elements] where elements = codeword.len() * 3
    // vec_encoding_length = 1 + codeword.len() * 3
    // Total = 1 + 1 + (1 + codeword.len() * 3) = 3 + codeword.len() * 3
    let expected_encoding_len = 3 + codeword.len() * 3;
    if encoding.len() != expected_encoding_len {
        eprintln!("[FFI ERROR] FriCodeword encoding length mismatch!");
        eprintln!("  Codeword length: {}", codeword.len());
        eprintln!("  Encoding length: {}", encoding.len());
        eprintln!("  Expected: {}", expected_encoding_len);
        eprintln!("  This will cause proof verification to fail!");
        // Don't return error, but log it
    }
    
    // Debug: Print encoding info
    if std::env::var("TVM_DEBUG_FFI_ENCODE").is_ok() {
        eprintln!("[FFI DEBUG] FriCodeword encoding:");
        eprintln!("  Codeword length: {} elements", codeword.len());
        eprintln!("  Encoding length: {} elements", encoding.len());
        if !encoding.is_empty() {
            eprintln!("  [0] Discriminant: {}", encoding[0].value());
        }
        if encoding.len() > 1 {
            eprintln!("  [1] Vec encoding length: {}", encoding[1].value());
            let vec_encoding_len = encoding[1].value() as usize;
            let expected_total = 1 + 1 + vec_encoding_len;
            eprintln!("  Expected total: 1 (disc) + 1 (vec_enc_len) + {} (vec_enc) = {}", vec_encoding_len, expected_total);
            if encoding.len() != expected_total {
                eprintln!("  ⚠️  MISMATCH: {} != {}", encoding.len(), expected_total);
            }
        }
        if encoding.len() > 2 {
            eprintln!("  [2] Vec length (in vec_encoding): {}", encoding[2].value());
        }
        if encoding.len() > 5 {
            eprintln!("  [3-5] First XFieldElement: [{}, {}, {}]", 
                     encoding[3].value(), encoding[4].value(), encoding[5].value());
        }
        eprintln!("  Last few elements: [{}, {}, {}]", 
                 encoding[encoding.len().saturating_sub(3)].value(),
                 encoding[encoding.len().saturating_sub(2)].value(),
                 encoding[encoding.len().saturating_sub(1)].value());
    }

    // Convert Vec<BFieldElement> to Vec<u64> and allocate
    let len = encoding.len();
    let layout = Layout::from_size_align(len * std::mem::size_of::<c_ulonglong>(), std::mem::align_of::<c_ulonglong>())
        .unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let data_ptr = unsafe { alloc(layout) as *mut c_ulonglong };
    if data_ptr.is_null() {
        return -1;
    }

    // Copy encoded data
    for (i, elem) in encoding.iter().enumerate() {
        unsafe {
            *data_ptr.add(i) = elem.value();
        }
    }

    *out_data = data_ptr;
    *out_len = len;
    0
}

/// General FFI function to encode any ProofItem
/// 
/// Takes the discriminant and raw data arrays, reconstructs the ProofItem in Rust,
/// and returns the encoded ProofItem as Vec<u64> (BFieldElement values)
/// 
/// # Safety
/// The caller must ensure:
/// - `discriminant` is a valid ProofItem variant discriminant
/// - For types requiring bfield_data: `bfield_data` points to `bfield_count` valid u64 values
/// - For types requiring xfield_data: `xfield_data` points to `xfield_count * 3` valid u64 values (each XFieldElement is 3 BFieldElements)
/// - For types requiring digest_data: `digest_data` points to `digest_count * 5` valid u64 values (each Digest is 5 BFieldElements)
/// - `out_data` points to a pointer that will be set to allocated memory (caller must free)
/// - `out_len` points to a valid usize that will be set to the length
/// 
/// # Returns
/// - 0 on success
/// - -1 on error
/// 
/// Note: The caller is responsible for freeing the memory allocated for `out_data` using proof_item_free_encoding
#[no_mangle]
pub unsafe extern "C" fn proof_item_encode_general(
    discriminant: u32,
    bfield_data: *const c_ulonglong,
    bfield_count: usize,
    xfield_data: *const c_ulonglong,
    xfield_count: usize,
    digest_data: *const c_ulonglong,
    digest_count: usize,
    u32_value: u32,
    out_data: *mut *mut c_ulonglong,
    out_len: *mut usize,
) -> c_int {
    if out_data.is_null() || out_len.is_null() {
        return -1;
    }

    // Reconstruct ProofItem based on discriminant
    let proof_item = match discriminant {
        0 => {
            // MerkleRoot(Digest)
            if digest_data.is_null() || digest_count != 1 {
                return -1;
            }
            let digest_slice = std::slice::from_raw_parts(digest_data, 5);
            let digest = Digest::new([
                BFieldElement::new(digest_slice[0]),
                BFieldElement::new(digest_slice[1]),
                BFieldElement::new(digest_slice[2]),
                BFieldElement::new(digest_slice[3]),
                BFieldElement::new(digest_slice[4]),
            ]);
            ProofItem::MerkleRoot(digest)
        }
        1 => {
            // OutOfDomainMainRow(Box<MainRow<XFieldElement>>)
            if xfield_data.is_null() || xfield_count != 379 {
                return -1;
            }
            let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
            let mut row_vec = vec![XFieldElement::new([BFieldElement::new(0), BFieldElement::new(0), BFieldElement::new(0)]); 379];
            for i in 0..xfield_count {
                let start = i * 3;
                let xfe = XFieldElement::new([
                    BFieldElement::new(xfield_slice[start]),
                    BFieldElement::new(xfield_slice[start + 1]),
                    BFieldElement::new(xfield_slice[start + 2]),
                ]);
                row_vec[i] = xfe;
            }
            let row: MainRow<XFieldElement> = row_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to MainRow"));
            ProofItem::OutOfDomainMainRow(Box::new(row))
        }
        2 => {
            // OutOfDomainAuxRow(Box<AuxiliaryRow>)
            if xfield_data.is_null() || xfield_count != 88 {
                return -1;
            }
            let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
            let mut row_vec = vec![XFieldElement::new([BFieldElement::new(0), BFieldElement::new(0), BFieldElement::new(0)]); 88];
            for i in 0..xfield_count {
                let start = i * 3;
                let xfe = XFieldElement::new([
                    BFieldElement::new(xfield_slice[start]),
                    BFieldElement::new(xfield_slice[start + 1]),
                    BFieldElement::new(xfield_slice[start + 2]),
                ]);
                row_vec[i] = xfe;
            }
            let row: AuxiliaryRow = row_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to AuxiliaryRow"));
            ProofItem::OutOfDomainAuxRow(Box::new(row))
        }
        3 => {
            // OutOfDomainQuotientSegments(QuotientSegments)
            if xfield_data.is_null() || xfield_count != 4 {
                return -1;
            }
            let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
            let mut segments_vec = vec![XFieldElement::new([BFieldElement::new(0), BFieldElement::new(0), BFieldElement::new(0)]); 4];
            for i in 0..xfield_count {
                let start = i * 3;
                let xfe = XFieldElement::new([
                    BFieldElement::new(xfield_slice[start]),
                    BFieldElement::new(xfield_slice[start + 1]),
                    BFieldElement::new(xfield_slice[start + 2]),
                ]);
                segments_vec[i] = xfe;
            }
            let segments: QuotientSegments = segments_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to QuotientSegments"));
            ProofItem::OutOfDomainQuotientSegments(segments)
        }
        4 => {
            // AuthenticationStructure(Vec<Digest>)
            if digest_data.is_null() {
                return -1;
            }
            let digest_slice = std::slice::from_raw_parts(digest_data, digest_count * 5);
            let mut digests = Vec::with_capacity(digest_count);
            for i in 0..digest_count {
                let start = i * 5;
                let digest = Digest::new([
                    BFieldElement::new(digest_slice[start]),
                    BFieldElement::new(digest_slice[start + 1]),
                    BFieldElement::new(digest_slice[start + 2]),
                    BFieldElement::new(digest_slice[start + 3]),
                    BFieldElement::new(digest_slice[start + 4]),
                ]);
                digests.push(digest);
            }
            ProofItem::AuthenticationStructure(digests)
        }
        5 => {
            // MasterMainTableRows(Vec<MainRow<BFieldElement>>)
            if bfield_data.is_null() || bfield_count % 379 != 0 {
                return -1;
            }
            let num_rows = bfield_count / 379;
            let bfield_slice = std::slice::from_raw_parts(bfield_data, bfield_count);
            let mut rows = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let start = i * 379;
                let mut row_vec = vec![BFieldElement::new(0); 379];
                for j in 0..379 {
                    row_vec[j] = BFieldElement::new(bfield_slice[start + j]);
                }
                let row: MainRow<BFieldElement> = row_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to MainRow"));
                rows.push(row);
            }
            ProofItem::MasterMainTableRows(rows)
        }
        6 => {
            // MasterAuxTableRows(Vec<AuxiliaryRow>)
            if xfield_data.is_null() || xfield_count % 88 != 0 {
                return -1;
            }
            let num_rows = xfield_count / 88;
            let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
            let mut rows = Vec::with_capacity(num_rows);
            for i in 0..num_rows {
                let mut row_vec = vec![XFieldElement::new([BFieldElement::new(0), BFieldElement::new(0), BFieldElement::new(0)]); 88];
                for j in 0..88 {
                    let start = (i * 88 + j) * 3;
                    let xfe = XFieldElement::new([
                        BFieldElement::new(xfield_slice[start]),
                        BFieldElement::new(xfield_slice[start + 1]),
                        BFieldElement::new(xfield_slice[start + 2]),
                    ]);
                    row_vec[j] = xfe;
                }
                let row: AuxiliaryRow = row_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to AuxiliaryRow"));
                rows.push(row);
            }
            ProofItem::MasterAuxTableRows(rows)
        }
        7 => {
            // Log2PaddedHeight(u32)
            ProofItem::Log2PaddedHeight(u32_value)
        }
        8 => {
            // QuotientSegmentsElements(Vec<QuotientSegments>)
            if xfield_data.is_null() || xfield_count % 4 != 0 {
                return -1;
            }
            let num_segments = xfield_count / 4;
            let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
            let mut segments_vec = Vec::with_capacity(num_segments);
            for i in 0..num_segments {
                let mut seg_vec = vec![XFieldElement::new([BFieldElement::new(0), BFieldElement::new(0), BFieldElement::new(0)]); 4];
                for j in 0..4 {
                    let start = (i * 4 + j) * 3;
                    let xfe = XFieldElement::new([
                        BFieldElement::new(xfield_slice[start]),
                        BFieldElement::new(xfield_slice[start + 1]),
                        BFieldElement::new(xfield_slice[start + 2]),
                    ]);
                    seg_vec[j] = xfe;
                }
                let segments: QuotientSegments = seg_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to QuotientSegments"));
                segments_vec.push(segments);
            }
            ProofItem::QuotientSegmentsElements(segments_vec)
        }
        9 => {
            // FriCodeword(Vec<XFieldElement>)
            if xfield_data.is_null() {
                return -1;
            }
            let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
            let mut codeword = Vec::with_capacity(xfield_count);
            for i in 0..xfield_count {
                let start = i * 3;
                let xfe = XFieldElement::new([
                    BFieldElement::new(xfield_slice[start]),
                    BFieldElement::new(xfield_slice[start + 1]),
                    BFieldElement::new(xfield_slice[start + 2]),
                ]);
                codeword.push(xfe);
            }
            ProofItem::FriCodeword(codeword)
        }
        10 => {
            // FriPolynomial(Polynomial<'static, XFieldElement>)
            if xfield_data.is_null() {
                return -1;
            }
            let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
            let mut coeffs = Vec::with_capacity(xfield_count);
            for i in 0..xfield_count {
                let start = i * 3;
                let xfe = XFieldElement::new([
                    BFieldElement::new(xfield_slice[start]),
                    BFieldElement::new(xfield_slice[start + 1]),
                    BFieldElement::new(xfield_slice[start + 2]),
                ]);
                coeffs.push(xfe);
            }
            // Polynomial::new trims trailing zeros, matching Rust's behavior
            let poly = Polynomial::new(coeffs);
            ProofItem::FriPolynomial(poly)
        }
        11 => {
            // FriResponse(FriResponse)
            if digest_data.is_null() || xfield_data.is_null() {
                return -1;
            }
            let digest_slice = std::slice::from_raw_parts(digest_data, digest_count * 5);
            let mut auth_structure = Vec::with_capacity(digest_count);
            for i in 0..digest_count {
                let start = i * 5;
                let digest = Digest::new([
                    BFieldElement::new(digest_slice[start]),
                    BFieldElement::new(digest_slice[start + 1]),
                    BFieldElement::new(digest_slice[start + 2]),
                    BFieldElement::new(digest_slice[start + 3]),
                    BFieldElement::new(digest_slice[start + 4]),
                ]);
                auth_structure.push(digest);
            }
            let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
            let mut revealed_leaves = Vec::with_capacity(xfield_count);
            for i in 0..xfield_count {
                let start = i * 3;
                let xfe = XFieldElement::new([
                    BFieldElement::new(xfield_slice[start]),
                    BFieldElement::new(xfield_slice[start + 1]),
                    BFieldElement::new(xfield_slice[start + 2]),
                ]);
                revealed_leaves.push(xfe);
            }
            let fri_response = FriResponse {
                auth_structure,
                revealed_leaves,
            };
            ProofItem::FriResponse(fri_response)
        }
        _ => {
            return -1; // Invalid discriminant
        }
    };

    // Encode the ProofItem
    let encoding = proof_item.encode();

    // Allocate memory and copy encoded data
    let len = encoding.len();
    let layout = Layout::from_size_align(len * std::mem::size_of::<c_ulonglong>(), std::mem::align_of::<c_ulonglong>())
        .unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let data_ptr = unsafe { alloc(layout) as *mut c_ulonglong };
    if data_ptr.is_null() {
        return -1;
    }

    // Copy encoded data
    for (i, elem) in encoding.iter().enumerate() {
        unsafe {
            *data_ptr.add(i) = elem.value();
        }
    }

    *out_data = data_ptr;
    *out_len = len;
    0
}

/// Free memory allocated by proof_item_encode_* functions
/// 
/// # Safety
/// 
/// The caller must ensure:
/// - `ptr` was allocated by a proof_item_encode_* function
/// - `ptr` is not null
#[no_mangle]
pub unsafe extern "C" fn proof_item_free_encoding(ptr: *mut c_ulonglong, len: usize) {
    if !ptr.is_null() && len > 0 {
        let layout = Layout::from_size_align(len * std::mem::size_of::<c_ulonglong>(), std::mem::align_of::<c_ulonglong>())
            .unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(ptr as *mut u8, layout);
    }
}

/// Free memory allocated by bincode_deserialize_vec_u64_from_file
/// 
/// # Safety
/// 
/// The caller must ensure:
/// - `ptr` was allocated by bincode_deserialize_vec_u64_from_file
/// - `ptr` is not null
#[no_mangle]
pub unsafe extern "C" fn bincode_free_vec_u64(ptr: *mut c_ulonglong, len: usize) {
    if !ptr.is_null() && len > 0 {
        let layout = Layout::from_size_align(len * std::mem::size_of::<c_ulonglong>(), std::mem::align_of::<c_ulonglong>())
            .unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(ptr as *mut u8, layout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::CString;
    use std::fs;
    use std::ptr;

    #[test]
    fn test_serialize_deserialize() {
        let test_data = vec![1u64, 2, 3, 4, 5];
        let test_path = "/tmp/test_bincode_ffi.bin";
        
        // Clean up if exists
        let _ = fs::remove_file(test_path);
        
        // Serialize
        let path_cstr = CString::new(test_path).unwrap();
        let result = unsafe {
            bincode_serialize_vec_u64_to_file(
                test_data.as_ptr() as *const c_ulonglong,
                test_data.len(),
                path_cstr.as_ptr(),
            )
        };
        assert_eq!(result, 0);
        
        // Deserialize
        let mut out_data: *mut c_ulonglong = ptr::null_mut();
        let mut out_len: usize = 0;
        let result = unsafe {
            bincode_deserialize_vec_u64_from_file(
                path_cstr.as_ptr(),
                &mut out_data,
                &mut out_len,
            )
        };
        assert_eq!(result, 0);
        assert_eq!(out_len, test_data.len());
        
        // Verify data
        unsafe {
            let slice = std::slice::from_raw_parts(out_data, out_len);
            assert_eq!(slice, test_data.as_slice());
            bincode_free_vec_u64(out_data, out_len);
        }
        
        // Clean up
        let _ = fs::remove_file(test_path);
    }
}

/// FFI function to encode proof stream and serialize to file entirely in Rust
/// 
/// Takes all proof items as raw data arrays, reconstructs ProofStream in Rust,
/// encodes it, serializes to bincode, and writes to file.
/// 
/// This ensures 100% compatibility with Rust's proof format.
/// 
/// # Safety
/// The caller must ensure:
/// - `discriminants` points to `num_items` valid u32 values
/// - For each item, the corresponding data arrays are valid (or nullptr if not needed)
/// - `file_path` is a valid null-terminated C string
/// 
/// # Returns
/// - 0 on success
/// - -1 on error
/// 
/// Note: This function handles steps 12 and 13 entirely in Rust:
///   Step 12: Proof Stream Construction (ProofStream::encode())
///   Step 13: Bincode Serialization (bincode::serialize_into)
#[no_mangle]
pub unsafe extern "C" fn proof_stream_encode_and_serialize(
    discriminants: *const u32,
    num_items: usize,
    // For each item, we need to pass its data
    // We'll use arrays of pointers and counts for each item
    bfield_data_array: *const *const c_ulonglong,
    bfield_count_array: *const usize,
    xfield_data_array: *const *const c_ulonglong,
    xfield_count_array: *const usize,
    digest_data_array: *const *const c_ulonglong,
    digest_count_array: *const usize,
    u32_value_array: *const u32,
    file_path: *const c_char,
) -> c_int {
    if discriminants.is_null() || file_path.is_null() || num_items == 0 {
        return -1;
    }

    use triton_vm::proof_stream::ProofStream;
    use triton_vm::proof::Proof;

    // Convert C string to Rust string
    let path_cstr = match std::ffi::CStr::from_ptr(file_path).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    // Read discriminants
    let disc_slice = std::slice::from_raw_parts(discriminants, num_items);

    // Reconstruct all ProofItems using the same logic as proof_item_encode_general
    let mut proof_items = Vec::with_capacity(num_items);
    
    for i in 0..num_items {
        let discriminant = disc_slice[i];
        
        // Get item data (if provided)
        let bfield_data = if !bfield_data_array.is_null() {
            let ptr_array = std::slice::from_raw_parts(bfield_data_array, num_items);
            ptr_array[i]
        } else {
            std::ptr::null()
        };
        let bfield_count = if !bfield_count_array.is_null() {
            let count_array = std::slice::from_raw_parts(bfield_count_array, num_items);
            count_array[i]
        } else {
            0
        };
        
        let xfield_data = if !xfield_data_array.is_null() {
            let ptr_array = std::slice::from_raw_parts(xfield_data_array, num_items);
            ptr_array[i]
        } else {
            std::ptr::null()
        };
        let xfield_count = if !xfield_count_array.is_null() {
            let count_array = std::slice::from_raw_parts(xfield_count_array, num_items);
            count_array[i]
        } else {
            0
        };
        
        let digest_data = if !digest_data_array.is_null() {
            let ptr_array = std::slice::from_raw_parts(digest_data_array, num_items);
            ptr_array[i]
        } else {
            std::ptr::null()
        };
        let digest_count = if !digest_count_array.is_null() {
            let count_array = std::slice::from_raw_parts(digest_count_array, num_items);
            count_array[i]
        } else {
            0
        };
        
        let u32_value = if !u32_value_array.is_null() {
            let value_array = std::slice::from_raw_parts(u32_value_array, num_items);
            value_array[i]
        } else {
            0
        };

        // Reconstruct ProofItem - reuse the logic from proof_item_encode_general
        // We'll call a helper function to avoid code duplication
        let proof_item = match discriminant {
            0 => {
                // MerkleRoot(Digest)
                if digest_data.is_null() || digest_count != 1 {
                    return -1;
                }
                let digest_slice = std::slice::from_raw_parts(digest_data, 5);
                let digest = Digest::new([
                    BFieldElement::new(digest_slice[0]),
                    BFieldElement::new(digest_slice[1]),
                    BFieldElement::new(digest_slice[2]),
                    BFieldElement::new(digest_slice[3]),
                    BFieldElement::new(digest_slice[4]),
                ]);
                ProofItem::MerkleRoot(digest)
            }
            1 => {
                // OutOfDomainMainRow(Box<MainRow<XFieldElement>>)
                if xfield_data.is_null() || xfield_count != 379 {
                    return -1;
                }
                let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
                let mut row_vec = vec![XFieldElement::new([BFieldElement::new(0), BFieldElement::new(0), BFieldElement::new(0)]); 379];
                for j in 0..xfield_count {
                    let start = j * 3;
                    let xfe = XFieldElement::new([
                        BFieldElement::new(xfield_slice[start]),
                        BFieldElement::new(xfield_slice[start + 1]),
                        BFieldElement::new(xfield_slice[start + 2]),
                    ]);
                    row_vec[j] = xfe;
                }
                let row: MainRow<XFieldElement> = row_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to MainRow"));
                ProofItem::OutOfDomainMainRow(Box::new(row))
            }
            2 => {
                // OutOfDomainAuxRow(Box<AuxiliaryRow>)
                if xfield_data.is_null() || xfield_count != 88 {
                    return -1;
                }
                let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
                let mut row_vec = vec![XFieldElement::new([BFieldElement::new(0), BFieldElement::new(0), BFieldElement::new(0)]); 88];
                for j in 0..xfield_count {
                    let start = j * 3;
                    let xfe = XFieldElement::new([
                        BFieldElement::new(xfield_slice[start]),
                        BFieldElement::new(xfield_slice[start + 1]),
                        BFieldElement::new(xfield_slice[start + 2]),
                    ]);
                    row_vec[j] = xfe;
                }
                let row: AuxiliaryRow = row_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to AuxiliaryRow"));
                ProofItem::OutOfDomainAuxRow(Box::new(row))
            }
            3 => {
                // OutOfDomainQuotientSegments(QuotientSegments)
                if xfield_data.is_null() || xfield_count != 4 {
                    return -1;
                }
                let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
                let mut segments_vec = vec![XFieldElement::new([BFieldElement::new(0), BFieldElement::new(0), BFieldElement::new(0)]); 4];
                for j in 0..xfield_count {
                    let start = j * 3;
                    let xfe = XFieldElement::new([
                        BFieldElement::new(xfield_slice[start]),
                        BFieldElement::new(xfield_slice[start + 1]),
                        BFieldElement::new(xfield_slice[start + 2]),
                    ]);
                    segments_vec[j] = xfe;
                }
                let segments: QuotientSegments = segments_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to QuotientSegments"));
                ProofItem::OutOfDomainQuotientSegments(segments)
            }
            4 => {
                // AuthenticationStructure(Vec<Digest>)
                if digest_data.is_null() {
                    return -1;
                }
                let digest_slice = std::slice::from_raw_parts(digest_data, digest_count * 5);
                let mut digests = Vec::with_capacity(digest_count);
                for j in 0..digest_count {
                    let start = j * 5;
                    let digest = Digest::new([
                        BFieldElement::new(digest_slice[start]),
                        BFieldElement::new(digest_slice[start + 1]),
                        BFieldElement::new(digest_slice[start + 2]),
                        BFieldElement::new(digest_slice[start + 3]),
                        BFieldElement::new(digest_slice[start + 4]),
                    ]);
                    digests.push(digest);
                }
                ProofItem::AuthenticationStructure(digests)
            }
            5 => {
                // MasterMainTableRows(Vec<MainRow<BFieldElement>>)
                if bfield_data.is_null() || bfield_count % 379 != 0 {
                    return -1;
                }
                let num_rows = bfield_count / 379;
                let bfield_slice = std::slice::from_raw_parts(bfield_data, bfield_count);
                let mut rows = Vec::with_capacity(num_rows);
                for j in 0..num_rows {
                    let start = j * 379;
                    let mut row_vec = vec![BFieldElement::new(0); 379];
                    for k in 0..379 {
                        row_vec[k] = BFieldElement::new(bfield_slice[start + k]);
                    }
                    let row: MainRow<BFieldElement> = row_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to MainRow"));
                    rows.push(row);
                }
                ProofItem::MasterMainTableRows(rows)
            }
            6 => {
                // MasterAuxTableRows(Vec<AuxiliaryRow>)
                if xfield_data.is_null() || xfield_count % 88 != 0 {
                    return -1;
                }
                let num_rows = xfield_count / 88;
                let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
                let mut rows = Vec::with_capacity(num_rows);
                for j in 0..num_rows {
                    let mut row_vec = vec![XFieldElement::new([BFieldElement::new(0), BFieldElement::new(0), BFieldElement::new(0)]); 88];
                    for k in 0..88 {
                        let start = (j * 88 + k) * 3;
                        let xfe = XFieldElement::new([
                            BFieldElement::new(xfield_slice[start]),
                            BFieldElement::new(xfield_slice[start + 1]),
                            BFieldElement::new(xfield_slice[start + 2]),
                        ]);
                        row_vec[k] = xfe;
                    }
                    let row: AuxiliaryRow = row_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to AuxiliaryRow"));
                    rows.push(row);
                }
                ProofItem::MasterAuxTableRows(rows)
            }
            7 => {
                // Log2PaddedHeight(u32)
                ProofItem::Log2PaddedHeight(u32_value)
            }
            8 => {
                // QuotientSegmentsElements(Vec<QuotientSegments>)
                if xfield_data.is_null() || xfield_count % 4 != 0 {
                    return -1;
                }
                let num_segments = xfield_count / 4;
                let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
                let mut segments_vec = Vec::with_capacity(num_segments);
                for j in 0..num_segments {
                    let mut seg_vec = vec![XFieldElement::new([BFieldElement::new(0), BFieldElement::new(0), BFieldElement::new(0)]); 4];
                    for k in 0..4 {
                        let start = (j * 4 + k) * 3;
                        let xfe = XFieldElement::new([
                            BFieldElement::new(xfield_slice[start]),
                            BFieldElement::new(xfield_slice[start + 1]),
                            BFieldElement::new(xfield_slice[start + 2]),
                        ]);
                        seg_vec[k] = xfe;
                    }
                    let segments: QuotientSegments = seg_vec.try_into().unwrap_or_else(|_| panic!("Failed to convert Vec to QuotientSegments"));
                    segments_vec.push(segments);
                }
                ProofItem::QuotientSegmentsElements(segments_vec)
            }
            9 => {
                // FriCodeword(Vec<XFieldElement>)
                if xfield_data.is_null() {
                    return -1;
                }
                let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
                let mut codeword = Vec::with_capacity(xfield_count);
                for j in 0..xfield_count {
                    let start = j * 3;
                    let xfe = XFieldElement::new([
                        BFieldElement::new(xfield_slice[start]),
                        BFieldElement::new(xfield_slice[start + 1]),
                        BFieldElement::new(xfield_slice[start + 2]),
                    ]);
                    codeword.push(xfe);
                }
                ProofItem::FriCodeword(codeword)
            }
            10 => {
                // FriPolynomial(Polynomial<'static, XFieldElement>)
                if xfield_data.is_null() {
                    return -1;
                }
                let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
                let mut coeffs = Vec::with_capacity(xfield_count);
                for j in 0..xfield_count {
                    let start = j * 3;
                    let xfe = XFieldElement::new([
                        BFieldElement::new(xfield_slice[start]),
                        BFieldElement::new(xfield_slice[start + 1]),
                        BFieldElement::new(xfield_slice[start + 2]),
                    ]);
                    coeffs.push(xfe);
                }
                let poly = Polynomial::new(coeffs);
                ProofItem::FriPolynomial(poly)
            }
            11 => {
                // FriResponse(FriResponse)
                if digest_data.is_null() || xfield_data.is_null() {
                    return -1;
                }
                let digest_slice = std::slice::from_raw_parts(digest_data, digest_count * 5);
                let mut auth_structure = Vec::with_capacity(digest_count);
                for j in 0..digest_count {
                    let start = j * 5;
                    let digest = Digest::new([
                        BFieldElement::new(digest_slice[start]),
                        BFieldElement::new(digest_slice[start + 1]),
                        BFieldElement::new(digest_slice[start + 2]),
                        BFieldElement::new(digest_slice[start + 3]),
                        BFieldElement::new(digest_slice[start + 4]),
                    ]);
                    auth_structure.push(digest);
                }
                let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
                let mut revealed_leaves = Vec::with_capacity(xfield_count);
                for j in 0..xfield_count {
                    let start = j * 3;
                    let xfe = XFieldElement::new([
                        BFieldElement::new(xfield_slice[start]),
                        BFieldElement::new(xfield_slice[start + 1]),
                        BFieldElement::new(xfield_slice[start + 2]),
                    ]);
                    revealed_leaves.push(xfe);
                }
                let fri_response = FriResponse {
                    auth_structure,
                    revealed_leaves,
                };
                ProofItem::FriResponse(fri_response)
            }
            _ => {
                return -1; // Invalid discriminant
            }
        };
        
        proof_items.push(proof_item);
    }

    // Create ProofStream and encode
    let mut proof_stream = ProofStream::new();
    proof_stream.items = proof_items;
    
    // Encode ProofStream to Vec<BFieldElement> and convert to Proof
    let proof: Proof = (&proof_stream).into();
    
    // Serialize to bincode and write to file
    match std::fs::File::create(path_cstr) {
        Ok(file) => {
            match bincode::serialize_into(file, &proof) {
                Ok(_) => 0,
                Err(_) => -1,
            }
        }
        Err(_) => -1,
    }
}

/// FFI: Interpolate a FRI last polynomial exactly like Rust.
///
/// Given `last_codeword` as `xfield_count` XFieldElements (flattened as u64 triplets),
/// compute `ArithmeticDomain::of_length(xfield_count).interpolate(&last_codeword)`
/// and return the trimmed coefficient vector (as u64 triplets).
///
/// # Returns
/// - 0 on success
/// - -1 on error
#[no_mangle]
pub unsafe extern "C" fn fri_interpolate_last_polynomial_rust(
    xfield_data: *const c_ulonglong,
    xfield_count: usize,
    output_ptr: *mut *mut c_ulonglong,
    output_len: *mut usize,
) -> c_int {
    if xfield_data.is_null() || output_ptr.is_null() || output_len.is_null() {
        return -1;
    }
    if xfield_count == 0 {
        *output_ptr = std::ptr::null_mut();
        *output_len = 0;
        return 0;
    }

    // Reconstruct codeword
    let xfield_slice = std::slice::from_raw_parts(xfield_data, xfield_count * 3);
    let mut codeword = Vec::with_capacity(xfield_count);
    for i in 0..xfield_count {
        let start = i * 3;
        let xfe = XFieldElement::new([
            BFieldElement::new(xfield_slice[start]),
            BFieldElement::new(xfield_slice[start + 1]),
            BFieldElement::new(xfield_slice[start + 2]),
        ]);
        codeword.push(xfe);
    }

    // Interpolate exactly like Rust FRI prover
    let domain = ArithmeticDomain::of_length(xfield_count).unwrap();
    let poly = domain.interpolate(&codeword);
    let coeffs = poly.coefficients(); // already trimmed

    // Flatten to u64 triplets
    let mut flat: Vec<c_ulonglong> = Vec::with_capacity(coeffs.len() * 3);
    for c in coeffs {
        flat.push(c.coefficients[0].value());
        flat.push(c.coefficients[1].value());
        flat.push(c.coefficients[2].value());
    }

    // Allocate and return
    let len = flat.len();
    if len == 0 {
        *output_ptr = std::ptr::null_mut();
        *output_len = 0;
        return 0;
    }
    let layout = Layout::array::<c_ulonglong>(len).unwrap();
    let ptr = alloc(layout) as *mut c_ulonglong;
    if ptr.is_null() {
        return -1;
    }
    std::ptr::copy_nonoverlapping(flat.as_ptr(), ptr, len);
    *output_ptr = ptr;
    *output_len = len;
    0
}

#[no_mangle]
pub unsafe extern "C" fn fri_interpolate_last_polynomial_free(ptr: *mut c_ulonglong, len: usize) {
    if ptr.is_null() || len == 0 {
        return;
    }
    let layout = Layout::array::<c_ulonglong>(len).unwrap();
    dealloc(ptr as *mut u8, layout);
}

/// FFI function to evaluate initial constraints using Rust implementation
/// Returns a pointer to an array of XFieldElement values
/// The caller must call constraint_evaluation_free to free the memory
#[no_mangle]
pub extern "C" fn evaluate_initial_constraints_rust(
    main_row: *const u64,
    aux_row: *const u64,
    challenges: *const u64,
    out_constraints: *mut *mut u64,
    out_len: *mut usize,
) -> c_int {
    if main_row.is_null() || aux_row.is_null() || challenges.is_null()
        || out_constraints.is_null() || out_len.is_null() {
        return -1;
    }

    unsafe {
        // Convert C arrays to Rust types
        // Hardcode the column counts: Main table has 379 columns, Aux table has 88 columns
        let main_row_slice = std::slice::from_raw_parts(main_row, 379);
        let aux_row_slice = std::slice::from_raw_parts(aux_row, 88 * 3); // 3 coefficients per XFieldElement

        // Convert to proper types
        let main_bfes: Vec<BFieldElement> = main_row_slice.iter().map(|&x| bfe!(x)).collect();
        let main_row: [BFieldElement; 379] = main_bfes.try_into().unwrap();

        let aux_xfes: Vec<XFieldElement> = (0..88)
            .map(|i| {
                let c0 = aux_row_slice[i * 3];
                let c1 = aux_row_slice[i * 3 + 1];
                let c2 = aux_row_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let aux_row: [XFieldElement; 88] = aux_xfes.try_into().unwrap();

        // Convert challenges - expect flattened array of XFieldElement coefficients
        let challenges_slice = std::slice::from_raw_parts(challenges, 59 * 3); // SAMPLE_COUNT * 3 coefficients
        let challenges_scalars: Vec<XFieldElement> = (0..59)
            .map(|i| {
                let c0 = challenges_slice[i * 3];
                let c1 = challenges_slice[i * 3 + 1];
                let c2 = challenges_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();

        // Create dummy claim for challenges (we only care about sampled challenges)
        let dummy_claim = triton_vm::proof::Claim {
            version: 0,
            program_digest: Default::default(),
            input: vec![],
            output: vec![],
        };
        let challenges = triton_vm::challenges::Challenges::new(challenges_scalars, &dummy_claim);

        // Evaluate constraints
        let constraints = triton_vm::table::master_table::MasterAuxTable::evaluate_initial_constraints(
            main_row.as_slice().into(),
            aux_row.as_slice().into(),
            &challenges,
        );


        // Convert to flat array (each XFieldElement becomes 3 u64s)
        let mut flat_constraints: Vec<u64> = Vec::new();
        for constraint in constraints {
            flat_constraints.push(constraint.coefficients[0].value());
            flat_constraints.push(constraint.coefficients[1].value());
            flat_constraints.push(constraint.coefficients[2].value());
        }

        // Allocate memory for the result
        let layout = Layout::array::<u64>(flat_constraints.len()).unwrap();
        let ptr = alloc(layout) as *mut u64;
        if ptr.is_null() {
            return -1;
        }

        // Copy data
        for (i, &val) in flat_constraints.iter().enumerate() {
            *ptr.add(i) = val;
        }

        *out_constraints = ptr;
        *out_len = flat_constraints.len();
        0
    }
}

/// FFI function to evaluate consistency constraints using Rust implementation
#[no_mangle]
pub extern "C" fn evaluate_consistency_constraints_rust(
    main_row: *const u64,
    aux_row: *const u64,
    challenges: *const u64,
    out_constraints: *mut *mut u64,
    out_len: *mut usize,
) -> c_int {
    if main_row.is_null() || aux_row.is_null() || challenges.is_null()
        || out_constraints.is_null() || out_len.is_null() {
        return -1;
    }

    unsafe {
        // Convert C arrays to Rust types
        let main_row_slice = std::slice::from_raw_parts(main_row, 379);
        let aux_row_slice = std::slice::from_raw_parts(aux_row, 88 * 3);

        let main_bfes: Vec<BFieldElement> = main_row_slice.iter().map(|&x| bfe!(x)).collect();
        let main_row: [BFieldElement; 379] = main_bfes.try_into().unwrap();

        let aux_xfes: Vec<XFieldElement> = (0..88)
            .map(|i| {
                let c0 = aux_row_slice[i * 3];
                let c1 = aux_row_slice[i * 3 + 1];
                let c2 = aux_row_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let aux_row: [XFieldElement; 88] = aux_xfes.try_into().unwrap();

        // Convert challenges
        let challenges_slice = std::slice::from_raw_parts(challenges, 59 * 3);
        let challenges_scalars: Vec<XFieldElement> = (0..59)
            .map(|i| {
                let c0 = challenges_slice[i * 3];
                let c1 = challenges_slice[i * 3 + 1];
                let c2 = challenges_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();

        let dummy_claim = triton_vm::proof::Claim {
            version: 0,
            program_digest: Default::default(),
            input: vec![],
            output: vec![],
        };
        let challenges = triton_vm::challenges::Challenges::new(challenges_scalars, &dummy_claim);

        // Evaluate constraints
        let constraints = triton_vm::table::master_table::MasterAuxTable::evaluate_consistency_constraints(
            main_row.as_slice().into(),
            aux_row.as_slice().into(),
            &challenges,
        );

        // Convert to flat array
        let mut flat_constraints: Vec<u64> = Vec::new();
        for constraint in constraints {
            flat_constraints.push(constraint.coefficients[0].value());
            flat_constraints.push(constraint.coefficients[1].value());
            flat_constraints.push(constraint.coefficients[2].value());
        }

        // Allocate memory
        let layout = Layout::array::<u64>(flat_constraints.len()).unwrap();
        let ptr = alloc(layout) as *mut u64;
        if ptr.is_null() {
            return -1;
        }

        // Copy data
        for (i, &val) in flat_constraints.iter().enumerate() {
            *ptr.add(i) = val;
        }

        *out_constraints = ptr;
        *out_len = flat_constraints.len();
        0
    }
}

/// FFI function to evaluate transition constraints using Rust implementation
#[no_mangle]
pub extern "C" fn evaluate_transition_constraints_rust(
    current_main_row: *const u64,
    current_aux_row: *const u64,
    next_main_row: *const u64,
    next_aux_row: *const u64,
    challenges: *const u64,
    out_constraints: *mut *mut u64,
    out_len: *mut usize,
) -> c_int {
    if current_main_row.is_null() || current_aux_row.is_null() ||
        next_main_row.is_null() || next_aux_row.is_null() || challenges.is_null()
        || out_constraints.is_null() || out_len.is_null() {
        return -1;
    }

    unsafe {
        // Convert C arrays to Rust types
        let current_main_slice = std::slice::from_raw_parts(current_main_row, 379);
        let current_aux_slice = std::slice::from_raw_parts(current_aux_row, 88 * 3);
        let next_main_slice = std::slice::from_raw_parts(next_main_row, 379);
        let next_aux_slice = std::slice::from_raw_parts(next_aux_row, 88 * 3);

        let current_main_bfes: Vec<BFieldElement> = current_main_slice.iter().map(|&x| bfe!(x)).collect();
        let current_main_row: [BFieldElement; 379] = current_main_bfes.try_into().unwrap();

        let current_aux_xfes: Vec<XFieldElement> = (0..88)
            .map(|i| {
                let c0 = current_aux_slice[i * 3];
                let c1 = current_aux_slice[i * 3 + 1];
                let c2 = current_aux_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let current_aux_row: [XFieldElement; 88] = current_aux_xfes.try_into().unwrap();

        let next_main_bfes: Vec<BFieldElement> = next_main_slice.iter().map(|&x| bfe!(x)).collect();
        let next_main_row: [BFieldElement; 379] = next_main_bfes.try_into().unwrap();

        let next_aux_xfes: Vec<XFieldElement> = (0..88)
            .map(|i| {
                let c0 = next_aux_slice[i * 3];
                let c1 = next_aux_slice[i * 3 + 1];
                let c2 = next_aux_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let next_aux_row: [XFieldElement; 88] = next_aux_xfes.try_into().unwrap();

        // Convert challenges
        let challenges_slice = std::slice::from_raw_parts(challenges, 59 * 3);
        let challenges_scalars: Vec<XFieldElement> = (0..59)
            .map(|i| {
                let c0 = challenges_slice[i * 3];
                let c1 = challenges_slice[i * 3 + 1];
                let c2 = challenges_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();

        let dummy_claim = triton_vm::proof::Claim {
            version: 0,
            program_digest: Default::default(),
            input: vec![],
            output: vec![],
        };
        let challenges = triton_vm::challenges::Challenges::new(challenges_scalars, &dummy_claim);

        // Evaluate constraints
        let constraints = triton_vm::table::master_table::MasterAuxTable::evaluate_transition_constraints(
            current_main_row.as_slice().into(),
            current_aux_row.as_slice().into(),
            next_main_row.as_slice().into(),
            next_aux_row.as_slice().into(),
            &challenges,
        );

        // Convert to flat array
        let mut flat_constraints: Vec<u64> = Vec::new();
        for constraint in constraints {
            flat_constraints.push(constraint.coefficients[0].value());
            flat_constraints.push(constraint.coefficients[1].value());
            flat_constraints.push(constraint.coefficients[2].value());
        }

        // Allocate memory
        let layout = Layout::array::<u64>(flat_constraints.len()).unwrap();
        let ptr = alloc(layout) as *mut u64;
        if ptr.is_null() {
            return -1;
        }

        // Copy data
        for (i, &val) in flat_constraints.iter().enumerate() {
            *ptr.add(i) = val;
        }

        *out_constraints = ptr;
        *out_len = flat_constraints.len();
        0
    }
}

/// FFI function to evaluate terminal constraints using Rust implementation
#[no_mangle]
pub extern "C" fn evaluate_terminal_constraints_rust(
    main_row: *const u64,
    aux_row: *const u64,
    challenges: *const u64,
    out_constraints: *mut *mut u64,
    out_len: *mut usize,
) -> c_int {
    if main_row.is_null() || aux_row.is_null() || challenges.is_null()
        || out_constraints.is_null() || out_len.is_null() {
        return -1;
    }

    unsafe {
        // Convert C arrays to Rust types
        let main_row_slice = std::slice::from_raw_parts(main_row, 379);
        let aux_row_slice = std::slice::from_raw_parts(aux_row, 88 * 3);

        let main_bfes: Vec<BFieldElement> = main_row_slice.iter().map(|&x| bfe!(x)).collect();
        let main_row: [BFieldElement; 379] = main_bfes.try_into().unwrap();

        let aux_xfes: Vec<XFieldElement> = (0..88)
            .map(|i| {
                let c0 = aux_row_slice[i * 3];
                let c1 = aux_row_slice[i * 3 + 1];
                let c2 = aux_row_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let aux_row: [XFieldElement; 88] = aux_xfes.try_into().unwrap();

        // Convert challenges
        let challenges_slice = std::slice::from_raw_parts(challenges, 59 * 3);
        let challenges_scalars: Vec<XFieldElement> = (0..59)
            .map(|i| {
                let c0 = challenges_slice[i * 3];
                let c1 = challenges_slice[i * 3 + 1];
                let c2 = challenges_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();

        let dummy_claim = triton_vm::proof::Claim {
            version: 0,
            program_digest: Default::default(),
            input: vec![],
            output: vec![],
        };
        let challenges = triton_vm::challenges::Challenges::new(challenges_scalars, &dummy_claim);

        // Evaluate constraints
        let constraints = triton_vm::table::master_table::MasterAuxTable::evaluate_terminal_constraints(
            main_row.as_slice().into(),
            aux_row.as_slice().into(),
            &challenges,
        );

        // Convert to flat array
        let mut flat_constraints: Vec<u64> = Vec::new();
        for constraint in constraints {
            flat_constraints.push(constraint.coefficients[0].value());
            flat_constraints.push(constraint.coefficients[1].value());
            flat_constraints.push(constraint.coefficients[2].value());
        }

        // Allocate memory
        let layout = Layout::array::<u64>(flat_constraints.len()).unwrap();
        let ptr = alloc(layout) as *mut u64;
        if ptr.is_null() {
            return -1;
        }

        // Copy data
        for (i, &val) in flat_constraints.iter().enumerate() {
            *ptr.add(i) = val;
        }

        *out_constraints = ptr;
        *out_len = flat_constraints.len();
        0
    }
}

/// FFI function to compute out-of-domain quotient value using Rust verifier logic
#[no_mangle]
pub extern "C" fn compute_out_of_domain_quotient_rust(
    main_row_curr: *const u64,
    aux_row_curr: *const u64,
    main_row_next: *const u64,
    aux_row_next: *const u64,
    challenges: *const u64,
    weights: *const u64,  // XFieldElement coefficients flattened
    num_weights: usize,
    trace_domain_length: u64,
    trace_domain_generator_inverse: u64,
    out_of_domain_point_c0: u64,
    out_of_domain_point_c1: u64,
    out_of_domain_point_c2: u64,
    out_quotient_value: *mut *mut u64,
    out_len: *mut usize,
) -> c_int {
    if main_row_curr.is_null() || aux_row_curr.is_null() || main_row_next.is_null() ||
        aux_row_next.is_null() || challenges.is_null() || weights.is_null() ||
        out_quotient_value.is_null() || out_len.is_null() {
        return -1;
    }

    unsafe {
        // Convert inputs
        let main_curr_bfes: Vec<BFieldElement> = std::slice::from_raw_parts(main_row_curr, 379).iter().map(|&x| bfe!(x)).collect();
        let main_curr_row: [BFieldElement; 379] = main_curr_bfes.try_into().unwrap();

        let aux_curr_xfes: Vec<XFieldElement> = (0..88)
            .map(|i| {
                let c0 = *aux_row_curr.add(i * 3);
                let c1 = *aux_row_curr.add(i * 3 + 1);
                let c2 = *aux_row_curr.add(i * 3 + 2);
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let aux_curr_row: [XFieldElement; 88] = aux_curr_xfes.try_into().unwrap();

        let main_next_bfes: Vec<BFieldElement> = std::slice::from_raw_parts(main_row_next, 379).iter().map(|&x| bfe!(x)).collect();
        let main_next_row: [BFieldElement; 379] = main_next_bfes.try_into().unwrap();

        let aux_next_xfes: Vec<XFieldElement> = (0..88)
            .map(|i| {
                let c0 = *aux_row_next.add(i * 3);
                let c1 = *aux_row_next.add(i * 3 + 1);
                let c2 = *aux_row_next.add(i * 3 + 2);
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let aux_next_row: [XFieldElement; 88] = aux_next_xfes.try_into().unwrap();

        let challenges_slice = std::slice::from_raw_parts(challenges, 59 * 3);
        let challenges_scalars: Vec<XFieldElement> = (0..59)
            .map(|i| {
                let c0 = challenges_slice[i * 3];
                let c1 = challenges_slice[i * 3 + 1];
                let c2 = challenges_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();

        let dummy_claim = triton_vm::proof::Claim {
            version: 0,
            program_digest: Default::default(),
            input: vec![],
            output: vec![],
        };
        let challenges = triton_vm::challenges::Challenges::new(challenges_scalars, &dummy_claim);

        let weights_slice = std::slice::from_raw_parts(weights, num_weights * 3);
        let quot_codeword_weights: Vec<XFieldElement> = (0..num_weights)
            .map(|i| {
                let c0 = weights_slice[i * 3];
                let c1 = weights_slice[i * 3 + 1];
                let c2 = weights_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();

        let out_of_domain_point = XFieldElement::new([
            bfe!(out_of_domain_point_c0),
            bfe!(out_of_domain_point_c1),
            bfe!(out_of_domain_point_c2)
        ]);

        let trace_domain_generator_inv = bfe!(trace_domain_generator_inverse);

        // Compute zerofier inverses (exactly like Rust verifier)
        let initial_zerofier_inv = (out_of_domain_point - bfe!(1)).inverse();
        let consistency_zerofier_inv = (out_of_domain_point.mod_pow_u32(trace_domain_length as u32) - bfe!(1)).inverse();
        let except_last_row = out_of_domain_point - trace_domain_generator_inv;
        let transition_zerofier_inv = except_last_row * consistency_zerofier_inv;
        let terminal_zerofier_inv = except_last_row.inverse();

        // Evaluate constraints (exactly like Rust verifier)
        let evaluated_initial_constraints = MasterAuxTable::evaluate_initial_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            &challenges,
        );
        let evaluated_consistency_constraints = MasterAuxTable::evaluate_consistency_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            &challenges,
        );
        let evaluated_transition_constraints = MasterAuxTable::evaluate_transition_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            main_next_row.as_slice().into(),
            aux_next_row.as_slice().into(),
            &challenges,
        );
        let evaluated_terminal_constraints = MasterAuxTable::evaluate_terminal_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            &challenges,
        );

        // Divide by zerofiers (exactly like Rust verifier)
        let divide = |constraints: Vec<_>, z_inv| constraints.into_iter().map(move |c| c * z_inv);
        let initial_quotients = divide(evaluated_initial_constraints, initial_zerofier_inv);
        let consistency_quotients = divide(evaluated_consistency_constraints, consistency_zerofier_inv);
        let transition_quotients = divide(evaluated_transition_constraints, transition_zerofier_inv);
        let terminal_quotients = divide(evaluated_terminal_constraints, terminal_zerofier_inv);

        let quotient_summands = initial_quotients
            .chain(consistency_quotients)
            .chain(transition_quotients)
            .chain(terminal_quotients)
            .collect::<Vec<_>>();

        // Inner product with weights (exactly like Rust verifier)
        let out_of_domain_quotient_value = quotient_summands.iter()
            .zip(quot_codeword_weights.iter())
            .map(|(constraint, weight)| *constraint * *weight)
            .sum::<XFieldElement>();

        // Return the result
        let layout = Layout::array::<u64>(3).unwrap();
        let ptr = alloc(layout) as *mut u64;
        if ptr.is_null() {
            return -1;
        }

        *ptr = out_of_domain_quotient_value.coefficients[0].value();
        *ptr.add(1) = out_of_domain_quotient_value.coefficients[1].value();
        *ptr.add(2) = out_of_domain_quotient_value.coefficients[2].value();

        *out_quotient_value = ptr;
        *out_len = 3;
        0
    }
}

/// FFI function to compute out-of-domain quotient value using Rust verifier logic,
/// with XFieldElement main rows and a fully-specified Challenges (63 XFEs).
///
/// This matches the verifier path where main rows are in the extension field.
#[no_mangle]
pub extern "C" fn compute_out_of_domain_quotient_xfe_main_challenges63_rust(
    main_row_curr_xfe: *const u64, // [379*3]
    aux_row_curr_xfe: *const u64,  // [88*3]
    main_row_next_xfe: *const u64, // [379*3]
    aux_row_next_xfe: *const u64,  // [88*3]
    challenges_63_xfe: *const u64, // [63*3]
    weights: *const u64,           // [num_weights*3]
    num_weights: usize,
    trace_domain_length: u64,
    trace_domain_generator_inverse: u64,
    out_of_domain_point_c0: u64,
    out_of_domain_point_c1: u64,
    out_of_domain_point_c2: u64,
    out_quotient_value: *mut *mut u64,
    out_len: *mut usize,
) -> c_int {
    if main_row_curr_xfe.is_null()
        || aux_row_curr_xfe.is_null()
        || main_row_next_xfe.is_null()
        || aux_row_next_xfe.is_null()
        || challenges_63_xfe.is_null()
        || weights.is_null()
        || out_quotient_value.is_null()
        || out_len.is_null()
    {
        return -1;
    }

    unsafe {
        // Convert main rows (XFE)
        let main_curr: Vec<XFieldElement> = (0..379)
            .map(|i| {
                let c0 = *main_row_curr_xfe.add(i * 3);
                let c1 = *main_row_curr_xfe.add(i * 3 + 1);
                let c2 = *main_row_curr_xfe.add(i * 3 + 2);
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let main_curr_row: [XFieldElement; 379] = main_curr.try_into().unwrap();

        let main_next: Vec<XFieldElement> = (0..379)
            .map(|i| {
                let c0 = *main_row_next_xfe.add(i * 3);
                let c1 = *main_row_next_xfe.add(i * 3 + 1);
                let c2 = *main_row_next_xfe.add(i * 3 + 2);
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let main_next_row: [XFieldElement; 379] = main_next.try_into().unwrap();

        // Aux rows (XFE)
        let aux_curr_xfes: Vec<XFieldElement> = (0..88)
            .map(|i| {
                let c0 = *aux_row_curr_xfe.add(i * 3);
                let c1 = *aux_row_curr_xfe.add(i * 3 + 1);
                let c2 = *aux_row_curr_xfe.add(i * 3 + 2);
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let aux_curr_row: [XFieldElement; 88] = aux_curr_xfes.try_into().unwrap();

        let aux_next_xfes: Vec<XFieldElement> = (0..88)
            .map(|i| {
                let c0 = *aux_row_next_xfe.add(i * 3);
                let c1 = *aux_row_next_xfe.add(i * 3 + 1);
                let c2 = *aux_row_next_xfe.add(i * 3 + 2);
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let aux_next_row: [XFieldElement; 88] = aux_next_xfes.try_into().unwrap();

        // Challenges (63)
        let ch_slice = std::slice::from_raw_parts(challenges_63_xfe, 63 * 3);
        let mut ch_vec: Vec<XFieldElement> = Vec::with_capacity(63);
        for i in 0..63 {
            let c0 = ch_slice[i * 3];
            let c1 = ch_slice[i * 3 + 1];
            let c2 = ch_slice[i * 3 + 2];
            ch_vec.push(XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)]));
        }
        let challenges_arr: [XFieldElement; triton_vm::challenges::Challenges::COUNT] =
            ch_vec.try_into().unwrap();
        let challenges = triton_vm::challenges::Challenges {
            challenges: challenges_arr,
        };

        // Weights
        let weights_slice = std::slice::from_raw_parts(weights, num_weights * 3);
        let quot_codeword_weights: Vec<XFieldElement> = (0..num_weights)
            .map(|i| {
                let c0 = weights_slice[i * 3];
                let c1 = weights_slice[i * 3 + 1];
                let c2 = weights_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();

        let out_of_domain_point = XFieldElement::new([
            bfe!(out_of_domain_point_c0),
            bfe!(out_of_domain_point_c1),
            bfe!(out_of_domain_point_c2),
        ]);

        let trace_domain_generator_inv = bfe!(trace_domain_generator_inverse);

        // Zerofier inverses (verifier logic)
        let initial_zerofier_inv = (out_of_domain_point - bfe!(1)).inverse();
        let consistency_zerofier_inv =
            (out_of_domain_point.mod_pow_u32(trace_domain_length as u32) - bfe!(1)).inverse();
        let except_last_row = out_of_domain_point - trace_domain_generator_inv;
        let transition_zerofier_inv = except_last_row * consistency_zerofier_inv;
        let terminal_zerofier_inv = except_last_row.inverse();

        // Evaluate constraints with FF = XFieldElement for main rows (matches verifier)
        let evaluated_initial_constraints = MasterAuxTable::evaluate_initial_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            &challenges,
        );
        let evaluated_consistency_constraints = MasterAuxTable::evaluate_consistency_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            &challenges,
        );
        let evaluated_transition_constraints = MasterAuxTable::evaluate_transition_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            main_next_row.as_slice().into(),
            aux_next_row.as_slice().into(),
            &challenges,
        );
        let evaluated_terminal_constraints = MasterAuxTable::evaluate_terminal_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            &challenges,
        );

        let divide = |constraints: Vec<_>, z_inv| constraints.into_iter().map(move |c| c * z_inv);
        let initial_quotients = divide(evaluated_initial_constraints, initial_zerofier_inv);
        let consistency_quotients = divide(evaluated_consistency_constraints, consistency_zerofier_inv);
        let transition_quotients = divide(evaluated_transition_constraints, transition_zerofier_inv);
        let terminal_quotients = divide(evaluated_terminal_constraints, terminal_zerofier_inv);

        let quotient_summands = initial_quotients
            .chain(consistency_quotients)
            .chain(transition_quotients)
            .chain(terminal_quotients)
            .collect::<Vec<_>>();

        if quotient_summands.len() != quot_codeword_weights.len() {
            return -1;
        }

        let out_of_domain_quotient_value = quot_codeword_weights
            .iter()
            .zip(quotient_summands.iter())
            .fold(XFieldElement::new_const(bfe!(0)), |acc, (w, s)| acc + (*w) * (*s));

        // Return as 3 u64s
        let out_vals = vec![
            out_of_domain_quotient_value.coefficients[0].value(),
            out_of_domain_quotient_value.coefficients[1].value(),
            out_of_domain_quotient_value.coefficients[2].value(),
        ];

        let layout = std::alloc::Layout::from_size_align(3 * std::mem::size_of::<u64>(), 8)
            .unwrap_or_else(|_| std::alloc::Layout::new::<u64>());
        let ptr = std::alloc::alloc(layout) as *mut u64;
        if ptr.is_null() {
            return -1;
        }
        for (i, &v) in out_vals.iter().enumerate() {
            *ptr.add(i) = v;
        }
        *out_quotient_value = ptr;
        *out_len = 3;
        0
    }
}

/// Evaluate a randomized trace column (BFE polynomial) at a point `x` (BFE), matching Rust's
/// randomized trace construction: `p(x) + zerofier_trace(x) * r(x)`, where `r` is a randomizer polynomial.
#[no_mangle]
pub extern "C" fn eval_randomized_bfe_column_at_point_rust(
    trace_values: *const u64,      // [trace_len]
    trace_len: usize,
    trace_offset: u64,
    randomizer_coeffs: *const u64, // [rand_len]
    rand_len: usize,
    x: u64,
    out_value: *mut u64,
) -> c_int {
    if trace_values.is_null() || randomizer_coeffs.is_null() || out_value.is_null() {
        return -1;
    }
    unsafe {
        let tv = std::slice::from_raw_parts(trace_values, trace_len)
            .iter()
            .map(|&v| bfe!(v))
            .collect::<Vec<BFieldElement>>();
        let rc = std::slice::from_raw_parts(randomizer_coeffs, rand_len)
            .iter()
            .map(|&v| bfe!(v))
            .collect::<Vec<BFieldElement>>();

        let domain = ArithmeticDomain::of_length(trace_len)
            .unwrap()
            .with_offset(bfe!(trace_offset));
        let interpolant = domain.interpolate(&tv);
        let rand_poly = Polynomial::new(rc);
        let randomized = interpolant + domain.mul_zerofier_with(rand_poly);
        let y: BFieldElement = randomized.evaluate(bfe!(x));
        *out_value = y.value();
        0
    }
}

/// Evaluate a randomized aux trace column (XFE polynomial) at a point `x` (BFE),
/// matching Rust's logic where the randomizer is BFE and is lifted into XFE (only c0 affected).
#[no_mangle]
pub extern "C" fn eval_randomized_xfe_column_at_point_rust(
    trace_values_xfe: *const u64, // [trace_len*3]
    trace_len: usize,
    trace_offset: u64,
    randomizer_coeffs_bfe: *const u64, // [rand_len]
    rand_len: usize,
    x: u64,
    out_xfe3: *mut u64, // [3]
) -> c_int {
    if trace_values_xfe.is_null() || randomizer_coeffs_bfe.is_null() || out_xfe3.is_null() {
        return -1;
    }
    unsafe {
        let tv = std::slice::from_raw_parts(trace_values_xfe, trace_len * 3);
        let mut c0_vals = Vec::with_capacity(trace_len);
        let mut c1_vals = Vec::with_capacity(trace_len);
        let mut c2_vals = Vec::with_capacity(trace_len);
        for i in 0..trace_len {
            c0_vals.push(bfe!(tv[i * 3 + 0]));
            c1_vals.push(bfe!(tv[i * 3 + 1]));
            c2_vals.push(bfe!(tv[i * 3 + 2]));
        }

        let rc = std::slice::from_raw_parts(randomizer_coeffs_bfe, rand_len)
            .iter()
            .map(|&v| bfe!(v))
            .collect::<Vec<BFieldElement>>();

        let domain = ArithmeticDomain::of_length(trace_len)
            .unwrap()
            .with_offset(bfe!(trace_offset));
        let p0 = domain.interpolate(&c0_vals);
        let p1 = domain.interpolate(&c1_vals);
        let p2 = domain.interpolate(&c2_vals);
        let rand_poly = Polynomial::new(rc);
        let rand_term = domain.mul_zerofier_with(rand_poly);

        let poly0 = p0 + rand_term; // lift into c0 only
        let y0: BFieldElement = poly0.evaluate(bfe!(x));
        let y1: BFieldElement = p1.evaluate(bfe!(x));
        let y2: BFieldElement = p2.evaluate(bfe!(x));

        *out_xfe3.add(0) = y0.value();
        *out_xfe3.add(1) = y1.value();
        *out_xfe3.add(2) = y2.value();
        0
    }
}

/// Compute out-of-domain quotient segment evaluations exactly like Rust's
/// `interpolate_quotient_segments` + OOD evaluation:
/// - interpolate `quotient_codeword` on `quotient_domain` (coset)
/// - split polynomial coefficients into 4 segments (step_by(4))
/// - evaluate each segment at `z^4`
#[no_mangle]
pub extern "C" fn compute_ood_quot_segments_from_quotient_codeword_rust(
    quotient_codeword_xfe: *const u64, // [quotient_len*3]
    quotient_len: usize,
    quotient_offset: u64,
    z0: u64,
    z1: u64,
    z2: u64,
    out_segments_xfe: *mut *mut u64,
    out_len: *mut usize,
) -> c_int {
    if quotient_codeword_xfe.is_null() || out_segments_xfe.is_null() || out_len.is_null() {
        return -1;
    }
    unsafe {
        let qw = std::slice::from_raw_parts(quotient_codeword_xfe, quotient_len * 3);
        let mut evals = Vec::with_capacity(quotient_len);
        for i in 0..quotient_len {
            evals.push(XFieldElement::new([
                bfe!(qw[i * 3 + 0]),
                bfe!(qw[i * 3 + 1]),
                bfe!(qw[i * 3 + 2]),
            ]));
        }
        let domain = ArithmeticDomain::of_length(quotient_len)
            .unwrap()
            .with_offset(bfe!(quotient_offset));
        let poly = domain.interpolate(&evals);
        let coeffs = poly.into_coefficients();
        // Split into 4 segments by interleaving coefficients
        let mut segs: [Vec<XFieldElement>; 4] = [(); 4].map(|_| Vec::new());
        for seg in 0..4 {
            segs[seg] = coeffs.iter().skip(seg).step_by(4).copied().collect();
        }
        let z = XFieldElement::new([bfe!(z0), bfe!(z1), bfe!(z2)]);
        let z4 = z.mod_pow_u32(4);
        let mut out = Vec::with_capacity(12);
        for seg in 0..4 {
            let seg_poly = Polynomial::new(segs[seg].clone());
            let v: XFieldElement = seg_poly.evaluate(z4);
            out.push(v.coefficients[0].value());
            out.push(v.coefficients[1].value());
            out.push(v.coefficients[2].value());
        }

        let layout = std::alloc::Layout::from_size_align(out.len() * 8, 8)
            .unwrap_or_else(|_| std::alloc::Layout::new::<u64>());
        let ptr = std::alloc::alloc(layout) as *mut u64;
        if ptr.is_null() {
            return -1;
        }
        for (i, &v) in out.iter().enumerate() {
            *ptr.add(i) = v;
        }
        *out_segments_xfe = ptr;
        *out_len = out.len();
        0
    }
}

/// Interpolate a coset-domain XFE codeword and evaluate the interpolant at an XFE point `z`.
#[no_mangle]
pub extern "C" fn eval_xfe_coset_codeword_at_xfe_point_rust(
    codeword_xfe: *const u64, // [len*3]
    len: usize,
    domain_offset: u64,
    z0: u64,
    z1: u64,
    z2: u64,
    out_xfe3: *mut u64, // [3]
) -> c_int {
    if codeword_xfe.is_null() || out_xfe3.is_null() {
        return -1;
    }
    unsafe {
        let cw = std::slice::from_raw_parts(codeword_xfe, len * 3);
        let mut evals = Vec::with_capacity(len);
        for i in 0..len {
            evals.push(XFieldElement::new([
                bfe!(cw[i * 3 + 0]),
                bfe!(cw[i * 3 + 1]),
                bfe!(cw[i * 3 + 2]),
            ]));
        }
        let domain = ArithmeticDomain::of_length(len)
            .unwrap()
            .with_offset(bfe!(domain_offset));
        let poly = domain.interpolate(&evals);
        let z = XFieldElement::new([bfe!(z0), bfe!(z1), bfe!(z2)]);
        let y: XFieldElement = poly.evaluate(z);
        *out_xfe3.add(0) = y.coefficients[0].value();
        *out_xfe3.add(1) = y.coefficients[1].value();
        *out_xfe3.add(2) = y.coefficients[2].value();
        0
    }
}

/// Compute quotient value at a base-field point `x` using Rust's constraint evaluation,
/// matching the verifier's combination logic:
///   q(x) = <weights, (constraints(x) / zerofiers(x))>
#[no_mangle]
pub extern "C" fn compute_quotient_value_at_bfe_point_rust(
    x: u64,
    trace_domain_length: u64,
    trace_domain_generator_inverse: u64,
    main_row_curr: *const u64,      // [379]
    aux_row_curr: *const u64,       // [88*3]
    main_row_next: *const u64,      // [379]
    aux_row_next: *const u64,       // [88*3]
    challenges_63_xfe: *const u64,  // [63*3]
    weights: *const u64,            // [num_weights*3]
    num_weights: usize,
    out_xfe3: *mut u64,             // [3]
) -> c_int {
    if main_row_curr.is_null()
        || aux_row_curr.is_null()
        || main_row_next.is_null()
        || aux_row_next.is_null()
        || challenges_63_xfe.is_null()
        || weights.is_null()
        || out_xfe3.is_null()
    {
        return -1;
    }

    unsafe {
        let main_curr_bfes: Vec<BFieldElement> =
            std::slice::from_raw_parts(main_row_curr, 379).iter().map(|&v| bfe!(v)).collect();
        let main_curr_row: [BFieldElement; 379] = main_curr_bfes.try_into().unwrap();
        let main_next_bfes: Vec<BFieldElement> =
            std::slice::from_raw_parts(main_row_next, 379).iter().map(|&v| bfe!(v)).collect();
        let main_next_row: [BFieldElement; 379] = main_next_bfes.try_into().unwrap();

        let aux_curr_xfes: Vec<XFieldElement> = (0..88)
            .map(|i| {
                let c0 = *aux_row_curr.add(i * 3);
                let c1 = *aux_row_curr.add(i * 3 + 1);
                let c2 = *aux_row_curr.add(i * 3 + 2);
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let aux_curr_row: [XFieldElement; 88] = aux_curr_xfes.try_into().unwrap();

        let aux_next_xfes: Vec<XFieldElement> = (0..88)
            .map(|i| {
                let c0 = *aux_row_next.add(i * 3);
                let c1 = *aux_row_next.add(i * 3 + 1);
                let c2 = *aux_row_next.add(i * 3 + 2);
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();
        let aux_next_row: [XFieldElement; 88] = aux_next_xfes.try_into().unwrap();

        let ch_slice = std::slice::from_raw_parts(challenges_63_xfe, 63 * 3);
        let mut ch_vec: Vec<XFieldElement> = Vec::with_capacity(63);
        for i in 0..63 {
            let c0 = ch_slice[i * 3];
            let c1 = ch_slice[i * 3 + 1];
            let c2 = ch_slice[i * 3 + 2];
            ch_vec.push(XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)]));
        }
        let challenges_arr: [XFieldElement; triton_vm::challenges::Challenges::COUNT] =
            ch_vec.try_into().unwrap();
        let challenges = triton_vm::challenges::Challenges { challenges: challenges_arr };

        let w_slice = std::slice::from_raw_parts(weights, num_weights * 3);
        let ws: Vec<XFieldElement> = (0..num_weights)
            .map(|i| {
                let c0 = w_slice[i * 3];
                let c1 = w_slice[i * 3 + 1];
                let c2 = w_slice[i * 3 + 2];
                XFieldElement::new([bfe!(c0), bfe!(c1), bfe!(c2)])
            })
            .collect();

        let x_xfe = XFieldElement::new_const(bfe!(x));
        let trace_gen_inv = bfe!(trace_domain_generator_inverse);

        let initial_inv = (x_xfe - bfe!(1)).inverse();
        let consistency_inv = (x_xfe.mod_pow_u32(trace_domain_length as u32) - bfe!(1)).inverse();
        let except_last = x_xfe - trace_gen_inv;
        let transition_inv = except_last * consistency_inv;
        let terminal_inv = except_last.inverse();

        let init = MasterAuxTable::evaluate_initial_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            &challenges,
        );
        let cons = MasterAuxTable::evaluate_consistency_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            &challenges,
        );
        let tran = MasterAuxTable::evaluate_transition_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            main_next_row.as_slice().into(),
            aux_next_row.as_slice().into(),
            &challenges,
        );
        let term = MasterAuxTable::evaluate_terminal_constraints(
            main_curr_row.as_slice().into(),
            aux_curr_row.as_slice().into(),
            &challenges,
        );

        let divide = |constraints: Vec<_>, z_inv| constraints.into_iter().map(move |c| c * z_inv);
        let summands = divide(init, initial_inv)
            .chain(divide(cons, consistency_inv))
            .chain(divide(tran, transition_inv))
            .chain(divide(term, terminal_inv))
            .collect::<Vec<_>>();
        if summands.len() != ws.len() {
            return -1;
        }
        let q = ws
            .iter()
            .zip(summands.iter())
            .fold(XFieldElement::new_const(bfe!(0)), |acc, (w, s)| acc + (*w) * (*s));

        *out_xfe3.add(0) = q.coefficients[0].value();
        *out_xfe3.add(1) = q.coefficients[1].value();
        *out_xfe3.add(2) = q.coefficients[2].value();
        0
    }
}

/// Evaluate a randomized main trace column (BFE codeword + BFE randomizer) at an XFE point `z`
/// using Rust's interpolation+zerofier construction.
#[no_mangle]
pub extern "C" fn eval_randomized_main_column_at_xfe_point_rust(
    trace_values: *const u64,      // [trace_len]
    trace_len: usize,
    trace_offset: u64,
    randomizer_coeffs: *const u64, // [rand_len]
    rand_len: usize,
    z0: u64,
    z1: u64,
    z2: u64,
    out_xfe3: *mut u64,            // [3]
) -> c_int {
    if trace_values.is_null() || randomizer_coeffs.is_null() || out_xfe3.is_null() {
        return -1;
    }
    unsafe {
        let tv = std::slice::from_raw_parts(trace_values, trace_len)
            .iter()
            .map(|&v| bfe!(v))
            .collect::<Vec<BFieldElement>>();
        let rc = std::slice::from_raw_parts(randomizer_coeffs, rand_len)
            .iter()
            .map(|&v| bfe!(v))
            .collect::<Vec<BFieldElement>>();

        let domain = ArithmeticDomain::of_length(trace_len)
            .unwrap()
            .with_offset(bfe!(trace_offset));
        let p = domain.interpolate(&tv);
        let rand_poly = Polynomial::new(rc);
        let randomized = p + domain.mul_zerofier_with(rand_poly);

        let z = XFieldElement::new([bfe!(z0), bfe!(z1), bfe!(z2)]);
        let y: XFieldElement = randomized.evaluate(z);
        *out_xfe3.add(0) = y.coefficients[0].value();
        *out_xfe3.add(1) = y.coefficients[1].value();
        *out_xfe3.add(2) = y.coefficients[2].value();
        0
    }
}

/// Evaluate a randomized aux trace column (XFE codeword + XFE randomizer polynomial) at an XFE point `z`
/// using Rust's interpolation + zerofier construction.
///
/// This is the aux-table analogue of `eval_randomized_main_column_at_xfe_point_rust`, and is intended
/// purely for debugging parity between the GPU/C++ prover and the Rust verifier/prover.
#[no_mangle]
pub extern "C" fn eval_randomized_aux_column_at_xfe_point_rust(
    trace_values_xfe: *const u64,       // [trace_len*3]
    trace_len: usize,
    trace_offset: u64,
    randomizer_coeffs_xfe: *const u64,  // [rand_len*3]
    rand_len: usize,
    z0: u64,
    z1: u64,
    z2: u64,
    out_xfe3: *mut u64,                // [3]
) -> c_int {
    if trace_values_xfe.is_null() || randomizer_coeffs_xfe.is_null() || out_xfe3.is_null() {
        return -1;
    }

    unsafe {
        // Trace codeword (XFE evaluations on trace domain)
        let tv = std::slice::from_raw_parts(trace_values_xfe, trace_len * 3);
        let mut evals: Vec<XFieldElement> = Vec::with_capacity(trace_len);
        for i in 0..trace_len {
            evals.push(XFieldElement::new([
                bfe!(tv[i * 3 + 0]),
                bfe!(tv[i * 3 + 1]),
                bfe!(tv[i * 3 + 2]),
            ]));
        }

        // Interpolate and evaluate at z
        let domain = ArithmeticDomain::of_length(trace_len)
            .unwrap()
            .with_offset(bfe!(trace_offset));
        let poly = domain.interpolate(&evals);
        let z = XFieldElement::new([bfe!(z0), bfe!(z1), bfe!(z2)]);
        let bary: XFieldElement = poly.evaluate(z);

        // Randomizer polynomial (XFE coefficients)
        let rc = std::slice::from_raw_parts(randomizer_coeffs_xfe, rand_len * 3);
        let mut coeffs: Vec<XFieldElement> = Vec::with_capacity(rand_len);
        for i in 0..rand_len {
            coeffs.push(XFieldElement::new([
                bfe!(rc[i * 3 + 0]),
                bfe!(rc[i * 3 + 1]),
                bfe!(rc[i * 3 + 2]),
            ]));
        }
        // Horner evaluation of polynomial with XFE coeffs at XFE point z
        let mut rand_at_z = XFieldElement::new_const(bfe!(0));
        for c in coeffs.iter().rev() {
            rand_at_z = rand_at_z * z + *c;
        }

        // Zerofier for trace domain coset
        let offset_pow = bfe!(trace_offset).mod_pow_u32(trace_len as u32);
        let zerofier = z.mod_pow_u32(trace_len as u32) - XFieldElement::new_const(offset_pow);

        let y = bary + zerofier * rand_at_z;

        *out_xfe3.add(0) = y.coefficients[0].value();
        *out_xfe3.add(1) = y.coefficients[1].value();
        *out_xfe3.add(2) = y.coefficients[2].value();
        0
    }
}

/// Evaluate the coset interpolant of a BFE codeword at an XFE point `z` (no randomizer).
#[no_mangle]
pub extern "C" fn eval_bfe_interpolant_at_xfe_point_rust(
    trace_values: *const u64, // [len]
    len: usize,
    domain_offset: u64,
    z0: u64,
    z1: u64,
    z2: u64,
    out_xfe3: *mut u64, // [3]
) -> c_int {
    if trace_values.is_null() || out_xfe3.is_null() {
        return -1;
    }
    unsafe {
        let tv = std::slice::from_raw_parts(trace_values, len)
            .iter()
            .map(|&v| bfe!(v))
            .collect::<Vec<BFieldElement>>();
        let domain = ArithmeticDomain::of_length(len)
            .unwrap()
            .with_offset(bfe!(domain_offset));
        let poly = domain.interpolate(&tv);
        let z = XFieldElement::new([bfe!(z0), bfe!(z1), bfe!(z2)]);
        let y: XFieldElement = poly.evaluate(z);
        *out_xfe3.add(0) = y.coefficients[0].value();
        *out_xfe3.add(1) = y.coefficients[1].value();
        *out_xfe3.add(2) = y.coefficients[2].value();
        0
    }
}

/// Debug helper: compute the first few `domain_over_domain_shift` values and the
/// `barycentric_eval_denominator_inverse` exactly like Rust's `out_of_domain_row`.
/// Returns 4 XFE values (12 u64) + denom_inv (3 u64) = 15 u64.
#[no_mangle]
pub extern "C" fn debug_barycentric_weights_rust(
    trace_len: usize,
    trace_offset: u64,
    z0: u64,
    z1: u64,
    z2: u64,
    out_u64: *mut *mut u64,
    out_len: *mut usize,
) -> c_int {
    if out_u64.is_null() || out_len.is_null() {
        return -1;
    }
    let domain = ArithmeticDomain::of_length(trace_len)
        .unwrap()
        .with_offset(bfe!(trace_offset));
    let z = XFieldElement::new([bfe!(z0), bfe!(z1), bfe!(z2)]);
    let dom_vals = domain.values();
    let domain_shift = dom_vals.iter().map(|&d| z - d).collect::<Vec<_>>();
    let invs = XFieldElement::batch_inversion(domain_shift);
    let dom_over = dom_vals
        .into_iter()
        .zip(invs.into_iter())
        .map(|(d, inv)| d * inv)
        .collect::<Vec<_>>();
    let denom_inv = dom_over.iter().copied().sum::<XFieldElement>().inverse();

    let mut out = Vec::with_capacity(15);
    for i in 0..4 {
        let v = dom_over[i];
        out.push(v.coefficients[0].value());
        out.push(v.coefficients[1].value());
        out.push(v.coefficients[2].value());
    }
    out.push(denom_inv.coefficients[0].value());
    out.push(denom_inv.coefficients[1].value());
    out.push(denom_inv.coefficients[2].value());

    unsafe {
        let layout = std::alloc::Layout::from_size_align(out.len() * 8, 8)
            .unwrap_or_else(|_| std::alloc::Layout::new::<u64>());
        let ptr = std::alloc::alloc(layout) as *mut u64;
        if ptr.is_null() {
            return -1;
        }
        for (i, &v) in out.iter().enumerate() {
            *ptr.add(i) = v;
        }
        *out_u64 = ptr;
        *out_len = out.len();
    }
    0
}

/// Free memory allocated by constraint evaluation functions
#[no_mangle]
pub extern "C" fn constraint_evaluation_free(ptr: *mut u64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let layout = Layout::array::<u64>(len).unwrap();
            dealloc(ptr as *mut u8, layout);
        }
    }
}

/// FFI function to encode a Claim using Rust's BFieldCodec (ensures exact compatibility)
#[no_mangle]
pub extern "C" fn claim_encode_rust(
    program_digest: *const u64,  // 5 elements
    version: u32,
    input: *const u64,
    input_len: usize,
    output: *const u64,
    output_len: usize,
    out_encoded: *mut *mut u64,
    out_len: *mut usize,
) -> c_int {
    if program_digest.is_null() || out_encoded.is_null() || out_len.is_null() {
        return -1;
    }

    unsafe {
        // Reconstruct Digest from 5 u64 values
        let digest_slice = std::slice::from_raw_parts(program_digest, 5);
        let digest = Digest::new([
            bfe!(digest_slice[0]),
            bfe!(digest_slice[1]),
            bfe!(digest_slice[2]),
            bfe!(digest_slice[3]),
            bfe!(digest_slice[4]),
        ]);

        // Reconstruct input Vec<BFieldElement>
        let input_vec = if !input.is_null() && input_len > 0 {
            let input_slice = std::slice::from_raw_parts(input, input_len);
            input_slice.iter().map(|&x| bfe!(x)).collect()
        } else {
            vec![]
        };

        // Reconstruct output Vec<BFieldElement>
        let output_vec = if !output.is_null() && output_len > 0 {
            let output_slice = std::slice::from_raw_parts(output, output_len);
            output_slice.iter().map(|&x| bfe!(x)).collect()
        } else {
            vec![]
        };

        // Create Claim
        let claim = triton_vm::proof::Claim {
            program_digest: digest,
            version,
            input: input_vec,
            output: output_vec,
        };

        // Encode using Rust's BFieldCodec
        let encoded = claim.encode();

        // Allocate memory for result
        let layout = Layout::array::<u64>(encoded.len()).unwrap();
        let ptr = alloc(layout) as *mut u64;
        if ptr.is_null() {
            return -1;
        }

        // Copy encoded data
        for (i, &elem) in encoded.iter().enumerate() {
            *ptr.add(i) = elem.value();
        }

        *out_encoded = ptr;
        *out_len = encoded.len();
        0
    }
}

/// Free memory allocated by claim_encode_rust
#[no_mangle]
pub extern "C" fn claim_encode_free(ptr: *mut u64, len: usize) {
    if !ptr.is_null() {
        unsafe {
            let layout = Layout::array::<u64>(len).unwrap();
            dealloc(ptr as *mut u8, layout);
        }
    }
}

/// Run Rust trace execution only (no table creation).
/// Returns AET data for C++ to use.
/// 
/// This is faster than C++ trace execution for large programs.
/// 
/// # Safety
/// Caller must ensure all pointers are valid and provide sufficient buffers.
#[no_mangle]
pub unsafe extern "C" fn tvm_trace_execution_rust_ffi(
    program_path: *const c_char,
    public_input_data: *const c_ulonglong,
    public_input_len: usize,
    // Output: processor trace (flat, row-major)
    out_processor_trace_data: *mut *mut c_ulonglong,
    out_processor_trace_rows: *mut usize,
    out_processor_trace_cols: *mut usize,
    // Output: program bwords
    out_program_bwords_data: *mut *mut c_ulonglong,
    out_program_bwords_len: *mut usize,
    // Output: instruction multiplicities
    out_instruction_multiplicities_data: *mut *mut u32,
    out_instruction_multiplicities_len: *mut usize,
    // Output: public output
    out_public_output_data: *mut *mut c_ulonglong,
    out_public_output_len: *mut usize,
    // Output: co-processor traces (all flat, row-major)
    out_op_stack_trace_data: *mut *mut c_ulonglong,
    out_op_stack_trace_rows: *mut usize,
    out_op_stack_trace_cols: *mut usize,
    out_ram_trace_data: *mut *mut c_ulonglong,
    out_ram_trace_rows: *mut usize,
    out_ram_trace_cols: *mut usize,
    out_program_hash_trace_data: *mut *mut c_ulonglong,
    out_program_hash_trace_rows: *mut usize,
    out_program_hash_trace_cols: *mut usize,
    out_hash_trace_data: *mut *mut c_ulonglong,
    out_hash_trace_rows: *mut usize,
    out_hash_trace_cols: *mut usize,
    out_sponge_trace_data: *mut *mut c_ulonglong,
    out_sponge_trace_rows: *mut usize,
    out_sponge_trace_cols: *mut usize,
    // Output: U32 entries (flat: [instruction, operand1, operand2, multiplicity] per entry)
    out_u32_entries_data: *mut *mut c_ulonglong,
    out_u32_entries_len: *mut usize,
    // Output: Cascade lookup multiplicities (flat: [limb, multiplicity] pairs)
    out_cascade_multiplicities_data: *mut *mut c_ulonglong,
    out_cascade_multiplicities_len: *mut usize,
    // Output: Lookup lookup multiplicities (array of 256 u64)
    out_lookup_multiplicities_256: *mut c_ulonglong,
    // Output: table lengths [program, processor, op_stack, ram, jump_stack, hash, cascade, lookup, u32]
    out_table_lengths_9: *mut usize,
) -> c_int {
    if program_path.is_null()
        || public_input_data.is_null()
        || out_processor_trace_data.is_null()
        || out_processor_trace_rows.is_null()
        || out_processor_trace_cols.is_null()
        || out_program_bwords_data.is_null()
        || out_program_bwords_len.is_null()
        || out_instruction_multiplicities_data.is_null()
        || out_instruction_multiplicities_len.is_null()
        || out_public_output_data.is_null()
        || out_public_output_len.is_null()
        || out_op_stack_trace_data.is_null()
        || out_op_stack_trace_rows.is_null()
        || out_op_stack_trace_cols.is_null()
        || out_ram_trace_data.is_null()
        || out_ram_trace_rows.is_null()
        || out_ram_trace_cols.is_null()
        || out_program_hash_trace_data.is_null()
        || out_program_hash_trace_rows.is_null()
        || out_program_hash_trace_cols.is_null()
        || out_hash_trace_data.is_null()
        || out_hash_trace_rows.is_null()
        || out_hash_trace_cols.is_null()
        || out_sponge_trace_data.is_null()
        || out_sponge_trace_rows.is_null()
        || out_sponge_trace_cols.is_null()
        || out_u32_entries_data.is_null()
        || out_u32_entries_len.is_null()
        || out_cascade_multiplicities_data.is_null()
        || out_cascade_multiplicities_len.is_null()
        || out_lookup_multiplicities_256.is_null()
        || out_table_lengths_9.is_null()
    {
        return -1;
    }

    let path_str = match std::ffi::CStr::from_ptr(program_path).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let code = match std::fs::read_to_string(path_str) {
        Ok(s) => s,
        Err(_) => return -1,
    };

    let program = match Program::from_code(&code) {
        Ok(p) => p,
        Err(_) => return -1,
    };

    let input_slice = std::slice::from_raw_parts(public_input_data, public_input_len);
    let input_vec: Vec<BFieldElement> = input_slice
        .iter()
        .copied()
        .map(BFieldElement::new)
        .collect();
    let public_input = PublicInput::new(input_vec.clone());

    // Run VM trace execution.
    let (aet, output) = match VM::trace_execution(program.clone(), public_input, NonDeterminism::default()) {
        Ok(res) => res,
        Err(_) => return -1,
    };

    // Extract processor trace (flat, row-major)
    let proc_trace = &aet.processor_trace;
    let proc_rows = proc_trace.nrows();
    let proc_cols = proc_trace.ncols();
    let proc_flat_len = proc_rows * proc_cols;
    
    let proc_layout = Layout::from_size_align(
        proc_flat_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let proc_ptr = alloc(proc_layout) as *mut c_ulonglong;
    if proc_ptr.is_null() {
        return -1;
    }
    
    // Copy processor trace (row-major)
    for i in 0..proc_rows {
        for j in 0..proc_cols {
            let bfe = proc_trace[[i, j]];
            *proc_ptr.add(i * proc_cols + j) = bfe.value();
        }
    }
    
    *out_processor_trace_data = proc_ptr;
    *out_processor_trace_rows = proc_rows;
    *out_processor_trace_cols = proc_cols;

    // Extract program bwords
    let bwords = program.to_bwords();
    let bwords_len = bwords.len();
    let bwords_layout = Layout::from_size_align(
        bwords_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let bwords_ptr = alloc(bwords_layout) as *mut c_ulonglong;
    if bwords_ptr.is_null() {
        dealloc(proc_ptr as *mut u8, proc_layout);
        return -1;
    }
    for (i, bfe) in bwords.iter().enumerate() {
        *bwords_ptr.add(i) = bfe.value();
    }
    *out_program_bwords_data = bwords_ptr;
    *out_program_bwords_len = bwords_len;

    // Extract instruction multiplicities
    let mults = &aet.instruction_multiplicities;
    let mults_len = mults.len();
    let mults_layout = Layout::from_size_align(
        mults_len * std::mem::size_of::<u32>(),
        std::mem::align_of::<u32>(),
    ).unwrap_or_else(|_| Layout::new::<u32>());
    let mults_ptr = alloc(mults_layout) as *mut u32;
    if mults_ptr.is_null() {
        dealloc(proc_ptr as *mut u8, proc_layout);
        dealloc(bwords_ptr as *mut u8, bwords_layout);
        return -1;
    }
    for (i, &mult) in mults.iter().enumerate() {
        *mults_ptr.add(i) = mult;
    }
    *out_instruction_multiplicities_data = mults_ptr;
    *out_instruction_multiplicities_len = mults_len;

    // Extract public output
    let out_len = output.len();
    let out_layout = Layout::from_size_align(
        out_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let out_ptr = alloc(out_layout) as *mut c_ulonglong;
    if out_ptr.is_null() {
        dealloc(proc_ptr as *mut u8, proc_layout);
        dealloc(bwords_ptr as *mut u8, bwords_layout);
        dealloc(mults_ptr as *mut u8, mults_layout);
        return -1;
    }
    for (i, bfe) in output.iter().enumerate() {
        *out_ptr.add(i) = bfe.value();
    }
    *out_public_output_data = out_ptr;
    *out_public_output_len = out_len;

    // Extract op_stack trace (flat, row-major)
    let op_stack_trace = &aet.op_stack_underflow_trace;
    let op_stack_rows = op_stack_trace.nrows();
    let op_stack_cols = op_stack_trace.ncols();
    let op_stack_flat_len = op_stack_rows * op_stack_cols;
    let op_stack_layout = Layout::from_size_align(
        op_stack_flat_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let op_stack_ptr = alloc(op_stack_layout) as *mut c_ulonglong;
    if op_stack_ptr.is_null() {
        dealloc(proc_ptr as *mut u8, proc_layout);
        dealloc(bwords_ptr as *mut u8, bwords_layout);
        dealloc(mults_ptr as *mut u8, mults_layout);
        dealloc(out_ptr as *mut u8, out_layout);
        return -1;
    }
    for i in 0..op_stack_rows {
        for j in 0..op_stack_cols {
            let bfe = op_stack_trace[[i, j]];
            *op_stack_ptr.add(i * op_stack_cols + j) = bfe.value();
        }
    }
    *out_op_stack_trace_data = op_stack_ptr;
    *out_op_stack_trace_rows = op_stack_rows;
    *out_op_stack_trace_cols = op_stack_cols;

    // Extract ram trace (flat, row-major)
    let ram_trace = &aet.ram_trace;
    let ram_rows = ram_trace.nrows();
    let ram_cols = ram_trace.ncols();
    let ram_flat_len = ram_rows * ram_cols;
    let ram_layout = Layout::from_size_align(
        ram_flat_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let ram_ptr = alloc(ram_layout) as *mut c_ulonglong;
    if ram_ptr.is_null() {
        dealloc(proc_ptr as *mut u8, proc_layout);
        dealloc(bwords_ptr as *mut u8, bwords_layout);
        dealloc(mults_ptr as *mut u8, mults_layout);
        dealloc(out_ptr as *mut u8, out_layout);
        dealloc(op_stack_ptr as *mut u8, op_stack_layout);
        return -1;
    }
    for i in 0..ram_rows {
        for j in 0..ram_cols {
            let bfe = ram_trace[[i, j]];
            *ram_ptr.add(i * ram_cols + j) = bfe.value();
        }
    }
    *out_ram_trace_data = ram_ptr;
    *out_ram_trace_rows = ram_rows;
    *out_ram_trace_cols = ram_cols;

    // Extract program_hash trace (flat, row-major)
    let prog_hash_trace = &aet.program_hash_trace;
    let prog_hash_rows = prog_hash_trace.nrows();
    let prog_hash_cols = prog_hash_trace.ncols();
    let prog_hash_flat_len = prog_hash_rows * prog_hash_cols;
    let prog_hash_layout = Layout::from_size_align(
        prog_hash_flat_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let prog_hash_ptr = alloc(prog_hash_layout) as *mut c_ulonglong;
    if prog_hash_ptr.is_null() {
        dealloc(proc_ptr as *mut u8, proc_layout);
        dealloc(bwords_ptr as *mut u8, bwords_layout);
        dealloc(mults_ptr as *mut u8, mults_layout);
        dealloc(out_ptr as *mut u8, out_layout);
        dealloc(op_stack_ptr as *mut u8, op_stack_layout);
        dealloc(ram_ptr as *mut u8, ram_layout);
        return -1;
    }
    for i in 0..prog_hash_rows {
        for j in 0..prog_hash_cols {
            let bfe = prog_hash_trace[[i, j]];
            *prog_hash_ptr.add(i * prog_hash_cols + j) = bfe.value();
        }
    }
    *out_program_hash_trace_data = prog_hash_ptr;
    *out_program_hash_trace_rows = prog_hash_rows;
    *out_program_hash_trace_cols = prog_hash_cols;

    // Extract hash trace (flat, row-major)
    let hash_trace = &aet.hash_trace;
    let hash_rows = hash_trace.nrows();
    let hash_cols = hash_trace.ncols();
    let hash_flat_len = hash_rows * hash_cols;
    let hash_layout = Layout::from_size_align(
        hash_flat_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let hash_ptr = alloc(hash_layout) as *mut c_ulonglong;
    if hash_ptr.is_null() {
        dealloc(proc_ptr as *mut u8, proc_layout);
        dealloc(bwords_ptr as *mut u8, bwords_layout);
        dealloc(mults_ptr as *mut u8, mults_layout);
        dealloc(out_ptr as *mut u8, out_layout);
        dealloc(op_stack_ptr as *mut u8, op_stack_layout);
        dealloc(ram_ptr as *mut u8, ram_layout);
        dealloc(prog_hash_ptr as *mut u8, prog_hash_layout);
        return -1;
    }
    for i in 0..hash_rows {
        for j in 0..hash_cols {
            let bfe = hash_trace[[i, j]];
            *hash_ptr.add(i * hash_cols + j) = bfe.value();
        }
    }
    *out_hash_trace_data = hash_ptr;
    *out_hash_trace_rows = hash_rows;
    *out_hash_trace_cols = hash_cols;

    // Extract sponge trace (flat, row-major)
    let sponge_trace = &aet.sponge_trace;
    let sponge_rows = sponge_trace.nrows();
    let sponge_cols = sponge_trace.ncols();
    let sponge_flat_len = sponge_rows * sponge_cols;
    let sponge_layout = Layout::from_size_align(
        sponge_flat_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let sponge_ptr = alloc(sponge_layout) as *mut c_ulonglong;
    if sponge_ptr.is_null() {
        dealloc(proc_ptr as *mut u8, proc_layout);
        dealloc(bwords_ptr as *mut u8, bwords_layout);
        dealloc(mults_ptr as *mut u8, mults_layout);
        dealloc(out_ptr as *mut u8, out_layout);
        dealloc(op_stack_ptr as *mut u8, op_stack_layout);
        dealloc(ram_ptr as *mut u8, ram_layout);
        dealloc(prog_hash_ptr as *mut u8, prog_hash_layout);
        dealloc(hash_ptr as *mut u8, hash_layout);
        return -1;
    }
    for i in 0..sponge_rows {
        for j in 0..sponge_cols {
            let bfe = sponge_trace[[i, j]];
            *sponge_ptr.add(i * sponge_cols + j) = bfe.value();
        }
    }
    *out_sponge_trace_data = sponge_ptr;
    *out_sponge_trace_rows = sponge_rows;
    *out_sponge_trace_cols = sponge_cols;

    // Extract U32 entries (flat: [instruction_opcode, left_operand, right_operand, multiplicity] per entry)
    let u32_entries = &aet.u32_entries;
    let u32_len = u32_entries.len();
    let u32_flat_len = u32_len * 4; // [instruction_opcode, left_operand, right_operand, mult]
    let u32_layout = Layout::from_size_align(
        u32_flat_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let u32_ptr = alloc(u32_layout) as *mut c_ulonglong;
    if u32_ptr.is_null() {
        dealloc(proc_ptr as *mut u8, proc_layout);
        dealloc(bwords_ptr as *mut u8, bwords_layout);
        dealloc(mults_ptr as *mut u8, mults_layout);
        dealloc(out_ptr as *mut u8, out_layout);
        dealloc(op_stack_ptr as *mut u8, op_stack_layout);
        dealloc(ram_ptr as *mut u8, ram_layout);
        dealloc(prog_hash_ptr as *mut u8, prog_hash_layout);
        dealloc(hash_ptr as *mut u8, hash_layout);
        dealloc(sponge_ptr as *mut u8, sponge_layout);
        return -1;
    }
    for (idx, (entry, mult)) in u32_entries.iter().enumerate() {
        *u32_ptr.add(idx * 4 + 0) = entry.instruction.opcode_b().value();
        *u32_ptr.add(idx * 4 + 1) = entry.left_operand.value();
        *u32_ptr.add(idx * 4 + 2) = entry.right_operand.value();
        *u32_ptr.add(idx * 4 + 3) = *mult as u64;
    }
    *out_u32_entries_data = u32_ptr;
    *out_u32_entries_len = u32_len;

    // Extract cascade lookup multiplicities (flat: [limb, multiplicity] pairs)
    let cascade_mults = &aet.cascade_table_lookup_multiplicities;
    let cascade_len = cascade_mults.len();
    let cascade_flat_len = cascade_len * 2; // [limb, mult]
    let cascade_layout = Layout::from_size_align(
        cascade_flat_len * std::mem::size_of::<c_ulonglong>(),
        std::mem::align_of::<c_ulonglong>(),
    ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
    let cascade_ptr = alloc(cascade_layout) as *mut c_ulonglong;
    if cascade_ptr.is_null() {
        dealloc(proc_ptr as *mut u8, proc_layout);
        dealloc(bwords_ptr as *mut u8, bwords_layout);
        dealloc(mults_ptr as *mut u8, mults_layout);
        dealloc(out_ptr as *mut u8, out_layout);
        dealloc(op_stack_ptr as *mut u8, op_stack_layout);
        dealloc(ram_ptr as *mut u8, ram_layout);
        dealloc(prog_hash_ptr as *mut u8, prog_hash_layout);
        dealloc(hash_ptr as *mut u8, hash_layout);
        dealloc(sponge_ptr as *mut u8, sponge_layout);
        dealloc(u32_ptr as *mut u8, u32_layout);
        return -1;
    }
    for (idx, (limb, mult)) in cascade_mults.iter().enumerate() {
        *cascade_ptr.add(idx * 2 + 0) = *limb as u64;
        *cascade_ptr.add(idx * 2 + 1) = *mult;
    }
    *out_cascade_multiplicities_data = cascade_ptr;
    *out_cascade_multiplicities_len = cascade_len;

    // Extract lookup lookup multiplicities (array of 256 u64)
    let lookup_mults = &aet.lookup_table_lookup_multiplicities;
    for i in 0..256 {
        *out_lookup_multiplicities_256.add(i) = lookup_mults[i];
    }

    // Extract table lengths using height_of_table
    use triton_vm::prelude::TableId;
    let table_lengths = [
        aet.height_of_table(TableId::Program),
        aet.height_of_table(TableId::Processor),
        aet.height_of_table(TableId::OpStack),
        aet.height_of_table(TableId::Ram),
        aet.height_of_table(TableId::JumpStack),
        aet.height_of_table(TableId::Hash),
        aet.height_of_table(TableId::Cascade),
        aet.height_of_table(TableId::Lookup),
        aet.height_of_table(TableId::U32),
    ];
    std::ptr::copy_nonoverlapping(table_lengths.as_ptr(), out_table_lengths_9, 9);

    0
}

/// Rust FFI: Execute VM trace with NonDeterminism JSON support
/// 
/// This is a variant of tvm_trace_execution_rust_ffi that accepts NonDeterminism
/// as a JSON string, allowing programs that need RAM/secret input to execute correctly.
/// 
/// # Safety
/// Caller must ensure all pointers are valid and provide sufficient buffers.
#[no_mangle]
pub unsafe extern "C" fn tvm_trace_execution_with_nondet(
    program_json: *const c_char,
    nondet_json: *const c_char,
    public_input_data: *const c_ulonglong,
    public_input_len: usize,
    // Output: processor trace (flat, row-major)
    out_processor_trace_data: *mut *mut c_ulonglong,
    out_processor_trace_rows: *mut usize,
    out_processor_trace_cols: *mut usize,
    // Output: program bwords
    out_program_bwords_data: *mut *mut c_ulonglong,
    out_program_bwords_len: *mut usize,
    // Output: instruction multiplicities
    out_instruction_multiplicities_data: *mut *mut u32,
    out_instruction_multiplicities_len: *mut usize,
    // Output: public output
    out_public_output_data: *mut *mut c_ulonglong,
    out_public_output_len: *mut usize,
    // Output: co-processor traces (all flat, row-major)
    out_op_stack_trace_data: *mut *mut c_ulonglong,
    out_op_stack_trace_rows: *mut usize,
    out_op_stack_trace_cols: *mut usize,
    out_ram_trace_data: *mut *mut c_ulonglong,
    out_ram_trace_rows: *mut usize,
    out_ram_trace_cols: *mut usize,
    out_program_hash_trace_data: *mut *mut c_ulonglong,
    out_program_hash_trace_rows: *mut usize,
    out_program_hash_trace_cols: *mut usize,
    out_hash_trace_data: *mut *mut c_ulonglong,
    out_hash_trace_rows: *mut usize,
    out_hash_trace_cols: *mut usize,
    out_sponge_trace_data: *mut *mut c_ulonglong,
    out_sponge_trace_rows: *mut usize,
    out_sponge_trace_cols: *mut usize,
    // Output: U32 entries (flat: [instruction, operand1, operand2, multiplicity] per entry)
    out_u32_entries_data: *mut *mut c_ulonglong,
    out_u32_entries_len: *mut usize,
    // Output: Cascade lookup multiplicities (flat: [limb, multiplicity] pairs)
    out_cascade_multiplicities_data: *mut *mut c_ulonglong,
    out_cascade_multiplicities_len: *mut usize,
    // Output: Lookup lookup multiplicities (array of 256 u64)
    out_lookup_multiplicities_256: *mut c_ulonglong,
    // Output: table lengths [program, processor, op_stack, ram, jump_stack, hash, cascade, lookup, u32]
    out_table_lengths_9: *mut usize,
) -> c_int {
    // Validate pointers
    if program_json.is_null()
        || nondet_json.is_null()
        || public_input_data.is_null()
        || out_processor_trace_data.is_null()
        || out_processor_trace_rows.is_null()
        || out_processor_trace_cols.is_null()
        || out_program_bwords_data.is_null()
        || out_program_bwords_len.is_null()
        || out_instruction_multiplicities_data.is_null()
        || out_instruction_multiplicities_len.is_null()
        || out_public_output_data.is_null()
        || out_public_output_len.is_null()
        || out_op_stack_trace_data.is_null()
        || out_op_stack_trace_rows.is_null()
        || out_op_stack_trace_cols.is_null()
        || out_ram_trace_data.is_null()
        || out_ram_trace_rows.is_null()
        || out_ram_trace_cols.is_null()
        || out_program_hash_trace_data.is_null()
        || out_program_hash_trace_rows.is_null()
        || out_program_hash_trace_cols.is_null()
        || out_hash_trace_data.is_null()
        || out_hash_trace_rows.is_null()
        || out_hash_trace_cols.is_null()
        || out_sponge_trace_data.is_null()
        || out_sponge_trace_rows.is_null()
        || out_sponge_trace_cols.is_null()
        || out_u32_entries_data.is_null()
        || out_u32_entries_len.is_null()
        || out_cascade_multiplicities_data.is_null()
        || out_cascade_multiplicities_len.is_null()
        || out_lookup_multiplicities_256.is_null()
        || out_table_lengths_9.is_null()
    {
        eprintln!("[FFI] tvm_trace_execution_with_nondet: null pointer argument");
        return -1;
    }

    // Parse program JSON
    let program_str = match std::ffi::CStr::from_ptr(program_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("[FFI] Invalid program_json UTF-8");
            return -1;
        }
    };
    
    let program: Program = match serde_json::from_str(program_str) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[FFI] Failed to parse program JSON: {}", e);
            return -1;
        }
    };
    
    // Parse NonDeterminism JSON
    let nondet_str = match std::ffi::CStr::from_ptr(nondet_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("[FFI] Invalid nondet_json UTF-8");
            return -1;
        }
    };
    
    let non_determinism: NonDeterminism = match serde_json::from_str(nondet_str) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[FFI] Failed to parse NonDeterminism JSON: {}", e);
            return -1;
        }
    };
    
    // Convert public input
    let input_slice = std::slice::from_raw_parts(public_input_data, public_input_len);
    let input_vec: Vec<BFieldElement> = input_slice
        .iter()
        .copied()
        .map(BFieldElement::new)
        .collect();
    let public_input = PublicInput::new(input_vec);
    
    eprintln!("[FFI] Running trace execution with NonDeterminism (ram_len={}, individual_tokens_len={})", 
              non_determinism.ram.len(), non_determinism.individual_tokens.len());
    
    // Run VM trace execution WITH NonDeterminism
    let (aet, output) = match VM::trace_execution(program.clone(), public_input, non_determinism) {
        Ok(res) => res,
        Err(e) => {
            eprintln!("[FFI] Trace execution failed: {:?}", e);
            return -1;
        }
    };
    
    eprintln!("[FFI] Trace execution succeeded, padded_height={}", aet.padded_height());

    // Extract processor trace (flat, row-major)
    let proc_trace = &aet.processor_trace;
    let proc_rows = proc_trace.nrows();
    let proc_cols = proc_trace.ncols();
    let proc_flat_len = proc_rows * proc_cols;
    
    let proc_layout = Layout::from_size_align(
        proc_flat_len * std::mem::size_of::<u64>(),
        std::mem::align_of::<u64>()
    ).unwrap();
    let proc_ptr = std::alloc::alloc(proc_layout) as *mut c_ulonglong;
    
    for row in 0..proc_rows {
        for col in 0..proc_cols {
            let val = proc_trace[(row, col)].value();
            *proc_ptr.add(row * proc_cols + col) = val;
        }
    }
    *out_processor_trace_data = proc_ptr;
    *out_processor_trace_rows = proc_rows;
    *out_processor_trace_cols = proc_cols;

    // Extract program bwords
    let program_bwords: Vec<BFieldElement> = program.to_bwords();
    let bwords_len = program_bwords.len();
    let bwords_layout = Layout::from_size_align(
        bwords_len * std::mem::size_of::<u64>(),
        std::mem::align_of::<u64>()
    ).unwrap();
    let bwords_ptr = std::alloc::alloc(bwords_layout) as *mut c_ulonglong;
    for (i, bw) in program_bwords.iter().enumerate() {
        *bwords_ptr.add(i) = bw.value();
    }
    *out_program_bwords_data = bwords_ptr;
    *out_program_bwords_len = bwords_len;

    // Extract instruction multiplicities
    let inst_mults = &aet.instruction_multiplicities;
    let inst_mults_len = inst_mults.len();
    let inst_mults_layout = Layout::from_size_align(
        inst_mults_len * std::mem::size_of::<u32>(),
        std::mem::align_of::<u32>()
    ).unwrap();
    let inst_mults_ptr = std::alloc::alloc(inst_mults_layout) as *mut u32;
    std::ptr::copy_nonoverlapping(inst_mults.as_ptr(), inst_mults_ptr, inst_mults_len);
    *out_instruction_multiplicities_data = inst_mults_ptr;
    *out_instruction_multiplicities_len = inst_mults_len;

    // Extract public output
    let output_len = output.len();
    let output_layout = Layout::from_size_align(
        output_len * std::mem::size_of::<u64>(),
        std::mem::align_of::<u64>()
    ).unwrap();
    let output_ptr = std::alloc::alloc(output_layout) as *mut c_ulonglong;
    for (i, o) in output.iter().enumerate() {
        *output_ptr.add(i) = o.value();
    }
    *out_public_output_data = output_ptr;
    *out_public_output_len = output_len;

    // Extract op_stack trace
    let op_stack_trace = &aet.op_stack_underflow_trace;
    let op_rows = op_stack_trace.nrows();
    let op_cols = op_stack_trace.ncols();
    let op_flat_len = op_rows * op_cols;
    let op_layout = Layout::from_size_align(
        op_flat_len * std::mem::size_of::<u64>(),
        std::mem::align_of::<u64>()
    ).unwrap();
    let op_ptr = std::alloc::alloc(op_layout) as *mut c_ulonglong;
    for row in 0..op_rows {
        for col in 0..op_cols {
            let val = op_stack_trace[(row, col)].value();
            *op_ptr.add(row * op_cols + col) = val;
        }
    }
    *out_op_stack_trace_data = op_ptr;
    *out_op_stack_trace_rows = op_rows;
    *out_op_stack_trace_cols = op_cols;

    // Extract RAM trace
    let ram_trace = &aet.ram_trace;
    let ram_rows = ram_trace.nrows();
    let ram_cols = ram_trace.ncols();
    let ram_flat_len = ram_rows * ram_cols;
    let ram_layout = Layout::from_size_align(
        ram_flat_len * std::mem::size_of::<u64>(),
        std::mem::align_of::<u64>()
    ).unwrap();
    let ram_ptr = std::alloc::alloc(ram_layout) as *mut c_ulonglong;
    for row in 0..ram_rows {
        for col in 0..ram_cols {
            let val = ram_trace[(row, col)].value();
            *ram_ptr.add(row * ram_cols + col) = val;
        }
    }
    *out_ram_trace_data = ram_ptr;
    *out_ram_trace_rows = ram_rows;
    *out_ram_trace_cols = ram_cols;

    // Extract program hash trace
    let ph_trace = &aet.program_hash_trace;
    let ph_rows = ph_trace.nrows();
    let ph_cols = ph_trace.ncols();
    let ph_flat_len = ph_rows * ph_cols;
    let ph_layout = Layout::from_size_align(
        ph_flat_len * std::mem::size_of::<u64>(),
        std::mem::align_of::<u64>()
    ).unwrap();
    let ph_ptr = std::alloc::alloc(ph_layout) as *mut c_ulonglong;
    for row in 0..ph_rows {
        for col in 0..ph_cols {
            let val = ph_trace[(row, col)].value();
            *ph_ptr.add(row * ph_cols + col) = val;
        }
    }
    *out_program_hash_trace_data = ph_ptr;
    *out_program_hash_trace_rows = ph_rows;
    *out_program_hash_trace_cols = ph_cols;

    // Extract hash trace
    let hash_trace = &aet.hash_trace;
    let hash_rows = hash_trace.nrows();
    let hash_cols = hash_trace.ncols();
    let hash_flat_len = hash_rows * hash_cols;
    let hash_layout = Layout::from_size_align(
        hash_flat_len * std::mem::size_of::<u64>(),
        std::mem::align_of::<u64>()
    ).unwrap();
    let hash_ptr = std::alloc::alloc(hash_layout) as *mut c_ulonglong;
    for row in 0..hash_rows {
        for col in 0..hash_cols {
            let val = hash_trace[(row, col)].value();
            *hash_ptr.add(row * hash_cols + col) = val;
        }
    }
    *out_hash_trace_data = hash_ptr;
    *out_hash_trace_rows = hash_rows;
    *out_hash_trace_cols = hash_cols;

    // Extract sponge trace
    let sponge_trace = &aet.sponge_trace;
    let sponge_rows = sponge_trace.nrows();
    let sponge_cols = sponge_trace.ncols();
    let sponge_flat_len = sponge_rows * sponge_cols;
    let sponge_layout = Layout::from_size_align(
        sponge_flat_len * std::mem::size_of::<u64>(),
        std::mem::align_of::<u64>()
    ).unwrap();
    let sponge_ptr = std::alloc::alloc(sponge_layout) as *mut c_ulonglong;
    for row in 0..sponge_rows {
        for col in 0..sponge_cols {
            let val = sponge_trace[(row, col)].value();
            *sponge_ptr.add(row * sponge_cols + col) = val;
        }
    }
    *out_sponge_trace_data = sponge_ptr;
    *out_sponge_trace_rows = sponge_rows;
    *out_sponge_trace_cols = sponge_cols;

    // Extract U32 entries (flat: [instruction_opcode, left_operand, right_operand, multiplicity])
    let u32_entries = &aet.u32_entries;
    let u32_len = u32_entries.len();
    let u32_flat_len = u32_len * 4;
    let u32_layout = Layout::from_size_align(
        u32_flat_len * std::mem::size_of::<u64>(),
        std::mem::align_of::<u64>()
    ).unwrap();
    let u32_ptr = std::alloc::alloc(u32_layout) as *mut c_ulonglong;
    for (idx, (entry, mult)) in u32_entries.iter().enumerate() {
        *u32_ptr.add(idx * 4 + 0) = entry.instruction.opcode_b().value();
        *u32_ptr.add(idx * 4 + 1) = entry.left_operand.value();
        *u32_ptr.add(idx * 4 + 2) = entry.right_operand.value();
        *u32_ptr.add(idx * 4 + 3) = *mult as u64;
    }
    *out_u32_entries_data = u32_ptr;
    *out_u32_entries_len = u32_len;

    // Extract cascade lookup multiplicities (flat: [limb, multiplicity] pairs)
    let cascade_mults = &aet.cascade_table_lookup_multiplicities;
    let cascade_len = cascade_mults.len();
    let cascade_flat_len = cascade_len * 2;
    let cascade_layout = Layout::from_size_align(
        cascade_flat_len * std::mem::size_of::<u64>(),
        std::mem::align_of::<u64>()
    ).unwrap();
    let cascade_ptr = std::alloc::alloc(cascade_layout) as *mut c_ulonglong;
    for (idx, (limb, mult)) in cascade_mults.iter().enumerate() {
        *cascade_ptr.add(idx * 2 + 0) = *limb as u64;
        *cascade_ptr.add(idx * 2 + 1) = *mult;
    }
    *out_cascade_multiplicities_data = cascade_ptr;
    *out_cascade_multiplicities_len = cascade_len;

    // Extract lookup lookup multiplicities (array of 256 u64)
    let lookup_mults = &aet.lookup_table_lookup_multiplicities;
    for i in 0..256 {
        *out_lookup_multiplicities_256.add(i) = lookup_mults[i];
    }

    // Extract table lengths
    let table_lengths: [usize; 9] = [
        aet.height_of_table(TableId::Program),
        aet.height_of_table(TableId::Processor),
        aet.height_of_table(TableId::OpStack),
        aet.height_of_table(TableId::Ram),
        aet.height_of_table(TableId::JumpStack),
        aet.height_of_table(TableId::Hash),
        aet.height_of_table(TableId::Cascade),
        aet.height_of_table(TableId::Lookup),
        aet.height_of_table(TableId::U32),
    ];
    std::ptr::copy_nonoverlapping(table_lengths.as_ptr(), out_table_lengths_9, 9);

    0
}

/// Free buffers allocated by tvm_trace_execution_rust_ffi
#[no_mangle]
pub unsafe extern "C" fn tvm_trace_execution_rust_ffi_free(
    processor_trace_data: *mut c_ulonglong,
    processor_trace_rows: usize,
    processor_trace_cols: usize,
    program_bwords_data: *mut c_ulonglong,
    program_bwords_len: usize,
    instruction_multiplicities_data: *mut u32,
    instruction_multiplicities_len: usize,
    public_output_data: *mut c_ulonglong,
    public_output_len: usize,
    op_stack_trace_data: *mut c_ulonglong,
    op_stack_trace_rows: usize,
    op_stack_trace_cols: usize,
    ram_trace_data: *mut c_ulonglong,
    ram_trace_rows: usize,
    ram_trace_cols: usize,
    program_hash_trace_data: *mut c_ulonglong,
    program_hash_trace_rows: usize,
    program_hash_trace_cols: usize,
    hash_trace_data: *mut c_ulonglong,
    hash_trace_rows: usize,
    hash_trace_cols: usize,
    sponge_trace_data: *mut c_ulonglong,
    sponge_trace_rows: usize,
    sponge_trace_cols: usize,
    u32_entries_data: *mut c_ulonglong,
    u32_entries_len: usize,
    cascade_multiplicities_data: *mut c_ulonglong,
    cascade_multiplicities_len: usize,
) {
    if !processor_trace_data.is_null() {
        let layout = Layout::from_size_align(
            processor_trace_rows * processor_trace_cols * std::mem::size_of::<c_ulonglong>(),
            std::mem::align_of::<c_ulonglong>(),
        ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(processor_trace_data as *mut u8, layout);
    }
    if !program_bwords_data.is_null() {
        let layout = Layout::from_size_align(
            program_bwords_len * std::mem::size_of::<c_ulonglong>(),
            std::mem::align_of::<c_ulonglong>(),
        ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(program_bwords_data as *mut u8, layout);
    }
    if !instruction_multiplicities_data.is_null() {
        let layout = Layout::from_size_align(
            instruction_multiplicities_len * std::mem::size_of::<u32>(),
            std::mem::align_of::<u32>(),
        ).unwrap_or_else(|_| Layout::new::<u32>());
        dealloc(instruction_multiplicities_data as *mut u8, layout);
    }
    if !public_output_data.is_null() {
        let layout = Layout::from_size_align(
            public_output_len * std::mem::size_of::<c_ulonglong>(),
            std::mem::align_of::<c_ulonglong>(),
        ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(public_output_data as *mut u8, layout);
    }
    if !op_stack_trace_data.is_null() {
        let layout = Layout::from_size_align(
            op_stack_trace_rows * op_stack_trace_cols * std::mem::size_of::<c_ulonglong>(),
            std::mem::align_of::<c_ulonglong>(),
        ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(op_stack_trace_data as *mut u8, layout);
    }
    if !ram_trace_data.is_null() {
        let layout = Layout::from_size_align(
            ram_trace_rows * ram_trace_cols * std::mem::size_of::<c_ulonglong>(),
            std::mem::align_of::<c_ulonglong>(),
        ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(ram_trace_data as *mut u8, layout);
    }
    if !program_hash_trace_data.is_null() {
        let layout = Layout::from_size_align(
            program_hash_trace_rows * program_hash_trace_cols * std::mem::size_of::<c_ulonglong>(),
            std::mem::align_of::<c_ulonglong>(),
        ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(program_hash_trace_data as *mut u8, layout);
    }
    if !hash_trace_data.is_null() {
        let layout = Layout::from_size_align(
            hash_trace_rows * hash_trace_cols * std::mem::size_of::<c_ulonglong>(),
            std::mem::align_of::<c_ulonglong>(),
        ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(hash_trace_data as *mut u8, layout);
    }
    if !sponge_trace_data.is_null() {
        let layout = Layout::from_size_align(
            sponge_trace_rows * sponge_trace_cols * std::mem::size_of::<c_ulonglong>(),
            std::mem::align_of::<c_ulonglong>(),
        ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(sponge_trace_data as *mut u8, layout);
    }
    if !u32_entries_data.is_null() {
        let layout = Layout::from_size_align(
            u32_entries_len * 4 * std::mem::size_of::<c_ulonglong>(),
            std::mem::align_of::<c_ulonglong>(),
        ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(u32_entries_data as *mut u8, layout);
    }
    if !cascade_multiplicities_data.is_null() {
        let layout = Layout::from_size_align(
            cascade_multiplicities_len * 2 * std::mem::size_of::<c_ulonglong>(),
            std::mem::align_of::<c_ulonglong>(),
        ).unwrap_or_else(|_| Layout::new::<c_ulonglong>());
        dealloc(cascade_multiplicities_data as *mut u8, layout);
    }
}

// ============================================================================
// Neptune Integration: Prove from JSON inputs
// ============================================================================

/// Prove from Neptune's JSON format and return bincode-serialized proof
/// 
/// This is the main entry point for GPU prover server integration.
/// Takes the same JSON inputs that Neptune sends to triton-vm-prover.
/// 
/// # Safety
/// - All string parameters must be valid null-terminated UTF-8 C strings
/// - Output pointers must be valid
/// - Caller must free output using tvm_prove_from_json_free
/// 
/// # Returns
/// - 0 on success
/// - 1 if padded height exceeds max_log2 (check out_observed_log2)
/// - -1 on error
#[no_mangle]
pub unsafe extern "C" fn tvm_prove_from_json(
    claim_json: *const c_char,
    program_json: *const c_char,
    nondet_json: *const c_char,
    max_log2_json: *const c_char,
    out_proof_bincode: *mut *mut u8,
    out_proof_len: *mut usize,
    out_observed_log2: *mut u8,
) -> c_int {
    use triton_vm::stark::Stark;
    
    // Parse inputs
    let claim_str = match std::ffi::CStr::from_ptr(claim_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("[FFI] Invalid claim_json UTF-8");
            return -1;
        }
    };
    
    let program_str = match std::ffi::CStr::from_ptr(program_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("[FFI] Invalid program_json UTF-8");
            return -1;
        }
    };
    
    let nondet_str = match std::ffi::CStr::from_ptr(nondet_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("[FFI] Invalid nondet_json UTF-8");
            return -1;
        }
    };
    
    let max_log2_str = match std::ffi::CStr::from_ptr(max_log2_json).to_str() {
        Ok(s) => s,
        Err(_) => {
            eprintln!("[FFI] Invalid max_log2_json UTF-8");
            return -1;
        }
    };
    
    // Parse JSON
    let claim: triton_vm::proof::Claim = match serde_json::from_str(claim_str) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[FFI] Failed to parse claim JSON: {}", e);
            return -1;
        }
    };
    
    let program: Program = match serde_json::from_str(program_str) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[FFI] Failed to parse program JSON: {}", e);
            return -1;
        }
    };
    
    let non_determinism: NonDeterminism = match serde_json::from_str(nondet_str) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[FFI] Failed to parse non_determinism JSON: {}", e);
            return -1;
        }
    };
    
    let max_log2_padded_height: Option<u8> = match serde_json::from_str(max_log2_str) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[FFI] Failed to parse max_log2 JSON: {}", e);
            return -1;
        }
    };
    
    eprintln!("[FFI] Parsed inputs successfully");
    eprintln!("[FFI] Claim: digest={}, input_len={}, output_len={}", 
              claim.program_digest, claim.input.len(), claim.output.len());
    
    // Verify program digest
    let computed_digest = program.hash();
    if computed_digest != claim.program_digest {
        eprintln!("[FFI] Program digest mismatch!");
        eprintln!("[FFI]   Claim:    {}", claim.program_digest);
        eprintln!("[FFI]   Computed: {}", computed_digest);
        return -1;
    }
    
    // Run trace execution
    eprintln!("[FFI] Running trace execution...");
    let public_input = PublicInput::new(claim.input.clone());
    let (aet, output) = match VM::trace_execution(program, public_input, non_determinism) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("[FFI] Trace execution failed: {:?}", e);
            return -1;
        }
    };
    
    // Verify output
    if output != claim.output {
        eprintln!("[FFI] Output mismatch!");
        return -1;
    }
    
    // Check padded height
    let padded_height = aet.padded_height();
    let log2_padded_height = padded_height.ilog2() as u8;
    *out_observed_log2 = log2_padded_height;
    
    eprintln!("[FFI] Padded height: {} (log2={})", padded_height, log2_padded_height);
    
    if let Some(limit) = max_log2_padded_height {
        if log2_padded_height > limit {
            eprintln!("[FFI] Padded height {} exceeds limit {}", log2_padded_height, limit);
            return 1; // Special return code for padded height too big
        }
    }
    
    // Run STARK prover
    eprintln!("[FFI] Running STARK prover...");
    let start = std::time::Instant::now();
    
    let stark = Stark::default();
    let proof = match stark.prove(&claim, &aet) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[FFI] STARK proving failed: {:?}", e);
            return -1;
        }
    };
    
    let prove_duration = start.elapsed();
    eprintln!("[FFI] STARK proof generated in {:?}", prove_duration);
    eprintln!("[FFI] Proof has {} BFieldElements", proof.0.len());
    
    // Serialize to bincode
    eprintln!("[FFI] Serializing proof to bincode...");
    let proof_bincode = match bincode::serialize(&proof) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("[FFI] Bincode serialization failed: {}", e);
            return -1;
        }
    };
    
    eprintln!("[FFI] Proof serialized: {} bytes", proof_bincode.len());
    
    // Allocate output buffer
    let layout = Layout::from_size_align(
        proof_bincode.len(),
        std::mem::align_of::<u8>(),
    ).unwrap_or_else(|_| Layout::new::<u8>());
    
    let proof_ptr = alloc(layout);
    if proof_ptr.is_null() {
        eprintln!("[FFI] Failed to allocate output buffer");
        return -1;
    }
    
    std::ptr::copy_nonoverlapping(proof_bincode.as_ptr(), proof_ptr, proof_bincode.len());
    
    *out_proof_bincode = proof_ptr;
    *out_proof_len = proof_bincode.len();
    
    eprintln!("[FFI] Proof ready to return");
    0
}

/// Free proof buffer allocated by tvm_prove_from_json
#[no_mangle]
pub unsafe extern "C" fn tvm_prove_from_json_free(
    proof_bincode: *mut u8,
    proof_len: usize,
) {
    if !proof_bincode.is_null() && proof_len > 0 {
        let layout = Layout::from_size_align(
            proof_len,
            std::mem::align_of::<u8>(),
        ).unwrap_or_else(|_| Layout::new::<u8>());
        dealloc(proof_bincode, layout);
    }
}

