use std::slice;

use ndarray::Array2;
use triton_vm::challenges::Challenges;
use triton_vm::prelude::{BFieldElement, XFieldElement};

// Include the full generated degree lowering table code (with fixed crate:: references)
include!("../degree_lowering_table_fixed.rs");

fn u64_to_bfe(value: u64) -> BFieldElement {
    BFieldElement::from(value)
}

fn u64_to_xfe(chunk: &[u64]) -> XFieldElement {
    debug_assert_eq!(chunk.len(), 3);
    XFieldElement::from([
        BFieldElement::from(chunk[0]),
        BFieldElement::from(chunk[1]),
        BFieldElement::from(chunk[2]),
    ])
}

fn xfe_to_u64(chunk: &mut [u64], xfe: &XFieldElement) {
    debug_assert_eq!(chunk.len(), 3);
    let coeffs = xfe.coefficients;
    chunk[0] = coeffs[0].value();
    chunk[1] = coeffs[1].value();
    chunk[2] = coeffs[2].value();
}

fn challenges_from_flat(slice: &[u64], count: usize) -> Challenges {
    assert_eq!(slice.len(), count * 3);
    let mut challenges = Vec::with_capacity(count);
    for coeffs in slice.chunks_exact(3) {
        challenges.push(u64_to_xfe(coeffs));
    }
    let array: [XFieldElement; Challenges::COUNT] = challenges.try_into().expect("invalid challenge count");
    Challenges { challenges: array }
}

fn array2_from_bfe_flat(rows: usize, cols: usize, data: &[u64]) -> Array2<BFieldElement> {
    let mut elements = Vec::with_capacity(rows * cols);
    elements.extend(data.iter().map(|&v| u64_to_bfe(v)));
    Array2::from_shape_vec((rows, cols), elements).expect("invalid main table shape")
}

fn xfe_array2_from_flat(rows: usize, cols: usize, data: &[u64]) -> Array2<XFieldElement> {
    let mut elements = Vec::with_capacity(rows * cols);
    for coeffs in data.chunks_exact(3) {
        elements.push(u64_to_xfe(coeffs));
    }
    assert_eq!(elements.len(), rows * cols);
    Array2::from_shape_vec((rows, cols), elements).expect("invalid aux table shape")
}

#[no_mangle]
pub extern "C" fn degree_lowering_fill_main_columns(
    table_ptr: *mut u64,
    num_rows: usize,
    num_cols: usize,
) {
    assert!(!table_ptr.is_null());
    let total = num_rows * num_cols;
    let slice = unsafe { slice::from_raw_parts_mut(table_ptr, total) };
    let mut table = array2_from_bfe_flat(num_rows, num_cols, slice);
    DegreeLoweringTable::fill_derived_main_columns(table.view_mut());
    for (dst, value) in slice.iter_mut().zip(table.iter()) {
        *dst = value.value();
    }
}

#[no_mangle]
pub extern "C" fn degree_lowering_fill_aux_columns(
    main_ptr: *const u64,
    num_rows: usize,
    main_cols: usize,
    aux_ptr: *mut u64,
    aux_cols: usize,
    challenges_ptr: *const u64,
    challenges_len: usize,
) {
    assert!(!main_ptr.is_null());
    assert!(!aux_ptr.is_null());
    assert!(!challenges_ptr.is_null());

    let main_slice = unsafe { slice::from_raw_parts(main_ptr, num_rows * main_cols) };
    let aux_slice = unsafe { slice::from_raw_parts_mut(aux_ptr, num_rows * aux_cols * 3) };
    let challenges_slice = unsafe { slice::from_raw_parts(challenges_ptr, challenges_len * 3) };

    let main_table = array2_from_bfe_flat(num_rows, main_cols, main_slice);
    let mut aux_table = xfe_array2_from_flat(num_rows, aux_cols, aux_slice);
    let challenges = challenges_from_flat(challenges_slice, challenges_len);

    // Use the generated fill_derived_aux_columns function from degree_lowering_table.rs
    DegreeLoweringTable::fill_derived_aux_columns(
        main_table.view(),
        aux_table.view_mut(),
        &challenges,
    );

    for (chunk, value) in aux_slice.chunks_exact_mut(3).zip(aux_table.iter()) {
        xfe_to_u64(chunk, value);
    }
}

// (debug helpers removed)
