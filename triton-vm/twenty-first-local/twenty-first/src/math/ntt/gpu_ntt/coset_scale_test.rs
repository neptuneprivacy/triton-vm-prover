// Unit tests for GPU coset scaling
// These tests verify correctness against CPU reference implementation

#[cfg(test)]
mod tests {
    use crate::math::b_field_element::BFieldElement;
    use crate::math::ntt::gpu_ntt::get_gpu_context;
    use num_traits::{One, Zero};

    /// CPU reference implementation of coset scaling
    fn cpu_coset_scale_reference(coeffs: &mut [BFieldElement], offset: BFieldElement) {
        let mut power = BFieldElement::one();
        for coeff in coeffs.iter_mut() {
            *coeff = *coeff * power;
            power = power * offset;
        }
    }

    #[test]
    fn test_coset_scale_small() {
        // Test with a small array (8 elements)
        let offset = BFieldElement::new(7);
        let mut cpu_data = vec![
            BFieldElement::new(1),
            BFieldElement::new(2),
            BFieldElement::new(3),
            BFieldElement::new(4),
            BFieldElement::new(5),
            BFieldElement::new(6),
            BFieldElement::new(7),
            BFieldElement::new(8),
        ];
        let mut gpu_data = cpu_data.clone();

        // CPU reference
        cpu_coset_scale_reference(&mut cpu_data, offset);

        // GPU implementation
        if let Some(ctx) = get_gpu_context() {
            let mut gpu_refs = vec![gpu_data.as_mut_slice()];
            ctx.coset_scale_bfield_chunked(&mut gpu_refs, offset, 1, "test")
                .expect("GPU coset scale failed");

            // Compare results
            for (i, (&cpu, &gpu)) in cpu_data.iter().zip(gpu_data.iter()).enumerate() {
                assert_eq!(
                    cpu.value(),
                    gpu.value(),
                    "Mismatch at index {}: CPU={}, GPU={}",
                    i,
                    cpu.value(),
                    gpu.value()
                );
            }
        } else {
            eprintln!("GPU context not available, skipping test");
        }
    }

    #[test]
    fn test_coset_scale_powers_of_two() {
        // Test with power-of-2 sizes
        for log2_size in 4..12 {
            // Test from 16 to 2048 elements
            let size = 1 << log2_size;
            let offset = BFieldElement::new(7);

            let mut cpu_data: Vec<BFieldElement> =
                (0..size).map(|i| BFieldElement::new(i as u64 + 1)).collect();
            let mut gpu_data = cpu_data.clone();

            // CPU reference
            cpu_coset_scale_reference(&mut cpu_data, offset);

            // GPU implementation
            if let Some(ctx) = get_gpu_context() {
                let mut gpu_refs = vec![gpu_data.as_mut_slice()];
                ctx.coset_scale_bfield_chunked(&mut gpu_refs, offset, 1, "test")
                    .expect("GPU coset scale failed");

                // Compare results
                for (i, (&cpu, &gpu)) in cpu_data.iter().zip(gpu_data.iter()).enumerate() {
                    assert_eq!(
                        cpu.value(),
                        gpu.value(),
                        "Size {}, index {}: CPU={}, GPU={}",
                        size,
                        i,
                        cpu.value(),
                        gpu.value()
                    );
                }
            } else {
                eprintln!("GPU context not available, skipping test");
                break;
            }
        }
    }

    #[test]
    fn test_coset_scale_zero_coefficients() {
        // Test with zero coefficients
        let offset = BFieldElement::new(7);
        let size = 64;
        let mut cpu_data = vec![BFieldElement::zero(); size];
        let mut gpu_data = cpu_data.clone();

        // CPU reference
        cpu_coset_scale_reference(&mut cpu_data, offset);

        // GPU implementation
        if let Some(ctx) = get_gpu_context() {
            let mut gpu_refs = vec![gpu_data.as_mut_slice()];
            ctx.coset_scale_bfield_chunked(&mut gpu_refs, offset, 1, "test")
                .expect("GPU coset scale failed");

            // All should still be zero
            for (i, (&cpu, &gpu)) in cpu_data.iter().zip(gpu_data.iter()).enumerate() {
                assert_eq!(
                    cpu.value(),
                    gpu.value(),
                    "Index {}: CPU={}, GPU={}",
                    i,
                    cpu.value(),
                    gpu.value()
                );
            }
        } else {
            eprintln!("GPU context not available, skipping test");
        }
    }

    #[test]
    fn test_coset_scale_batch() {
        // Test batching multiple arrays
        let offset = BFieldElement::new(7);
        let size = 128;
        let batch_size = 5;

        let mut cpu_arrays: Vec<Vec<BFieldElement>> = (0..batch_size)
            .map(|b| {
                (0..size)
                    .map(|i| BFieldElement::new((b * 1000 + i) as u64))
                    .collect()
            })
            .collect();

        let mut gpu_arrays = cpu_arrays.clone();

        // CPU reference
        for arr in cpu_arrays.iter_mut() {
            cpu_coset_scale_reference(arr, offset);
        }

        // GPU implementation
        if let Some(ctx) = get_gpu_context() {
            let mut gpu_refs: Vec<&mut [BFieldElement]> =
                gpu_arrays.iter_mut().map(|v| v.as_mut_slice()).collect();
            ctx.coset_scale_bfield_chunked(&mut gpu_refs, offset, batch_size, "test")
                .expect("GPU coset scale failed");

            // Compare results
            for (batch_idx, (cpu_arr, gpu_arr)) in
                cpu_arrays.iter().zip(gpu_arrays.iter()).enumerate()
            {
                for (i, (&cpu, &gpu)) in cpu_arr.iter().zip(gpu_arr.iter()).enumerate() {
                    assert_eq!(
                        cpu.value(),
                        gpu.value(),
                        "Batch {}, index {}: CPU={}, GPU={}",
                        batch_idx,
                        i,
                        cpu.value(),
                        gpu.value()
                    );
                }
            }
        } else {
            eprintln!("GPU context not available, skipping test");
        }
    }
}
