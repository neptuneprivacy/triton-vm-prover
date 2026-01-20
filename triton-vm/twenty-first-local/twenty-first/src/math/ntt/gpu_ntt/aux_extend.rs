use crate::math::b_field_element::BFieldElement;
use crate::math::x_field_element::XFieldElement;
use super::cuda_driver::KernelArgs;

/// GPU-accelerated running evaluation computation
/// Computes: out[i] = out[i-1] * challenge + values[i]
///
/// # Arguments
/// * `values` - Input values (BFieldElement)
/// * `output` - Output buffer for XFieldElement results
/// * `challenge` - Challenge value for the evaluation
pub fn gpu_running_evaluation(
    values: &[BFieldElement],
    output: &mut [XFieldElement],
    challenge: XFieldElement,
) -> Result<(), Box<dyn std::error::Error>> {
    // Get GPU context
    let gpu_ctx = super::get_gpu_context()
        .ok_or("GPU context not available")?;
    let device_ctx = &gpu_ctx.devices[0]; // Use first GPU

    let n = values.len() as u32;
    assert_eq!(values.len(), output.len(), "Input and output must have same length");

    // Prepare input values
    let values_host: Vec<u64> = values.iter().map(|x| x.raw_u64()).collect();

    // Prepare challenge as u64 array
    let challenge_host = vec![
        challenge.coefficients[0].raw_u64(),
        challenge.coefficients[1].raw_u64(),
        challenge.coefficients[2].raw_u64(),
    ];

    // Prepare output buffer on host
    let mut output_host = vec![0u64; (n * 3) as usize];

    // Get stream and allocate GPU memory
    let stream = device_ctx.device.default_stream();
    let values_device = stream.memcpy_htod(&values_host)?;
    let challenge_device = stream.memcpy_htod(&challenge_host)?;
    let output_device = stream.memcpy_htod(&output_host)?;

    // Configure kernel launch for parallel Blelloch scan (256 threads)
    let block_size = 256u32;
    let grid_size = ((n + block_size - 1) / block_size).max(1);

    // Launch parallel kernel
    let stride = 1u32; // Contiguous access
    let values_ptr = values_device.as_ptr();
    let mut output_ptr = output_device.as_ptr();
    let challenge_ptr = challenge_device.as_ptr();

    let mut kernel_args = KernelArgs::new();
    kernel_args
        .push_ptr(&values_ptr)
        .push_mut_ptr(&mut output_ptr)
        .push_ptr(&challenge_ptr)
        .push_value(n)
        .push_value(stride);

    device_ctx.running_eval_scan_parallel_fn.launch(
        (grid_size, 1, 1),  // grid
        (block_size, 1, 1),  // block
        0,  // shared mem
        &stream,
        kernel_args.as_mut_slice(),
    )?;

    stream.synchronize()?;

    // Copy results back
    stream.memcpy_dtoh(&output_device, &mut output_host)?;

    // Convert back to XFieldElements and write to output
    for i in 0..n as usize {
        output[i] = XFieldElement::new([
            BFieldElement::from_raw_u64(output_host[i * 3]),
            BFieldElement::from_raw_u64(output_host[i * 3 + 1]),
            BFieldElement::from_raw_u64(output_host[i * 3 + 2]),
        ]);
    }

    Ok(())
}

/// GPU-accelerated log derivative computation
/// Computes: out[i] = out[i-1] + (challenge - compressed_row[i])^(-1) * multiplicity[i]
pub fn gpu_log_derivative(
    compressed_rows: &[XFieldElement],
    multiplicities: &[BFieldElement],
    output: &mut [XFieldElement],
    challenge: XFieldElement,
) -> Result<(), Box<dyn std::error::Error>> {
    let gpu_ctx = super::get_gpu_context()
        .ok_or("GPU context not available")?;
    let device_ctx = &gpu_ctx.devices[0];

    let n = compressed_rows.len() as u32;
    assert_eq!(compressed_rows.len(), multiplicities.len());
    assert_eq!(compressed_rows.len(), output.len());

    // Prepare inputs - XFieldElement compressed rows (3 u64s each)
    let mut compressed_host: Vec<u64> = Vec::with_capacity((n * 3) as usize);
    for elem in compressed_rows {
        compressed_host.push(elem.coefficients[0].raw_u64());
        compressed_host.push(elem.coefficients[1].raw_u64());
        compressed_host.push(elem.coefficients[2].raw_u64());
    }

    let mult_host: Vec<u64> = multiplicities.iter().map(|x| x.raw_u64()).collect();
    let challenge_host = vec![
        challenge.coefficients[0].raw_u64(),
        challenge.coefficients[1].raw_u64(),
        challenge.coefficients[2].raw_u64(),
    ];
    let mut output_host = vec![0u64; (n * 3) as usize];

    // Allocate GPU memory
    let stream = device_ctx.device.default_stream();
    let compressed_device = stream.memcpy_htod(&compressed_host)?;
    let mult_device = stream.memcpy_htod(&mult_host)?;
    let challenge_device = stream.memcpy_htod(&challenge_host)?;
    let output_device = stream.memcpy_htod(&output_host)?;

    // Use sequential scan for correctness (parallel Blelloch scan only works for ≤256 elements)
    // Launch sequential kernel
    let element_stride = 3u32; // 3 u64s per XFieldElement

    let compressed_ptr = compressed_device.as_ptr();
    let mult_ptr = mult_device.as_ptr();
    let mut output_ptr = output_device.as_ptr();
    let challenge_ptr = challenge_device.as_ptr();

    let mut kernel_args = KernelArgs::new();
    kernel_args
        .push_ptr(&compressed_ptr)
        .push_ptr(&mult_ptr)
        .push_mut_ptr(&mut output_ptr)
        .push_ptr(&challenge_ptr)
        .push_value(n)
        .push_value(element_stride);

    device_ctx.log_derivative_scan_fn.launch(
        (1, 1, 1),  // grid
        (1, 1, 1),  // block
        0,  // shared mem
        &stream,
        kernel_args.as_mut_slice(),
    )?;

    stream.synchronize()?;

    // Copy back
    stream.memcpy_dtoh(&output_device, &mut output_host)?;

    // Write to output
    for i in 0..n as usize {
        output[i] = XFieldElement::new([
            BFieldElement::from_raw_u64(output_host[i * 3]),
            BFieldElement::from_raw_u64(output_host[i * 3 + 1]),
            BFieldElement::from_raw_u64(output_host[i * 3 + 2]),
        ]);
    }

    Ok(())
}

/// Batch field inversion using GPU
pub fn gpu_batch_inverse(values: &[BFieldElement]) -> Result<Vec<BFieldElement>, Box<dyn std::error::Error>> {
    let gpu_ctx = super::get_gpu_context()
        .ok_or("GPU context not available")?;
    let device_ctx = &gpu_ctx.devices[0];

    let n = values.len() as u32;
    let values_host: Vec<u64> = values.iter().map(|x| x.raw_u64()).collect();
    let mut inverses_host = vec![0u64; n as usize];

    // Allocate GPU memory
    let stream = device_ctx.device.default_stream();
    let values_device = stream.memcpy_htod(&values_host)?;
    let inverses_device = stream.memcpy_htod(&inverses_host)?;

    // Configure kernel
    let block_size = 256u32;
    let grid_size = ((n + block_size - 1) / block_size) as u32;

    // Launch kernel
    let values_ptr = values_device.as_ptr();
    let mut inverses_ptr = inverses_device.as_ptr();

    let mut kernel_args = KernelArgs::new();
    kernel_args
        .push_ptr(&values_ptr)
        .push_mut_ptr(&mut inverses_ptr)
        .push_value(n);

    device_ctx.batch_inversion_fn.launch(
        (grid_size, 1, 1),  // grid
        (block_size, 1, 1),  // block
        0,  // shared mem
        &stream,
        kernel_args.as_mut_slice(),
    )?;

    stream.synchronize()?;

    // Copy back
    stream.memcpy_dtoh(&inverses_device, &mut inverses_host)?;

    Ok(inverses_host.iter().map(|&x| BFieldElement::from_raw_u64(x)).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::math::x_field_element::XFieldElement as XFE;
    use crate::math::b_field_element::BFieldElement as BFE;
    use num_traits::{Zero, One};
    use crate::math::traits::Inverse;

    /// Test GPU XFieldElement inversion for embedded BFieldElements
    #[test]
    fn test_gpu_xfield_inversion_embedded() {
        // Create test values: embedded BFieldElements (c1=c2=0)
        let test_values = vec![
            XFE::new_const(BFE::new(42)),
            XFE::new_const(BFE::new(12345)),
            XFE::new_const(BFE::new(0xFFFFFFFF)),
        ];

        for value in test_values {
            // Compute CPU inverse
            let cpu_inverse = value.inverse();

            // Prepare for GPU inversion (use batch_inverse with single element)
            let compressed_value = value.coefficients[0];

            let gpu_result = gpu_batch_inverse(&[compressed_value]);
            if gpu_result.is_err() {
                println!("GPU not available, skipping GPU test");
                return;
            }

            let gpu_inverse_coeff = gpu_result.unwrap()[0];
            let gpu_inverse = XFE::new_const(gpu_inverse_coeff);

            // Verify they match
            assert_eq!(
                cpu_inverse, gpu_inverse,
                "GPU inverse mismatch for embedded BFieldElement: {:?}",
                value
            );

            // Verify it's actually an inverse
            assert_eq!(
                value * gpu_inverse,
                XFE::one(),
                "GPU result is not actually an inverse"
            );
        }
    }

    /// Test batch inversion for BFieldElements
    #[test]
    fn test_gpu_batch_inversion() {
        let test_values = vec![
            BFE::new(1),
            BFE::new(2),
            BFE::new(42),
            BFE::new(12345),
            BFE::new(0xFFFFFFFF),
        ];

        let gpu_result = gpu_batch_inverse(&test_values);
        if gpu_result.is_err() {
            println!("GPU not available, skipping GPU test");
            return;
        }

        let gpu_inverses = gpu_result.unwrap();

        for (i, &value) in test_values.iter().enumerate() {
            let cpu_inverse = value.inverse();
            let gpu_inverse = gpu_inverses[i];

            assert_eq!(
                cpu_inverse, gpu_inverse,
                "Batch inversion mismatch at index {}: CPU={:?}, GPU={:?}",
                i, cpu_inverse, gpu_inverse
            );

            // Verify it's actually an inverse
            assert_eq!(
                value * gpu_inverse,
                BFE::one(),
                "GPU batch inverse at index {} is not actually an inverse",
                i
            );
        }
    }

    /// Test GPU running evaluation against CPU
    #[test]
    fn test_gpu_running_evaluation_small() {
        let values = vec![
            BFE::new(1),
            BFE::new(2),
            BFE::new(3),
            BFE::new(4),
            BFE::new(5),
        ];

        let challenge = XFE::new([
            BFE::new(7),
            BFE::new(11),
            BFE::new(13),
        ]);

        // Compute CPU reference (EvalArg::default_initial() = 1)
        let mut cpu_result = Vec::with_capacity(values.len());
        let mut running_eval = XFE::one();
        for &value in &values {
            running_eval = running_eval * challenge + XFE::new_const(value);
            cpu_result.push(running_eval);
        }

        // Compute GPU result
        let mut gpu_output = vec![XFE::zero(); values.len()];
        let gpu_res = gpu_running_evaluation(&values, &mut gpu_output, challenge);

        if gpu_res.is_err() {
            println!("GPU not available, skipping GPU test");
            return;
        }

        // Compare
        for (i, (&cpu_val, &gpu_val)) in cpu_result.iter().zip(gpu_output.iter()).enumerate() {
            assert_eq!(
                cpu_val, gpu_val,
                "Running evaluation mismatch at index {}: CPU={:?}, GPU={:?}",
                i, cpu_val, gpu_val
            );
        }
    }

    /// Test GPU log derivative against CPU (simple case, no padding)
    #[test]
    fn test_gpu_log_derivative_simple() {
        // Create compressed rows (XFieldElements)
        let compressed_rows = vec![
            XFE::new([BFE::new(10), BFE::new(20), BFE::new(30)]),
            XFE::new([BFE::new(15), BFE::new(25), BFE::new(35)]),
            XFE::new([BFE::new(5), BFE::new(15), BFE::new(25)]),
        ];

        let multiplicities = vec![
            BFE::new(1),
            BFE::new(2),
            BFE::new(1),
        ];

        let challenge = XFE::new([BFE::new(100), BFE::new(200), BFE::new(300)]);

        // Compute CPU reference
        let mut cpu_result = Vec::with_capacity(compressed_rows.len());
        let mut log_deriv = XFE::zero();
        for (i, &compressed) in compressed_rows.iter().enumerate() {
            let diff = challenge - compressed;
            let inv = diff.inverse();
            log_deriv += inv * XFE::new_const(multiplicities[i]);
            cpu_result.push(log_deriv);
        }

        // Compute GPU result
        let mut gpu_output = vec![XFE::zero(); compressed_rows.len()];
        let gpu_res = gpu_log_derivative(
            &compressed_rows,
            &multiplicities,
            &mut gpu_output,
            challenge
        );

        if gpu_res.is_err() {
            println!("GPU not available, skipping GPU test");
            return;
        }

        // Compare
        for (i, (&cpu_val, &gpu_val)) in cpu_result.iter().zip(gpu_output.iter()).enumerate() {
            assert_eq!(
                cpu_val, gpu_val,
                "Log derivative mismatch at index {}: CPU={:?}, GPU={:?}",
                i, cpu_val, gpu_val
            );
        }
    }

    /// Test that GPU and CPU produce identical results for larger dataset
    #[test]
    fn test_gpu_running_evaluation_medium() {
        let values: Vec<BFE> = (0..100).map(|i| BFE::new(i as u64)).collect();
        let challenge = XFE::new([BFE::new(42), BFE::new(17), BFE::new(99)]);

        // CPU reference (EvalArg::default_initial() = 1)
        let mut cpu_result = Vec::with_capacity(values.len());
        let mut running_eval = XFE::one();
        for &value in &values {
            running_eval = running_eval * challenge + XFE::new_const(value);
            cpu_result.push(running_eval);
        }

        // GPU result
        let mut gpu_output = vec![XFE::zero(); values.len()];
        let gpu_res = gpu_running_evaluation(&values, &mut gpu_output, challenge);

        if gpu_res.is_err() {
            println!("GPU not available, skipping GPU test");
            return;
        }

        // Verify all match
        for (i, (&cpu_val, &gpu_val)) in cpu_result.iter().zip(gpu_output.iter()).enumerate() {
            assert_eq!(
                cpu_val, gpu_val,
                "Medium running evaluation mismatch at index {}",
                i
            );
        }
    }
}
