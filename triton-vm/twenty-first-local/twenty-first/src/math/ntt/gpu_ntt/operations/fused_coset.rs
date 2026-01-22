use super::super::types::GpuNttContext;
use crate::math::b_field_element::BFieldElement;
use crate::math::x_field_element::XFieldElement;
use super::super::cuda_driver::KernelArgs;
use std::time::Instant;

impl GpuNttContext {
    /// Execute fused coset+NTT batch operation on a specific stream
    /// Stream-aware version for BFieldElement
    pub(crate) fn execute_fused_coset_ntt_on_stream_bfield(
        dev_ctx: &super::super::types::GpuDeviceContext,
        stream: &super::super::cuda_driver::CudaStream,
        data_arrays: &mut [&mut [BFieldElement]],
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
    ) -> Result<
        (
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
        ),
        Box<dyn std::error::Error>,
    > {
        if data_arrays.is_empty() {
            return Ok((
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
            ));
        }

        let len = data_arrays[0].len();
        assert!(len.is_power_of_two(), "Coset+NTT length must be power of 2");

        // Verify all arrays are same length
        for arr in data_arrays.iter() {
            assert_eq!(arr.len(), len, "All arrays in batch must have same length");
        }

        let log2_len = len.trailing_zeros();
        let batch_size = data_arrays.len();

        // Prep phase: Serialize all arrays
        let prep_start = Instant::now();
        let mut raw_data: Vec<u64> = Vec::with_capacity(len * batch_size);
        for arr in data_arrays.iter() {
            for elem in arr.iter() {
                raw_data.push(elem.raw_u64());
            }
        }
        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);
        let prep_time = prep_start.elapsed();

        // Upload phase
        let upload_start = Instant::now();
        let d_data = stream.memcpy_htod(&raw_data)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let upload_time = upload_start.elapsed();

        // Kernel phase - fused coset + NTT on stream
        let kernel_start = Instant::now();
        let mut d_data_ptr = d_data.as_ptr();
        let d_twiddles_ptr = d_twiddles.as_ptr();
        let offset_raw = offset.raw_u64();

        let mut kernel_args = KernelArgs::new();
        kernel_args
            .push_mut_ptr(&mut d_data_ptr)
            .push_value(offset_raw)
            .push_value(len as u64)
            .push_ptr(&d_twiddles_ptr)
            .push_value(log2_len);

        dev_ctx.ntt_bfield_fused_coset_fn.launch(
            (batch_size as u32, 1, 1),  // grid
            (super::super::get_gpu_ntt_block_size(), 1, 1),  // block
            0,  // shared mem
            stream,
            kernel_args.as_mut_slice(),
        )?;
        // Synchronize stream to wait for kernel to complete before download
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download phase
        let download_start = Instant::now();
        stream.memcpy_dtoh(&d_data, &mut raw_data)?;
        let download_time = download_start.elapsed();

        // Postproc phase: Deserialize results
        let postproc_start = Instant::now();
        for (batch_idx, arr) in data_arrays.iter_mut().enumerate() {
            let start = batch_idx * len;
            for (i, elem) in arr.iter_mut().enumerate() {
                *elem = BFieldElement::from_raw_u64(raw_data[start + i]);
            }
        }
        let postproc_time = postproc_start.elapsed();

        Ok((
            prep_time + postproc_time,
            upload_time,
            kernel_time,
            download_time,
        ))
    }

    /// Execute fused coset+NTT batch operation on a specific stream
    /// Stream-aware version for XFieldElement
    pub(crate) fn execute_fused_coset_ntt_on_stream_xfield(
        dev_ctx: &super::super::types::GpuDeviceContext,
        stream: &super::super::cuda_driver::CudaStream,
        data_arrays: &mut [&mut [XFieldElement]],
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
    ) -> Result<
        (
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
        ),
        Box<dyn std::error::Error>,
    > {
        if data_arrays.is_empty() {
            return Ok((
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
            ));
        }

        let len = data_arrays[0].len();
        assert!(len.is_power_of_two(), "Coset+NTT length must be power of 2");

        // Verify all arrays are same length
        for arr in data_arrays.iter() {
            assert_eq!(arr.len(), len, "All arrays in batch must have same length");
        }

        let log2_len = len.trailing_zeros();
        let batch_size = data_arrays.len();

        // Prep phase: Serialize all arrays (3 u64s per XFieldElement)
        let prep_start = Instant::now();
        let mut raw_data: Vec<u64> = Vec::with_capacity(len * batch_size * 3);
        for arr in data_arrays.iter() {
            for elem in arr.iter() {
                raw_data.push(elem.coefficients[0].raw_u64());
                raw_data.push(elem.coefficients[1].raw_u64());
                raw_data.push(elem.coefficients[2].raw_u64());
            }
        }
        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);
        let prep_time = prep_start.elapsed();

        // Upload phase
        let upload_start = Instant::now();
        let d_data = stream.memcpy_htod(&raw_data)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let upload_time = upload_start.elapsed();

        // Kernel phase - fused coset + NTT on stream
        let kernel_start = Instant::now();
        let mut d_data_ptr = d_data.as_ptr();
        let d_twiddles_ptr = d_twiddles.as_ptr();
        let offset_raw = offset.raw_u64();

        let mut kernel_args = KernelArgs::new();
        kernel_args
            .push_mut_ptr(&mut d_data_ptr)
            .push_value(offset_raw)
            .push_value(len as u64)
            .push_ptr(&d_twiddles_ptr)
            .push_value(log2_len);

        dev_ctx.ntt_xfield_fused_coset_fn.launch(
            (batch_size as u32, 1, 1),  // grid
            (super::super::get_gpu_ntt_block_size(), 1, 1),  // block
            0,  // shared mem
            stream,
            kernel_args.as_mut_slice(),
        )?;
        // Synchronize stream to wait for kernel to complete before download
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download phase
        let download_start = Instant::now();
        stream.memcpy_dtoh(&d_data, &mut raw_data)?;
        let download_time = download_start.elapsed();

        // Postproc phase: Deserialize results
        let postproc_start = Instant::now();
        for (batch_idx, arr) in data_arrays.iter_mut().enumerate() {
            let start = batch_idx * len * 3;
            for (i, elem) in arr.iter_mut().enumerate() {
                let idx = start + i * 3;
                *elem = XFieldElement::new([
                    BFieldElement::from_raw_u64(raw_data[idx]),
                    BFieldElement::from_raw_u64(raw_data[idx + 1]),
                    BFieldElement::from_raw_u64(raw_data[idx + 2]),
                ]);
            }
        }
        let postproc_time = postproc_start.elapsed();

        Ok((
            prep_time + postproc_time,
            upload_time,
            kernel_time,
            download_time,
        ))
    }

    /// Execute fused coset+NTT on ALL arrays in a single batch (no chunking)
    /// This eliminates per-chunk serialization overhead by processing everything at once
    /// For BFieldElement arrays
    pub(crate) fn execute_fused_coset_ntt_batch_unchunked_bfield(
        &self,
        data_arrays: &mut [&mut [BFieldElement]],
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if data_arrays.is_empty() {
            return Ok(());
        }

        let operation_start = Instant::now();
        let len = data_arrays[0].len();
        assert!(len.is_power_of_two(), "Coset+NTT length must be power of 2");

        // Verify all arrays are same length
        for arr in data_arrays.iter() {
            assert_eq!(arr.len(), len, "All arrays in batch must have same length");
        }

        let log2_len = len.trailing_zeros();
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        eprintln!(
            "\n[GPU Batch BField Fused Coset+NTT - {}] Processing {} arrays (UNCHUNKED - single GPU call)",
            phase_name,
            batch_size
        );

        // Prep phase: Serialize all arrays in ONE go (not per chunk)
        let prep_start = Instant::now();
        let mut raw_data: Vec<u64> = Vec::with_capacity(len * batch_size);
        for arr in data_arrays.iter() {
            for elem in arr.iter() {
                raw_data.push(elem.raw_u64());
            }
        }
        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);
        let prep_time = prep_start.elapsed();

        // Upload phase: Single upload for all data
        let upload_start = Instant::now();
        let stream = dev_ctx.device.default_stream();
        let d_data = stream.memcpy_htod(&raw_data)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let upload_time = upload_start.elapsed();

        // Kernel phase: Single kernel launch for all arrays
        let kernel_start = Instant::now();
        let mut d_data_ptr = d_data.as_ptr();
        let d_twiddles_ptr = d_twiddles.as_ptr();
        let offset_raw = offset.raw_u64();

        let mut kernel_args = KernelArgs::new();
        kernel_args
            .push_mut_ptr(&mut d_data_ptr)
            .push_value(offset_raw)
            .push_value(len as u64)
            .push_ptr(&d_twiddles_ptr)
            .push_value(log2_len);

        dev_ctx.ntt_bfield_fused_coset_fn.launch(
            (batch_size as u32, 1, 1),  // grid
            (super::super::get_gpu_ntt_block_size(), 1, 1),  // block
            0,  // shared mem
            &stream,
            kernel_args.as_mut_slice(),
        )?;
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download phase: Single download for all results
        let download_start = Instant::now();
        stream.memcpy_dtoh(&d_data, &mut raw_data)?;
        let download_time = download_start.elapsed();

        // Postproc phase: Deserialize all results in ONE go (not per chunk)
        let postproc_start = Instant::now();
        for (batch_idx, arr) in data_arrays.iter_mut().enumerate() {
            let start = batch_idx * len;
            for (i, elem) in arr.iter_mut().enumerate() {
                *elem = BFieldElement::from_raw_u64(raw_data[start + i]);
            }
        }
        let postproc_time = postproc_start.elapsed();

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Single Batch] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms, postproc={:.1}ms)",
            batch_size,
            total_time.as_secs_f64() * 1000.0,
            prep_time.as_secs_f64() * 1000.0,
            upload_time.as_secs_f64() * 1000.0,
            kernel_time.as_secs_f64() * 1000.0,
            download_time.as_secs_f64() * 1000.0,
            postproc_time.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// Execute fused coset+NTT on ALL arrays in a single batch (no chunking)
    /// This eliminates per-chunk serialization overhead by processing everything at once
    /// For XFieldElement arrays
    ///
    /// Execute fused coset+NTT for XFieldElement batch operation (UNCHUNKED)
    /// Uses dual-GPU batch-splitting parallelization when available
    pub(crate) fn execute_fused_coset_ntt_batch_unchunked_xfield(
        &self,
        data_arrays: &mut [&mut [XFieldElement]],
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if data_arrays.is_empty() {
            return Ok(());
        }

        let operation_start = Instant::now();
        let len = data_arrays[0].len();
        assert!(len.is_power_of_two(), "Coset+NTT length must be power of 2");

        // Verify all arrays are same length
        for arr in data_arrays.iter() {
            assert_eq!(arr.len(), len, "All arrays in batch must have same length");
        }

        let log2_len = len.trailing_zeros();
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        // Check if dual-GPU is available and beneficial
        let use_dual_gpu = self.devices.len() >= 2 && batch_size >= 2;
        let mid_batch = if use_dual_gpu { batch_size / 2 } else { batch_size };

        if use_dual_gpu {
            eprintln!(
                "\n[Dual-GPU Batch XField Fused Coset+NTT - {}] Processing {} arrays",
                phase_name,
                batch_size
            );
            eprintln!("  GPU 0: arrays [0, {})", mid_batch);
            eprintln!("  GPU 1: arrays [{}, {})", mid_batch, batch_size);
        } else {
            eprintln!(
                "\n[GPU Batch XField Fused Coset+NTT - {}] Processing {} arrays (UNCHUNKED - single GPU call)",
                phase_name,
                batch_size
            );
        }

        // Prep phase: Serialize all arrays in ONE go (not per chunk)
        let prep_start = Instant::now();
        let mut raw_data: Vec<u64> = Vec::with_capacity(len * batch_size * 3);
        for arr in data_arrays.iter() {
            for elem in arr.iter() {
                raw_data.push(elem.coefficients[0].raw_u64());
                raw_data.push(elem.coefficients[1].raw_u64());
                raw_data.push(elem.coefficients[2].raw_u64());
            }
        }
        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);
        let prep_time = prep_start.elapsed();

        // Upload phase: Single upload for all data
        let upload_start = Instant::now();
        let stream = dev_ctx.device.default_stream();
        let d_data = stream.memcpy_htod(&raw_data)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let upload_time = upload_start.elapsed();

        // Kernel phase: Dual-GPU or single GPU
        let kernel_start = Instant::now();
        let offset_raw = offset.raw_u64();

        let (gpu0_kernel_time, gpu1_kernel_time, sync0_time, sync1_time, p2p_time) = if use_dual_gpu {
            let dev_1 = &self.devices[1];
            let stream_1 = dev_1.device.create_stream()?;

            // Split data for GPU 1 (second half of batch)
            let gpu1_batch_size = batch_size - mid_batch;
            let gpu1_data_start = mid_batch * len * 3;
            let gpu1_data_size = gpu1_batch_size * len * 3;
            let gpu1_data = &raw_data[gpu1_data_start..gpu1_data_start + gpu1_data_size];

            // Upload GPU 1's data and twiddles
            let d_data_1 = stream_1.memcpy_htod(gpu1_data)?;
            let d_twiddles_1 = stream_1.memcpy_htod(&raw_twiddles)?;

            // Launch GPU 0 kernel (first half)
            let gpu0_start = Instant::now();
            let mut d_data_ptr = d_data.as_ptr();
            let d_twiddles_ptr = d_twiddles.as_ptr();

            let mut kernel_args_0 = KernelArgs::new();
            kernel_args_0
                .push_mut_ptr(&mut d_data_ptr)
                .push_value(offset_raw)
                .push_value(len as u64)
                .push_ptr(&d_twiddles_ptr)
                .push_value(log2_len);

            dev_ctx.ntt_xfield_fused_coset_fn.launch(
                (mid_batch as u32, 1, 1),
                (super::super::get_gpu_ntt_block_size(), 1, 1),
                0,
                &stream,
                kernel_args_0.as_mut_slice(),
            )?;
            let gpu0_kernel = gpu0_start.elapsed();

            // Launch GPU 1 kernel (second half)
            let gpu1_start = Instant::now();
            let mut d_data_ptr_1 = d_data_1.as_ptr();
            let d_twiddles_ptr_1 = d_twiddles_1.as_ptr();

            let mut kernel_args_1 = KernelArgs::new();
            kernel_args_1
                .push_mut_ptr(&mut d_data_ptr_1)
                .push_value(offset_raw)
                .push_value(len as u64)
                .push_ptr(&d_twiddles_ptr_1)
                .push_value(log2_len);

            dev_1.ntt_xfield_fused_coset_fn.launch(
                (gpu1_batch_size as u32, 1, 1),
                (super::super::get_gpu_ntt_block_size(), 1, 1),
                0,
                &stream_1,
                kernel_args_1.as_mut_slice(),
            )?;
            let gpu1_kernel = gpu1_start.elapsed();

            // Wait for both GPUs
            let sync0_start = Instant::now();
            stream.synchronize()?;
            let sync0 = sync0_start.elapsed();

            let sync1_start = Instant::now();
            stream_1.synchronize()?;
            let sync1 = sync1_start.elapsed();

            // Download GPU 1 results and merge into raw_data
            let download1_start = Instant::now();
            let mut gpu1_results = vec![0u64; gpu1_data_size];
            stream_1.memcpy_dtoh(&d_data_1, &mut gpu1_results)?;
            // Merge GPU 1 results into raw_data
            raw_data[gpu1_data_start..gpu1_data_start + gpu1_data_size].copy_from_slice(&gpu1_results);
            let download1 = download1_start.elapsed();

            eprintln!("  [Dual-GPU Timing]");
            eprintln!("    GPU 0 kernel launch: {:.1}ms", gpu0_kernel.as_secs_f64() * 1000.0);
            eprintln!("    GPU 1 kernel launch: {:.1}ms", gpu1_kernel.as_secs_f64() * 1000.0);
            eprintln!("    GPU 0 sync wait: {:.1}ms", sync0.as_secs_f64() * 1000.0);
            eprintln!("    GPU 1 sync wait: {:.1}ms", sync1.as_secs_f64() * 1000.0);
            eprintln!("    GPU 1 download+merge: {:.1}ms ({:.2} GB @ {:.1} GB/s)",
                     download1.as_secs_f64() * 1000.0,
                     gpu1_data_size as f64 * 8.0 / 1_073_741_824.0,
                     (gpu1_data_size as f64 * 8.0 / 1_073_741_824.0) / download1.as_secs_f64());

            (gpu0_kernel, gpu1_kernel, sync0, sync1, download1)
        } else {
            // Single GPU path
            let mut d_data_ptr = d_data.as_ptr();
            let d_twiddles_ptr = d_twiddles.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_mut_ptr(&mut d_data_ptr)
                .push_value(offset_raw)
                .push_value(len as u64)
                .push_ptr(&d_twiddles_ptr)
                .push_value(log2_len);

            dev_ctx.ntt_xfield_fused_coset_fn.launch(
                (batch_size as u32, 1, 1),
                (super::super::get_gpu_ntt_block_size(), 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
            stream.synchronize()?;

            (std::time::Duration::ZERO, std::time::Duration::ZERO,
             std::time::Duration::ZERO, std::time::Duration::ZERO,
             std::time::Duration::ZERO)
        };
        let kernel_time = kernel_start.elapsed();

        // Download phase: Download GPU 0 results (GPU 1 already downloaded if dual-GPU)
        let download_start = Instant::now();
        if use_dual_gpu {
            // Only download GPU 0's portion (first mid_batch arrays)
            let gpu0_data_size = mid_batch * len * 3;
            stream.memcpy_dtoh(&d_data, &mut raw_data[..gpu0_data_size])?;
        } else {
            // Single GPU: download everything
            stream.memcpy_dtoh(&d_data, &mut raw_data)?;
        }
        let download_time = download_start.elapsed();

        // Postproc phase: Deserialize all results in ONE go (not per chunk)
        let postproc_start = Instant::now();
        for (batch_idx, arr) in data_arrays.iter_mut().enumerate() {
            let start = batch_idx * len * 3;
            for (i, elem) in arr.iter_mut().enumerate() {
                let idx = start + i * 3;
                *elem = XFieldElement::new([
                    BFieldElement::from_raw_u64(raw_data[idx]),
                    BFieldElement::from_raw_u64(raw_data[idx + 1]),
                    BFieldElement::from_raw_u64(raw_data[idx + 2]),
                ]);
            }
        }
        let postproc_time = postproc_start.elapsed();

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Single Batch] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms, postproc={:.1}ms)",
            batch_size,
            total_time.as_secs_f64() * 1000.0,
            prep_time.as_secs_f64() * 1000.0,
            upload_time.as_secs_f64() * 1000.0,
            kernel_time.as_secs_f64() * 1000.0,
            download_time.as_secs_f64() * 1000.0,
            postproc_time.as_secs_f64() * 1000.0
        );

        Ok(())
    }
}
