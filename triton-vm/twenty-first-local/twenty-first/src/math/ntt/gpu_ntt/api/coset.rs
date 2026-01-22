use super::super::types::{GpuError, GpuNttContext};
use crate::math::b_field_element::BFieldElement;
use crate::math::x_field_element::XFieldElement;
use super::super::cuda_driver::KernelArgs;
use std::sync::Mutex;
use std::time::Instant;

impl GpuNttContext {
    // =============================================================================
    // Public API: Coset Scaling Operations
    // =============================================================================

    /// Perform GPU coset scaling on multiple BFieldElement arrays
    /// Computes coefficients[i] = coefficients[i] * offset^i for each polynomial
    ///
    /// This is used for coset evaluation preprocessing before NTT
    ///
    /// # Arguments
    /// * `data_arrays` - Mutable slices of BFieldElements to scale
    /// * `offset` - Coset offset value (generator for the coset)
    /// * `chunk_size` - Maximum number of arrays to process in one GPU batch
    ///
    /// # Performance
    /// This function batches multiple polynomials together to amortize GPU transfer overhead
    pub fn coset_scale_bfield_chunked(
        &self,
        data_arrays: &mut [&mut [BFieldElement]],
        offset: BFieldElement,
        chunk_size: usize,
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if data_arrays.is_empty() {
            return Ok(());
        }

        use super::super::types::GpuError;
        use std::sync::Mutex;
        use std::time::Instant;

        let operation_start = Instant::now();
        let num_gpus = self.devices.len();

        eprintln!(
            "\n[GPU Batch BField Coset Scale - {}] Processing {} arrays in chunks of {} across {} GPU(s)",
            phase_name,
            data_arrays.len(),
            chunk_size,
            num_gpus
        );
        eprintln!("  Offset: {}", offset.value());
        eprintln!("  Using {} parallel workers (one per GPU)", num_gpus);

        // Track per-chunk timing
        let chunk_times = Mutex::new(Vec::new());

        // Create thread pool with one thread per GPU
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_gpus)
            .build()
            .map_err(|e| {
                Box::new(GpuError(format!("Failed to create thread pool: {}", e)))
                    as Box<dyn std::error::Error>
            })?;

        // Process chunks in parallel
        pool.install(|| {
            rayon::scope(|s| {
                for (chunk_idx, chunk) in data_arrays.chunks_mut(chunk_size).enumerate() {
                    let chunk_times_ref = &chunk_times;
                    s.spawn(move |_| {
                        let chunk_start = Instant::now();
                        let chunk_len = chunk.len();

                        let result = self.execute_coset_scale_batch_bfield(chunk, offset);

                        let chunk_time = chunk_start.elapsed();

                        match result {
                            Ok((prep, upload, kernel, download)) => {
                                eprintln!("    [Chunk {}] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms)",
                                    chunk_idx, chunk_len,
                                    chunk_time.as_secs_f64() * 1000.0,
                                    prep.as_secs_f64() * 1000.0,
                                    upload.as_secs_f64() * 1000.0,
                                    kernel.as_secs_f64() * 1000.0,
                                    download.as_secs_f64() * 1000.0
                                );
                                if let Ok(mut times) = chunk_times_ref.lock() {
                                    times.push((chunk_idx, chunk_len, chunk_time, prep, upload, kernel, download));
                                }
                            }
                            Err(e) => {
                                eprintln!("    [Chunk {}] ERROR: GPU Coset Scale error: {}", chunk_idx, e);
                            }
                        }
                    });
                }
            });
        });

        let total_time = operation_start.elapsed();

        // Print summary
        super::super::timing::print_chunked_summary(phase_name, data_arrays.len(), total_time, &chunk_times);

        Ok(())
    }

    /// Execute coset scaling for a batch of BFieldElement arrays
    /// Internal helper function for coset_scale_bfield_chunked
    fn execute_coset_scale_batch_bfield(
        &self,
        data_arrays: &mut [&mut [BFieldElement]],
        offset: BFieldElement,
    ) -> Result<
        (
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
        ),
        Box<dyn std::error::Error>,
    > {
        use std::time::Instant;

        if data_arrays.is_empty() {
            return Ok((
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
                std::time::Duration::ZERO,
            ));
        }

        let len = data_arrays[0].len();
        assert!(len.is_power_of_two(), "Coset scale length must be power of 2");

        // Verify all arrays are same length
        for arr in data_arrays.iter() {
            assert_eq!(arr.len(), len, "All arrays in batch must have same length");
        }

        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        // Prep phase: Serialize all arrays to raw u64
        let prep_start = Instant::now();
        let mut raw_data: Vec<u64> = Vec::with_capacity(len * batch_size);
        for arr in data_arrays.iter() {
            for elem in arr.iter() {
                raw_data.push(elem.raw_u64());
            }
        }
        let prep_time = prep_start.elapsed();

        // Upload phase
        let upload_start = Instant::now();
        let stream = dev_ctx.device.default_stream();
        let d_data = stream.memcpy_htod(&raw_data)?;
        let upload_time = upload_start.elapsed();

        // Kernel phase
        let kernel_start = Instant::now();
        let mut d_data_ptr = d_data.as_ptr();
        let offset_raw = offset.raw_u64();

        let mut kernel_args = KernelArgs::new();
        kernel_args
            .push_mut_ptr(&mut d_data_ptr)
            .push_value(offset_raw)
            .push_value(len as u64)
            .push_value(batch_size as u64);

        dev_ctx.coset_scale_bfield_fn.launch(
            (batch_size as u32, 1, 1),
            (super::super::get_gpu_ntt_block_size(), 1, 1),
            0,
            &stream,
            kernel_args.as_mut_slice(),
        )?;
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download phase
        let download_start = Instant::now();
        stream.memcpy_dtoh(&d_data, &mut raw_data)?;
        let download_time = download_start.elapsed();

        // Postproc phase: Deserialize results back to arrays
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

    // =============================================================================
    // Public API: Fused Coset Scaling + NTT Operations
    // =============================================================================

    /// Perform fused GPU coset scaling + NTT on BFieldElement arrays
    /// This eliminates PCIe overhead by combining coset scaling and NTT in one kernel
    ///
    /// Performance advantage over separate operations:
    /// - Separate: Upload -> Scale -> Download -> Upload -> NTT -> Download (4 PCIe transfers)
    /// - Fused:    Upload -> Scale+NTT -> Download (2 PCIe transfers)
    /// - Expected speedup: ~2x by eliminating duplicate PCIe transfers
    ///
    /// # Arguments
    /// * `data_arrays` - Mutable slices of BFieldElements to scale and transform
    /// * `offset` - Coset offset value (generator for the coset)
    /// * `twiddle_factors` - Precomputed twiddle factors for NTT
    /// * `chunk_size` - Maximum number of arrays to process in one GPU batch
    /// * `phase_name` - Name for logging purposes
    pub fn ntt_bfield_fused_coset_chunked(
        &self,
        data_arrays: &mut [&mut [BFieldElement]],
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
        chunk_size: usize,
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use super::super::types::GpuError;
        use std::sync::Mutex;
        use std::time::Instant;

        if data_arrays.is_empty() {
            return Ok(());
        }

        let operation_start = Instant::now();
        let num_gpus = self.devices.len();

        eprintln!(
            "\n[GPU Batch BField Fused Coset+NTT - {}] Processing {} arrays in chunks of {} across {} GPU(s)",
            phase_name,
            data_arrays.len(),
            chunk_size,
            num_gpus
        );
        eprintln!("  Offset: {}", offset.value());

        // Calculate configured streams per GPU
        let streams_per_gpu = self.devices.first().map(|d| d.num_streams).unwrap_or(0);
        if streams_per_gpu > 1 {
            eprintln!("  Multi-stream mode:     {} streams per GPU", streams_per_gpu);
        } else {
            eprintln!("  Using {} parallel workers (one per GPU)", num_gpus);
        }

        // Track per-chunk timing
        let chunk_times = Mutex::new(Vec::new());

        // Create thread pool with one thread per GPU
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_gpus)
            .build()
            .map_err(|e| {
                Box::new(GpuError(format!("Failed to create thread pool: {}", e)))
                    as Box<dyn std::error::Error>
            })?;

        // Process chunks using multi-stream async execution
        pool.install(|| {
            use std::cell::RefCell;
            use std::collections::HashMap;

            // Thread-local storage for streams per device
            thread_local! {
                static STREAM_CACHE: RefCell<HashMap<usize, Vec<super::super::cuda_driver::CudaStream>>> = RefCell::new(HashMap::new());
            }

            rayon::scope(|s| {
                for (chunk_idx, chunk) in data_arrays.chunks_mut(chunk_size).enumerate() {
                    let chunk_times_ref = &chunk_times;
                    let device_idx = chunk_idx % num_gpus;
                    let dev_ctx = &self.devices[device_idx];

                    s.spawn(move |_| {
                        let chunk_start = Instant::now();
                        let chunk_len = chunk.len();

                        let result = if dev_ctx.num_streams > 0 {
                            STREAM_CACHE.with(|cache| {
                                let mut cache_map = cache.borrow_mut();
                                let streams = cache_map.entry(device_idx).or_insert_with(|| {
                                    let mut new_streams = Vec::new();
                                    for stream_idx in 0..dev_ctx.num_streams {
                                        match dev_ctx.device.create_stream() {
                                            Ok(stream) => new_streams.push(stream),
                                            Err(e) => {
                                                eprintln!("Warning: Failed to create stream {} for GPU {}: {}",
                                                    stream_idx, device_idx, e);
                                                break;
                                            }
                                        }
                                    }
                                    new_streams
                                });

                                if !streams.is_empty() {
                                    let stream_idx = chunk_idx % streams.len();
                                    let stream = &streams[stream_idx];
                                    Self::execute_fused_coset_ntt_on_stream_bfield(
                                        dev_ctx, stream, chunk, offset, twiddle_factors)
                                } else {
                                    self.execute_fused_coset_ntt_batch_bfield(chunk, offset, twiddle_factors)
                                }
                            })
                        } else {
                            self.execute_fused_coset_ntt_batch_bfield(chunk, offset, twiddle_factors)
                        };

                        let chunk_time = chunk_start.elapsed();

                        match result {
                            Ok((prep, upload, kernel, download)) => {
                                let stream_info = if dev_ctx.num_streams > 0 {
                                    format!(" Stream {}", chunk_idx % dev_ctx.num_streams)
                                } else {
                                    String::new()
                                };
                                eprintln!("    [Chunk {} GPU {}{}] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms)",
                                    chunk_idx, device_idx, stream_info, chunk_len,
                                    chunk_time.as_secs_f64() * 1000.0,
                                    prep.as_secs_f64() * 1000.0,
                                    upload.as_secs_f64() * 1000.0,
                                    kernel.as_secs_f64() * 1000.0,
                                    download.as_secs_f64() * 1000.0
                                );
                                if let Ok(mut times) = chunk_times_ref.lock() {
                                    times.push((chunk_idx, chunk_len, chunk_time, prep, upload, kernel, download));
                                }
                            }
                            Err(e) => {
                                eprintln!("    [Chunk {} GPU {}] ERROR: GPU Fused Coset+NTT error: {}", chunk_idx, device_idx, e);
                            }
                        }
                    });
                }
            });
        });

        let total_time = operation_start.elapsed();

        // Print summary
        super::super::timing::print_chunked_summary(phase_name, data_arrays.len(), total_time, &chunk_times);

        Ok(())
    }

    /// Execute fused coset scaling + NTT for a batch of BFieldElement arrays
    /// Internal helper function for ntt_bfield_fused_coset_chunked
    fn execute_fused_coset_ntt_batch_bfield(
        &self,
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
        use std::time::Instant;

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
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        // Prep phase: Serialize all arrays to raw u64
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
        let stream = dev_ctx.device.default_stream();
        let d_data = stream.memcpy_htod(&raw_data)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let upload_time = upload_start.elapsed();

        // Kernel phase: Fused coset scaling + NTT
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
            (batch_size as u32, 1, 1),
            (super::super::get_gpu_ntt_block_size(), 1, 1),
            0,
            &stream,
            kernel_args.as_mut_slice(),
        )?;
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download phase
        let download_start = Instant::now();
        stream.memcpy_dtoh(&d_data, &mut raw_data)?;
        let download_time = download_start.elapsed();

        // Postproc phase: Deserialize results back to arrays
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

    /// Perform fused GPU coset scaling + NTT on XFieldElement arrays
    /// This eliminates PCIe overhead by combining coset scaling and NTT in one kernel
    pub fn ntt_xfield_fused_coset_chunked(
        &self,
        data_arrays: &mut [&mut [XFieldElement]],
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
        chunk_size: usize,
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use super::super::types::GpuError;
        use std::sync::Mutex;
        use std::time::Instant;

        if data_arrays.is_empty() {
            return Ok(());
        }

        let operation_start = Instant::now();
        let num_gpus = self.devices.len();

        eprintln!(
            "\n[GPU Batch XField Fused Coset+NTT - {}] Processing {} arrays in chunks of {} across {} GPU(s)",
            phase_name,
            data_arrays.len(),
            chunk_size,
            num_gpus
        );
        eprintln!("  Offset: {}", offset.value());

        // Calculate configured streams per GPU
        let streams_per_gpu = self.devices.first().map(|d| d.num_streams).unwrap_or(0);
        if streams_per_gpu > 1 {
            eprintln!("  Multi-stream mode:     {} streams per GPU", streams_per_gpu);
        } else {
            eprintln!("  Using {} parallel workers (one per GPU)", num_gpus);
        }

        // Track per-chunk timing
        let chunk_times = Mutex::new(Vec::new());

        // Create thread pool with one thread per GPU
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_gpus)
            .build()
            .map_err(|e| {
                Box::new(GpuError(format!("Failed to create thread pool: {}", e)))
                    as Box<dyn std::error::Error>
            })?;

        // Process chunks using multi-stream async execution
        pool.install(|| {
            use std::cell::RefCell;
            use std::collections::HashMap;

            // Thread-local storage for streams per device
            thread_local! {
                static STREAM_CACHE: RefCell<HashMap<usize, Vec<super::super::cuda_driver::CudaStream>>> = RefCell::new(HashMap::new());
            }

            rayon::scope(|s| {
                for (chunk_idx, chunk) in data_arrays.chunks_mut(chunk_size).enumerate() {
                    let chunk_times_ref = &chunk_times;
                    let device_idx = chunk_idx % num_gpus;
                    let dev_ctx = &self.devices[device_idx];

                    s.spawn(move |_| {
                        let chunk_start = Instant::now();
                        let chunk_len = chunk.len();

                        let result = if dev_ctx.num_streams > 0 {
                            STREAM_CACHE.with(|cache| {
                                let mut cache_map = cache.borrow_mut();
                                let streams = cache_map.entry(device_idx).or_insert_with(|| {
                                    let mut new_streams = Vec::new();
                                    for stream_idx in 0..dev_ctx.num_streams {
                                        match dev_ctx.device.create_stream() {
                                            Ok(stream) => new_streams.push(stream),
                                            Err(e) => {
                                                eprintln!("Warning: Failed to create stream {} for GPU {}: {}",
                                                    stream_idx, device_idx, e);
                                                break;
                                            }
                                        }
                                    }
                                    new_streams
                                });

                                if !streams.is_empty() {
                                    let stream_idx = chunk_idx % streams.len();
                                    let stream = &streams[stream_idx];
                                    Self::execute_fused_coset_ntt_on_stream_xfield(
                                        dev_ctx, stream, chunk, offset, twiddle_factors)
                                } else {
                                    self.execute_fused_coset_ntt_batch_xfield(chunk, offset, twiddle_factors)
                                }
                            })
                        } else {
                            self.execute_fused_coset_ntt_batch_xfield(chunk, offset, twiddle_factors)
                        };

                        let chunk_time = chunk_start.elapsed();

                        match result {
                            Ok((prep, upload, kernel, download)) => {
                                let stream_info = if dev_ctx.num_streams > 0 {
                                    format!(" Stream {}", chunk_idx % dev_ctx.num_streams)
                                } else {
                                    String::new()
                                };
                                eprintln!("    [Chunk {} GPU {}{}] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms)",
                                    chunk_idx, device_idx, stream_info, chunk_len,
                                    chunk_time.as_secs_f64() * 1000.0,
                                    prep.as_secs_f64() * 1000.0,
                                    upload.as_secs_f64() * 1000.0,
                                    kernel.as_secs_f64() * 1000.0,
                                    download.as_secs_f64() * 1000.0
                                );
                                if let Ok(mut times) = chunk_times_ref.lock() {
                                    times.push((chunk_idx, chunk_len, chunk_time, prep, upload, kernel, download));
                                }
                            }
                            Err(e) => {
                                eprintln!("    [Chunk {} GPU {}] ERROR: GPU Fused Coset+NTT error: {}", chunk_idx, device_idx, e);
                            }
                        }
                    });
                }
            });
        });

        let total_time = operation_start.elapsed();

        // Print summary
        super::super::timing::print_chunked_summary(phase_name, data_arrays.len(), total_time, &chunk_times);

        Ok(())
    }

    /// Execute fused coset scaling + NTT for a batch of XFieldElement arrays
    /// Internal helper function for ntt_xfield_fused_coset_chunked
    fn execute_fused_coset_ntt_batch_xfield(
        &self,
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
        use std::time::Instant;

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
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        // Prep phase: Serialize all arrays to raw u64 (3 u64s per XFieldElement)
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
        let stream = dev_ctx.device.default_stream();
        let d_data = stream.memcpy_htod(&raw_data)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let upload_time = upload_start.elapsed();

        // Kernel phase: Fused coset scaling + NTT
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
            (batch_size as u32, 1, 1),
            (super::super::get_gpu_ntt_block_size(), 1, 1),
            0,
            &stream,
            kernel_args.as_mut_slice(),
        )?;
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download phase
        let download_start = Instant::now();
        stream.memcpy_dtoh(&d_data, &mut raw_data)?;
        let download_time = download_start.elapsed();

        // Postproc phase: Deserialize results back to arrays
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
}
