use super::super::types::{GpuFieldElement, GpuNttContext, GpuTimingStats};
use crate::math::b_field_element::BFieldElement;
use super::super::cuda_driver::{CudaFunction, KernelArgs};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

impl GpuNttContext {
    /// Extract root twiddle factors from twiddle factor stages
    pub(crate) fn extract_twiddle_roots(twiddle_factors: &[Vec<BFieldElement>]) -> Vec<u64> {
        twiddle_factors
            .iter()
            .map(|stage| {
                if stage.len() >= 2 {
                    stage[1].raw_u64()
                } else {
                    stage[0].raw_u64()
                }
            })
            .collect()
    }

    /// Generic single operation handler for NTT/INTT on any field type
    /// Eliminates duplication between ntt_bfield, intt_bfield, ntt_xfield, intt_xfield
    pub(crate) fn execute_single<T: GpuFieldElement>(
        &self,
        data: &mut [T],
        twiddle_factors: &[Vec<BFieldElement>],
        kernel_fn: &CudaFunction,
        stats_lock: &OnceLock<Mutex<GpuTimingStats>>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let total_start = Instant::now();

        let len = data.len();
        assert!(len.is_power_of_two(), "NTT length must be power of 2");
        let log2_len = len.trailing_zeros();

        let dev_ctx = self.select_device();

        // Prep phase: Serialize field elements to raw u64
        let prep_start = Instant::now();
        let mut raw_data: Vec<u64> = Vec::with_capacity(len * T::U64_COUNT);
        for elem in data.iter() {
            elem.to_raw_u64s(&mut raw_data);
        }
        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);
        let prep_time = prep_start.elapsed();

        // Upload phase
        let upload_start = Instant::now();
        let upload_bytes = raw_data.len() * 8 + raw_twiddles.len() * 8;
        let stream = dev_ctx.device.default_stream();
        let d_data = stream.memcpy_htod(&raw_data)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let upload_time = upload_start.elapsed();

        // Kernel phase
        let kernel_start = Instant::now();
        let mut d_data_ptr = d_data.as_ptr();
        let d_twiddles_ptr = d_twiddles.as_ptr();

        let mut kernel_args = KernelArgs::new();
        kernel_args
            .push_mut_ptr(&mut d_data_ptr)
            .push_value(len as u64)
            .push_ptr(&d_twiddles_ptr)
            .push_value(log2_len);

        kernel_fn.launch(
            (1, 1, 1),  // grid
            (super::super::get_gpu_ntt_block_size(), 1, 1),  // block
            0,  // shared mem
            &stream,
            kernel_args.as_mut_slice(),
        )?;
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download phase
        let download_start = Instant::now();
        let download_bytes = raw_data.len() * 8;
        stream.memcpy_dtoh(&d_data, &mut raw_data)?;
        let download_time = download_start.elapsed();

        // Postproc phase: Deserialize back to field elements
        let postproc_start = Instant::now();
        for i in 0..len {
            data[i] = T::from_raw_u64s(&raw_data, i * T::U64_COUNT);
        }
        let postproc_time = postproc_start.elapsed();

        let total_time = total_start.elapsed();

        // Record timing statistics
        use std::sync::Mutex;
        let stats = stats_lock.get_or_init(|| Mutex::new(GpuTimingStats::default()));
        if let Ok(mut stats) = stats.lock() {
            stats.count += 1;
            stats.total_prep += prep_time;
            stats.total_upload += upload_time;
            stats.total_kernel += kernel_time;
            stats.total_download += download_time;
            stats.total_postproc += postproc_time;
            stats.total_time += total_time;
            stats.total_upload_bytes += upload_bytes;
            stats.total_download_bytes += download_bytes;
        }

        Ok(())
    }

    /// Generic batch operation handler for NTT/INTT on any field type
    /// Eliminates duplication across all 4 batch operations
    pub(crate) fn execute_batch<T: GpuFieldElement>(
        &self,
        data_arrays: &mut [&mut [T]],
        twiddle_factors: &[Vec<BFieldElement>],
        kernel_fn: &CudaFunction,
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
        assert!(len.is_power_of_two(), "NTT length must be power of 2");

        // Verify all arrays are same length
        for arr in data_arrays.iter() {
            assert_eq!(arr.len(), len, "All arrays in batch must have same length");
        }

        let log2_len = len.trailing_zeros();
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        // Prep phase: Serialize all arrays
        let prep_start = Instant::now();
        let mut raw_data: Vec<u64> = Vec::with_capacity(len * batch_size * T::U64_COUNT);
        for arr in data_arrays.iter() {
            for elem in arr.iter() {
                elem.to_raw_u64s(&mut raw_data);
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

        // Kernel phase
        let kernel_start = Instant::now();
        let mut d_data_ptr = d_data.as_ptr();
        let d_twiddles_ptr = d_twiddles.as_ptr();

        let mut kernel_args = KernelArgs::new();
        kernel_args
            .push_mut_ptr(&mut d_data_ptr)
            .push_value(len as u64)
            .push_ptr(&d_twiddles_ptr)
            .push_value(log2_len);

        kernel_fn.launch(
            (batch_size as u32, 1, 1),  // grid
            (super::super::get_gpu_ntt_block_size(), 1, 1),  // block
            0,  // shared mem
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
            let start = batch_idx * len * T::U64_COUNT;
            for (i, elem) in arr.iter_mut().enumerate() {
                *elem = T::from_raw_u64s(&raw_data, start + i * T::U64_COUNT);
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

    /// Execute batch operation with fused INTT + unscaling kernel
    /// This variant adds the n_inv parameter for the unscaling step
    pub(crate) fn execute_batch_fused_unscale<T: GpuFieldElement>(
        &self,
        data_arrays: &mut [&mut [T]],
        twiddle_factors: &[Vec<BFieldElement>],
        n_inv: BFieldElement,
        kernel_fn: &CudaFunction,
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
        assert!(len.is_power_of_two(), "NTT length must be power of 2");

        // Verify all arrays are same length
        for arr in data_arrays.iter() {
            assert_eq!(arr.len(), len, "All arrays in batch must have same length");
        }

        let log2_len = len.trailing_zeros();
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        // Prep phase: Serialize all arrays
        let prep_start = Instant::now();
        let mut raw_data: Vec<u64> = Vec::with_capacity(len * batch_size * T::U64_COUNT);
        for arr in data_arrays.iter() {
            for elem in arr.iter() {
                elem.to_raw_u64s(&mut raw_data);
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

        // Kernel phase - includes n_inv parameter for fused unscaling
        let kernel_start = Instant::now();
        let mut d_data_ptr = d_data.as_ptr();
        let d_twiddles_ptr = d_twiddles.as_ptr();

        let mut kernel_args = KernelArgs::new();
        kernel_args
            .push_mut_ptr(&mut d_data_ptr)
            .push_value(len as u64)
            .push_ptr(&d_twiddles_ptr)
            .push_value(n_inv.raw_u64())  // Additional parameter for unscaling
            .push_value(log2_len);

        kernel_fn.launch(
            (batch_size as u32, 1, 1),  // grid
            (super::super::get_gpu_ntt_block_size(), 1, 1),  // block
            0,  // shared mem
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
            let start = batch_idx * len * T::U64_COUNT;
            for (i, elem) in arr.iter_mut().enumerate() {
                *elem = T::from_raw_u64s(&raw_data, start + i * T::U64_COUNT);
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

    /// Execute batch operation with fused INTT + unscaling + randomizer kernel
    /// This variant pre-allocates larger buffers and adds randomizers on GPU
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn execute_batch_fused_unscale_randomize<T: GpuFieldElement>(
        &self,
        data_arrays: &mut [Vec<T>],  // Note: Vec instead of slice to allow resizing
        twiddle_factors: &[Vec<BFieldElement>],
        n_inv: BFieldElement,
        randomizers: &[Vec<T>],  // One randomizer per column
        offset_power_n: BFieldElement,
        kernel_fn: &CudaFunction,
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
        let num_randomizers = randomizers[0].len();
        let total_len = len + num_randomizers;

        assert!(len.is_power_of_two(), "NTT length must be power of 2");

        // Verify all arrays are same length
        for arr in data_arrays.iter() {
            assert_eq!(arr.len(), len, "All arrays in batch must have same length");
        }
        for rand in randomizers.iter() {
            assert_eq!(rand.len(), num_randomizers, "All randomizers must have same length");
        }

        let log2_len = len.trailing_zeros();
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        // Prep phase: Resize arrays and serialize
        let prep_start = Instant::now();

        // Resize all arrays to accommodate randomizers
        for arr in data_arrays.iter_mut() {
            arr.resize(total_len, T::zero());
        }

        // Serialize all arrays (now with extra space)
        /*
        let mut raw_data: Vec<u64> = Vec::with_capacity(total_len * batch_size * T::U64_COUNT);
        for arr in data_arrays.iter() {
            for elem in arr.iter() {
                elem.to_raw_u64s(&mut raw_data);
            }
        }
        */

        // Serialize randomizers
        let mut raw_randomizers: Vec<u64> = Vec::with_capacity(num_randomizers * batch_size * T::U64_COUNT);
        for rand in randomizers.iter() {
            for elem in rand.iter() {
                elem.to_raw_u64s(&mut raw_randomizers);
            }
        }

        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);
        let prep_time = prep_start.elapsed();

        // Upload phase
        let upload_start = Instant::now();
        let stream = dev_ctx.device.default_stream();
        let d_data = stream.alloc::<u64>(total_len * batch_size * T::U64_COUNT)?;

        for (idx,arr) in data_arrays.iter().enumerate() {
            if T::U64_COUNT == 1 {
                let raw_arr: & [u64] = unsafe {
                    std::slice::from_raw_parts(
                        arr.as_ptr() as *const u64,
                        arr.len()
                    )
                };
                let offset = total_len * T::U64_COUNT * idx;
                stream.memcpy_htod_to_slice(raw_arr, &d_data, offset)?;
            } else if T::U64_COUNT == 3 {
                let raw_arr: & [u64] = unsafe {
                    std::slice::from_raw_parts(
                        arr.as_ptr() as *const u64,
                        arr.len() * 3
                    )
                };
                let offset = total_len * T::U64_COUNT * idx;
                stream.memcpy_htod_to_slice(raw_arr, &d_data, offset)?;
            } else {
                panic!("Uknown T::U64_COUNT {}",T::U64_COUNT);
            }
        }

        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let d_randomizers = stream.memcpy_htod(&raw_randomizers)?;
        stream.synchronize()?;
        let upload_time = upload_start.elapsed();

        // Kernel phase - includes n_inv, randomizers, and offset_power_n
        let kernel_start = Instant::now();
        let mut d_data_ptr = d_data.as_ptr();
        let d_twiddles_ptr = d_twiddles.as_ptr();
        let d_randomizers_ptr = d_randomizers.as_ptr();

        let mut kernel_args = KernelArgs::new();
        kernel_args
            .push_mut_ptr(&mut d_data_ptr)
            .push_value(len as u64)
            .push_value(num_randomizers as u32)
            .push_ptr(&d_twiddles_ptr)
            .push_value(n_inv.raw_u64())
            .push_ptr(&d_randomizers_ptr)
            .push_value(offset_power_n.raw_u64())
            .push_value(log2_len);

        kernel_fn.launch(
            (batch_size as u32, 1, 1),  // grid
            (super::super::get_gpu_ntt_block_size(), 1, 1),  // block
            0,  // shared mem
            &stream,
            kernel_args.as_mut_slice(),
        )?;
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download phase
        //let download_start = Instant::now();
        //stream.memcpy_dtoh(&d_data, &mut raw_data)?;
        //let download_time = download_start.elapsed();

        // Postproc phase: Deserialize results back to arrays
        let postproc_start = Instant::now();
        let download_start = Instant::now();
        for (batch_idx, arr) in data_arrays.iter_mut().enumerate() {
            if T::U64_COUNT == 1 {
                let raw_arr: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        arr.as_ptr() as *mut u64,
                        arr.len()
                    )
                };
                let offset = total_len * T::U64_COUNT * batch_idx;
                stream.memcpy_dtoh_from_slice(&d_data, offset, raw_arr)?;
            } else if T::U64_COUNT == 3 {
                let raw_arr: &mut [u64] = unsafe {
                    std::slice::from_raw_parts_mut(
                        arr.as_ptr() as *mut u64,
                        arr.len() * 3
                    )
                };
                let offset = total_len * T::U64_COUNT * batch_idx;
                stream.memcpy_dtoh_from_slice(&d_data, offset, raw_arr)?;
            } else {
                panic!("Uknown T::U64_COUNT {}",T::U64_COUNT);
            }
        }
        let download_time = download_start.elapsed();
        let postproc_time = postproc_start.elapsed();

        Ok((
            prep_time + postproc_time,
            upload_time,
            kernel_time,
            download_time,
        ))
    }

    /// Execute batch operation on a specific stream for async overlapping
    /// This is the stream-aware version of execute_batch that enables concurrent execution
    pub(crate) fn execute_batch_on_stream<T: GpuFieldElement>(
        _dev_ctx: &super::super::types::GpuDeviceContext,
        stream: &super::super::cuda_driver::CudaStream,
        data_arrays: &mut [&mut [T]],
        twiddle_factors: &[Vec<BFieldElement>],
        kernel_fn: &CudaFunction,
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
        assert!(len.is_power_of_two(), "NTT length must be power of 2");

        // Verify all arrays are same length
        for arr in data_arrays.iter() {
            assert_eq!(arr.len(), len, "All arrays in batch must have same length");
        }

        let log2_len = len.trailing_zeros();
        let batch_size = data_arrays.len();

        // Prep phase: Serialize all arrays
        let prep_start = Instant::now();
        let mut raw_data: Vec<u64> = Vec::with_capacity(len * batch_size * T::U64_COUNT);
        for arr in data_arrays.iter() {
            for elem in arr.iter() {
                elem.to_raw_u64s(&mut raw_data);
            }
        }
        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);
        let prep_time = prep_start.elapsed();

        // Upload phase
        let upload_start = Instant::now();
        let d_data = stream.memcpy_htod(&raw_data)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let upload_time = upload_start.elapsed();

        // Kernel phase - launch on specific stream for async execution
        let kernel_start = Instant::now();
        let mut d_data_ptr = d_data.as_ptr();
        let d_twiddles_ptr = d_twiddles.as_ptr();

        let mut kernel_args = KernelArgs::new();
        kernel_args
            .push_mut_ptr(&mut d_data_ptr)
            .push_value(len as u64)
            .push_ptr(&d_twiddles_ptr)
            .push_value(log2_len);

        kernel_fn.launch(
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

        // Postproc phase: Deserialize results back to arrays
        let postproc_start = Instant::now();
        for (batch_idx, arr) in data_arrays.iter_mut().enumerate() {
            let start = batch_idx * len * T::U64_COUNT;
            for (i, elem) in arr.iter_mut().enumerate() {
                *elem = T::from_raw_u64s(&raw_data, start + i * T::U64_COUNT);
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
