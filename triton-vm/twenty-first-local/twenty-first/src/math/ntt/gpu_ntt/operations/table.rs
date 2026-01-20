use super::super::types::GpuNttContext;
use crate::math::b_field_element::BFieldElement;
use crate::math::x_field_element::XFieldElement;
use crate::tip5::Digest;
use super::super::cuda_driver::{DeviceBuffer, KernelArgs};
use std::time::Instant;
use crate::prelude::Inverse;

impl GpuNttContext {
    /// GPU-accelerated FRI INTT (Inverse NTT) with fused unscaling
    /// This performs stage-by-stage INTT directly on GPU, fusing the unscaling operation
    /// into the kernel to avoid a separate pass.
    pub fn execute_fri_intt(
        &self,
        values: &mut [XFieldElement],
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let slice_len = values.len();
        let log2_len = values.len().trailing_zeros();
        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);

        let dev_ctx = self.select_device();
        let stream = dev_ctx.device.default_stream();

        // Allocate GPU buffers
        let d_data = stream.alloc::<u64>(slice_len * 3)?;
        let d_omegas_extra = stream.alloc::<u64>(slice_len)?;

        // Zero-copy: Cast XFieldElement slice directly to u64 slice
        let raw_data_slice_poly: &mut [u64] = unsafe {
            std::slice::from_raw_parts_mut(
                values.as_mut_ptr() as *mut u64,
                values.len() * 3
            )
        };

        // Upload data and twiddles
        stream.memcpy_htod_to_slice(raw_data_slice_poly, &d_data, 0)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;

        let offset_raw = offset.raw_u64();

        // Calculate unscaling parameter (1/n for INTT)
        let n = BFieldElement::from(values.len());
        let n_inv = n.inverse_or_zero();
        let unscale_param_raw = n_inv.raw_u64();

        // Initialize omega values
        let mut d_omegas_ptr = d_omegas_extra.as_ptr();
        let d_twiddles_ptr = d_twiddles.as_ptr();
        let len_u64 = slice_len as u64;

        let mut init_args = KernelArgs::new();
        init_args
            .push_value(len_u64)
            .push_ptr(&d_twiddles_ptr)
            .push_mut_ptr(&mut d_omegas_ptr);

        dev_ctx.ntt_xfield_init_omegas_fn.launch(
            (log2_len as u32, 1, 1),
            (256, 1, 1),
            0,
            &stream,
            init_args.as_mut_slice(),
        )?;
        stream.synchronize()?;

        // Stage-by-stage cooperative INTT with fused unscaling
        for ntt_stage in 0..log2_len {
            let mut d_data_ptr = d_data.as_ptr();

            let mut stage_args = KernelArgs::new();
            stage_args
                .push_mut_ptr(&mut d_data_ptr)
                .push_value(len_u64)
                .push_value(ntt_stage as u64)
                .push_ptr(&d_twiddles_ptr)
                .push_ptr(&d_omegas_ptr)
                .push_value(offset_raw)
                .push_value(log2_len as u64)
                .push_value(unscale_param_raw);

            dev_ctx.ntt_xfield_fused_coset_single_interpolate_fn.launch_cooperative(
                (dev_ctx.sm_count as u32, 1, 1),
                (256, 1, 1),
                0,
                &stream,
                stage_args.as_mut_slice(),
            )?;
        }

        stream.synchronize()?;

        // Download results back
        stream.memcpy_dtoh_from_slice(&d_data, 0, raw_data_slice_poly)?;

        Ok(())
    }

    /// Execute fused coset+NTT on a row-major table using strided GPU kernels
    /// This eliminates column extraction overhead by working directly with table data
    /// For BFieldElement tables
    ///
    /// Returns: (Vec<Digest>, DeviceBuffer) containing hashes and GPU-resident buffer
    /// The GPU buffer is kept on device to avoid wasteful GPU→RAM→GPU transfers
    pub(crate) fn execute_fused_coset_ntt_table_bfield(
        &self,
        table_data: DeviceBuffer,
        num_rows: usize,
        num_columns: usize,
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(Vec<Digest>, DeviceBuffer), Box<dyn std::error::Error>> {
        let operation_start = Instant::now();

        assert!(num_rows.is_power_of_two(), "Table rows must be power of 2");

        let log2_len = num_rows.trailing_zeros();

        // Check if dual-GPU mode is available
        let use_dual_gpu = self.devices.len() >= 2;
        let mid_column = if use_dual_gpu { num_columns / 2 } else { num_columns };

        if use_dual_gpu {
            eprintln!(
                "\n[Dual-GPU Table BField Fused Coset+NTT - {}] Processing table: {} rows × {} columns",
                phase_name, num_rows, num_columns
            );
            eprintln!("  GPU 0: columns [0, {})", mid_column);
            eprintln!("  GPU 1: columns [{}, {})", mid_column, num_columns);
        } else {
            eprintln!(
                "\n[GPU Table BField Fused Coset+NTT - {}] Processing table: {} rows × {} columns (row-major, strided access)",
                phase_name,
                num_rows,
                num_columns
            );
        }

        let dev_ctx = &self.devices[0];

        // ZERO-COPY: Cast BFieldElement slice directly to u64 slice
        let prep_start = Instant::now();
        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);
        let prep_time = prep_start.elapsed();

        // Calculate memory requirements
        let twiddle_bytes = raw_twiddles.len() * 8;
        let total_bytes =  0 + twiddle_bytes;
        eprintln!(
            "  GPU Memory Required: {:.2} GB (data: {:?} GB, twiddles: {:.2} MB)",
            total_bytes as f64 / 1_073_741_824.0,
            "unknown",
            twiddle_bytes as f64 / 1_048_576.0
        );

        // Upload entire table (zero-copy - no serialization!)
        let upload_start = Instant::now();
        let stream = dev_ctx.device.default_stream();
        let d_data = table_data;
        let d_data_extra = stream.alloc::<u64>(num_rows)?;
        let d_omegas_extra = stream.alloc::<u64>(num_rows)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let offset_raw = offset.raw_u64();
        let upload_time = upload_start.elapsed();

        let kernel_start = Instant::now();

        let gpu0_init_start = Instant::now();
        {
            let mut d_omegas_extra_ptr = d_omegas_extra.as_ptr();
            let d_twiddles_ptr = d_twiddles.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_value(num_rows as u64)
                .push_ptr(&d_twiddles_ptr)
                .push_mut_ptr(&mut d_omegas_extra_ptr);

            dev_ctx.ntt_bfield_init_omegas_fn.launch(
                (log2_len as u32, 1, 1),
                (512, 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
        }
        stream.synchronize()?;
        let gpu0_init_time = gpu0_init_start.elapsed();

        // Dual-GPU setup
        let data_size_gb = (num_rows * num_columns * 8) as f64 / 1_000_000_000.0;
        let mut gpu1_setup_time = std::time::Duration::from_secs(0);
        let mut gpu1_ntt_time = std::time::Duration::from_secs(0);
        let mut sync0_time = std::time::Duration::from_secs(0);
        let mut sync1_time = std::time::Duration::from_secs(0);
        let mut p2p_return_time = std::time::Duration::from_secs(0);
        let mut merge_time = std::time::Duration::from_secs(0);
        let (dev_ctx_1, stream_1, d_data_1, d_data_extra_1, d_omegas_extra_1, d_twiddles_1) = if use_dual_gpu {
            let gpu1_setup_start = Instant::now();
            let dev_1 = &self.devices[1];
            let stream_1 = dev_1.device.create_stream()?;

            eprintln!("  [GPU 1] Allocating buffers and P2P copying table...");
            // P2P copy entire table from GPU 0 to GPU 1
            let d_data_1 = stream_1.alloc::<u64>(num_rows * num_columns)?;
            let p2p_copy_start = Instant::now();
            stream.memcpy_peer_async(&d_data_1, &dev_1.device.context, &d_data, &dev_ctx.device.context, num_rows * num_columns * 8)?;
            stream.synchronize()?;
            let p2p_copy_time = p2p_copy_start.elapsed();
            let bandwidth_gbs = data_size_gb / p2p_copy_time.as_secs_f64();
            eprintln!("  [P2P Copy GPU 0→1] {:.2} GB in {:.1}ms = {:.1} GB/s",
                     data_size_gb, p2p_copy_time.as_secs_f64() * 1000.0, bandwidth_gbs);

            // Allocate GPU 1's working buffers
            let d_data_extra_1 = stream_1.alloc::<u64>(num_rows)?;
            let d_omegas_extra_1 = stream_1.alloc::<u64>(num_rows)?;
            let d_twiddles_1 = stream_1.memcpy_htod(&raw_twiddles)?;

            // Init omegas on GPU 1
            {
                let mut d_omegas_extra_ptr_1 = d_omegas_extra_1.as_ptr();
                let d_twiddles_ptr_1 = d_twiddles_1.as_ptr();
                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_value(num_rows as u64)
                    .push_ptr(&d_twiddles_ptr_1)
                    .push_mut_ptr(&mut d_omegas_extra_ptr_1);
                dev_1.ntt_bfield_init_omegas_fn.launch(
                    (log2_len as u32, 1, 1),
                    (512, 1, 1),
                    0,
                    &stream_1,
                    kernel_args.as_mut_slice(),
                )?;
            }
            stream_1.synchronize()?;
            gpu1_setup_time = gpu1_setup_start.elapsed();
            eprintln!("  [GPU 1 Setup] {:.1}ms (alloc + P2P + init omegas)", gpu1_setup_time.as_secs_f64() * 1000.0);

            (Some(dev_1), Some(stream_1), Some(d_data_1), Some(d_data_extra_1), Some(d_omegas_extra_1), Some(d_twiddles_1))
        } else {
            (None, None, None, None, None, None)
        };

        // GPU 0: Process first half of columns using strided kernel (single launch!)
        let gpu0_ntt_start = Instant::now();
        if mid_column > 0 {
            let mut d_data_ptr = d_data.as_ptr();
            let d_omegas_ptr = d_twiddles.as_ptr();  // Use raw twiddles, not precomputed powers!

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_mut_ptr(&mut d_data_ptr)
                .push_value(offset_raw)
                .push_value(num_rows as u64)
                .push_value(num_columns as u64)  // stride
                .push_ptr(&d_omegas_ptr)
                .push_value(log2_len as u32);

            dev_ctx.ntt_bfield_fused_coset_strided_fn.launch_cooperative(
                (mid_column as u32, 1, 1),  // one block per column
                (256, 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
        }
        let gpu0_ntt_time = gpu0_ntt_start.elapsed();
        eprintln!("  [GPU 0 NTT] {:.1}ms for {} columns", gpu0_ntt_time.as_secs_f64() * 1000.0, mid_column);

        // GPU 1: Process second half of columns (async via CUDA streams)
        if let (Some(dev_1), Some(stream_1), Some(d_data_1), Some(d_data_extra_1), Some(d_omegas_extra_1), Some(d_twiddles_1)) =
            (dev_ctx_1, stream_1, &d_data_1, &d_data_extra_1, &d_omegas_extra_1, &d_twiddles_1) {

            eprintln!("  [GPU 1] Processing columns [{}, {})...", mid_column, num_columns);
            let gpu1_ntt_start = Instant::now();

            // GPU 1: Process second half using strided kernel (single launch!)
            let num_cols_gpu1 = num_columns - mid_column;
            if num_cols_gpu1 > 0 {
                // Offset data pointer to start at mid_column (each element is 8 bytes)
                let mut d_data_ptr_1 = d_data_1.as_ptr() + (mid_column as u64 * 8);
                let d_omegas_ptr_1 = d_twiddles_1.as_ptr();  // Use raw twiddles, not precomputed powers!

                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_mut_ptr(&mut d_data_ptr_1)
                    .push_value(offset_raw)
                    .push_value(num_rows as u64)
                    .push_value(num_columns as u64)  // stride
                    .push_ptr(&d_omegas_ptr_1)
                    .push_value(log2_len as u32);

                dev_1.ntt_bfield_fused_coset_strided_fn.launch_cooperative(
                    (num_cols_gpu1 as u32, 1, 1),  // one block per column
                    (256, 1, 1),
                    0,
                    &stream_1,
                    kernel_args.as_mut_slice(),
                )?;
            }
            gpu1_ntt_time = gpu1_ntt_start.elapsed();
            eprintln!("  [GPU 1 NTT] {:.1}ms for {} columns (strided, 1 launch)", gpu1_ntt_time.as_secs_f64() * 1000.0, num_columns - mid_column);

            // Wait for GPU 0 to finish
            // NOTE: The "sync wait" time reported here is the actual kernel execution time,
            // not overhead. Kernels are launched async, and synchronize() blocks until completion.
            // Large sync times indicate the kernel is compute-bound, not a synchronization bug.
            let sync0_start = Instant::now();
            stream.synchronize()?;
            sync0_time = sync0_start.elapsed();
            eprintln!("  [GPU 0] Completed columns [0, {}) - sync wait: {:.1}ms", mid_column, sync0_time.as_secs_f64() * 1000.0);

            // Wait for GPU 1 to finish
            let sync1_start = Instant::now();
            stream_1.synchronize()?;
            sync1_time = sync1_start.elapsed();
            eprintln!("  [GPU 1] Completed columns [{}, {}) - sync wait: {:.1}ms", mid_column, num_columns, sync1_time.as_secs_f64() * 1000.0);

            // P2P copy GPU 1's table back to GPU 0 temp buffer
            eprintln!("  [Merge] P2P copying GPU 1 results back...");
            let d_data_temp = stream.alloc::<u64>(num_rows * num_columns)?;
            let p2p_return_start = Instant::now();
            stream.memcpy_peer_async(&d_data_temp, &dev_ctx.device.context, d_data_1, &dev_1.device.context, num_rows * num_columns * 8)?;
            stream.synchronize()?;
            p2p_return_time = p2p_return_start.elapsed();
            let return_bandwidth_gbs = data_size_gb / p2p_return_time.as_secs_f64();
            eprintln!("  [P2P Copy GPU 1→0] {:.2} GB in {:.1}ms = {:.1} GB/s",
                     data_size_gb, p2p_return_time.as_secs_f64() * 1000.0, return_bandwidth_gbs);

            // Merge GPU 1's columns into main table
            eprintln!("  [Merge] Copying columns [{}, {}) from GPU 1 into main table...", mid_column, num_columns);
            let merge_start = Instant::now();
            {
                let d_data_temp_ptr = d_data_temp.as_ptr();
                let mut d_data_ptr = d_data.as_ptr();
                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_ptr(&d_data_temp_ptr)
                    .push_mut_ptr(&mut d_data_ptr)
                    .push_value(num_rows as u64)
                    .push_value(num_columns as u64)
                    .push_value(mid_column as u64)
                    .push_value(num_columns as u64);

                let total_elements = num_rows * (num_columns - mid_column);
                let threads_per_block = 256;
                let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

                dev_ctx.copy_columns_bfield_fn.launch(
                    (num_blocks as u32, 1, 1),
                    (threads_per_block as u32, 1, 1),
                    0,
                    &stream,
                    kernel_args.as_mut_slice(),
                )?;
            }
            stream.synchronize()?;
            merge_time = merge_start.elapsed();
            eprintln!("  [Merge Kernel] {:.1}ms", merge_time.as_secs_f64() * 1000.0);
            eprintln!("  [Merge] Complete!");

            // Detailed kernel timing breakdown
            let total_ntt_overhead = gpu0_init_time.as_secs_f64() * 1000.0
                                   + gpu1_setup_time.as_secs_f64() * 1000.0
                                   + sync0_time.as_secs_f64() * 1000.0
                                   + sync1_time.as_secs_f64() * 1000.0
                                   + p2p_return_time.as_secs_f64() * 1000.0
                                   + merge_time.as_secs_f64() * 1000.0;
            let actual_compute = gpu0_ntt_time.as_secs_f64().max(gpu1_ntt_time.as_secs_f64()) * 1000.0;
            eprintln!("  [Kernel Timing Breakdown]");
            eprintln!("    GPU 0 init omegas: {:.1}ms", gpu0_init_time.as_secs_f64() * 1000.0);
            eprintln!("    GPU 1 setup (alloc+P2P+init): {:.1}ms", gpu1_setup_time.as_secs_f64() * 1000.0);
            eprintln!("    GPU 0 NTT compute: {:.1}ms", gpu0_ntt_time.as_secs_f64() * 1000.0);
            eprintln!("    GPU 1 NTT compute: {:.1}ms", gpu1_ntt_time.as_secs_f64() * 1000.0);
            eprintln!("    Sync waits (0+1): {:.1}ms + {:.1}ms", sync0_time.as_secs_f64() * 1000.0, sync1_time.as_secs_f64() * 1000.0);
            eprintln!("    Merge (P2P+kernel): {:.1}ms + {:.1}ms", p2p_return_time.as_secs_f64() * 1000.0, merge_time.as_secs_f64() * 1000.0);
            eprintln!("    Actual parallel compute: {:.1}ms", actual_compute);
            eprintln!("    Total overhead: {:.1}ms", total_ntt_overhead);

            // Now perform dual-GPU parallel hashing while we still have access to dev_1 and stream_1
            let hash_start_dual = Instant::now();
            const DIGEST_LEN: usize = 5;
            const THREADS_PER_BLOCK: u32 = 256;

            // Dual-GPU parallel hashing: split rows 50/50
            let mid_row = num_rows / 2;
            let num_rows_gpu0 = mid_row;
            let num_rows_gpu1 = num_rows - mid_row;

            eprintln!("  [Dual-GPU Hash] GPU 0: rows [0, {}), GPU 1: rows [{}, {})", mid_row, mid_row, num_rows);

            // P2P copy merged table from GPU 0 to GPU 1 for hashing
            let d_data_1_hash = stream_1.alloc::<u64>(num_rows * num_columns)?;
            let p2p_hash_copy_start = Instant::now();
            stream.memcpy_peer_async(&d_data_1_hash, &dev_1.device.context, &d_data, &dev_ctx.device.context, num_rows * num_columns * 8)?;
            stream.synchronize()?;
            let p2p_hash_copy_time = p2p_hash_copy_start.elapsed();
            let hash_copy_gb = (num_rows * num_columns * 8) as f64 / 1_000_000_000.0;
            let hash_bandwidth_gbs = hash_copy_gb / p2p_hash_copy_time.as_secs_f64();
            eprintln!("  [P2P Hash Copy GPU 0→1] {:.2} GB in {:.1}ms = {:.1} GB/s",
                     hash_copy_gb, p2p_hash_copy_time.as_secs_f64() * 1000.0, hash_bandwidth_gbs);

            // Allocate digest buffers on both GPUs
            let d_digests_0 = stream.alloc::<u64>(num_rows_gpu0 * DIGEST_LEN)?;
            let d_digests_1 = stream_1.alloc::<u64>(num_rows_gpu1 * DIGEST_LEN)?;

            // Launch hash on GPU 0 for first half (rows 0 to mid_row)
            let num_blocks_0 = (num_rows_gpu0 as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            {
                let d_data_ptr = d_data.as_ptr();
                let mut d_digests_ptr_0 = d_digests_0.as_ptr();
                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_ptr(&d_data_ptr)
                    .push_mut_ptr(&mut d_digests_ptr_0)
                    .push_value(num_columns as u32)
                    .push_value(num_rows_gpu0 as u32)
                    .push_value(0u32); // row_start = 0
                dev_ctx.hash_rows_bfield_fn.launch(
                    (num_blocks_0, 1, 1),
                    (THREADS_PER_BLOCK, 1, 1),
                    0,
                    &stream,
                    kernel_args.as_mut_slice(),
                )?;
            }

            // Launch hash on GPU 1 for second half (rows mid_row to num_rows)
            let num_blocks_1 = (num_rows_gpu1 as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
            {
                let d_data_ptr_1 = d_data_1_hash.as_ptr();
                let mut d_digests_ptr_1 = d_digests_1.as_ptr();
                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_ptr(&d_data_ptr_1)
                    .push_mut_ptr(&mut d_digests_ptr_1)
                    .push_value(num_columns as u32)
                    .push_value(num_rows_gpu1 as u32)
                    .push_value(mid_row as u32); // row_start = mid_row
                dev_1.hash_rows_bfield_fn.launch(
                    (num_blocks_1, 1, 1),
                    (THREADS_PER_BLOCK, 1, 1),
                    0,
                    &stream_1,
                    kernel_args.as_mut_slice(),
                )?;
            }

            // Wait for both GPUs to finish hashing
            stream.synchronize()?;
            stream_1.synchronize()?;
            eprintln!("  [Dual-GPU Hash] Both GPUs completed hashing");

            // Download digests from both GPUs
            let mut digest_buffer_0 = vec![0u64; num_rows_gpu0 * DIGEST_LEN];
            let mut digest_buffer_1 = vec![0u64; num_rows_gpu1 * DIGEST_LEN];
            stream.memcpy_dtoh(&d_digests_0, &mut digest_buffer_0)?;
            stream_1.memcpy_dtoh(&d_digests_1, &mut digest_buffer_1)?;

            let hash_time_dual = hash_start_dual.elapsed();

            // Merge digest buffers
            let digest_buffer_size = num_rows * DIGEST_LEN;
            let mut digest_buffer_host = vec![0u64; digest_buffer_size];
            digest_buffer_host[0..digest_buffer_0.len()].copy_from_slice(&digest_buffer_0);
            digest_buffer_host[digest_buffer_0.len()..].copy_from_slice(&digest_buffer_1);

            // Drop extra buffers
            drop(d_data_extra);
            drop(d_omegas_extra);

            let kernel_time = kernel_start.elapsed();

            // Print hash timing and continue to digest conversion
            eprintln!(
                "  [GPU Hash] hash={:.1}ms ({} rows, {:.1} MB digests)",
                hash_time_dual.as_secs_f64() * 1000.0,
                num_rows,
                (digest_buffer_size * 8) as f64 / 1_048_576.0
            );

            // Skip full LDE download - only revealed rows will be fetched on-demand later
            let download_start = Instant::now();
            eprintln!("  [Optimization] Skipping {:.2} GB download - will fetch revealed rows on demand",
                     0 as f64 / 1_073_741_824.0);
            eprintln!("  [GPU-Resident LDE] Cached {:.2} GB GPU buffer (for quotient reuse & partial row extraction)",
                     0 as f64 / 1_073_741_824.0);
            let download_time = download_start.elapsed();

            // Convert digest buffer to Vec<Digest>
            let postproc_start = Instant::now();
            let digests: Vec<Digest> = (0..num_rows)
                .map(|row_idx| {
                    let start = row_idx * DIGEST_LEN;
                    Digest::new([
                        BFieldElement::from_raw_u64(digest_buffer_host[start]),
                        BFieldElement::from_raw_u64(digest_buffer_host[start + 1]),
                        BFieldElement::from_raw_u64(digest_buffer_host[start + 2]),
                        BFieldElement::from_raw_u64(digest_buffer_host[start + 3]),
                        BFieldElement::from_raw_u64(digest_buffer_host[start + 4]),
                    ])
                })
                .collect();
            let postproc_time = postproc_start.elapsed();

            let total_time = operation_start.elapsed();

            eprintln!(
                "  [Table Operation GPU-Resident] total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, hash={:.1}ms, download={:.1}ms, postproc={:.1}ms)",
                total_time.as_secs_f64() * 1000.0,
                prep_time.as_secs_f64() * 1000.0,
                upload_time.as_secs_f64() * 1000.0,
                kernel_time.as_secs_f64() * 1000.0,
                hash_time_dual.as_secs_f64() * 1000.0,
                download_time.as_secs_f64() * 1000.0,
                postproc_time.as_secs_f64() * 1000.0
            );

            // Return digests AND GPU buffer
            return Ok((digests, d_data));
        } else {
            // Single-GPU path: just wait for GPU 0 to finish
            stream.synchronize()?;
            eprintln!("  [GPU 0] Completed all {} columns", num_columns);
        }

        drop(d_data_extra);
        drop(d_omegas_extra);

        let kernel_time = kernel_start.elapsed();

        // GPU Row Hashing: Hash rows while data is still on GPU (Single-GPU path)
        let hash_start = Instant::now();
        const DIGEST_LEN: usize = 5;
        const THREADS_PER_BLOCK: u32 = 256;
        let digest_buffer_size = num_rows * DIGEST_LEN;
        let mut digest_buffer_host = vec![0u64; digest_buffer_size];

        let d_digests = stream.memcpy_htod(&digest_buffer_host)?;
        let num_blocks = (num_rows as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        {
            let d_data_ptr = d_data.as_ptr();
            let mut d_digests_ptr = d_digests.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_ptr(&d_data_ptr)
                .push_mut_ptr(&mut d_digests_ptr)
                .push_value(num_columns as u32)
                .push_value(num_rows as u32)
                .push_value(0u32); // row_start = 0

            dev_ctx.hash_rows_bfield_fn.launch(
                (num_blocks, 1, 1),
                (THREADS_PER_BLOCK, 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
        }
        stream.synchronize()?;

        // Download digests (512MB instead of 47GB!)
        stream.memcpy_dtoh(&d_digests, &mut digest_buffer_host)?;
        let hash_time = hash_start.elapsed();

        eprintln!(
            "  [GPU Hash] hash={:.1}ms ({} rows, {:.1} MB digests)",
            hash_time.as_secs_f64() * 1000.0,
            num_rows,
            (digest_buffer_size * 8) as f64 / 1_048_576.0
        );

        // Skip full LDE download - only revealed rows will be fetched on-demand later
        // This optimization saves ~6 seconds by avoiding large transfer for small needed data
        let download_start = Instant::now();
        eprintln!("  [Optimization] Skipping {:.2} GB download - will fetch revealed rows on demand",
                 0 as f64 / 1_073_741_824.0);
        eprintln!("  [GPU-Resident LDE] Cached {:.2} GB GPU buffer (for quotient reuse & partial row extraction)",
                 0 as f64 / 1_073_741_824.0);
        let download_time = download_start.elapsed();

        // Convert digest buffer to Vec<Digest>
        let postproc_start = Instant::now();
        let digests: Vec<Digest> = (0..num_rows)
            .map(|row_idx| {
                let start = row_idx * DIGEST_LEN;
                Digest::new([
                    BFieldElement::from_raw_u64(digest_buffer_host[start]),
                    BFieldElement::from_raw_u64(digest_buffer_host[start + 1]),
                    BFieldElement::from_raw_u64(digest_buffer_host[start + 2]),
                    BFieldElement::from_raw_u64(digest_buffer_host[start + 3]),
                    BFieldElement::from_raw_u64(digest_buffer_host[start + 4]),
                ])
            })
            .collect();
        let postproc_time = postproc_start.elapsed();

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Table Operation GPU-Resident] total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, hash={:.1}ms, download={:.1}ms, postproc={:.1}ms)",
            total_time.as_secs_f64() * 1000.0,
            prep_time.as_secs_f64() * 1000.0,
            upload_time.as_secs_f64() * 1000.0,
            kernel_time.as_secs_f64() * 1000.0,
            hash_time.as_secs_f64() * 1000.0,
            download_time.as_secs_f64() * 1000.0,
            postproc_time.as_secs_f64() * 1000.0
        );

        // Return digests AND GPU buffer
        // The buffer stays GPU-resident and can be reused for quotient evaluation
        // (avoiding wasteful re-upload)
        Ok((digests, d_data))
    }


    pub(crate) fn execute_poly_fill_table_bfield(
        &self,
        table_data: &mut [BFieldElement],
        num_rows: usize,
        num_columns: usize,
        polys_bfield: Vec<&[BFieldElement]>
    ) -> Result<DeviceBuffer,Box<dyn std::error::Error>> {

        assert!(num_rows.is_power_of_two(), "Table rows must be power of 2");

        // Check if dual-GPU mode is available
        let use_dual_gpu = self.devices.len() >= 2;
        let mid_column = if use_dual_gpu { num_columns / 2 } else { num_columns };

        if use_dual_gpu {
            eprintln!(
                "\n[Dual-GPU Poly init bfield] Processing table: {} rows × {} columns",
                num_rows, num_columns
            );
            eprintln!("  GPU 0: columns [0, {})", mid_column);
            eprintln!("  GPU 1: columns [{}, {})", mid_column, num_columns);
        } else {
            eprintln!(
                "\n[GPU Poly init bfield] Processing table: {} rows × {} columns (row-major, strided access)",
                num_rows,
                num_columns
            );
        }

        let dev_ctx = &self.devices[0];

        // Prepare polynomial metadata
        let mut total_polys_length = 0;
        let mut polys_start_idx = Vec::new();
        let mut polys_len_idx = Vec::new();
        for i in 0..polys_bfield.len(){
            polys_start_idx.push(total_polys_length as u64);
            polys_len_idx.push(polys_bfield[i].len() as u64);
            total_polys_length += polys_bfield[i].len() as u64;
        }

        // Batch all polynomial data into a single buffer
        let mut combined_poly_data = Vec::with_capacity(total_polys_length as usize);
        for poly in polys_bfield.iter() {
            let raw_data_slice: &[u64] = unsafe {
                std::slice::from_raw_parts(
                    poly.as_ptr() as *const u64,
                    poly.len()
                )
            };
            combined_poly_data.extend_from_slice(raw_data_slice);
        }

        let alloc_start = Instant::now();
        let stream = dev_ctx.device.default_stream();

        // GPU 0: Allocate and process first subset of columns [0, mid_column)
        let d_data = stream.alloc::<u64>(num_rows * num_columns)?;
        let d_poly_lengths = stream.memcpy_htod(&polys_len_idx)?;
        let d_poly_start = stream.memcpy_htod(&polys_start_idx)?;
        let d_all_poly = stream.memcpy_htod(&combined_poly_data)?;

        let alloc_time = alloc_start.elapsed();
        eprintln!("  [GPU 0] Alloc+copy time: {:.1}ms", alloc_time.as_secs_f64() * 1000.0);

        // Dual-GPU setup
        let mut gpu1_setup_time = std::time::Duration::from_secs(0);
        let mut gpu0_kernel_time = std::time::Duration::from_secs(0);
        let mut gpu1_kernel_time = std::time::Duration::from_secs(0);
        let mut sync0_time = std::time::Duration::from_secs(0);
        let mut sync1_time = std::time::Duration::from_secs(0);
        let mut p2p_return_time = std::time::Duration::from_secs(0);
        let mut merge_time = std::time::Duration::from_secs(0);

        let (dev_ctx_1, stream_1, d_data_1, d_poly_lengths_1, d_poly_start_1, d_all_poly_1) = if use_dual_gpu {
            let gpu1_setup_start = Instant::now();
            let dev_1 = &self.devices[1];
            let stream_1 = dev_1.device.create_stream()?;

            eprintln!("  [GPU 1] Allocating buffers and uploading data...");
            // GPU 1 needs its own allocations for the full table (but will only fill its columns)
            let d_data_1 = stream_1.alloc::<u64>(num_rows * num_columns)?;
            let d_poly_lengths_1 = stream_1.memcpy_htod(&polys_len_idx)?;
            let d_poly_start_1 = stream_1.memcpy_htod(&polys_start_idx)?;
            let d_all_poly_1 = stream_1.memcpy_htod(&combined_poly_data)?;

            gpu1_setup_time = gpu1_setup_start.elapsed();
            eprintln!("  [GPU 1 Setup] {:.1}ms (alloc + upload)", gpu1_setup_time.as_secs_f64() * 1000.0);

            (Some(dev_1), Some(stream_1), Some(d_data_1), Some(d_poly_lengths_1), Some(d_poly_start_1), Some(d_all_poly_1))
        } else {
            (None, None, None, None, None, None)
        };

        // GPU 0: Process first subset of polynomials [0, mid_column)
        let kernel_start = Instant::now();
        if mid_column > 0 {
            let d_data_ptr = d_data.as_ptr();
            let d_all_poly_ptr = d_all_poly.as_ptr();
            let d_poly_lengths_ptr = d_poly_lengths.as_ptr();
            let d_poly_start_ptr = d_poly_start.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_ptr(&d_data_ptr)
                .push_value(num_rows as u64)
                .push_value(num_columns as u64)
                .push_ptr(&d_all_poly_ptr)
                .push_ptr(&d_poly_lengths_ptr)
                .push_ptr(&d_poly_start_ptr)
                .push_value(0u64)          // col_start
                .push_value(mid_column as u64); // col_end

            dev_ctx.bfield_poly_fill_table_fn.launch(
                (mid_column as u32, 1, 1),
                (512, 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
        }
        gpu0_kernel_time = kernel_start.elapsed();
        eprintln!("  [GPU 0 Kernel] {:.1}ms for {} columns", gpu0_kernel_time.as_secs_f64() * 1000.0, mid_column);

        // GPU 1: Process second subset of polynomials [mid_column, num_columns)
        if let (Some(dev_1), Some(stream_1), Some(d_data_1), Some(d_poly_lengths_1), Some(d_poly_start_1), Some(d_all_poly_1)) =
            (dev_ctx_1, &stream_1, &d_data_1, &d_poly_lengths_1, &d_poly_start_1, &d_all_poly_1) {

            eprintln!("  [GPU 1] Processing columns [{}, {})...", mid_column, num_columns);
            let gpu1_kernel_start = Instant::now();

            let num_cols_gpu1 = num_columns - mid_column;
            if num_cols_gpu1 > 0 {
                let d_data_ptr_1 = d_data_1.as_ptr();
                let d_all_poly_ptr_1 = d_all_poly_1.as_ptr();
                let d_poly_lengths_ptr_1 = d_poly_lengths_1.as_ptr();
                let d_poly_start_ptr_1 = d_poly_start_1.as_ptr();

                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_ptr(&d_data_ptr_1)
                    .push_value(num_rows as u64)
                    .push_value(num_columns as u64)
                    .push_ptr(&d_all_poly_ptr_1)
                    .push_ptr(&d_poly_lengths_ptr_1)
                    .push_ptr(&d_poly_start_ptr_1)
                    .push_value(mid_column as u64)     // col_start
                    .push_value(num_columns as u64);   // col_end

                dev_1.bfield_poly_fill_table_fn.launch(
                    (num_cols_gpu1 as u32, 1, 1),
                    (512, 1, 1),
                    0,
                    &stream_1,
                    kernel_args.as_mut_slice(),
                )?;
            }
            gpu1_kernel_time = gpu1_kernel_start.elapsed();
            eprintln!("  [GPU 1 Kernel] {:.1}ms for {} columns", gpu1_kernel_time.as_secs_f64() * 1000.0, num_cols_gpu1);

            // Wait for GPU 0 to finish
            let sync0_start = Instant::now();
            stream.synchronize()?;
            sync0_time = sync0_start.elapsed();
            eprintln!("  [GPU 0] Completed columns [0, {}) - sync wait: {:.1}ms", mid_column, sync0_time.as_secs_f64() * 1000.0);

            // Wait for GPU 1 to finish
            let sync1_start = Instant::now();
            stream_1.synchronize()?;
            sync1_time = sync1_start.elapsed();
            eprintln!("  [GPU 1] Completed columns [{}, {}) - sync wait: {:.1}ms", mid_column, num_columns, sync1_time.as_secs_f64() * 1000.0);

            // P2P copy GPU 1's table back to GPU 0 temp buffer
            eprintln!("  [Merge] P2P copying GPU 1 results back...");
            let d_data_temp = stream.alloc::<u64>(num_rows * num_columns)?;
            let p2p_return_start = Instant::now();
            let data_size_gb = (num_rows * num_columns * 8) as f64 / 1_000_000_000.0;
            stream.memcpy_peer_async(&d_data_temp, &dev_ctx.device.context, d_data_1, &dev_1.device.context, num_rows * num_columns * 8)?;
            stream.synchronize()?;
            p2p_return_time = p2p_return_start.elapsed();
            let return_bandwidth_gbs = data_size_gb / p2p_return_time.as_secs_f64();
            eprintln!("  [P2P Copy GPU 1→0] {:.2} GB in {:.1}ms = {:.1} GB/s",
                     data_size_gb, p2p_return_time.as_secs_f64() * 1000.0, return_bandwidth_gbs);

            // Merge GPU 1's columns into main table
            eprintln!("  [Merge] Copying columns [{}, {}) from GPU 1 into main table...", mid_column, num_columns);
            let merge_start = Instant::now();
            {
                let d_data_temp_ptr = d_data_temp.as_ptr();
                let mut d_data_ptr = d_data.as_ptr();
                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_ptr(&d_data_temp_ptr)
                    .push_mut_ptr(&mut d_data_ptr)
                    .push_value(num_rows as u64)
                    .push_value(num_columns as u64)
                    .push_value(mid_column as u64)
                    .push_value(num_columns as u64);

                let total_elements = num_rows * (num_columns - mid_column);
                let threads_per_block = 256;
                let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

                dev_ctx.copy_columns_bfield_fn.launch(
                    (num_blocks as u32, 1, 1),
                    (threads_per_block as u32, 1, 1),
                    0,
                    &stream,
                    kernel_args.as_mut_slice(),
                )?;
            }
            stream.synchronize()?;
            merge_time = merge_start.elapsed();
            eprintln!("  [Merge Kernel] {:.1}ms", merge_time.as_secs_f64() * 1000.0);
            eprintln!("  [Merge] Complete!");

            // Timing breakdown
            let total_overhead = gpu1_setup_time.as_secs_f64() * 1000.0
                               + sync0_time.as_secs_f64() * 1000.0
                               + sync1_time.as_secs_f64() * 1000.0
                               + p2p_return_time.as_secs_f64() * 1000.0
                               + merge_time.as_secs_f64() * 1000.0;
            let actual_compute = gpu0_kernel_time.as_secs_f64().max(gpu1_kernel_time.as_secs_f64()) * 1000.0;
            eprintln!("  [Kernel Timing Breakdown]");
            eprintln!("    GPU 1 setup (alloc+upload): {:.1}ms", gpu1_setup_time.as_secs_f64() * 1000.0);
            eprintln!("    GPU 0 kernel compute: {:.1}ms", gpu0_kernel_time.as_secs_f64() * 1000.0);
            eprintln!("    GPU 1 kernel compute: {:.1}ms", gpu1_kernel_time.as_secs_f64() * 1000.0);
            eprintln!("    Sync waits (0+1): {:.1}ms + {:.1}ms", sync0_time.as_secs_f64() * 1000.0, sync1_time.as_secs_f64() * 1000.0);
            eprintln!("    Merge (P2P+kernel): {:.1}ms + {:.1}ms", p2p_return_time.as_secs_f64() * 1000.0, merge_time.as_secs_f64() * 1000.0);
            eprintln!("    Actual parallel compute: {:.1}ms", actual_compute);
            eprintln!("    Total overhead: {:.1}ms", total_overhead);

            // Clean up GPU 1 resources
            drop(d_poly_lengths_1);
            drop(d_all_poly_1);
            drop(d_poly_start_1);
        } else {
            // Single-GPU path
            stream.synchronize()?;
            let kernel_duration = kernel_start.elapsed();
            eprintln!("  [GPU 0] Completed all {} columns", num_columns);
            eprintln!("[GPU Poly init] Kernel time: {:.1}ms", kernel_duration.as_secs_f64() * 1000.0);
        }

        drop(d_poly_lengths);
        drop(d_all_poly);
        drop(d_poly_start);

        Ok(d_data)
    }



    pub(crate) fn execute_poly_fill_table_xfield(
        &self,
        table_data: &mut [XFieldElement],
        num_rows: usize,
        num_columns: usize,
        polys_xfield: Vec<&[XFieldElement]>
    ) -> Result<DeviceBuffer,Box<dyn std::error::Error>> {

        let operation_start = Instant::now();

        assert!(num_rows.is_power_of_two(), "Table rows must be power of 2");

        // Check if dual-GPU mode is available
        let use_dual_gpu = self.devices.len() >= 2;
        // For XField (3 u64s per element), ensure mid_column * 3 * 8 is aligned to 128 bytes
        // This requires: (mid_column * 24) % 128 == 0, which simplifies to: mid_column % 16 == 0
        let mid_column = if use_dual_gpu {
            let half = num_columns / 2;
            // Round UP to nearest multiple of 16 to ensure proper alignment
            ((half + 15) / 16) * 16
        } else {
            num_columns
        };

        if use_dual_gpu {
            eprintln!(
                "\n[Dual-GPU Poly init xfield] Processing table: {} rows × {} columns",
                num_rows, num_columns
            );
            eprintln!("  GPU 0: columns [0, {})", mid_column);
            eprintln!("  GPU 1: columns [{}, {})", mid_column, num_columns);
        } else {
            eprintln!(
                "\n[GPU Poly init xfield] Processing table: {} rows × {} columns (row-major, strided access)",
                num_rows,
                num_columns
            );
        }

        let dev_ctx = &self.devices[0];

        // Prepare polynomial metadata
        let mut total_polys_length = 0;
        let mut polys_start_idx = Vec::new();
        let mut polys_len_idx = Vec::new();
        for i in 0..polys_xfield.len(){
            polys_start_idx.push(total_polys_length as u64);
            polys_len_idx.push(polys_xfield[i].len() as u64);
            total_polys_length += polys_xfield[i].len() as u64;
        }

        // Batch all polynomial data into a single buffer
        let mut combined_poly_data = Vec::with_capacity((total_polys_length * 3) as usize);
        for poly in polys_xfield.iter() {
            let raw_data_slice: &[u64] = unsafe {
                std::slice::from_raw_parts(
                    poly.as_ptr() as *const u64,
                    poly.len() * 3  // XField has 3 u64s per element
                )
            };
            combined_poly_data.extend_from_slice(raw_data_slice);
        }

        let alloc_start = Instant::now();
        let stream = dev_ctx.device.default_stream();

        // GPU 0: Allocate and process first subset of columns [0, mid_column)
        let d_data = stream.alloc::<u64>(num_rows * num_columns * 3)?;
        let d_poly_lengths = stream.memcpy_htod(&polys_len_idx)?;
        let d_poly_start = stream.memcpy_htod(&polys_start_idx)?;
        let d_all_poly = stream.memcpy_htod(&combined_poly_data)?;

        let alloc_time = alloc_start.elapsed();
        eprintln!("  [GPU 0] Alloc+copy time: {:.1}ms", alloc_time.as_secs_f64() * 1000.0);

        // Dual-GPU setup
        let mut gpu1_setup_time = std::time::Duration::from_secs(0);
        let mut gpu0_kernel_time = std::time::Duration::from_secs(0);
        let mut gpu1_kernel_time = std::time::Duration::from_secs(0);
        let mut sync0_time = std::time::Duration::from_secs(0);
        let mut sync1_time = std::time::Duration::from_secs(0);
        let mut p2p_return_time = std::time::Duration::from_secs(0);
        let mut merge_time = std::time::Duration::from_secs(0);

        let (dev_ctx_1, stream_1, d_data_1, d_poly_lengths_1, d_poly_start_1, d_all_poly_1) = if use_dual_gpu {
            let gpu1_setup_start = Instant::now();
            let dev_1 = &self.devices[1];
            let stream_1 = dev_1.device.create_stream()?;

            eprintln!("  [GPU 1] Allocating buffers and uploading data...");
            // GPU 1 needs its own allocations for the full table (but will only fill its columns)
            let d_data_1 = stream_1.alloc::<u64>(num_rows * num_columns * 3)?;
            let d_poly_lengths_1 = stream_1.memcpy_htod(&polys_len_idx)?;
            let d_poly_start_1 = stream_1.memcpy_htod(&polys_start_idx)?;
            let d_all_poly_1 = stream_1.memcpy_htod(&combined_poly_data)?;

            gpu1_setup_time = gpu1_setup_start.elapsed();
            eprintln!("  [GPU 1 Setup] {:.1}ms (alloc + upload)", gpu1_setup_time.as_secs_f64() * 1000.0);

            (Some(dev_1), Some(stream_1), Some(d_data_1), Some(d_poly_lengths_1), Some(d_poly_start_1), Some(d_all_poly_1))
        } else {
            (None, None, None, None, None, None)
        };

        // GPU 0: Process first subset of polynomials [0, mid_column)
        let kernel_start = Instant::now();
        if mid_column > 0 {
            let d_data_ptr = d_data.as_ptr();
            let d_all_poly_ptr = d_all_poly.as_ptr();
            let d_poly_lengths_ptr = d_poly_lengths.as_ptr();
            let d_poly_start_ptr = d_poly_start.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_ptr(&d_data_ptr)
                .push_value(num_rows as u64)
                .push_value(num_columns as u64)
                .push_ptr(&d_all_poly_ptr)
                .push_ptr(&d_poly_lengths_ptr)
                .push_ptr(&d_poly_start_ptr)
                .push_value(0u64)          // col_start
                .push_value(mid_column as u64); // col_end

            dev_ctx.xfield_poly_fill_table_fn.launch(
                (mid_column as u32, 1, 1),
                (512, 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
        }
        gpu0_kernel_time = kernel_start.elapsed();
        eprintln!("  [GPU 0 Kernel] {:.1}ms for {} columns", gpu0_kernel_time.as_secs_f64() * 1000.0, mid_column);

        // GPU 1: Process second subset of polynomials [mid_column, num_columns)
        if let (Some(dev_1), Some(stream_1), Some(d_data_1), Some(d_poly_lengths_1), Some(d_poly_start_1), Some(d_all_poly_1)) =
            (dev_ctx_1, &stream_1, &d_data_1, &d_poly_lengths_1, &d_poly_start_1, &d_all_poly_1) {

            eprintln!("  [GPU 1] Processing columns [{}, {})...", mid_column, num_columns);
            let gpu1_kernel_start = Instant::now();

            let num_cols_gpu1 = num_columns - mid_column;
            if num_cols_gpu1 > 0 {
                let d_data_ptr_1 = d_data_1.as_ptr();
                let d_all_poly_ptr_1 = d_all_poly_1.as_ptr();
                let d_poly_lengths_ptr_1 = d_poly_lengths_1.as_ptr();
                let d_poly_start_ptr_1 = d_poly_start_1.as_ptr();

                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_ptr(&d_data_ptr_1)
                    .push_value(num_rows as u64)
                    .push_value(num_columns as u64)
                    .push_ptr(&d_all_poly_ptr_1)
                    .push_ptr(&d_poly_lengths_ptr_1)
                    .push_ptr(&d_poly_start_ptr_1)
                    .push_value(mid_column as u64)     // col_start
                    .push_value(num_columns as u64);   // col_end

                dev_1.xfield_poly_fill_table_fn.launch(
                    (num_cols_gpu1 as u32, 1, 1),
                    (512, 1, 1),
                    0,
                    &stream_1,
                    kernel_args.as_mut_slice(),
                )?;
            }
            gpu1_kernel_time = gpu1_kernel_start.elapsed();
            eprintln!("  [GPU 1 Kernel] {:.1}ms for {} columns", gpu1_kernel_time.as_secs_f64() * 1000.0, num_cols_gpu1);

            // Wait for GPU 0 to finish
            let sync0_start = Instant::now();
            stream.synchronize()?;
            sync0_time = sync0_start.elapsed();
            eprintln!("  [GPU 0] Completed columns [0, {}) - sync wait: {:.1}ms", mid_column, sync0_time.as_secs_f64() * 1000.0);

            // Wait for GPU 1 to finish
            let sync1_start = Instant::now();
            stream_1.synchronize()?;
            sync1_time = sync1_start.elapsed();
            eprintln!("  [GPU 1] Completed columns [{}, {}) - sync wait: {:.1}ms", mid_column, num_columns, sync1_time.as_secs_f64() * 1000.0);

            // P2P copy GPU 1's table back to GPU 0 temp buffer (XField = 3 u64s per element)
            eprintln!("  [Merge] P2P copying GPU 1 results back...");
            let d_data_temp = stream.alloc::<u64>(num_rows * num_columns * 3)?;
            let p2p_return_start = Instant::now();
            let data_size_gb = (num_rows * num_columns * 3 * 8) as f64 / 1_000_000_000.0;
            stream.memcpy_peer_async(&d_data_temp, &dev_ctx.device.context, d_data_1, &dev_1.device.context, num_rows * num_columns * 3 * 8)?;
            stream.synchronize()?;
            p2p_return_time = p2p_return_start.elapsed();
            let return_bandwidth_gbs = data_size_gb / p2p_return_time.as_secs_f64();
            eprintln!("  [P2P Copy GPU 1→0] {:.2} GB in {:.1}ms = {:.1} GB/s",
                     data_size_gb, p2p_return_time.as_secs_f64() * 1000.0, return_bandwidth_gbs);

            // Merge GPU 1's columns into main table
            eprintln!("  [Merge] Copying columns [{}, {}) from GPU 1 into main table...", mid_column, num_columns);
            let merge_start = Instant::now();
            {
                let d_data_temp_ptr = d_data_temp.as_ptr();
                let mut d_data_ptr = d_data.as_ptr();
                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_ptr(&d_data_temp_ptr)
                    .push_mut_ptr(&mut d_data_ptr)
                    .push_value(num_rows as u64)
                    .push_value(num_columns as u64)    // total columns (XField elements)
                    .push_value(mid_column as u64)     // col_start (XField elements)
                    .push_value(num_columns as u64);   // col_end (XField elements)

                let total_elements = num_rows * (num_columns - mid_column);
                let threads_per_block = 256;
                let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

                dev_ctx.copy_columns_xfield_fn.launch(
                    (num_blocks as u32, 1, 1),
                    (threads_per_block as u32, 1, 1),
                    0,
                    &stream,
                    kernel_args.as_mut_slice(),
                )?;
            }
            stream.synchronize()?;
            merge_time = merge_start.elapsed();
            eprintln!("  [Merge Kernel] {:.1}ms", merge_time.as_secs_f64() * 1000.0);
            eprintln!("  [Merge] Complete!");

            // Timing breakdown
            let total_overhead = gpu1_setup_time.as_secs_f64() * 1000.0
                               + sync0_time.as_secs_f64() * 1000.0
                               + sync1_time.as_secs_f64() * 1000.0
                               + p2p_return_time.as_secs_f64() * 1000.0
                               + merge_time.as_secs_f64() * 1000.0;
            let actual_compute = gpu0_kernel_time.as_secs_f64().max(gpu1_kernel_time.as_secs_f64()) * 1000.0;
            eprintln!("  [Kernel Timing Breakdown]");
            eprintln!("    GPU 1 setup (alloc+upload): {:.1}ms", gpu1_setup_time.as_secs_f64() * 1000.0);
            eprintln!("    GPU 0 kernel compute: {:.1}ms", gpu0_kernel_time.as_secs_f64() * 1000.0);
            eprintln!("    GPU 1 kernel compute: {:.1}ms", gpu1_kernel_time.as_secs_f64() * 1000.0);
            eprintln!("    Sync waits (0+1): {:.1}ms + {:.1}ms", sync0_time.as_secs_f64() * 1000.0, sync1_time.as_secs_f64() * 1000.0);
            eprintln!("    Merge (P2P+kernel): {:.1}ms + {:.1}ms", p2p_return_time.as_secs_f64() * 1000.0, merge_time.as_secs_f64() * 1000.0);
            eprintln!("    Actual parallel compute: {:.1}ms", actual_compute);
            eprintln!("    Total overhead: {:.1}ms", total_overhead);

            // Clean up GPU 1 resources
            drop(d_poly_lengths_1);
            drop(d_all_poly_1);
            drop(d_poly_start_1);
        } else {
            // Single-GPU path
            stream.synchronize()?;
            let kernel_duration = kernel_start.elapsed();
            eprintln!("  [GPU 0] Completed all {} columns", num_columns);
            eprintln!("[GPU Poly init] Kernel time: {:.1}ms", kernel_duration.as_secs_f64() * 1000.0);
        }

        drop(d_poly_lengths);
        drop(d_all_poly);
        drop(d_poly_start);

        Ok(d_data)
    }


    /// Execute fused coset+NTT on a row-major table using strided GPU kernels
    /// For XFieldElement tables
    ///
    /// Returns: (Vec<Digest>, DeviceBuffer) containing hashes and GPU-resident buffer
    /// The GPU buffer is kept on device to avoid wasteful GPU→RAM→GPU transfers
    pub(crate) fn execute_fused_coset_ntt_table_xfield(
        &self,
        table_data: DeviceBuffer,
        num_rows: usize,
        num_columns: usize,
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(Vec<Digest>, DeviceBuffer), Box<dyn std::error::Error>> {
        let operation_start = Instant::now();

        assert!(num_rows.is_power_of_two(), "Table rows must be power of 2");

        let log2_len = num_rows.trailing_zeros();
        let dev_ctx = self.select_device();

        // Check if dual-GPU is available
        let use_dual_gpu = self.devices.len() >= 2;
        // For XField (3 u64s per element), ensure mid_column * 3 * 8 is aligned to 128 bytes
        // This requires: (mid_column * 24) % 128 == 0, which simplifies to: mid_column % 16 == 0
        let mid_column = if use_dual_gpu {
            let half = num_columns / 2;
            // Round UP to nearest multiple of 16 to ensure proper alignment
            ((half + 15) / 16) * 16
        } else {
            num_columns
        };

        if use_dual_gpu {
            eprintln!(
                "\n[Dual-GPU Table XField Fused Coset+NTT - {}] Processing table: {} rows × {} columns",
                phase_name,
                num_rows,
                num_columns
            );
            eprintln!("  GPU 0: columns [0, {})", mid_column);
            eprintln!("  GPU 1: columns [{}, {})", mid_column, num_columns);
        } else {
            eprintln!(
                "\n[GPU Table XField Fused Coset+NTT - {}] Processing table: {} rows × {} columns (row-major, strided access)",
                phase_name,
                num_rows,
                num_columns
            );
        }

        // ZERO-COPY: Cast XFieldElement slice directly to u64 slice
        let prep_start = Instant::now();
        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);
        let prep_time = prep_start.elapsed();

        // Calculate memory requirements
        let twiddle_bytes = raw_twiddles.len() * 8;
        let total_bytes = 0 + twiddle_bytes;
        eprintln!(
            "  GPU Memory Required: {:.2} GB (data: {:?} GB, twiddles: {:.2} MB)",
            total_bytes as f64 / 1_073_741_824.0,
            "unknown",
            twiddle_bytes as f64 / 1_048_576.0
        );

        // Upload entire table (zero-copy - no serialization!)
        let upload_start = Instant::now();
        let stream = dev_ctx.device.default_stream();
        let d_data = table_data;
        let d_data_extra = stream.alloc::<u64>(num_rows * 3)?;
        let d_omegas_extra = stream.alloc::<u64>(num_rows)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let upload_time = upload_start.elapsed();

        // Launch dual-GPU strided kernel
        let offset_raw = offset.raw_u64();
        let kernel_start = Instant::now();

        // Dual-GPU setup
        let data_size_gb = (num_rows * num_columns * 3 * 8) as f64 / 1_000_000_000.0;
        let (dev_ctx_1, stream_1, d_data_1, d_data_extra_1, d_omegas_extra_1, d_twiddles_1) = if use_dual_gpu {
            let gpu1_setup_start = Instant::now();
            let dev_1 = &self.devices[1];
            let stream_1 = dev_1.device.create_stream()?;

            eprintln!("  [GPU 1] Allocating buffers and P2P copying table...");
            // P2P copy entire table from GPU 0 to GPU 1
            let d_data_1 = stream_1.alloc::<u64>(num_rows * num_columns * 3)?;
            let p2p_copy_start = Instant::now();
            stream.memcpy_peer_async(&d_data_1, &dev_1.device.context, &d_data, &dev_ctx.device.context, num_rows * num_columns * 3 * 8)?;
            stream.synchronize()?;
            let p2p_copy_time = p2p_copy_start.elapsed();
            let bandwidth_gbs = data_size_gb / p2p_copy_time.as_secs_f64();
            eprintln!("  [P2P Copy GPU 0→1] {:.2} GB in {:.1}ms = {:.1} GB/s",
                     data_size_gb, p2p_copy_time.as_secs_f64() * 1000.0, bandwidth_gbs);

            // Allocate GPU 1's working buffers
            let d_data_extra_1 = stream_1.alloc::<u64>(num_rows * 3)?;
            let d_omegas_extra_1 = stream_1.alloc::<u64>(num_rows)?;
            let d_twiddles_1 = stream_1.memcpy_htod(&raw_twiddles)?;

            // Init omegas on GPU 1
            {
                let mut d_omegas_extra_ptr_1 = d_omegas_extra_1.as_ptr();
                let d_twiddles_ptr_1 = d_twiddles_1.as_ptr();
                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_value(num_rows as u64)
                    .push_ptr(&d_twiddles_ptr_1)
                    .push_mut_ptr(&mut d_omegas_extra_ptr_1);
                dev_1.ntt_xfield_init_omegas_fn.launch(
                    (log2_len as u32, 1, 1),
                    (512, 1, 1),
                    0,
                    &stream_1,
                    kernel_args.as_mut_slice(),
                )?;
            }
            stream_1.synchronize()?;
            let gpu1_setup_time = gpu1_setup_start.elapsed();
            eprintln!("  [GPU 1 Setup] {:.1}ms (alloc + P2P + init omegas)", gpu1_setup_time.as_secs_f64() * 1000.0);

            (Some(dev_1), Some(stream_1), Some(d_data_1), Some(d_data_extra_1), Some(d_omegas_extra_1), Some(d_twiddles_1))
        } else {
            (None, None, None, None, None, None)
        };

        // GPU 0: Process first half of columns using strided kernel (single launch!)
        let gpu0_ntt_start = Instant::now();
        if mid_column > 0 {
            let mut d_data_ptr = d_data.as_ptr();
            let d_omegas_ptr = d_twiddles.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_mut_ptr(&mut d_data_ptr)
                .push_value(offset_raw)
                .push_value(num_rows as u64)
                .push_value((num_columns * 3) as u64)  // stride for XField (3 u64s per element)
                .push_ptr(&d_omegas_ptr)
                .push_value(log2_len as u32);

            dev_ctx.ntt_xfield_fused_coset_strided_fn.launch_cooperative(
                (mid_column as u32, 1, 1),  // one block per column
                (256, 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
        }
        let gpu0_ntt_time = gpu0_ntt_start.elapsed();
        eprintln!("  [GPU 0 NTT] {:.1}ms for {} columns", gpu0_ntt_time.as_secs_f64() * 1000.0, mid_column);

        // GPU 1: Process second half of columns (async via CUDA streams)
        if let (Some(dev_1), Some(stream_1), Some(d_data_1), Some(d_data_extra_1), Some(d_omegas_extra_1), Some(d_twiddles_1)) =
            (&dev_ctx_1, &stream_1, &d_data_1, &d_data_extra_1, &d_omegas_extra_1, &d_twiddles_1) {

            eprintln!("  [GPU 1] Processing columns [{}, {})...", mid_column, num_columns);
            let gpu1_ntt_start = Instant::now();

            // GPU 1: Process second half using strided kernel (single launch!)
            let num_cols_gpu1 = num_columns - mid_column;
            if num_cols_gpu1 > 0 {
                // Offset data pointer to start at mid_column (XField = 3 u64s * 8 bytes = 24 bytes per element)
                let mut d_data_ptr_1 = d_data_1.as_ptr() + (mid_column as u64 * 3 * 8);
                let d_omegas_ptr_1 = d_twiddles_1.as_ptr();

                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_mut_ptr(&mut d_data_ptr_1)
                    .push_value(offset_raw)
                    .push_value(num_rows as u64)
                    .push_value((num_columns * 3) as u64)  // stride for XField
                    .push_ptr(&d_omegas_ptr_1)
                    .push_value(log2_len as u32);

                dev_1.ntt_xfield_fused_coset_strided_fn.launch_cooperative(
                    (num_cols_gpu1 as u32, 1, 1),
                    (256, 1, 1),
                    0,
                    &stream_1,
                    kernel_args.as_mut_slice(),
                )?;
            }
            let gpu1_ntt_time = gpu1_ntt_start.elapsed();
            eprintln!("  [GPU 1 NTT] {:.1}ms for {} columns (strided, 1 launch)", gpu1_ntt_time.as_secs_f64() * 1000.0, num_cols_gpu1);
        }

        // Synchronize and merge results
        let sync0_start = Instant::now();
        stream.synchronize()?;
        let sync0_time = sync0_start.elapsed();
        eprintln!("  [GPU 0] Completed columns [0, {}) - sync wait: {:.1}ms", mid_column, sync0_time.as_secs_f64() * 1000.0);

        if use_dual_gpu {
            if let (Some(dev_1), Some(stream_1), Some(d_data_1), Some(_d_data_extra_1), Some(_d_omegas_extra_1), Some(_d_twiddles_1)) =
                   (&dev_ctx_1, &stream_1, &d_data_1, &d_data_extra_1, &d_omegas_extra_1, &d_twiddles_1) {
                let sync1_start = Instant::now();
                stream_1.synchronize()?;
                let sync1_time = sync1_start.elapsed();
                eprintln!("  [GPU 1] Completed columns [{}, {}) - sync wait: {:.1}ms", mid_column, num_columns, sync1_time.as_secs_f64() * 1000.0);

                // P2P copy GPU 1's table back to GPU 0 temp buffer (like BField does)
                eprintln!("  [Merge] P2P copying GPU 1 results back...");
                let d_data_temp = stream.alloc::<u64>(num_rows * num_columns * 3)?;
                let p2p_return_start = Instant::now();
                stream.memcpy_peer_async(&d_data_temp, &dev_ctx.device.context, d_data_1, &dev_1.device.context, num_rows * num_columns * 3 * 8)?;
                stream.synchronize()?;
                let p2p_return_time = p2p_return_start.elapsed();
                let return_bandwidth_gbs = data_size_gb / p2p_return_time.as_secs_f64();
                eprintln!("  [P2P Copy GPU 1→0] {:.2} GB in {:.1}ms = {:.1} GB/s",
                         data_size_gb, p2p_return_time.as_secs_f64() * 1000.0, return_bandwidth_gbs);

                // Merge GPU 1's columns into main table
                eprintln!("  [Merge] Copying columns [{}, {}) from GPU 1 into main table...", mid_column, num_columns);
                let merge_start = Instant::now();

                let gpu1_num_cols = num_columns - mid_column;

                // Use copy kernel to merge into correct columns
                let d_data_temp_ptr = d_data_temp.as_ptr();
                let mut d_data_ptr = d_data.as_ptr();
                let mut kernel_args = KernelArgs::new();
                kernel_args
                    .push_ptr(&d_data_temp_ptr)
                    .push_mut_ptr(&mut d_data_ptr)
                    .push_value(num_rows as u64)
                    .push_value(num_columns as u64)    // total columns (XField elements)
                    .push_value(mid_column as u64)     // col_start (XField elements)
                    .push_value(num_columns as u64);   // col_end (XField elements)

                let total_elements = num_rows * gpu1_num_cols;
                let threads_per_block = 256;
                let num_blocks = (total_elements + threads_per_block - 1) / threads_per_block;

                dev_ctx.copy_columns_xfield_fn.launch(
                    (num_blocks as u32, 1, 1),
                    (threads_per_block as u32, 1, 1),
                    0,
                    &stream,
                    kernel_args.as_mut_slice(),
                )?;
                stream.synchronize()?;
                let merge_time = merge_start.elapsed();
                eprintln!("  [Merge] Complete in {:.1}ms!", merge_time.as_secs_f64() * 1000.0);

                // Now perform dual-GPU parallel hashing while we still have access to dev_1 and stream_1
                let hash_start_dual = Instant::now();
                const DIGEST_LEN: usize = 5;
                const THREADS_PER_BLOCK: u32 = 256;

                // Dual-GPU parallel hashing: split rows 50/50
                let mid_row = num_rows / 2;
                let num_rows_gpu0 = mid_row;
                let num_rows_gpu1 = num_rows - mid_row;

                eprintln!("  [Dual-GPU Hash] GPU 0: rows [0, {}), GPU 1: rows [{}, {})", mid_row, mid_row, num_rows);

                // P2P copy merged table from GPU 0 to GPU 1 for hashing (XField = 3 u64s per element)
                let d_data_1_hash = stream_1.alloc::<u64>(num_rows * num_columns * 3)?;
                let p2p_hash_copy_start = Instant::now();
                stream.memcpy_peer_async(&d_data_1_hash, &dev_1.device.context, &d_data, &dev_ctx.device.context, num_rows * num_columns * 3 * 8)?;
                stream.synchronize()?;
                let p2p_hash_copy_time = p2p_hash_copy_start.elapsed();
                let hash_copy_gb = (num_rows * num_columns * 3 * 8) as f64 / 1_000_000_000.0;
                let hash_bandwidth_gbs = hash_copy_gb / p2p_hash_copy_time.as_secs_f64();
                eprintln!("  [P2P Hash Copy GPU 0→1] {:.2} GB in {:.1}ms = {:.1} GB/s",
                         hash_copy_gb, p2p_hash_copy_time.as_secs_f64() * 1000.0, hash_bandwidth_gbs);

                // Allocate digest buffers on both GPUs
                let d_digests_0 = stream.alloc::<u64>(num_rows_gpu0 * DIGEST_LEN)?;
                let d_digests_1 = stream_1.alloc::<u64>(num_rows_gpu1 * DIGEST_LEN)?;

                // Row length in BFieldElements: num_columns * 3 (since XField = 3 BField)
                let row_length_bfield = (num_columns * 3) as u32;

                // Launch hash on GPU 0 for first half (rows 0 to mid_row)
                let num_blocks_0 = (num_rows_gpu0 as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                {
                    let d_data_ptr = d_data.as_ptr();
                    let mut d_digests_ptr_0 = d_digests_0.as_ptr();
                    let mut kernel_args = KernelArgs::new();
                    kernel_args
                        .push_ptr(&d_data_ptr)
                        .push_mut_ptr(&mut d_digests_ptr_0)
                        .push_value(row_length_bfield)
                        .push_value(num_rows_gpu0 as u32)
                        .push_value(0u32); // row_start = 0
                    dev_ctx.hash_rows_xfield_fn.launch(
                        (num_blocks_0, 1, 1),
                        (THREADS_PER_BLOCK, 1, 1),
                        0,
                        &stream,
                        kernel_args.as_mut_slice(),
                    )?;
                }

                // Launch hash on GPU 1 for second half (rows mid_row to num_rows)
                let num_blocks_1 = (num_rows_gpu1 as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
                {
                    let d_data_ptr_1 = d_data_1_hash.as_ptr();
                    let mut d_digests_ptr_1 = d_digests_1.as_ptr();
                    let mut kernel_args = KernelArgs::new();
                    kernel_args
                        .push_ptr(&d_data_ptr_1)
                        .push_mut_ptr(&mut d_digests_ptr_1)
                        .push_value(row_length_bfield)
                        .push_value(num_rows_gpu1 as u32)
                        .push_value(mid_row as u32); // row_start = mid_row
                    dev_1.hash_rows_xfield_fn.launch(
                        (num_blocks_1, 1, 1),
                        (THREADS_PER_BLOCK, 1, 1),
                        0,
                        &stream_1,
                        kernel_args.as_mut_slice(),
                    )?;
                }

                // Wait for both GPUs to finish hashing
                stream.synchronize()?;
                stream_1.synchronize()?;
                eprintln!("  [Dual-GPU Hash] Both GPUs completed hashing");

                // Download digests from both GPUs
                let mut digest_buffer_0 = vec![0u64; num_rows_gpu0 * DIGEST_LEN];
                let mut digest_buffer_1 = vec![0u64; num_rows_gpu1 * DIGEST_LEN];
                stream.memcpy_dtoh(&d_digests_0, &mut digest_buffer_0)?;
                stream_1.memcpy_dtoh(&d_digests_1, &mut digest_buffer_1)?;

                let hash_time_dual = hash_start_dual.elapsed();

                // Merge digest buffers
                let digest_buffer_size = num_rows * DIGEST_LEN;
                let mut digest_buffer_host = vec![0u64; digest_buffer_size];
                digest_buffer_host[0..digest_buffer_0.len()].copy_from_slice(&digest_buffer_0);
                digest_buffer_host[digest_buffer_0.len()..].copy_from_slice(&digest_buffer_1);

                // Drop extra buffers
                drop(d_data_extra);
                drop(d_omegas_extra);

                let kernel_time = kernel_start.elapsed();

                // Print hash timing and continue to digest conversion
                eprintln!(
                    "  [GPU Hash] hash={:.1}ms ({} rows, {:.1} MB digests)",
                    hash_time_dual.as_secs_f64() * 1000.0,
                    num_rows,
                    (digest_buffer_size * 8) as f64 / 1_048_576.0
                );

                // Skip full LDE download - only revealed rows will be fetched on-demand later
                let download_start = Instant::now();
                eprintln!("  [Optimization] Skipping {:.2} GB download - will fetch revealed rows on demand",
                         0 as f64 / 1_073_741_824.0);
                eprintln!("  [GPU-Resident LDE] Cached {:.2} GB GPU buffer (for quotient reuse & partial row extraction)",
                         0 as f64 / 1_073_741_824.0);
                let download_time = download_start.elapsed();

                // Convert digest buffer to Vec<Digest>
                let postproc_start = Instant::now();
                let digests: Vec<Digest> = (0..num_rows)
                    .map(|row_idx| {
                        let start = row_idx * DIGEST_LEN;
                        Digest::new([
                            BFieldElement::from_raw_u64(digest_buffer_host[start]),
                            BFieldElement::from_raw_u64(digest_buffer_host[start + 1]),
                            BFieldElement::from_raw_u64(digest_buffer_host[start + 2]),
                            BFieldElement::from_raw_u64(digest_buffer_host[start + 3]),
                            BFieldElement::from_raw_u64(digest_buffer_host[start + 4]),
                        ])
                    })
                    .collect();
                let postproc_time = postproc_start.elapsed();

                let total_time = operation_start.elapsed();

                eprintln!(
                    "  [Table Operation GPU-Resident] total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, hash={:.1}ms, download={:.1}ms, postproc={:.1}ms)",
                    total_time.as_secs_f64() * 1000.0,
                    prep_time.as_secs_f64() * 1000.0,
                    upload_time.as_secs_f64() * 1000.0,
                    kernel_time.as_secs_f64() * 1000.0,
                    hash_time_dual.as_secs_f64() * 1000.0,
                    download_time.as_secs_f64() * 1000.0,
                    postproc_time.as_secs_f64() * 1000.0
                );

                eprintln!("[GPU-Resident LDE] Cached {:.2} GB GPU buffer (avoiding wasteful download)",
                         (num_rows * num_columns * 3 * 8) as f64 / 1_073_741_824.0);

                // Return digests AND GPU buffer (for quotient reuse & partial row extraction)
                return Ok((digests, d_data));
            }
        }

        let kernel_time = kernel_start.elapsed();

        // GPU Row Hashing: Hash rows while data is still on GPU
        // XField: Each XFieldElement = 3 BFieldElements, so row_length = num_columns * 3
        let hash_start = Instant::now();
        const DIGEST_LEN: usize = 5;
        let digest_buffer_size = num_rows * DIGEST_LEN;
        let mut digest_buffer_host = vec![0u64; digest_buffer_size];
        let d_digests = stream.memcpy_htod(&digest_buffer_host)?;

        // Launch config: 256 threads per block, each thread processes different row
        // This improves SM occupancy and reduces scheduling overhead
        const THREADS_PER_BLOCK: u32 = 256;
        let num_blocks = (num_rows as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        // Row length in BFieldElements: num_columns * 3 (since XField = 3 BField)
        let row_length_bfield = (num_columns * 3) as u32;

        {
            let d_data_ptr = d_data.as_ptr();
            let mut d_digests_ptr = d_digests.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_ptr(&d_data_ptr)
                .push_mut_ptr(&mut d_digests_ptr)
                .push_value(row_length_bfield)
                .push_value(num_rows as u32)
                .push_value(0u32); // row_start = 0

            dev_ctx.hash_rows_xfield_fn.launch(
                (num_blocks, 1, 1),
                (THREADS_PER_BLOCK, 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
        }
        stream.synchronize()?;

        // Download digests (512MB instead of 33GB!)
        stream.memcpy_dtoh(&d_digests, &mut digest_buffer_host)?;
        let hash_time = hash_start.elapsed();

        eprintln!(
            "  [GPU Hash] hash={:.1}ms ({} rows, {:.1} MB digests)",
            hash_time.as_secs_f64() * 1000.0,
            num_rows,
            (digest_buffer_size * 8) as f64 / 1_048_576.0
        );

        // Skip full LDE download - only revealed rows will be fetched on-demand later
        // This optimization saves ~6 seconds by avoiding large transfer for small needed data
        let download_start = Instant::now();
        eprintln!("  [Optimization] Skipping {:.2} GB download - will fetch revealed rows on demand",
                 0 as f64 / 1_073_741_824.0);
        eprintln!("  [GPU-Resident LDE] Cached {:.2} GB GPU buffer (for quotient reuse & partial row extraction)",
                 0 as f64 / 1_073_741_824.0);
        let download_time = download_start.elapsed();

        // Convert digest buffer to Vec<Digest>
        let postproc_start = Instant::now();
        let digests: Vec<Digest> = (0..num_rows)
            .map(|row_idx| {
                let start = row_idx * DIGEST_LEN;
                Digest::new([
                    BFieldElement::from_raw_u64(digest_buffer_host[start]),
                    BFieldElement::from_raw_u64(digest_buffer_host[start + 1]),
                    BFieldElement::from_raw_u64(digest_buffer_host[start + 2]),
                    BFieldElement::from_raw_u64(digest_buffer_host[start + 3]),
                    BFieldElement::from_raw_u64(digest_buffer_host[start + 4]),
                ])
            })
            .collect();
        let postproc_time = postproc_start.elapsed();

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Table Operation GPU-Resident] total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, hash={:.1}ms, download={:.1}ms, postproc={:.1}ms)",
            total_time.as_secs_f64() * 1000.0,
            prep_time.as_secs_f64() * 1000.0,
            upload_time.as_secs_f64() * 1000.0,
            kernel_time.as_secs_f64() * 1000.0,
            hash_time.as_secs_f64() * 1000.0,
            download_time.as_secs_f64() * 1000.0,
            postproc_time.as_secs_f64() * 1000.0
        );

        // Return digests AND GPU buffer
        // The buffer stays GPU-resident and can be reused for quotient evaluation
        // (avoiding wasteful re-upload)
        Ok((digests, d_data))
    }

    /// Execute INTT on a row-major table using strided GPU kernels
    /// For BFieldElement tables
    pub(crate) fn execute_intt_table_bfield(
        &self,
        table_data: &mut [BFieldElement],
        num_rows: usize,
        num_columns: usize,
        twiddle_factors: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let operation_start = Instant::now();

        assert!(num_rows.is_power_of_two(), "Table rows must be power of 2");

        let log2_len = num_rows.trailing_zeros();
        let dev_ctx = self.select_device();

        eprintln!(
            "\n[GPU Table BField INTT - {}] Processing table: {} rows × {} columns (row-major, strided access)",
            phase_name,
            num_rows,
            num_columns
        );

        // ZERO-COPY: Cast BFieldElement slice directly to u64 slice
        let prep_start = Instant::now();
        let raw_data_slice: &mut [u64] = unsafe {
            std::slice::from_raw_parts_mut(
                table_data.as_mut_ptr() as *mut u64,
                table_data.len()
            )
        };
        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);
        let prep_time = prep_start.elapsed();

        // Calculate memory requirements
        let data_bytes = raw_data_slice.len() * 8;
        let twiddle_bytes = raw_twiddles.len() * 8;
        let total_bytes = data_bytes + twiddle_bytes;
        eprintln!(
            "  GPU Memory Required: {:.2} GB (data: {:.2} GB, twiddles: {:.2} MB)",
            total_bytes as f64 / 1_073_741_824.0,
            data_bytes as f64 / 1_073_741_824.0,
            twiddle_bytes as f64 / 1_048_576.0
        );

        // Upload entire table (zero-copy - no serialization!)
        let upload_start = Instant::now();
        let stream = dev_ctx.device.default_stream();
        let mut d_data = stream.memcpy_htod(raw_data_slice)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let upload_time = upload_start.elapsed();

        // Launch strided INTT kernel
        let kernel_start = Instant::now();
        let stride = num_columns as u64;

        {
            let mut d_data_ptr = d_data.as_ptr();
            let d_twiddles_ptr = d_twiddles.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_mut_ptr(&mut d_data_ptr)
                .push_value(num_rows as u64)
                .push_value(stride)
                .push_ptr(&d_twiddles_ptr)
                .push_value(log2_len);

            dev_ctx.intt_bfield_strided_fn.launch(
                (num_columns as u32, 1, 1),
                (super::super::get_gpu_ntt_block_size(), 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
        }
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download results (zero-copy - no deserialization!)
        let download_start = Instant::now();
        stream.memcpy_dtoh(&d_data, raw_data_slice)?;
        let download_time = download_start.elapsed();

        let postproc_start = Instant::now();
        let postproc_time = postproc_start.elapsed();

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Table Operation] total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms, postproc={:.1}ms)",
            total_time.as_secs_f64() * 1000.0,
            prep_time.as_secs_f64() * 1000.0,
            upload_time.as_secs_f64() * 1000.0,
            kernel_time.as_secs_f64() * 1000.0,
            download_time.as_secs_f64() * 1000.0,
            postproc_time.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// Execute INTT on a row-major table using strided GPU kernels
    /// For XFieldElement tables
    pub(crate) fn execute_intt_table_xfield(
        &self,
        table_data: &mut [XFieldElement],
        num_rows: usize,
        num_columns: usize,
        twiddle_factors: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let operation_start = Instant::now();

        assert!(num_rows.is_power_of_two(), "Table rows must be power of 2");

        let log2_len = num_rows.trailing_zeros();
        let dev_ctx = self.select_device();

        eprintln!(
            "\n[GPU Table XField INTT - {}] Processing table: {} rows × {} columns (row-major, strided access)",
            phase_name,
            num_rows,
            num_columns
        );

        // ZERO-COPY: Cast XFieldElement slice directly to u64 slice
        let prep_start = Instant::now();
        let raw_data_slice: &mut [u64] = unsafe {
            std::slice::from_raw_parts_mut(
                table_data.as_mut_ptr() as *mut u64,
                table_data.len() * 3
            )
        };
        let raw_twiddles = Self::extract_twiddle_roots(twiddle_factors);
        let prep_time = prep_start.elapsed();

        // Calculate memory requirements
        let data_bytes = raw_data_slice.len() * 8;
        let twiddle_bytes = raw_twiddles.len() * 8;
        let total_bytes = data_bytes + twiddle_bytes;
        eprintln!(
            "  GPU Memory Required: {:.2} GB (data: {:.2} GB, twiddles: {:.2} MB)",
            total_bytes as f64 / 1_073_741_824.0,
            data_bytes as f64 / 1_073_741_824.0,
            twiddle_bytes as f64 / 1_048_576.0
        );

        // Upload entire table (zero-copy - no serialization!)
        let upload_start = Instant::now();
        let stream = dev_ctx.device.default_stream();
        let mut d_data = stream.memcpy_htod(raw_data_slice)?;
        let d_twiddles = stream.memcpy_htod(&raw_twiddles)?;
        let upload_time = upload_start.elapsed();

        // Launch strided INTT kernel
        let kernel_start = Instant::now();
        let stride = (num_columns * 3) as u64;

        {
            let mut d_data_ptr = d_data.as_ptr();
            let d_twiddles_ptr = d_twiddles.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_mut_ptr(&mut d_data_ptr)
                .push_value(num_rows as u64)
                .push_value(stride)
                .push_ptr(&d_twiddles_ptr)
                .push_value(log2_len);

            dev_ctx.intt_xfield_strided_fn.launch(
                (num_columns as u32, 1, 1),
                (super::super::get_gpu_ntt_block_size(), 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
        }
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download results (zero-copy - no deserialization!)
        let download_start = Instant::now();
        stream.memcpy_dtoh(&d_data, raw_data_slice)?;
        let download_time = download_start.elapsed();

        let postproc_start = Instant::now();
        let postproc_time = postproc_start.elapsed();

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Table Operation] total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms, postproc={:.1}ms)",
            total_time.as_secs_f64() * 1000.0,
            prep_time.as_secs_f64() * 1000.0,
            upload_time.as_secs_f64() * 1000.0,
            kernel_time.as_secs_f64() * 1000.0,
            download_time.as_secs_f64() * 1000.0,
            postproc_time.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// Extract specific rows from GPU-resident BField table
    /// This avoids downloading the full LDE table when only specific rows are needed
    /// (e.g., for FRI proof revelation where ~80 rows out of millions are needed)
    ///
    /// # Arguments
    /// * `gpu_buffer` - GPU-resident table buffer (from GpuLdeBuffer)
    /// * `row_indices` - Which rows to extract
    /// * `num_columns` - Number of columns in the table
    ///
    /// # Returns
    /// Vec of rows, where each row is a Vec<BFieldElement>
    pub fn extract_rows_from_gpu_bfield(
        &self,
        gpu_buffer: &DeviceBuffer,
        row_indices: &[usize],
        num_columns: usize,
    ) -> Result<Vec<Vec<BFieldElement>>, Box<dyn std::error::Error>> {
        let operation_start = Instant::now();
        let dev_ctx = self.select_device();
        let stream = dev_ctx.device.default_stream();

        let num_rows_to_extract = row_indices.len();
        let total_output_elements = num_rows_to_extract * num_columns;

        eprintln!(
            "\n[GPU Row Extraction BField] Extracting {} rows × {} columns = {} elements",
            num_rows_to_extract, num_columns, total_output_elements
        );

        // Upload row indices to GPU
        let upload_start = Instant::now();
        let row_indices_u32: Vec<u32> = row_indices.iter().map(|&idx| idx as u32).collect();
        let d_row_indices = stream.memcpy_htod(&row_indices_u32)?;
        let upload_time = upload_start.elapsed();

        // Allocate output buffer on GPU
        let mut output_host = vec![0u64; total_output_elements];
        let d_output = stream.memcpy_htod(&output_host)?;

        // Launch extraction kernel
        let kernel_start = Instant::now();
        const THREADS_PER_BLOCK: u32 = 256;
        let num_blocks = (total_output_elements as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        {
            let gpu_buffer_ptr = gpu_buffer.as_ptr();
            let d_row_indices_ptr = d_row_indices.as_ptr();
            let mut d_output_ptr = d_output.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_ptr(&gpu_buffer_ptr)
                .push_ptr(&d_row_indices_ptr)
                .push_mut_ptr(&mut d_output_ptr)
                .push_value(num_rows_to_extract as u32)
                .push_value(num_columns as u32);

            dev_ctx.extract_rows_bfield_fn.launch(
                (num_blocks, 1, 1),
                (THREADS_PER_BLOCK, 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
        }
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download extracted rows
        let download_start = Instant::now();
        stream.memcpy_dtoh(&d_output, &mut output_host)?;
        let download_time = download_start.elapsed();

        // Convert to Vec<Vec<BFieldElement>>
        let postproc_start = Instant::now();
        let rows: Vec<Vec<BFieldElement>> = (0..num_rows_to_extract)
            .map(|row_idx| {
                let start = row_idx * num_columns;
                let end = start + num_columns;
                output_host[start..end]
                    .iter()
                    .map(|&val| BFieldElement::from_raw_u64(val))
                    .collect()
            })
            .collect();
        let postproc_time = postproc_start.elapsed();

        let total_time = operation_start.elapsed();
        let data_kb = (total_output_elements * 8) as f64 / 1024.0;

        eprintln!(
            "  [Row Extraction] total={:.1}ms (upload={:.1}ms, kernel={:.1}ms, download={:.1}ms, postproc={:.1}ms, {:.1} KB)",
            total_time.as_secs_f64() * 1000.0,
            upload_time.as_secs_f64() * 1000.0,
            kernel_time.as_secs_f64() * 1000.0,
            download_time.as_secs_f64() * 1000.0,
            postproc_time.as_secs_f64() * 1000.0,
            data_kb
        );

        Ok(rows)
    }

    /// Extract specific rows from GPU-resident XField table
    /// This avoids downloading the full LDE table when only specific rows are needed
    ///
    /// # Arguments
    /// * `gpu_buffer` - GPU-resident table buffer (from GpuLdeBuffer)
    /// * `row_indices` - Which rows to extract
    /// * `num_columns` - Number of XField columns in the table
    ///
    /// # Returns
    /// Vec of rows, where each row is a Vec<XFieldElement>
    pub fn extract_rows_from_gpu_xfield(
        &self,
        gpu_buffer: &DeviceBuffer,
        row_indices: &[usize],
        num_columns: usize,
    ) -> Result<Vec<Vec<XFieldElement>>, Box<dyn std::error::Error>> {
        let operation_start = Instant::now();
        let dev_ctx = self.select_device();
        let stream = dev_ctx.device.default_stream();

        let num_rows_to_extract = row_indices.len();
        let total_output_xfields = num_rows_to_extract * num_columns;
        let total_output_u64s = total_output_xfields * 3; // XField = 3 u64s

        eprintln!(
            "\n[GPU Row Extraction XField] Extracting {} rows × {} columns = {} XField elements ({} u64s)",
            num_rows_to_extract, num_columns, total_output_xfields, total_output_u64s
        );

        // Upload row indices to GPU
        let upload_start = Instant::now();
        let row_indices_u32: Vec<u32> = row_indices.iter().map(|&idx| idx as u32).collect();
        let d_row_indices = stream.memcpy_htod(&row_indices_u32)?;
        let upload_time = upload_start.elapsed();

        // Allocate output buffer on GPU
        let mut output_host = vec![0u64; total_output_u64s];
        let d_output = stream.memcpy_htod(&output_host)?;

        // Launch extraction kernel
        let kernel_start = Instant::now();
        const THREADS_PER_BLOCK: u32 = 256;
        let num_blocks = (total_output_xfields as u32 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        {
            let gpu_buffer_ptr = gpu_buffer.as_ptr();
            let d_row_indices_ptr = d_row_indices.as_ptr();
            let mut d_output_ptr = d_output.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_ptr(&gpu_buffer_ptr)
                .push_ptr(&d_row_indices_ptr)
                .push_mut_ptr(&mut d_output_ptr)
                .push_value(num_rows_to_extract as u32)
                .push_value(num_columns as u32);

            dev_ctx.extract_rows_xfield_fn.launch(
                (num_blocks, 1, 1),
                (THREADS_PER_BLOCK, 1, 1),
                0,
                &stream,
                kernel_args.as_mut_slice(),
            )?;
        }
        stream.synchronize()?;
        let kernel_time = kernel_start.elapsed();

        // Download extracted rows
        let download_start = Instant::now();
        stream.memcpy_dtoh(&d_output, &mut output_host)?;
        let download_time = download_start.elapsed();

        // Convert to Vec<Vec<XFieldElement>>
        let postproc_start = Instant::now();
        let rows: Vec<Vec<XFieldElement>> = (0..num_rows_to_extract)
            .map(|row_idx| {
                let start = row_idx * num_columns;
                let end = start + num_columns;
                (start..end)
                    .map(|xfield_idx| {
                        let u64_idx = xfield_idx * 3;
                        XFieldElement::new([
                            BFieldElement::from_raw_u64(output_host[u64_idx]),
                            BFieldElement::from_raw_u64(output_host[u64_idx + 1]),
                            BFieldElement::from_raw_u64(output_host[u64_idx + 2]),
                        ])
                    })
                    .collect()
            })
            .collect();
        let postproc_time = postproc_start.elapsed();

        let total_time = operation_start.elapsed();
        let data_kb = (total_output_u64s * 8) as f64 / 1024.0;

        eprintln!(
            "  [Row Extraction] total={:.1}ms (upload={:.1}ms, kernel={:.1}ms, download={:.1}ms, postproc={:.1}ms, {:.1} KB)",
            total_time.as_secs_f64() * 1000.0,
            upload_time.as_secs_f64() * 1000.0,
            kernel_time.as_secs_f64() * 1000.0,
            download_time.as_secs_f64() * 1000.0,
            postproc_time.as_secs_f64() * 1000.0,
            data_kb
        );

        Ok(rows)
    }
}
