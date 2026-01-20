// GPU-accelerated quotient evaluation module
// This module provides GPU-based evaluation of AIR constraints for quotient computation

use crate::arithmetic_domain::ArithmeticDomain;
use crate::challenges::Challenges;
use crate::table::master_table::MasterAuxTable;
use crate::table::auxiliary_table::Evaluable;
use ndarray::ArrayView2;
use twenty_first::prelude::*;
use std::fs;

#[cfg(feature = "gpu")]
use twenty_first::math::ntt::gpu_ntt::cuda_driver::{CudaDevice, CudaFunction, CudaStream, DeviceBuffer, KernelArgs};
#[cfg(feature = "gpu")]
use std::sync::Arc;

/// GPU context for quotient evaluation
#[cfg(feature = "gpu")]
pub struct GpuQuotientContext {
    device_0: CudaDevice,
    device_1: Option<CudaDevice>,
    evaluate_quotient_fn_0: CudaFunction,
    evaluate_quotient_fn_1: Option<CudaFunction>,
    use_dual_gpu: bool,
}

fn read_ptx(path:&str) -> String{
    let res = fs::read_to_string(path);
    match res {
        Ok(s)=>{
            s
        },
        Err(err)=>{
            panic!("Error reading from {} -> {}",path,err);
        }
    }
}

#[cfg(feature = "gpu")]
impl GpuQuotientContext {
    /// Initialize GPU context and load the quotient evaluation kernel
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        // Initialize first GPU
        let device_0 = CudaDevice::new(0)?;

        // Load the CUBIN module on GPU 0
        let cubin_data = std::fs::read("built_kernels/quotient_evaluation.cubin")?;
        let module_0 = device_0.load_module(&cubin_data)?;

        // Get the kernel function for GPU 0
        let evaluate_quotient_fn_0 = module_0.get_function("evaluate_quotient_kernel")?;

        // Try to initialize second GPU for dual-GPU mode
        let (device_1, evaluate_quotient_fn_1, use_dual_gpu) = match CudaDevice::new(1) {
            Ok(dev_1) => {
                eprintln!("[GPU Quotient Init] Found 2 GPUs - enabling dual-GPU mode");

                // Load kernel on GPU 1
                let module_1 = dev_1.load_module(&cubin_data)?;
                let fn_1 = module_1.get_function("evaluate_quotient_kernel")?;

                // Enable P2P access between GPUs (if not already enabled)
                // Note: This may have already been done by the NTT initialization
                match device_0.enable_peer_access(&dev_1) {
                    Ok(_) => eprintln!("[GPU Quotient Init] P2P enabled: GPU 0 -> GPU 1"),
                    Err(e) => eprintln!("[GPU Quotient Init] P2P already enabled or not needed: {}", e),
                }
                match dev_1.enable_peer_access(&device_0) {
                    Ok(_) => eprintln!("[GPU Quotient Init] P2P enabled: GPU 1 -> GPU 0"),
                    Err(e) => eprintln!("[GPU Quotient Init] P2P already enabled or not needed: {}", e),
                }

                (Some(dev_1), Some(fn_1), true)
            }
            Err(e) => {
                eprintln!("[GPU Quotient Init] Only 1 GPU available - using single-GPU mode: {}", e);
                (None, None, false)
            }
        };

        Ok(Self {
            device_0,
            device_1,
            evaluate_quotient_fn_0,
            evaluate_quotient_fn_1,
            use_dual_gpu,
        })
    }

    /// Evaluate quotients on GPU in a single batch (requires JIT mode for small domain)
    /// This replaces the CPU parallel computation in all_quotients_combined
    ///
    /// # GPU-Resident Optimization
    /// If `gpu_main_buffer` and `gpu_aux_buffer` are provided, they will be used directly,
    /// avoiding wasteful GPU→RAM→GPU transfers. Otherwise, falls back to uploading from CPU.
    pub fn evaluate_quotients_gpu(
        &self,
        quotient_domain_master_main_table: ArrayView2<BFieldElement>,
        quotient_domain_master_aux_table: ArrayView2<XFieldElement>,
        trace_domain: ArithmeticDomain,
        quotient_domain: ArithmeticDomain,
        challenges: &Challenges,
        quotient_weights: &[XFieldElement],
        gpu_main_buffer: Option<&DeviceBuffer>,
        gpu_aux_buffer: Option<&DeviceBuffer>,
    ) -> Result<Vec<XFieldElement>, Box<dyn std::error::Error>> {
        use std::time::Instant;

        let start = Instant::now();

        // Calculate section boundaries
        // credits to allfather team
        let init_section_end = MasterAuxTable::NUM_INITIAL_CONSTRAINTS;
        let cons_section_end = init_section_end + MasterAuxTable::NUM_CONSISTENCY_CONSTRAINTS;
        let tran_section_end = cons_section_end + MasterAuxTable::NUM_TRANSITION_CONSTRAINTS;
        let num_total_constraints = MasterAuxTable::NUM_CONSTRAINTS;

        // Calculate unit_distance
        let unit_distance = quotient_domain.len() / trace_domain.len();

        let num_rows = quotient_domain.len();
        let num_main_cols = quotient_domain_master_main_table.ncols();
        let num_aux_cols = quotient_domain_master_aux_table.ncols();

        // With JIT mode (no LDE caching), the quotient domain is much smaller (262K rows instead of 2M)
        // We can process all rows in one batch without chunking!
        // Memory: 262K rows × (379 + 88*3) elements = ~1.3GB - fits easily on RTX 3080

        eprintln!("\n[GPU Quotient] Evaluating {} rows in single batch", num_rows);
        eprintln!("  Main cols: {}, Aux cols: {}", num_main_cols, num_aux_cols);
        eprintln!("  Constraints: init={}, cons={}, tran={}, term={}",
                  MasterAuxTable::NUM_INITIAL_CONSTRAINTS,
                  MasterAuxTable::NUM_CONSISTENCY_CONSTRAINTS,
                  MasterAuxTable::NUM_TRANSITION_CONSTRAINTS,
                  MasterAuxTable::NUM_TERMINAL_CONSTRAINTS);

        // Prepare static data (challenges, weights - same for all chunks)
        let prep_start = Instant::now();

        // Convert challenges to raw u64 (3 u64s per XFieldElement)
        let mut challenges_raw = Vec::with_capacity(challenges.challenges.len() * 3);
        for challenge in &challenges.challenges {
            challenges_raw.push(challenge.coefficients[0].raw_u64());
            challenges_raw.push(challenge.coefficients[1].raw_u64());
            challenges_raw.push(challenge.coefficients[2].raw_u64());
        }

        // Compute zerofier inverses (CPU - fast enough, not a bottleneck)
        // Do this once for all rows
        let initial_zerofier_inv = super::master_table::initial_quotient_zerofier_inverse(quotient_domain);
        let consistency_zerofier_inv = super::master_table::consistency_quotient_zerofier_inverse(trace_domain, quotient_domain);
        let transition_zerofier_inv = super::master_table::transition_quotient_zerofier_inverse(trace_domain, quotient_domain);
        let terminal_zerofier_inv = super::master_table::terminal_quotient_zerofier_inverse(trace_domain, quotient_domain);

        // Convert quotient weights to raw u64
        let mut weights_raw = Vec::with_capacity(quotient_weights.len() * 3);
        for weight in quotient_weights {
            weights_raw.push(weight.coefficients[0].raw_u64());
            weights_raw.push(weight.coefficients[1].raw_u64());
            weights_raw.push(weight.coefficients[2].raw_u64());
        }

        let prep_time = prep_start.elapsed();

        // Upload static data once (challenges, weights - shared across all chunks)
        let upload_start = Instant::now();
        let stream_0 = self.device_0.default_stream();

        // Determine if we're using dual-GPU mode
        let use_dual_gpu = self.use_dual_gpu;
        let mid_row = if use_dual_gpu { num_rows / 2 } else { num_rows };

        if use_dual_gpu {
            eprintln!("  [Dual-GPU Quotient] Splitting {} rows across 2 GPUs", num_rows);
            eprintln!("    GPU 0: rows [0, {})", mid_row);
            eprintln!("    GPU 1: rows [{}, {})", mid_row, num_rows);
        }

        // Upload challenges and weights to GPU 0
        let d_challenges_0 = stream_0.memcpy_htod(&challenges_raw)?;
        let d_weights_0 = stream_0.memcpy_htod(&weights_raw)?;

        let static_upload_time = upload_start.elapsed();

        // Check if we have GPU-resident buffers (fast path)
        let use_gpu_resident = gpu_main_buffer.is_some() && gpu_aux_buffer.is_some();

        if use_gpu_resident {
            eprintln!("  [GPU-Resident Fast Path] Using cached GPU buffers - SKIPPING wasteful upload!");
        } else {
            eprintln!("  [GPU Upload Path] No cached buffers available - uploading from CPU");
        }

        // Prepare data for GPU upload (only if not using GPU-resident path)
        let data_prep_start = Instant::now();

        let (main_raw, aux_raw) = if !use_gpu_resident {
            eprintln!("  [GPU Debug] Preparing data for upload:");
            eprintln!("    Table dims: {} rows × {} main cols × {} aux cols", num_rows, num_main_cols, num_aux_cols);
            eprintln!("    unit_distance: {}", unit_distance);
            eprintln!("    Main table shape from ArrayView: {:?}", quotient_domain_master_main_table.dim());
            eprintln!("    Aux table shape from ArrayView: {:?}", quotient_domain_master_aux_table.dim());

            // Convert main table to raw u64
            let main_raw: Vec<u64> = if let Some(slice) = quotient_domain_master_main_table.as_slice() {
                eprintln!("    Using fast path for main table (contiguous)");
                slice.iter().map(|x| x.raw_u64()).collect()
            } else {
                eprintln!("    Using slow path for main table (non-contiguous)");
                let mut main_raw = Vec::with_capacity(num_rows * num_main_cols);
                for row_idx in 0..num_rows {
                    for col_idx in 0..num_main_cols {
                        main_raw.push(quotient_domain_master_main_table[[row_idx, col_idx]].raw_u64());
                    }
                }
                main_raw
            };

            // Convert aux table to raw u64 (3 u64s per XFieldElement)
            let aux_raw: Vec<u64> = if let Some(slice) = quotient_domain_master_aux_table.as_slice() {
                eprintln!("    Using fast path for aux table (contiguous)");
                let mut aux_raw = Vec::with_capacity(num_rows * num_aux_cols * 3);
                for xfe in slice {
                    aux_raw.push(xfe.coefficients[0].raw_u64());
                    aux_raw.push(xfe.coefficients[1].raw_u64());
                    aux_raw.push(xfe.coefficients[2].raw_u64());
                }
                aux_raw
            } else {
                eprintln!("    Using slow path for aux table (non-contiguous)");
                let mut aux_raw = Vec::with_capacity(num_rows * num_aux_cols * 3);
                for row_idx in 0..num_rows {
                    for col_idx in 0..num_aux_cols {
                        let xfe = quotient_domain_master_aux_table[[row_idx, col_idx]];
                        aux_raw.push(xfe.coefficients[0].raw_u64());
                        aux_raw.push(xfe.coefficients[1].raw_u64());
                        aux_raw.push(xfe.coefficients[2].raw_u64());
                    }
                }
                aux_raw
            };

            (main_raw, aux_raw)
        } else {
            // GPU-resident path: no conversion needed
            (Vec::new(), Vec::new())
        };

        // Convert zerofiers to raw u64 (BFieldElement - no padding needed, kernel lifts to XFieldElement)
        // This optimization saves 2/3 of zerofier upload bandwidth (64 MB instead of 192 MB in cached mode)
        let init_zerofier_raw: Vec<u64> = initial_zerofier_inv.iter().map(|x| x.raw_u64()).collect();
        let cons_zerofier_raw: Vec<u64> = consistency_zerofier_inv.iter().map(|x| x.raw_u64()).collect();
        let tran_zerofier_raw: Vec<u64> = transition_zerofier_inv.iter().map(|x| x.raw_u64()).collect();
        let term_zerofier_raw: Vec<u64> = terminal_zerofier_inv.iter().map(|x| x.raw_u64()).collect();

        let data_prep_time = data_prep_start.elapsed();

        if !use_gpu_resident {
            eprintln!("    Prepared {} main elements, {} aux elements", main_raw.len(), aux_raw.len());
            eprintln!("    Expected: {} main, {} aux", num_rows * num_main_cols, num_rows * num_aux_cols * 3);
            if !main_raw.is_empty() {
                eprintln!("    main_raw[0] = {}, main_raw[1] = {}", main_raw[0], main_raw[1]);
            }
            eprintln!("    Source table [0,0] = {:?}", quotient_domain_master_main_table[[0,0]].raw_u64());
        }

        // Upload data to GPU (or use cached buffers)
        let upload_start = Instant::now();

        // For the upload path, we need owned buffers that live long enough
        let d_main_owned;
        let d_aux_owned;

        let (d_main_0, d_aux_0): (&DeviceBuffer, &DeviceBuffer) = if use_gpu_resident {
            // Fast path: use GPU-resident buffers directly (no upload!)
            let main_buf = gpu_main_buffer.unwrap();
            let aux_buf = gpu_aux_buffer.unwrap();
            eprintln!("    [OPTIMIZATION] Using GPU-resident buffers ({:.2} MB + {:.2} MB) - saved ~{:.2} GB upload!",
                     (num_rows * num_main_cols * 8) as f64 / 1_048_576.0,
                     (num_rows * num_aux_cols * 24) as f64 / 1_048_576.0,
                     ((num_rows * num_main_cols * 8) + (num_rows * num_aux_cols * 24)) as f64 / 1_073_741_824.0);
            (main_buf, aux_buf)
        } else {
            // Slow path: upload from CPU
            d_main_owned = stream_0.memcpy_htod(&main_raw)?;
            d_aux_owned = stream_0.memcpy_htod(&aux_raw)?;
            (&d_main_owned, &d_aux_owned)
        };

        // Upload zerofiers to GPU 0
        let d_init_zerofier_0 = stream_0.memcpy_htod(&init_zerofier_raw)?;
        let d_cons_zerofier_0 = stream_0.memcpy_htod(&cons_zerofier_raw)?;
        let d_tran_zerofier_0 = stream_0.memcpy_htod(&tran_zerofier_raw)?;
        let d_term_zerofier_0 = stream_0.memcpy_htod(&term_zerofier_raw)?;

        // Allocate output buffer for GPU 0
        let output_size_0 = if use_dual_gpu { mid_row * 3 } else { num_rows * 3 };
        let d_output_0 = stream_0.alloc::<u64>(output_size_0)?;

        // Dual-GPU setup: P2P copy buffers and upload shared data to GPU 1
        let (device_1, stream_1, d_main_1, d_aux_1, d_challenges_1, d_weights_1,
             d_init_zerofier_1, d_cons_zerofier_1, d_tran_zerofier_1, d_term_zerofier_1, d_output_1) =
        if use_dual_gpu {
            let dev_1 = self.device_1.as_ref().unwrap();
            let stream_1 = dev_1.default_stream();

            eprintln!("  [Dual-GPU Setup] P2P copying GPU-resident buffers GPU 0 → GPU 1...");
            let p2p_start = Instant::now();

            // P2P copy main and aux tables from GPU 0 to GPU 1
            let d_main_1 = if use_gpu_resident {
                let buf = stream_1.alloc::<u64>(num_rows * num_main_cols)?;
                stream_0.memcpy_peer_async(&buf, &dev_1.context, d_main_0, &self.device_0.context,
                                          num_rows * num_main_cols * 8)?;
                buf
            } else {
                stream_1.memcpy_htod(&main_raw)?
            };

            let d_aux_1 = if use_gpu_resident {
                let buf = stream_1.alloc::<u64>(num_rows * num_aux_cols * 3)?;
                stream_0.memcpy_peer_async(&buf, &dev_1.context, d_aux_0, &self.device_0.context,
                                          num_rows * num_aux_cols * 3 * 8)?;
                buf
            } else {
                stream_1.memcpy_htod(&aux_raw)?
            };

            stream_0.synchronize()?;
            let p2p_time = p2p_start.elapsed();
            let data_size_gb = ((num_rows * num_main_cols * 8) + (num_rows * num_aux_cols * 3 * 8)) as f64 / 1_000_000_000.0;
            eprintln!("  [Dual-GPU Setup] P2P copy complete in {:.1}ms ({:.2} GB @ {:.1} GB/s)",
                     p2p_time.as_secs_f64() * 1000.0, data_size_gb, data_size_gb / p2p_time.as_secs_f64());

            // Upload shared data to GPU 1
            let d_challenges_1 = stream_1.memcpy_htod(&challenges_raw)?;
            let d_weights_1 = stream_1.memcpy_htod(&weights_raw)?;
            let d_init_zerofier_1 = stream_1.memcpy_htod(&init_zerofier_raw)?;
            let d_cons_zerofier_1 = stream_1.memcpy_htod(&cons_zerofier_raw)?;
            let d_tran_zerofier_1 = stream_1.memcpy_htod(&tran_zerofier_raw)?;
            let d_term_zerofier_1 = stream_1.memcpy_htod(&term_zerofier_raw)?;

            // Allocate output buffer for GPU 1
            let output_size_1 = (num_rows - mid_row) * 3;
            let d_output_1 = stream_1.alloc::<u64>(output_size_1)?;

            (Some(dev_1), Some(stream_1), Some(d_main_1), Some(d_aux_1), Some(d_challenges_1), Some(d_weights_1),
             Some(d_init_zerofier_1), Some(d_cons_zerofier_1), Some(d_tran_zerofier_1), Some(d_term_zerofier_1), Some(d_output_1))
        } else {
            (None, None, None, None, None, None, None, None, None, None, None)
        };

        let upload_time = upload_start.elapsed();
        eprintln!("    Upload complete. Launching kernel with {} blocks × {} threads = {} total threads",
                  (num_rows as u32 + 255) / 256, 256, ((num_rows as u32 + 255) / 256) * 256);

        // Launch kernel on GPU 0
        let kernel_start_0 = Instant::now();

        // Use larger block size for B200 (Blackwell) - 512 threads gives better occupancy
        // Can be tuned via environment variable
        let threads_per_block = std::env::var("GPU_QUOTIENT_BLOCK_SIZE")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(64);  // Default 64 for modern GPUs

        let num_rows_0 = if use_dual_gpu { mid_row } else { num_rows };
        let num_blocks_0 = (num_rows_0 as u32 + threads_per_block - 1) / threads_per_block;

        eprintln!("  [GPU 0] Kernel config: {} blocks × {} threads = {} total (processing {} rows)",
                  num_blocks_0, threads_per_block, num_blocks_0 * threads_per_block, num_rows_0);

        // Build kernel arguments for GPU 0
        let mut d_main_ptr_0 = d_main_0.as_ptr();
        let mut d_aux_ptr_0 = d_aux_0.as_ptr();
        let mut d_challenges_ptr_0 = d_challenges_0.as_ptr();
        let mut d_init_zerofier_ptr_0 = d_init_zerofier_0.as_ptr();
        let mut d_cons_zerofier_ptr_0 = d_cons_zerofier_0.as_ptr();
        let mut d_tran_zerofier_ptr_0 = d_tran_zerofier_0.as_ptr();
        let mut d_term_zerofier_ptr_0 = d_term_zerofier_0.as_ptr();
        let mut d_weights_ptr_0 = d_weights_0.as_ptr();
        let mut d_output_ptr_0 = d_output_0.as_ptr();

        let mut args_0 = KernelArgs::new();
        args_0.push_mut_ptr(&mut d_main_ptr_0)
            .push_mut_ptr(&mut d_aux_ptr_0)
            .push_mut_ptr(&mut d_challenges_ptr_0)
            .push_value(challenges.challenges.len() as u32)
            .push_mut_ptr(&mut d_init_zerofier_ptr_0)
            .push_mut_ptr(&mut d_cons_zerofier_ptr_0)
            .push_mut_ptr(&mut d_tran_zerofier_ptr_0)
            .push_mut_ptr(&mut d_term_zerofier_ptr_0)
            .push_mut_ptr(&mut d_weights_ptr_0)
            .push_value(init_section_end as u32)
            .push_value(cons_section_end as u32)
            .push_value(tran_section_end as u32)
            .push_value(num_total_constraints as u32)
            .push_value(num_rows_0 as u32)          // num_rows_local
            .push_value(num_rows as u32)            // num_rows_total
            .push_value(0u32)                       // row_offset (GPU 0 starts at 0)
            .push_value(num_main_cols as u32)
            .push_value(num_aux_cols as u32)
            .push_value(unit_distance as u32)
            .push_mut_ptr(&mut d_output_ptr_0);

        self.evaluate_quotient_fn_0.launch(
            (num_blocks_0, 1, 1),
            (threads_per_block, 1, 1),
            0,
            &stream_0,
            args_0.as_mut_slice(),
        )?;

        // Launch kernel on GPU 1 (if dual-GPU)
        let kernel_start_1 = if use_dual_gpu {
            let start = Instant::now();
            let dev_1 = device_1.unwrap();
            let str_1 = stream_1.as_ref().unwrap();
            let d_main_1_ref = d_main_1.as_ref().unwrap();
            let d_aux_1_ref = d_aux_1.as_ref().unwrap();
            let d_challenges_1_ref = d_challenges_1.as_ref().unwrap();
            let d_weights_1_ref = d_weights_1.as_ref().unwrap();
            let d_init_zerofier_1_ref = d_init_zerofier_1.as_ref().unwrap();
            let d_cons_zerofier_1_ref = d_cons_zerofier_1.as_ref().unwrap();
            let d_tran_zerofier_1_ref = d_tran_zerofier_1.as_ref().unwrap();
            let d_term_zerofier_1_ref = d_term_zerofier_1.as_ref().unwrap();
            let d_output_1_ref = d_output_1.as_ref().unwrap();

            let num_rows_1 = num_rows - mid_row;
            let num_blocks_1 = (num_rows_1 as u32 + threads_per_block - 1) / threads_per_block;

            eprintln!("  [GPU 1] Kernel config: {} blocks × {} threads = {} total (processing {} rows)",
                      num_blocks_1, threads_per_block, num_blocks_1 * threads_per_block, num_rows_1);

            // Don't offset pointers - kernel will use row_offset parameter
            let mut d_main_ptr_1 = d_main_1_ref.as_ptr();
            let mut d_aux_ptr_1 = d_aux_1_ref.as_ptr();
            let mut d_challenges_ptr_1 = d_challenges_1_ref.as_ptr();
            let mut d_init_zerofier_ptr_1 = d_init_zerofier_1_ref.as_ptr();
            let mut d_cons_zerofier_ptr_1 = d_cons_zerofier_1_ref.as_ptr();
            let mut d_tran_zerofier_ptr_1 = d_tran_zerofier_1_ref.as_ptr();
            let mut d_term_zerofier_ptr_1 = d_term_zerofier_1_ref.as_ptr();
            let mut d_weights_ptr_1 = d_weights_1_ref.as_ptr();
            let mut d_output_ptr_1 = d_output_1_ref.as_ptr();

            let mut args_1 = KernelArgs::new();
            args_1.push_mut_ptr(&mut d_main_ptr_1)
                .push_mut_ptr(&mut d_aux_ptr_1)
                .push_mut_ptr(&mut d_challenges_ptr_1)
                .push_value(challenges.challenges.len() as u32)
                .push_mut_ptr(&mut d_init_zerofier_ptr_1)
                .push_mut_ptr(&mut d_cons_zerofier_ptr_1)
                .push_mut_ptr(&mut d_tran_zerofier_ptr_1)
                .push_mut_ptr(&mut d_term_zerofier_ptr_1)
                .push_mut_ptr(&mut d_weights_ptr_1)
                .push_value(init_section_end as u32)
                .push_value(cons_section_end as u32)
                .push_value(tran_section_end as u32)
                .push_value(num_total_constraints as u32)
                .push_value(num_rows_1 as u32)          // num_rows_local
                .push_value(num_rows as u32)            // num_rows_total
                .push_value(mid_row as u32)             // row_offset (GPU 1 starts at mid_row)
                .push_value(num_main_cols as u32)
                .push_value(num_aux_cols as u32)
                .push_value(unit_distance as u32)
                .push_mut_ptr(&mut d_output_ptr_1);

            self.evaluate_quotient_fn_1.as_ref().unwrap().launch(
                (num_blocks_1, 1, 1),
                (threads_per_block, 1, 1),
                0,
                str_1,
                args_1.as_mut_slice(),
            )?;

            Some(start)
        } else {
            None
        };

        // Wait for GPU 0 to complete
        stream_0.synchronize()?;
        let kernel_time_0 = kernel_start_0.elapsed();
        eprintln!("  [GPU 0] Kernel completed in {:.1}ms ({} rows)",
                  kernel_time_0.as_secs_f64() * 1000.0, num_rows_0);

        // Wait for GPU 1 to complete (if dual-GPU)
        let kernel_time_1 = if use_dual_gpu {
            let str_1 = stream_1.as_ref().unwrap();
            str_1.synchronize()?;
            let time = kernel_start_1.unwrap().elapsed();
            eprintln!("  [GPU 1] Kernel completed in {:.1}ms ({} rows)",
                      time.as_secs_f64() * 1000.0, num_rows - mid_row);
            Some(time)
        } else {
            None
        };

        // Download results from GPU 0
        let download_start = Instant::now();
        let mut output_host_0 = vec![0u64; output_size_0];
        stream_0.memcpy_dtoh(&d_output_0, &mut output_host_0)?;

        // Download results from GPU 1 (if dual-GPU)
        let output_host_1 = if use_dual_gpu {
            let str_1 = stream_1.as_ref().unwrap();
            let output_size_1 = (num_rows - mid_row) * 3;
            let mut output = vec![0u64; output_size_1];
            str_1.memcpy_dtoh(d_output_1.as_ref().unwrap(), &mut output)?;
            Some(output)
        } else {
            None
        };

        let download_time = download_start.elapsed();

        // Merge results
        let output_combined = if use_dual_gpu {
            let mut combined = output_host_0;
            combined.extend_from_slice(&output_host_1.unwrap());
            eprintln!("  [Dual-GPU] Merged results from both GPUs ({} total rows)", num_rows);
            combined
        } else {
            output_host_0
        };

        // Convert results to XFieldElement
        let mut quotient_codeword = Vec::with_capacity(num_rows);
        for i in 0..num_rows {
            let idx = i * 3;
            quotient_codeword.push(XFieldElement::new([
                BFieldElement::from_raw_u64(output_combined[idx]),
                BFieldElement::from_raw_u64(output_combined[idx + 1]),
                BFieldElement::from_raw_u64(output_combined[idx + 2]),
            ]));
        }

        // CPU Validation: Compare GPU vs CPU for first row
        // Skip validation when using GPU-resident path (main_raw/aux_raw are empty)
        if !use_gpu_resident {
            eprintln!("  [GPU Validation] Comparing GPU vs CPU for row 0:");
            let row_idx = 0;

            // Reconstruct row 0 from uploaded main_raw data (row-major layout)
            let mut current_main_vec: Vec<BFieldElement> = Vec::with_capacity(num_main_cols);
            for col in 0..num_main_cols {
                current_main_vec.push(BFieldElement::from_raw_u64(main_raw[row_idx * num_main_cols + col]));
            }
        let current_main = ndarray::ArrayView1::from(&current_main_vec);

        eprintln!("    Uploaded main_raw[0] = {}", main_raw[0]);
        eprintln!("    CPU reconstructed main_row[0] = {} (from main_raw[0])", current_main[0].raw_u64());
        eprintln!("    Should match: {}", main_raw[0] == current_main[0].raw_u64());

        // Reconstruct row 0 from uploaded aux_raw data
        let mut current_aux_vec: Vec<XFieldElement> = Vec::with_capacity(num_aux_cols);
        for col in 0..num_aux_cols {
            let base_idx = (row_idx * num_aux_cols + col) * 3;
            current_aux_vec.push(XFieldElement::new([
                BFieldElement::from_raw_u64(aux_raw[base_idx]),
                BFieldElement::from_raw_u64(aux_raw[base_idx + 1]),
                BFieldElement::from_raw_u64(aux_raw[base_idx + 2]),
            ]));
        }
        let current_aux = ndarray::ArrayView1::from(&current_aux_vec);

        // Reconstruct next row similarly
        let next_idx = unit_distance % num_rows;
        let mut next_main_vec: Vec<BFieldElement> = Vec::with_capacity(num_main_cols);
        for col in 0..num_main_cols {
            next_main_vec.push(BFieldElement::from_raw_u64(main_raw[next_idx * num_main_cols + col]));
        }
        let next_main = ndarray::ArrayView1::from(&next_main_vec);

        let mut next_aux_vec: Vec<XFieldElement> = Vec::with_capacity(num_aux_cols);
        for col in 0..num_aux_cols {
            let base_idx = (next_idx * num_aux_cols + col) * 3;
            next_aux_vec.push(XFieldElement::new([
                BFieldElement::from_raw_u64(aux_raw[base_idx]),
                BFieldElement::from_raw_u64(aux_raw[base_idx + 1]),
                BFieldElement::from_raw_u64(aux_raw[base_idx + 2]),
            ]));
        }
        let next_aux = ndarray::ArrayView1::from(&next_aux_vec);

        // Test lift() directly
        let test_lift = current_main[0].lift();
        eprintln!("    TEST: current_main[0] = {}", current_main[0].raw_u64());
        eprintln!("    TEST: current_main[0].lift() = {:?}", test_lift);
        eprintln!("    TEST: test_lift.coefficients[0].raw_u64() = {}", test_lift.coefficients[0].raw_u64());
        eprintln!("    TEST: test_lift.coefficients[1].raw_u64() = {}", test_lift.coefficients[1].raw_u64());
        eprintln!("    TEST: test_lift.coefficients[2].raw_u64() = {}", test_lift.coefficients[2].raw_u64());

        let cpu_init = MasterAuxTable::evaluate_initial_constraints(current_main, current_aux, challenges);
        let cpu_cons = MasterAuxTable::evaluate_consistency_constraints(current_main, current_aux, challenges);
        let cpu_tran = MasterAuxTable::evaluate_transition_constraints(current_main, current_aux, next_main, next_aux, challenges);
        let cpu_term = MasterAuxTable::evaluate_terminal_constraints(current_main, current_aux, challenges);

        eprintln!("    CPU: {} init, {} cons, {} tran, {} term constraints",
            cpu_init.len(), cpu_cons.len(), cpu_tran.len(), cpu_term.len());
        eprintln!("    CPU init[0] RAW = ({}, {}, {})",
            cpu_init[0].coefficients[0].raw_u64(),
            cpu_init[0].coefficients[1].raw_u64(),
            cpu_init[0].coefficients[2].raw_u64());
        eprintln!("    For first initial constraint, expected RAW: ({}, 0, 0)", current_main[0].raw_u64());
        eprintln!("    CPU cons[0] RAW = ({}, {}, {})",
            cpu_cons[0].coefficients[0].raw_u64(),
            cpu_cons[0].coefficients[1].raw_u64(),
            cpu_cons[0].coefficients[2].raw_u64());
        eprintln!("    CPU tran[0] = {:?}", cpu_tran[0]);
        eprintln!("    CPU term[0] = {:?}", cpu_term[0]);

        // Compute weighted quotient on CPU for row 0 (step by step for debugging)
        eprintln!("    CPU init_values[0] RAW = ({}, {}, {})",
            cpu_init[0].coefficients[0].raw_u64(),
            cpu_init[0].coefficients[1].raw_u64(),
            cpu_init[0].coefficients[2].raw_u64());
        eprintln!("    CPU init_weights[0] RAW = ({}, {}, {})",
            quotient_weights[0].coefficients[0].raw_u64(),
            quotient_weights[0].coefficients[1].raw_u64(),
            quotient_weights[0].coefficients[2].raw_u64());
        eprintln!("    CPU num_initial = {}", init_section_end);

        // Check first few and last few constraints to find where shift occurs
        eprintln!("    CPU: Total {} init constraints", init_section_end);
        eprintln!("    CPU: First 5 constraints:");
        for i in 0..5.min(init_section_end) {
            eprintln!("    CPU init[{}] RAW = ({}, {}, {})",
                i,
                cpu_init[i].coefficients[0].raw_u64(),
                cpu_init[i].coefficients[1].raw_u64(),
                cpu_init[i].coefficients[2].raw_u64());
        }
        eprintln!("    CPU: Last 7 constraints (74-80):");
        for i in 74..init_section_end.min(81) {
            eprintln!("    CPU init[{}] RAW = ({}, {}, {})",
                i,
                cpu_init[i].coefficients[0].raw_u64(),
                cpu_init[i].coefficients[1].raw_u64(),
                cpu_init[i].coefficients[2].raw_u64());
        }

        let cpu_init_inner_product: XFieldElement = cpu_init.iter().zip(&quotient_weights[..init_section_end])
            .map(|(c, w)| *c * *w).sum();
        let cpu_init_contrib = cpu_init_inner_product * initial_zerofier_inv[row_idx];

        eprintln!("    CPU init_inner_product RAW = ({}, {}, {})",
            cpu_init_inner_product.coefficients[0].raw_u64(),
            cpu_init_inner_product.coefficients[1].raw_u64(),
            cpu_init_inner_product.coefficients[2].raw_u64());
        eprintln!("    CPU init_zerofier RAW = ({}, {}, {})",
            initial_zerofier_inv[row_idx].raw_u64(), 0, 0);
        eprintln!("    CPU init_contrib RAW = ({}, {}, {})",
            cpu_init_contrib.coefficients[0].raw_u64(),
            cpu_init_contrib.coefficients[1].raw_u64(),
            cpu_init_contrib.coefficients[2].raw_u64());

        let cpu_cons_inner_product: XFieldElement = cpu_cons.iter().zip(&quotient_weights[init_section_end..cons_section_end])
            .map(|(c, w)| *c * *w).sum();
        let cpu_cons_contrib = cpu_cons_inner_product * consistency_zerofier_inv[row_idx];

        let cpu_tran_inner_product: XFieldElement = cpu_tran.iter().zip(&quotient_weights[cons_section_end..tran_section_end])
            .map(|(c, w)| *c * *w).sum();
        let cpu_tran_contrib = cpu_tran_inner_product * transition_zerofier_inv[row_idx];

        let cpu_term_inner_product: XFieldElement = cpu_term.iter().zip(&quotient_weights[tran_section_end..])
            .map(|(c, w)| *c * *w).sum();
        let cpu_term_contrib = cpu_term_inner_product * terminal_zerofier_inv[row_idx];

        let cpu_quotient = cpu_init_contrib + cpu_cons_contrib + cpu_tran_contrib + cpu_term_contrib;

        eprintln!("    CPU quotient[0] = {:?}", cpu_quotient);
        eprintln!("    GPU quotient[0] = {:?}", quotient_codeword[0]);

            if cpu_quotient == quotient_codeword[0] {
                eprintln!("    ✓ MATCH! GPU is correct");
            } else {
                eprintln!("    ✗ MISMATCH! GPU is incorrect");
                eprintln!("    Difference: {:?}", quotient_codeword[0] - cpu_quotient);
            }
        } else {
            eprintln!("  [GPU Validation] Skipped validation (using GPU-resident buffers)");
        }

        let total_time = start.elapsed();

        eprintln!("  [GPU Quotient Timing Summary]");
        eprintln!("    Challenges Prep:     {:.1}ms", prep_time.as_secs_f64() * 1000.0);
        eprintln!("    Challenges Upload:   {:.1}ms", static_upload_time.as_secs_f64() * 1000.0);
        eprintln!("    Data Preparation:    {:.1}ms", data_prep_time.as_secs_f64() * 1000.0);
        eprintln!("    Data Upload:         {:.1}ms", upload_time.as_secs_f64() * 1000.0);
        if use_dual_gpu {
            eprintln!("    GPU 0 Kernel:        {:.1}ms", kernel_time_0.as_secs_f64() * 1000.0);
            eprintln!("    GPU 1 Kernel:        {:.1}ms", kernel_time_1.unwrap().as_secs_f64() * 1000.0);
        } else {
            eprintln!("    Kernel Execution:    {:.1}ms", kernel_time_0.as_secs_f64() * 1000.0);
        }
        eprintln!("    Result Download:     {:.1}ms", download_time.as_secs_f64() * 1000.0);
        eprintln!("    TOTAL:               {:.1}ms", total_time.as_secs_f64() * 1000.0);

        Ok(quotient_codeword)
    }
}
