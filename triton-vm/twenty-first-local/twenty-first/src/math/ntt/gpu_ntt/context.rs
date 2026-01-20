use super::types::{GpuDeviceContext, GpuNttContext};
use super::cuda_driver::CudaDevice;
use std::fs;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

impl GpuNttContext {
    /// Initialize GPU context and load CUDA kernels on all available GPUs
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        let start_initialization = Instant::now();

        let mut devices = Vec::new();
        let mut device_id = 0;

        // Read CUBIN module as bytes
        let cubin_bytes = fs::read("built_kernels/big_package.cubin")
            .map_err(|e| format!("Failed to read CUBIN file: {}", e))?;

        // Try to initialize all available GPUs (limit to 2 for now)
        loop {
        match CudaDevice::new(device_id) {
            Ok(device) => {
                // Try to load CUBIN module on this device
                match device.load_module(&cubin_bytes) {
                    Ok(all_module) => {
                        // Get function handles
                        let ntt_bfield_fn = all_module.get_function("ntt_bfield")?;
                        let intt_bfield_fn = all_module.get_function("intt_bfield")?;
                        let ntt_xfield_fn = all_module.get_function("ntt_xfield")?;
                        let intt_xfield_fn = all_module.get_function("intt_xfield")?;
                        let coset_scale_bfield_fn =
                            all_module.get_function("coset_scale_bfield_fast")?;
                        let ntt_bfield_fused_coset_fn =
                            all_module.get_function("ntt_bfield_fused_coset")?;
                        let ntt_xfield_fused_coset_fn =
                            all_module.get_function("ntt_xfield_fused_coset")?;

                        // Load strided kernels for row-major table data
                        let bfield_poly_fill_table_fn =
                            all_module.get_function("poly_fill_table_bfield")?;
                        let ntt_bfield_init_omegas_fn =
                            all_module.get_function("ntt_bfield_init_omegas")?;
                        let ntt_bfield_extract_fn =
                            all_module.get_function("ntt_bfield_extract")?;
                        let ntt_bfield_fused_coset_single_fn =
                            all_module.get_function("ntt_bfield_fused_coset_single")?;
                        let ntt_bfield_restore_fn =
                            all_module.get_function("ntt_bfield_restore")?;

                        let ntt_bfield_fused_coset_strided_fn =
                            all_module.get_function("ntt_bfield_fused_coset_strided")?;

                        let xfield_poly_fill_table_fn =
                            all_module.get_function("poly_fill_table_xfield")?;
                        let ntt_xfield_init_omegas_fn =
                            all_module.get_function("ntt_xfield_init_omegas")?;
                        let ntt_xfield_extract_fn =
                            all_module.get_function("ntt_xfield_extract")?;
                        let ntt_xfield_fused_coset_single_fn =
                            all_module.get_function("ntt_xfield_fused_coset_single")?;
                        let ntt_xfield_fused_coset_single_interpolate_fn =
                            all_module.get_function("ntt_xfield_fused_coset_single_interpolate")?;
                        let ntt_xfield_restore_fn =
                            all_module.get_function("ntt_xfield_restore")?;

                        let ntt_xfield_fused_coset_strided_fn =
                            all_module.get_function("ntt_xfield_fused_coset_strided")?;
                        let intt_bfield_strided_fn =
                            all_module.get_function("intt_bfield_strided")?;
                        let intt_xfield_strided_fn =
                            all_module.get_function("intt_xfield_strided")?;

                        // Load fused INTT + unscaling kernels
                        let intt_bfield_fused_unscale_fn =
                            all_module.get_function("intt_bfield_fused_unscale")?;
                        let intt_xfield_fused_unscale_fn =
                            all_module.get_function("intt_xfield_fused_unscale")?;

                        // Load fused INTT + unscaling + randomizer kernels
                        let intt_bfield_fused_unscale_randomize_fn =
                            all_module.get_function("intt_bfield_fused_unscale_randomize")?;
                        let intt_xfield_fused_unscale_randomize_fn =
                            all_module.get_function("intt_xfield_fused_unscale_randomize")?;

                        let running_eval_scan_fn =
                            all_module.get_function("running_evaluation_scan")?;
                        let log_derivative_scan_fn =
                            all_module.get_function("log_derivative_scan")?;
                        let running_eval_scan_parallel_fn =
                            all_module.get_function("running_evaluation_scan_parallel")?;
                        let log_derivative_scan_parallel_fn =
                            all_module.get_function("log_derivative_scan_parallel")?;
                        let batch_inversion_fn =
                            all_module.get_function("batch_inversion")?;

                        let hash_rows_bfield_fn =
                            all_module.get_function("tip5_hash_batch")?;
                        let hash_rows_xfield_fn =
                            all_module.get_function("tip5_hash_batch")?;

                        let extract_rows_bfield_fn =
                            all_module.get_function("extract_rows_bfield")?;
                        let extract_rows_xfield_fn =
                            all_module.get_function("extract_rows_xfield")?;

                        let copy_columns_bfield_fn =
                            all_module.get_function("copy_columns_bfield")?;
                        let copy_columns_xfield_fn =
                            all_module.get_function("copy_columns_xfield")?;

                        // Configure number of streams for async overlapping execution
                        // Default: 3 streams for triple-buffering (upload, kernel, download)
                        // Streams will be created on-demand in worker threads
                        let num_streams = std::env::var("GPU_NUM_STREAMS")
                            .ok()
                            .and_then(|s| s.parse().ok())
                            .unwrap_or(3);

                        // Get SM count for cooperative kernel launches
                        let sm_count = unsafe {
                            let mut count: i32 = 0;
                            super::cuda_driver::check_cuda(
                                cudarc::driver::sys::cuDeviceGetAttribute(
                                    &mut count,
                                    cudarc::driver::sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT,
                                    device.device,
                                ),
                                "cuDeviceGetAttribute(SM_COUNT)",
                            )?;
                            count
                        };

                        devices.push(GpuDeviceContext {
                            sm_count,
                            device,
                            num_streams,
                            ntt_bfield_fn,
                            intt_bfield_fn,
                            ntt_xfield_fn,
                            intt_xfield_fn,
                            coset_scale_bfield_fn,
                            ntt_bfield_fused_coset_fn,
                            ntt_xfield_fused_coset_fn,

                            bfield_poly_fill_table_fn,
                            ntt_bfield_init_omegas_fn,
                            ntt_bfield_extract_fn,
                            ntt_bfield_restore_fn,
                            ntt_bfield_fused_coset_single_fn,
                            ntt_bfield_fused_coset_strided_fn,

                            xfield_poly_fill_table_fn,
                            ntt_xfield_init_omegas_fn,
                            ntt_xfield_extract_fn,
                            ntt_xfield_restore_fn,
                            ntt_xfield_fused_coset_single_fn,
                            ntt_xfield_fused_coset_single_interpolate_fn,
                            ntt_xfield_fused_coset_strided_fn,

                            intt_bfield_strided_fn,
                            intt_xfield_strided_fn,
                            intt_bfield_fused_unscale_fn,
                            intt_xfield_fused_unscale_fn,
                            intt_bfield_fused_unscale_randomize_fn,
                            intt_xfield_fused_unscale_randomize_fn,
                            running_eval_scan_fn,
                            log_derivative_scan_fn,
                            running_eval_scan_parallel_fn,
                            log_derivative_scan_parallel_fn,
                            batch_inversion_fn,
                            hash_rows_bfield_fn,
                            hash_rows_xfield_fn,
                            extract_rows_bfield_fn,
                            extract_rows_xfield_fn,
                            copy_columns_bfield_fn,
                            copy_columns_xfield_fn,
                        });

                        device_id += 1;
                        eprintln!("GPU {} initialized!", device_id);

                        // Limit to 2 GPUs for now
                        if device_id >= 2 {
                            break;
                        }
                    }
                    Err(e) => {
                        eprintln!(
                            "Failed to load CUBIN module on GPU {}: {}. Stopping enumeration.",
                            device_id, e
                        );
                        break;
                    }
                }
            }
            Err(_) => {
                // No more GPUs available
                break;
            }
        }
        }

        if devices.is_empty() {
            return Err("No GPUs could be initialized".into());
        }

        // Enable P2P access between all GPU pairs
        if devices.len() >= 2 {
            eprintln!("Enabling P2P access between {} GPUs...", devices.len());
            for i in 0..devices.len() {
                for j in 0..devices.len() {
                    if i != j {
                        match devices[i].device.can_access_peer(&devices[j].device) {
                            Ok(true) => {
                                if let Err(e) = devices[i].device.enable_peer_access(&devices[j].device) {
                                    eprintln!("  Warning: Failed to enable P2P GPU {} -> GPU {}: {}", i, j, e);
                                } else {
                                    eprintln!("  ✓ P2P enabled: GPU {} -> GPU {}", i, j);
                                }
                            }
                            Ok(false) => {
                                eprintln!("  ✗ P2P not supported: GPU {} -> GPU {}", i, j);
                            }
                            Err(e) => {
                                eprintln!("  Warning: Failed to check P2P GPU {} -> GPU {}: {}", i, j, e);
                            }
                        }
                    }
                }
            }
        }

        eprintln!("Multi-GPU NTT: {} GPU(s) initialized", devices.len());
        eprintln!("GPU Configuration:");
        eprintln!(
            "  Block size (threads):  {} (via GPU_NTT_BLOCK_SIZE env var, default: 256)",
            super::get_gpu_ntt_block_size()
        );
        eprintln!(
            "  Batch chunk size:      {} columns (via GPU_BATCH_CHUNK_SIZE env var, default: 30)",
            super::get_gpu_batch_chunk_size()
        );

        // Report stream configuration
        let streams_per_gpu = devices.first().map(|d| d.num_streams).unwrap_or(0);
        eprintln!(
            "  Streams per GPU:       {} streams (via GPU_NUM_STREAMS env var, default: 3)",
            streams_per_gpu
        );
        if streams_per_gpu > 0 {
            eprintln!("  Multi-stream mode:     ENABLED (async kernel overlapping)");
        }

        let duration = start_initialization.elapsed();
        eprintln!("Kernel initialization took: {:?}", duration);

        Ok(Self {
            devices,
            next_device: AtomicUsize::new(0),
        })
    }

    /// Select device for operation
    /// Always returns GPU 0 for now - dual-GPU is only used explicitly in table NTT
    pub(crate) fn select_device(&self) -> &GpuDeviceContext {
        // Always use GPU 0 for normal operations
        // GPU 1 is only used explicitly in dual-GPU table NTT path
        // credits to allfather team
        &self.devices[0]
    }
}
