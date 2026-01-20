///! Dual-GPU test module for NVLink P2P verification
///!
///! This module tests multi-GPU functionality by:
///! 1. Running the same NTT kernel on both GPUs
///! 2. Using P2P to transfer results between GPUs
///! 3. Verifying both GPUs produced identical results

use super::cuda_driver::{CudaDevice, DeviceBuffer, KernelArgs};
use std::sync::Arc;
use std::fs;

/// Dual-GPU context for testing
pub struct DualGpuContext {
    pub gpu0: CudaDevice,
    pub gpu1: CudaDevice,
}

impl DualGpuContext {
    /// Initialize both GPUs and enable P2P access
    pub fn new() -> Result<Self, Box<dyn std::error::Error>> {
        eprintln!("\n=== Initializing Dual-GPU Context ===");

        // Initialize both devices
        eprintln!("Creating GPU 0...");
        let gpu0 = CudaDevice::new(0)?;

        eprintln!("Creating GPU 1...");
        let gpu1 = CudaDevice::new(1)?;

        // Check P2P capability
        eprintln!("Checking P2P capability...");
        let can_p2p_0_to_1 = gpu0.can_access_peer(&gpu1)?;
        let can_p2p_1_to_0 = gpu1.can_access_peer(&gpu0)?;

        eprintln!("  GPU 0 -> GPU 1: {}", can_p2p_0_to_1);
        eprintln!("  GPU 1 -> GPU 0: {}", can_p2p_1_to_0);

        if !can_p2p_0_to_1 || !can_p2p_1_to_0 {
            return Err("P2P not supported between GPUs".into());
        }

        // Enable P2P access in both directions
        eprintln!("Enabling P2P access...");
        gpu0.enable_peer_access(&gpu1)?;
        gpu1.enable_peer_access(&gpu0)?;

        eprintln!("✓ Dual-GPU context initialized with P2P enabled\n");

        Ok(Self { gpu0, gpu1 })
    }

    /// Test NTT execution on both GPUs with P2P transfer and verification
    pub fn test_dual_ntt_with_verification(
        &self,
        test_data: &[u64],
    ) -> Result<(), Box<dyn std::error::Error>> {
        eprintln!("\n=== Dual-GPU NTT Test ===");
        eprintln!("Test data size: {} elements ({} bytes)", test_data.len(), test_data.len() * 8);

        // Load CUBIN module on both GPUs
        let cubin_bytes = fs::read("built_kernels/big_package.cubin")
            .map_err(|e| format!("Failed to read CUBIN file: {}", e))?;

        eprintln!("Loading module on GPU 0...");
        let module0 = self.gpu0.load_module(&cubin_bytes)?;
        let ntt_fn_0 = module0.get_function("ntt_bfield")?;
        let verify_fn_0 = module0.get_function("verify_buffers_bfield")?;

        eprintln!("Loading module on GPU 1...");
        let module1 = self.gpu1.load_module(&cubin_bytes)?;
        let ntt_fn_1 = module1.get_function("ntt_bfield")?;

        // Create streams for both GPUs
        let stream0 = self.gpu0.create_stream()?;
        let stream1 = self.gpu1.create_stream()?;

        // Step 1: Upload test data to GPU 0
        eprintln!("\n[Step 1] Uploading test data to GPU 0...");
        let d_data_gpu0 = stream0.memcpy_htod(test_data)?;

        // Step 2: P2P copy from GPU 0 to GPU 1
        eprintln!("[Step 2] P2P copying data from GPU 0 to GPU 1...");
        let d_data_gpu1 = stream1.alloc::<u64>(test_data.len())?;
        stream0.memcpy_peer_async(
            &d_data_gpu1,
            &self.gpu1.context,
            &d_data_gpu0,
            &self.gpu0.context,
            test_data.len() * 8,
        )?;
        stream0.synchronize()?;
        eprintln!("✓ P2P copy completed");

        // Step 3: Allocate twiddle factors (simplified - using same data for test)
        eprintln!("[Step 3] Setting up kernel parameters...");
        let d_twiddles_gpu0 = stream0.memcpy_htod(test_data)?;
        let d_twiddles_gpu1 = stream1.memcpy_htod(test_data)?;

        let len = test_data.len() as u64;
        let log2_len = len.trailing_zeros() as u64;

        // Step 4: Launch NTT kernel on both GPUs
        eprintln!("[Step 4] Launching NTT kernels on both GPUs...");

        // Launch on GPU 0
        // ntt_bfield signature: (u64* poly_global, u64 slice_len, u64* omegas, u32 log2_slice_len)
        {
            let mut d_data_ptr = d_data_gpu0.as_ptr();
            let d_twiddles_ptr = d_twiddles_gpu0.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_mut_ptr(&mut d_data_ptr)
                .push_value(len)
                .push_ptr(&d_twiddles_ptr)
                .push_value(log2_len as u32); // Note: u32 not u64

            ntt_fn_0.launch(
                (1, 1, 1),  // Single block since slice_len = full array
                (256, 1, 1),
                0,
                &stream0,
                kernel_args.as_mut_slice(),
            )?;
        }

        // Launch on GPU 1
        {
            let mut d_data_ptr = d_data_gpu1.as_ptr();
            let d_twiddles_ptr = d_twiddles_gpu1.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_mut_ptr(&mut d_data_ptr)
                .push_value(len)
                .push_ptr(&d_twiddles_ptr)
                .push_value(log2_len as u32); // Note: u32 not u64

            ntt_fn_1.launch(
                (1, 1, 1),  // Single block since slice_len = full array
                (256, 1, 1),
                0,
                &stream1,
                kernel_args.as_mut_slice(),
            )?;
        }

        stream0.synchronize()?;
        stream1.synchronize()?;
        eprintln!("✓ NTT kernels completed on both GPUs");

        // Step 5: P2P copy result from GPU 1 to GPU 0 for verification
        eprintln!("[Step 5] P2P copying result from GPU 1 to GPU 0...");
        let d_data_gpu1_copy = stream0.alloc::<u64>(test_data.len())?;
        stream0.memcpy_peer_async(
            &d_data_gpu1_copy,
            &self.gpu0.context,
            &d_data_gpu1,
            &self.gpu1.context,
            test_data.len() * 8,
        )?;
        stream0.synchronize()?;
        eprintln!("✓ P2P copy completed");

        // Step 6: Run verification kernel on GPU 0
        eprintln!("[Step 6] Running verification kernel...");
        let d_mismatch_count = stream0.memcpy_htod(&[0u64])?;

        {
            let d_data_gpu0_ptr = d_data_gpu0.as_ptr();
            let d_data_gpu1_copy_ptr = d_data_gpu1_copy.as_ptr();
            let mut d_mismatch_count_ptr = d_mismatch_count.as_ptr();

            let mut kernel_args = KernelArgs::new();
            kernel_args
                .push_ptr(&d_data_gpu0_ptr)
                .push_ptr(&d_data_gpu1_copy_ptr)
                .push_value(len)
                .push_mut_ptr(&mut d_mismatch_count_ptr);

            verify_fn_0.launch(
                (128, 1, 1),
                (256, 1, 1),
                0,
                &stream0,
                kernel_args.as_mut_slice(),
            )?;
        }

        stream0.synchronize()?;

        // Download mismatch count
        let mut mismatch_count = vec![0u64; 1];
        stream0.memcpy_dtoh(&d_mismatch_count, &mut mismatch_count)?;
        stream0.synchronize()?;

        eprintln!("\n=== Verification Results ===");
        eprintln!("Mismatch count: {}", mismatch_count[0]);

        if mismatch_count[0] == 0 {
            eprintln!("✓ SUCCESS: Both GPUs produced identical results!");
            eprintln!("✓ P2P transfers working correctly via NVLink");
            Ok(())
        } else {
            Err(format!(
                "FAILED: Found {} mismatches between GPU outputs",
                mismatch_count[0]
            ).into())
        }
    }
}
