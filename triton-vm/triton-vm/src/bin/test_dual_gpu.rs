///! Test binary for dual-GPU NVLink functionality
///!
///! This tests:
///! - Initializing 2 GPUs
///! - Enabling P2P access
///! - Running same NTT on both GPUs
///! - P2P transfer between GPUs
///! - Verification that results match

use twenty_first::math::ntt::gpu_ntt::dual_gpu_test::DualGpuContext;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    eprintln!("╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║         Dual-GPU NVLink Test for 2x B200 GPUs              ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    // Initialize dual-GPU context
    let dual_ctx = DualGpuContext::new()?;

    // Create test data (power of 2 size for NTT)
    let test_size: usize = 1 << 16; // 65,536 elements
    eprintln!("\nGenerating test data: {} elements", test_size);
    let test_data: Vec<u64> = (0..test_size)
        .map(|i| ((i as u64 * 12345 + 67890) % 2147483647)) // Simple pseudo-random data
        .collect();

    eprintln!("Test data sample: [{}, {}, {}, ..., {}]",
        test_data[0], test_data[1], test_data[2], test_data[test_size - 1]);

    // Run the dual-GPU test
    dual_ctx.test_dual_ntt_with_verification(&test_data)?;

    eprintln!("\n╔══════════════════════════════════════════════════════════════╗");
    eprintln!("║                    TEST PASSED ✓                            ║");
    eprintln!("║                                                              ║");
    eprintln!("║  NVLink P2P transfers working correctly on 2x B200          ║");
    eprintln!("║  Both GPUs produced identical NTT results                   ║");
    eprintln!("╚══════════════════════════════════════════════════════════════╝");

    Ok(())
}
