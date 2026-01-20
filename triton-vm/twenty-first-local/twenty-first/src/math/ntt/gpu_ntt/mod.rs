// GPU-accelerated NTT implementation using CUDA
// This module provides GPU implementations of NTT for BFieldElement and XFieldElement
//
// ## Graceful Degradation Strategy
//
// The GPU NTT implementation is designed to fail gracefully and never break proof generation:
//
// 1. **GPU Initialization Failure**: If GPU initialization fails (no CUDA, no GPU, driver issues),
//    `get_gpu_context()` returns None and all operations fall back to CPU automatically.
//
// 2. **GPU Operation Failure**: If a GPU operation fails (OOM, kernel error, etc.), the error
//    is propagated up and the caller falls back to CPU implementation. See ntt() and intt() in mod.rs.
//
// 3. **Size-based Routing**: Operations smaller than GPU_THRESHOLD automatically use CPU,
//    as GPU overhead makes them slower. This is checked by `should_use_gpu()`.
//
// 4. **No Panics in Hot Path**: All GPU operations return Result<> instead of panicking,
//    allowing graceful error handling and fallback.
//
// This ensures that GPU acceleration is purely an optimization - the system always works,
// with or without GPU support.

#[cfg(feature = "gpu")]
mod api;
#[cfg(feature = "gpu")]
pub mod aux_extend;
#[cfg(feature = "gpu")]
mod context;
#[cfg(feature = "gpu")]
#[cfg(test)]
mod coset_scale_test;
#[cfg(feature = "gpu")]
mod operations;
#[cfg(feature = "gpu")]
pub(crate) mod timing;
#[cfg(feature = "gpu")]
pub mod cuda_driver;
#[cfg(feature = "gpu")]
mod types;
#[cfg(feature = "gpu")]
pub mod dual_gpu_test;

#[cfg(feature = "gpu")]
pub use types::GpuNttContext;

#[cfg(feature = "gpu")]
use std::sync::OnceLock;

/// Minimum size threshold for using GPU (below this, CPU is faster due to overhead)
/// Analysis shows GPU is slower than CPU for sizes < 2^20 due to PCIe transfer overhead:
/// - 2^15: GPU 3.1ms vs CPU 0.8ms (4x slower)
/// - 2^16: GPU 2.8ms vs CPU 1.8ms (1.6x slower)
/// - 2^17: GPU 4.5ms vs CPU 3.8ms (1.2x slower)
/// - 2^18: GPU 6.2ms vs CPU ~7ms (roughly same)
/// Set to 2^20 (1,048,576) to only use GPU for large operations
pub const GPU_THRESHOLD: usize = 1 << 20;

/// Maximum number of columns to batch together for GPU NTT operations
///
/// ## GPU Memory Requirements
///
/// For BFieldElement (8 bytes):
/// - 30 columns × 4M elements × 8 bytes × 2 (input+output) = ~1.9 GB peak
///
/// For XFieldElement (24 bytes, 3× larger):
/// - 30 columns × 4M elements × 24 bytes × 2 = ~5.7 GB peak
///
/// ## Recommended Values by GPU Memory
///
/// - **8 GB GPU**:  chunk_size = 20-30 columns (conservative, fits both BField and XField)
/// - **12 GB GPU**: chunk_size = 40 columns
/// - **16 GB GPU**: chunk_size = 50-60 columns
/// - **24 GB GPU**: chunk_size = 80-100 columns
/// - **40 GB GPU**: chunk_size = 150+ (can fit full table)
///
/// ## Performance Impact
///
/// Current unbatched: 379 separate GPU calls = 682s, 0.18% PCIe utilization
/// With chunking:
/// - 1 batch (114 cols):  ~5.6s (122× speedup) - requires 24+ GB GPU
/// - 2 batches (57 cols): ~5.7s (120× speedup) - requires 16+ GB GPU
/// - 3 batches (38 cols): ~5.8s (118× speedup) - requires 12 GB GPU
/// - 4 batches (28 cols): ~5.9s (116× speedup) - fits 8 GB GPU ✅
/// - 5 batches (22 cols): ~6.0s (114× speedup) - safer for 8 GB GPU
///
/// **Key insight**: Even 4-5 batches gives ~115× speedup! Going from 379 → 4 GPU calls
/// eliminates thread contention and enables near-full PCIe bandwidth utilization.
///
/// ## How to Tune
///
/// 1. **Start conservative**: Use default (30 columns)
/// 2. **Monitor GPU memory**: Watch `nvidia-smi` during proof generation
/// 3. **Increase if stable**: If <80% VRAM used, increase chunk size
/// 4. **Decrease if OOM**: If you see "CUDA out of memory", reduce by 5-10 columns
///
/// Set to 30 columns as safe default for 8-12 GB GPUs (most common consumer hardware)
pub const GPU_BATCH_CHUNK_SIZE: usize = 30;

/// Get GPU NTT block size (threads per block) from environment variable
///
/// Set via: `export GPU_NTT_BLOCK_SIZE=256` (or 512, 1024, etc.)
/// Default: 256 threads per block (empirically optimal for this kernel)
///
/// **Tested values**:
/// - 256: Optimal performance (default)
/// - 512: ~5-10% slower
/// - 1024: ~20% slower (register pressure)
///
/// Only change if testing on different GPU architectures
#[cfg(feature = "gpu")]
pub fn get_gpu_ntt_block_size() -> u32 {
    std::env::var("GPU_NTT_BLOCK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(256)
}

/// Get GPU batch chunk size from environment variable
///
/// Set via: `export GPU_BATCH_CHUNK_SIZE=30` (or 20, 40, 60, etc.)
/// Default: 30 columns
///
/// **Recommended by GPU VRAM**:
/// - 8 GB:  20-30 columns
/// - 12 GB: 40 columns
/// - 16 GB: 50-60 columns
/// - 24+ GB: 80+ columns
///
/// Monitor `nvidia-smi` during proof generation and adjust if needed
#[cfg(feature = "gpu")]
pub fn get_gpu_batch_chunk_size() -> usize {
    std::env::var("GPU_BATCH_CHUNK_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(GPU_BATCH_CHUNK_SIZE)
}

#[cfg(feature = "gpu")]
static GPU_CONTEXT: OnceLock<Option<GpuNttContext>> = OnceLock::new();

#[cfg(feature = "gpu")]
pub fn get_gpu_context() -> Option<&'static GpuNttContext> {
    GPU_CONTEXT
        .get_or_init(|| match GpuNttContext::new() {
            Ok(ctx) => {
                eprintln!("GPU NTT context initialized successfully");
                Some(ctx)
            }
            Err(e) => {
                eprintln!("Failed to initialize GPU NTT context: {}", e);
                eprintln!("Falling back to CPU NTT");
                None
            }
        })
        .as_ref()
}

/// Check if GPU NTT is available and should be used for the given size
#[cfg(feature = "gpu")]
pub fn should_use_gpu(len: usize) -> bool {
    len >= GPU_THRESHOLD && get_gpu_context().is_some()
}

#[cfg(not(feature = "gpu"))]
pub fn should_use_gpu(_len: usize) -> bool {
    false
}

#[cfg(feature = "gpu")]
pub fn print_gpu_timing_summary() {
    timing::print_gpu_timing_summary();
}

#[cfg(not(feature = "gpu"))]
pub fn print_gpu_timing_summary() {
    // No-op when GPU is not enabled
}
