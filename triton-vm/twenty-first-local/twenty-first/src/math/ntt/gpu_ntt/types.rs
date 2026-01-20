use crate::math::b_field_element::BFieldElement;
use crate::math::x_field_element::XFieldElement;
use super::cuda_driver::{CudaDevice, CudaFunction};
use std::sync::atomic::AtomicUsize;

#[derive(Debug)]
pub(crate) struct GpuError(pub String);

impl std::fmt::Display for GpuError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for GpuError {}

/// Trait for field elements that can be processed on GPU
/// This eliminates duplication between BFieldElement and XFieldElement operations
pub(crate) trait GpuFieldElement: Sized + Send + Sync + Clone {
    /// Number of u64s needed to represent this element
    const U64_COUNT: usize;

    /// Zero element
    fn zero() -> Self;

    /// Serialize element to u64 array
    fn to_raw_u64s(&self, output: &mut Vec<u64>);

    /// Deserialize element from u64 array at given index
    fn from_raw_u64s(data: &[u64], index: usize) -> Self;

    /// Get the name for this field type (for logging)
    fn field_name() -> &'static str;
}

impl GpuFieldElement for BFieldElement {
    const U64_COUNT: usize = 1;

    fn zero() -> Self {
        BFieldElement::new(0)
    }

    fn to_raw_u64s(&self, output: &mut Vec<u64>) {
        output.push(self.raw_u64());
    }

    fn from_raw_u64s(data: &[u64], index: usize) -> Self {
        BFieldElement::from_raw_u64(data[index])
    }

    fn field_name() -> &'static str {
        "BField"
    }
}

impl GpuFieldElement for XFieldElement {
    const U64_COUNT: usize = 3;

    fn zero() -> Self {
        XFieldElement::new([BFieldElement::new(0); 3])
    }

    fn to_raw_u64s(&self, output: &mut Vec<u64>) {
        output.push(self.coefficients[0].raw_u64());
        output.push(self.coefficients[1].raw_u64());
        output.push(self.coefficients[2].raw_u64());
    }

    fn from_raw_u64s(data: &[u64], index: usize) -> Self {
        XFieldElement::new([
            BFieldElement::from_raw_u64(data[index]),
            BFieldElement::from_raw_u64(data[index + 1]),
            BFieldElement::from_raw_u64(data[index + 2]),
        ])
    }

    fn field_name() -> &'static str {
        "XField"
    }
}

/// Per-GPU device context holding device and functions
pub(crate) struct GpuDeviceContext {
    pub sm_count: i32,
    pub device: CudaDevice,
    pub num_streams: usize,                 // Number of streams to create per device
    pub ntt_bfield_fn: CudaFunction,
    pub intt_bfield_fn: CudaFunction,
    pub ntt_xfield_fn: CudaFunction,
    pub intt_xfield_fn: CudaFunction,
    pub coset_scale_bfield_fn: CudaFunction,
    pub ntt_bfield_fused_coset_fn: CudaFunction,
    pub ntt_xfield_fused_coset_fn: CudaFunction,

    pub bfield_poly_fill_table_fn: CudaFunction,
    pub ntt_bfield_init_omegas_fn: CudaFunction,
    pub ntt_bfield_extract_fn: CudaFunction,
    pub ntt_bfield_fused_coset_single_fn: CudaFunction,
    pub ntt_bfield_restore_fn: CudaFunction,
    pub ntt_bfield_fused_coset_strided_fn: CudaFunction,  // For row-major table data

    pub xfield_poly_fill_table_fn: CudaFunction,
    pub ntt_xfield_init_omegas_fn: CudaFunction,
    pub ntt_xfield_extract_fn: CudaFunction,
    pub ntt_xfield_fused_coset_single_fn: CudaFunction,
    pub ntt_xfield_fused_coset_single_interpolate_fn: CudaFunction,  // For FRI INTT with coset
    pub ntt_xfield_restore_fn: CudaFunction,
    pub ntt_xfield_fused_coset_strided_fn: CudaFunction,  // For row-major table data

    pub intt_bfield_strided_fn: CudaFunction,             // Strided INTT for row-major table data
    pub intt_xfield_strided_fn: CudaFunction,             // Strided INTT for row-major table data
    pub intt_bfield_fused_unscale_fn: CudaFunction,       // Fused INTT + unscaling for BField
    pub intt_xfield_fused_unscale_fn: CudaFunction,       // Fused INTT + unscaling for XField
    pub intt_bfield_fused_unscale_randomize_fn: CudaFunction,  // Fused INTT + unscaling + randomizer for BField
    pub intt_xfield_fused_unscale_randomize_fn: CudaFunction,  // Fused INTT + unscaling + randomizer for XField
    pub running_eval_scan_fn: CudaFunction,               // GPU sequential scan for running evaluations
    pub log_derivative_scan_fn: CudaFunction,             // GPU sequential scan for log derivative accumulation
    pub running_eval_scan_parallel_fn: CudaFunction,      // GPU parallel scan for running evaluations (Blelloch)
    pub log_derivative_scan_parallel_fn: CudaFunction,    // GPU parallel scan for log derivative (Blelloch)
    pub batch_inversion_fn: CudaFunction,                 // Batch field inversion using Montgomery's trick
    pub hash_rows_bfield_fn: CudaFunction,                // Tip5 hash for BField rows
    pub hash_rows_xfield_fn: CudaFunction,                // Tip5 hash for XField rows
    pub extract_rows_bfield_fn: CudaFunction,             // Extract specific rows from BField tables
    pub extract_rows_xfield_fn: CudaFunction,             // Extract specific rows from XField tables
    pub copy_columns_bfield_fn: CudaFunction,             // Copy specific columns between BField tables
    pub copy_columns_xfield_fn: CudaFunction,             // Copy specific columns between XField tables
}

pub struct GpuNttContext {
    pub(crate) devices: Vec<GpuDeviceContext>,
    pub(crate) next_device: AtomicUsize, // For round-robin device selection
}

impl std::fmt::Debug for GpuNttContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use std::sync::atomic::Ordering;
        f.debug_struct("GpuNttContext")
            .field("num_devices", &self.devices.len())
            .field("next_device", &self.next_device.load(Ordering::Relaxed))
            .finish()
    }
}

#[derive(Debug, Default)]
pub(crate) struct GpuTimingStats {
    pub count: usize,
    pub total_prep: std::time::Duration,
    pub total_upload: std::time::Duration,
    pub total_kernel: std::time::Duration,
    pub total_download: std::time::Duration,
    pub total_postproc: std::time::Duration,
    pub total_time: std::time::Duration,
    pub total_upload_bytes: usize,
    pub total_download_bytes: usize,
}
