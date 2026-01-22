//! GPU Buffer Management for LDE Tables
//!
//! This module provides types for keeping LDE-extended tables GPU-resident,
//! eliminating wasteful GPU→RAM→GPU transfers.

use std::sync::Arc;

#[cfg(feature = "gpu")]
use twenty_first::math::ntt::gpu_ntt::cuda_driver::DeviceBuffer;

/// GPU-resident LDE table buffer
///
/// Wraps a CUDA buffer with metadata about the table dimensions.
/// The buffer is kept alive via Arc, allowing it to be shared between
/// LDE computation and quotient evaluation phases.
#[derive(Clone)]
pub struct GpuLdeBuffer {
    /// GPU device buffer (u64 representation of field elements)
    /// Arc ensures the buffer stays alive until all references are dropped
    #[cfg(feature = "gpu")]
    pub(crate) device_buffer: Arc<DeviceBuffer>,

    /// Number of rows in the table
    pub(crate) num_rows: usize,

    /// Number of columns in the table
    pub(crate) num_cols: usize,

    /// Size of each element in bytes (8 for BField, 24 for XField)
    pub(crate) element_size: usize,
}

impl GpuLdeBuffer {
    /// Create a new GPU buffer wrapper
    #[cfg(feature = "gpu")]
    pub fn new(
        device_buffer: Arc<DeviceBuffer>,
        num_rows: usize,
        num_cols: usize,
        element_size: usize,
    ) -> Self {
        Self {
            device_buffer,
            num_rows,
            num_cols,
            element_size,
        }
    }

    /// Get the device buffer (for passing to GPU operations)
    #[cfg(feature = "gpu")]
    pub fn device_buffer(&self) -> &DeviceBuffer {
        &self.device_buffer
    }

    /// Get total number of u64 elements
    pub fn total_elements(&self) -> usize {
        self.num_rows * self.num_cols * (self.element_size / 8)
    }

    /// Get table dimensions
    pub fn dimensions(&self) -> (usize, usize) {
        (self.num_rows, self.num_cols)
    }
}

impl std::fmt::Debug for GpuLdeBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuLdeBuffer")
            .field("num_rows", &self.num_rows)
            .field("num_cols", &self.num_cols)
            .field("element_size", &self.element_size)
            .field("total_bytes", &(self.total_elements() * 8))
            .finish()
    }
}

// Safety: GpuLdeBuffer can be sent between threads as long as CUDA context is valid
// credits to allfather team
unsafe impl Send for GpuLdeBuffer {}
unsafe impl Sync for GpuLdeBuffer {}
