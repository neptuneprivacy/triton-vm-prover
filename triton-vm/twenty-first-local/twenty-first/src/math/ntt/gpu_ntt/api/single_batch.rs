use super::super::types::GpuNttContext;
use super::super::timing::{GPU_TIMING_BFIELD_INTT, GPU_TIMING_BFIELD_NTT, GPU_TIMING_XFIELD_INTT, GPU_TIMING_XFIELD_NTT};
use crate::math::b_field_element::BFieldElement;
use crate::math::x_field_element::XFieldElement;

impl GpuNttContext {
    // =============================================================================
    // Public API: Single Operations (BField)
    // =============================================================================

    /// Perform GPU NTT on BFieldElement array (unbatched for now)
    ///
    /// # Arguments
    /// * `data` - Mutable slice of BFieldElements (will be transformed in-place)
    /// * `twiddle_factors` - Precomputed twiddle factors (roots of unity powers)
    ///
    /// # Panics
    /// Panics if data length is not a power of 2 or if GPU operations fail
    pub fn ntt_bfield(
        &self,
        data: &mut [BFieldElement],
        twiddle_factors: &[Vec<BFieldElement>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dev_ctx = self.select_device();
        self.execute_single(data, twiddle_factors, &dev_ctx.ntt_bfield_fn, &GPU_TIMING_BFIELD_NTT)
    }

    /// Perform GPU inverse NTT on BFieldElement array (unbatched)
    pub fn intt_bfield(
        &self,
        data: &mut [BFieldElement],
        twiddle_factors_inv: &[Vec<BFieldElement>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dev_ctx = self.select_device();
        self.execute_single(data, twiddle_factors_inv, &dev_ctx.intt_bfield_fn, &GPU_TIMING_BFIELD_INTT)
        // Note: Caller must still divide by len (unscaling)
    }

    // =============================================================================
    // Public API: Single Operations (XField)
    // =============================================================================

    /// Perform GPU NTT on XFieldElement array
    pub fn ntt_xfield(
        &self,
        data: &mut [XFieldElement],
        twiddle_factors: &[Vec<BFieldElement>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dev_ctx = self.select_device();
        self.execute_single(data, twiddle_factors, &dev_ctx.ntt_xfield_fn, &GPU_TIMING_XFIELD_NTT)
    }

    /// Perform GPU inverse NTT on XFieldElement array
    pub fn intt_xfield(
        &self,
        data: &mut [XFieldElement],
        twiddle_factors_inv: &[Vec<BFieldElement>],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dev_ctx = self.select_device();
        self.execute_single(data, twiddle_factors_inv, &dev_ctx.intt_xfield_fn, &GPU_TIMING_XFIELD_INTT)
        // Note: Caller must still divide by len (unscaling)
    }

    // =============================================================================
    // Public API: Batch Operations
    // =============================================================================

    /// Batch process multiple NTT operations to amortize PCIe transfer overhead
    ///
    /// ## Performance Impact
    /// Batching reduces PCIe transfer overhead by combining multiple operations:
    /// - **Current**: N operations = 2N transfers (N upload + N download)
    /// - **Batched**: N operations = 2 transfers (1 upload + 1 download)
    /// - **Expected speedup**: 2-3x for typical workloads
    ///
    /// ## Requirements
    /// - All arrays must have the same length
    /// - Length must be a power of 2
    /// - All arrays use the same twiddle factors
    ///
    /// ## Integration Status
    /// This API is implemented and tested but not yet integrated into call sites.
    /// Integration requires collecting consecutive same-size operations before calling NTT.
    ///
    /// ## Example Usage (Future)
    /// ```ignore
    /// // Collect multiple arrays that need NTT
    /// let mut arrays: Vec<&mut [BFieldElement]> = vec![&mut arr1, &mut arr2, &mut arr3];
    /// ctx.ntt_bfield_batch(&mut arrays, twiddle_factors)?;
    /// // All arrays transformed in one GPU roundtrip
    /// ```
    ///
    /// All arrays must be the same length and power of 2
    pub fn ntt_bfield_batch(
        &self,
        data_arrays: &mut [&mut [BFieldElement]],
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
        let dev_ctx = self.select_device();
        self.execute_batch(data_arrays, twiddle_factors, &dev_ctx.ntt_bfield_fn)
    }

    /// Batch process multiple BField INTT operations
    pub fn intt_bfield_batch(
        &self,
        data_arrays: &mut [&mut [BFieldElement]],
        twiddle_factors_inv: &[Vec<BFieldElement>],
    ) -> Result<
        (
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
        ),
        Box<dyn std::error::Error>,
    > {
        let dev_ctx = self.select_device();
        self.execute_batch(data_arrays, twiddle_factors_inv, &dev_ctx.intt_bfield_fn)
    }

    /// Batch process multiple XField NTT operations
    pub fn ntt_xfield_batch(
        &self,
        data_arrays: &mut [&mut [XFieldElement]],
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
        let dev_ctx = self.select_device();
        self.execute_batch(data_arrays, twiddle_factors, &dev_ctx.ntt_xfield_fn)
    }

    /// Batch process multiple XField INTT operations
    pub fn intt_xfield_batch(
        &self,
        data_arrays: &mut [&mut [XFieldElement]],
        twiddle_factors_inv: &[Vec<BFieldElement>],
    ) -> Result<
        (
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
            std::time::Duration,
        ),
        Box<dyn std::error::Error>,
    > {
        let dev_ctx = self.select_device();
        self.execute_batch(data_arrays, twiddle_factors_inv, &dev_ctx.intt_xfield_fn)
    }

    // =============================================================================
    // Public API: Chunked Operations
    // =============================================================================

    /// Perform chunked batched NTT on BFieldElement arrays to fit GPU memory constraints
    ///
    /// This function splits the arrays into chunks of size GPU_BATCH_CHUNK_SIZE and
    /// processes each chunk as a batch, reducing PCIe transfer overhead while managing
    /// GPU memory usage.
    ///
    /// # Arguments
    /// * `data_arrays` - Mutable slices of BFieldElements to transform
    /// * `twiddle_factors` - Precomputed twiddle factors
    /// * `chunk_size` - Maximum number of arrays to batch together (e.g., GPU_BATCH_CHUNK_SIZE)
    ///
    /// # Performance
    /// - Reduces 379 separate GPU calls to ~14 batches
    /// - Eliminates thread contention on GPU queue
    /// - Achieves near-full PCIe bandwidth utilization
    /// - With multiple GPUs, chunks are processed in parallel automatically
    pub fn ntt_bfield_chunked(
        &self,
        data_arrays: &mut [&mut [BFieldElement]],
        twiddle_factors: &[Vec<BFieldElement>],
        chunk_size: usize,
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dev_ctx = self.select_device();
        self.execute_chunked(data_arrays, twiddle_factors, &dev_ctx.ntt_bfield_fn, chunk_size, phase_name, "NTT")
    }

    /// Perform chunked batched INTT on BFieldElement arrays
    pub fn intt_bfield_chunked(
        &self,
        data_arrays: &mut [&mut [BFieldElement]],
        twiddle_factors_inv: &[Vec<BFieldElement>],
        chunk_size: usize,
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dev_ctx = self.select_device();
        self.execute_chunked(data_arrays, twiddle_factors_inv, &dev_ctx.intt_bfield_fn, chunk_size, phase_name, "INTT")
    }

    /// Perform chunked batched NTT on XFieldElement arrays
    pub fn ntt_xfield_chunked(
        &self,
        data_arrays: &mut [&mut [XFieldElement]],
        twiddle_factors: &[Vec<BFieldElement>],
        chunk_size: usize,
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dev_ctx = self.select_device();
        self.execute_chunked(data_arrays, twiddle_factors, &dev_ctx.ntt_xfield_fn, chunk_size, phase_name, "NTT")
    }

    /// Perform chunked batched INTT on XFieldElement arrays
    pub fn intt_xfield_chunked(
        &self,
        data_arrays: &mut [&mut [XFieldElement]],
        twiddle_factors_inv: &[Vec<BFieldElement>],
        chunk_size: usize,
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let dev_ctx = self.select_device();
        self.execute_chunked(data_arrays, twiddle_factors_inv, &dev_ctx.intt_xfield_fn, chunk_size, phase_name, "INTT")
    }
}
