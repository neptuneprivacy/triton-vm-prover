use super::super::types::GpuNttContext;
use crate::math::b_field_element::BFieldElement;
use crate::math::x_field_element::XFieldElement;
use super::super::cuda_driver::DeviceBuffer;

impl GpuNttContext {
    // =============================================================================
    // Public API: Unchunked Batch Operations (Single GPU Call)
    // =============================================================================

    /// Perform fused GPU coset scaling + NTT on ALL BFieldElement arrays in a single batch
    /// This eliminates per-chunk serialization overhead by processing everything at once
    ///
    /// **Performance Benefits:**
    /// - Single serialization for all data (not per chunk)
    /// - Single PCIe upload (not per chunk)
    /// - Single kernel launch for all arrays
    /// - Single PCIe download (not per chunk)
    /// - Single deserialization for all data (not per chunk)
    ///
    /// **Use When:**
    /// - GPU memory can hold all data (~3.6GB for 114 BField columns × 4M elements)
    /// - Maximum performance is needed
    /// - No memory constraints
    ///
    /// # Arguments
    /// * `data_arrays` - Mutable slices of BFieldElements to scale and transform
    /// * `offset` - Coset offset value (generator for the coset)
    /// * `twiddle_factors` - Precomputed twiddle factors for NTT
    /// * `phase_name` - Name for logging purposes
    pub fn ntt_bfield_fused_coset_unchunked(
        &self,
        data_arrays: &mut [&mut [BFieldElement]],
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.execute_fused_coset_ntt_batch_unchunked_bfield(
            data_arrays,
            offset,
            twiddle_factors,
            phase_name,
        )
    }

    /// Perform fused GPU coset scaling + NTT on ALL XFieldElement arrays in a single batch
    /// This eliminates per-chunk serialization overhead by processing everything at once
    ///
    /// **Performance Benefits:**
    /// - Single serialization for all data (not per chunk)
    /// - Single PCIe upload (not per chunk)
    /// - Single kernel launch for all arrays
    /// - Single PCIe download (not per chunk)
    /// - Single deserialization for all data (not per chunk)
    ///
    /// **Use When:**
    /// - GPU memory can hold all data (~10.8GB for 114 XField columns × 4M elements)
    /// - Maximum performance is needed
    /// - No memory constraints
    ///
    /// # Arguments
    /// * `data_arrays` - Mutable slices of XFieldElements to scale and transform
    /// * `offset` - Coset offset value (generator for the coset)
    /// * `twiddle_factors` - Precomputed twiddle factors for NTT
    /// * `phase_name` - Name for logging purposes
    pub fn ntt_xfield_fused_coset_unchunked(
        &self,
        data_arrays: &mut [&mut [XFieldElement]],
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.execute_fused_coset_ntt_batch_unchunked_xfield(
            data_arrays,
            offset,
            twiddle_factors,
            phase_name,
        )
    }

    /// Perform GPU INTT on ALL BFieldElement arrays in a single batch (no chunking)
    /// This eliminates per-chunk serialization overhead by processing everything at once
    ///
    /// **Performance Benefits:**
    /// - Single serialization for all data (not per chunk)
    /// - Single PCIe upload (not per chunk)
    /// - Single kernel launch for all arrays
    /// - Single PCIe download (not per chunk)
    /// - Single deserialization for all data (not per chunk)
    ///
    /// **Use When:**
    /// - GPU memory can hold all data
    /// - Maximum performance is needed
    /// - Processing many columns at once
    ///
    /// # Arguments
    /// * `data_arrays` - Mutable slices of BFieldElements to transform
    /// * `twiddle_factors_inv` - Precomputed inverse twiddle factors for INTT
    /// * `phase_name` - Name for logging purposes
    ///
    /// # Note
    /// Caller must still divide by len (unscaling) after INTT
    pub fn intt_bfield_unchunked(
        &self,
        data_arrays: &mut [&mut [BFieldElement]],
        twiddle_factors_inv: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if data_arrays.is_empty() {
            return Ok(());
        }

        let operation_start = std::time::Instant::now();
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        eprintln!(
            "\n[GPU Batch BField INTT - {}] Processing {} arrays (UNCHUNKED - single GPU call)",
            phase_name,
            batch_size
        );

        let (prep, upload, kernel, download) =
            self.execute_batch(data_arrays, twiddle_factors_inv, &dev_ctx.intt_bfield_fn)?;

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Single Batch] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms)",
            batch_size,
            total_time.as_secs_f64() * 1000.0,
            prep.as_secs_f64() * 1000.0,
            upload.as_secs_f64() * 1000.0,
            kernel.as_secs_f64() * 1000.0,
            download.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// Perform GPU INTT on ALL XFieldElement arrays in a single batch (no chunking)
    /// This eliminates per-chunk serialization overhead by processing everything at once
    ///
    /// **Performance Benefits:**
    /// - Single serialization for all data (not per chunk)
    /// - Single PCIe upload (not per chunk)
    /// - Single kernel launch for all arrays
    /// - Single PCIe download (not per chunk)
    /// - Single deserialization for all data (not per chunk)
    ///
    /// **Use When:**
    /// - GPU memory can hold all data
    /// - Maximum performance is needed
    /// - Processing many columns at once
    ///
    /// # Arguments
    /// * `data_arrays` - Mutable slices of XFieldElements to transform
    /// * `twiddle_factors_inv` - Precomputed inverse twiddle factors for INTT
    /// * `phase_name` - Name for logging purposes
    ///
    /// # Note
    /// Caller must still divide by len (unscaling) after INTT
    pub fn intt_xfield_unchunked(
        &self,
        data_arrays: &mut [&mut [XFieldElement]],
        twiddle_factors_inv: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if data_arrays.is_empty() {
            return Ok(());
        }

        let operation_start = std::time::Instant::now();
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        eprintln!(
            "\n[GPU Batch XField INTT - {}] Processing {} arrays (UNCHUNKED - single GPU call)",
            phase_name,
            batch_size
        );

        let (prep, upload, kernel, download) =
            self.execute_batch(data_arrays, twiddle_factors_inv, &dev_ctx.intt_xfield_fn)?;

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Single Batch] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms)",
            batch_size,
            total_time.as_secs_f64() * 1000.0,
            prep.as_secs_f64() * 1000.0,
            upload.as_secs_f64() * 1000.0,
            kernel.as_secs_f64() * 1000.0,
            download.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// Perform GPU INTT with fused unscaling on BFieldElement arrays (single batch)
    /// This eliminates CPU postprocessing by performing the unscaling step on GPU
    ///
    /// **Performance Benefits over separate INTT + CPU unscaling:**
    /// - No CPU loop to multiply by n_inv
    /// - GPU parallelism for unscaling
    /// - Eliminates ~10s of CPU postprocessing time
    ///
    /// # Arguments
    /// * `data_arrays` - Mutable slices of BFieldElements to transform
    /// * `twiddle_factors_inv` - Precomputed inverse twiddle factors for INTT
    /// * `n_inv` - Inverse of the domain length (for unscaling)
    /// * `phase_name` - Name for logging purposes
    pub fn intt_bfield_fused_unscale(
        &self,
        data_arrays: &mut [&mut [BFieldElement]],
        twiddle_factors_inv: &[Vec<BFieldElement>],
        n_inv: BFieldElement,
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if data_arrays.is_empty() {
            return Ok(());
        }

        let operation_start = std::time::Instant::now();
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        eprintln!(
            "\n[GPU Batch BField INTT Fused Unscale - {}] Processing {} arrays",
            phase_name,
            batch_size
        );

        let (prep, upload, kernel, download) = self.execute_batch_fused_unscale(
            data_arrays,
            twiddle_factors_inv,
            n_inv,
            &dev_ctx.intt_bfield_fused_unscale_fn,
        )?;

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Fused Batch] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms)",
            batch_size,
            total_time.as_secs_f64() * 1000.0,
            prep.as_secs_f64() * 1000.0,
            upload.as_secs_f64() * 1000.0,
            kernel.as_secs_f64() * 1000.0,
            download.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// Perform GPU INTT with fused unscaling on XFieldElement arrays (single batch)
    /// This eliminates CPU postprocessing by performing the unscaling step on GPU
    ///
    /// **Performance Benefits over separate INTT + CPU unscaling:**
    /// - No CPU loop to multiply by n_inv
    /// - GPU parallelism for unscaling
    /// - Eliminates ~5s of CPU postprocessing time
    ///
    /// # Arguments
    /// * `data_arrays` - Mutable slices of XFieldElements to transform
    /// * `twiddle_factors_inv` - Precomputed inverse twiddle factors for INTT
    /// * `n_inv` - Inverse of the domain length (for unscaling)
    /// * `phase_name` - Name for logging purposes
    pub fn intt_xfield_fused_unscale(
        &self,
        data_arrays: &mut [&mut [XFieldElement]],
        twiddle_factors_inv: &[Vec<BFieldElement>],
        n_inv: BFieldElement,
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if data_arrays.is_empty() {
            return Ok(());
        }

        let operation_start = std::time::Instant::now();
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        eprintln!(
            "\n[GPU Batch XField INTT Fused Unscale - {}] Processing {} arrays",
            phase_name,
            batch_size
        );

        let (prep, upload, kernel, download) = self.execute_batch_fused_unscale(
            data_arrays,
            twiddle_factors_inv,
            n_inv,
            &dev_ctx.intt_xfield_fused_unscale_fn,
        )?;

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Fused Batch] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms)",
            batch_size,
            total_time.as_secs_f64() * 1000.0,
            prep.as_secs_f64() * 1000.0,
            upload.as_secs_f64() * 1000.0,
            kernel.as_secs_f64() * 1000.0,
            download.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// Perform GPU INTT with fused unscaling and randomizer on BFieldElement arrays
    /// This combines INTT + unscaling + randomizer addition in a single GPU kernel
    /// **Eliminates ALL CPU postprocessing**
    ///
    /// # Arguments
    /// * `data_arrays` - Mutable vectors (will be resized to include randomizers)
    /// * `twiddle_factors_inv` - Precomputed inverse twiddle factors
    /// * `n_inv` - Inverse of domain length (for unscaling)
    /// * `randomizers` - Random coefficients for zero-knowledge (one per column)
    /// * `offset_power_n` - offset^trace_length for zerofier multiplication
    /// * `phase_name` - Name for logging
    pub fn intt_bfield_fused_unscale_randomize(
        &self,
        data_arrays: &mut [Vec<BFieldElement>],
        twiddle_factors_inv: &[Vec<BFieldElement>],
        n_inv: BFieldElement,
        randomizers: &[Vec<BFieldElement>],
        offset_power_n: BFieldElement,
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if data_arrays.is_empty() {
            return Ok(());
        }

        let operation_start = std::time::Instant::now();
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        eprintln!(
            "\n[GPU Batch BField INTT Fused Unscale+Randomize - {}] Processing {} arrays",
            phase_name,
            batch_size
        );

        let (prep, upload, kernel, download) = self.execute_batch_fused_unscale_randomize(
            data_arrays,
            twiddle_factors_inv,
            n_inv,
            randomizers,
            offset_power_n,
            &dev_ctx.intt_bfield_fused_unscale_randomize_fn,
        )?;

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Fused Batch] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms)",
            batch_size,
            total_time.as_secs_f64() * 1000.0,
            prep.as_secs_f64() * 1000.0,
            upload.as_secs_f64() * 1000.0,
            kernel.as_secs_f64() * 1000.0,
            download.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    /// Perform GPU INTT with fused unscaling and randomizer on XFieldElement arrays
    /// This combines INTT + unscaling + randomizer addition in a single GPU kernel
    /// **Eliminates ALL CPU postprocessing**
    pub fn intt_xfield_fused_unscale_randomize(
        &self,
        data_arrays: &mut [Vec<XFieldElement>],
        twiddle_factors_inv: &[Vec<BFieldElement>],
        n_inv: BFieldElement,
        randomizers: &[Vec<XFieldElement>],
        offset_power_n: BFieldElement,
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if data_arrays.is_empty() {
            return Ok(());
        }

        let operation_start = std::time::Instant::now();
        let dev_ctx = self.select_device();
        let batch_size = data_arrays.len();

        eprintln!(
            "\n[GPU Batch XField INTT Fused Unscale+Randomize - {}] Processing {} arrays",
            phase_name,
            batch_size
        );

        let (prep, upload, kernel, download) = self.execute_batch_fused_unscale_randomize(
            data_arrays,
            twiddle_factors_inv,
            n_inv,
            randomizers,
            offset_power_n,
            &dev_ctx.intt_xfield_fused_unscale_randomize_fn,
        )?;

        let total_time = operation_start.elapsed();

        eprintln!(
            "  [Fused Batch] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms)",
            batch_size,
            total_time.as_secs_f64() * 1000.0,
            prep.as_secs_f64() * 1000.0,
            upload.as_secs_f64() * 1000.0,
            kernel.as_secs_f64() * 1000.0,
            download.as_secs_f64() * 1000.0
        );

        Ok(())
    }

    // =============================================================================
    // Public API: Table-Based Operations (Row-Major, Strided GPU Access)
    // =============================================================================

    /// Perform fused GPU coset scaling + NTT on row-major table using strided access
    /// **Maximum performance** - eliminates all column extraction overhead
    ///
    /// **Performance Benefits:**
    /// - No column extraction (works with table directly)
    /// - Single serialization pass (row-major → u64 array)
    /// - Strided GPU kernels access columns directly
    /// - Single upload/download cycle
    ///
    /// # Arguments
    /// * `table_data` - Mutable slice of BFieldElements in row-major layout
    /// * `num_rows` - Number of rows in the table
    /// * `num_columns` - Number of columns in the table
    /// * `offset` - Coset offset value
    /// * `twiddle_factors` - Precomputed twiddle factors for NTT
    /// * `phase_name` - Name for logging
    ///
    /// # Returns
    /// * `(Vec<Digest>, DeviceBuffer)` - Row digests and GPU-resident buffer
    pub fn ntt_bfield_fused_coset_table(
        &self,
        table_data: DeviceBuffer,
        num_rows: usize,
        num_columns: usize,
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(Vec<crate::tip5::Digest>, DeviceBuffer), Box<dyn std::error::Error>> {
        self.execute_fused_coset_ntt_table_bfield(
            table_data,
            num_rows,
            num_columns,
            offset,
            twiddle_factors,
            phase_name,
        )
    }


    pub fn poly_fill_table_bfield(
        &self,
        table_data: &mut [BFieldElement],
        num_rows: usize,
        num_columns: usize,
        poly_bfield: Vec<&[BFieldElement]>
    ) -> Result<DeviceBuffer,Box<dyn std::error::Error>> {
        self.execute_poly_fill_table_bfield(
            table_data,
            num_rows,
            num_columns,
            poly_bfield
        )
    }

    pub fn poly_fill_table_xfield(
        &self,
        table_data: &mut [XFieldElement],
        num_rows: usize,
        num_columns: usize,
        poly_xfield: Vec<&[XFieldElement]>
    ) -> Result<DeviceBuffer,Box<dyn std::error::Error>> {
        self.execute_poly_fill_table_xfield(
            table_data,
            num_rows,
            num_columns,
            poly_xfield
        )
    }

    /// Perform fused GPU coset scaling + NTT on row-major XField table using strided access
    ///
    /// # Returns
    /// * `(Vec<Digest>, DeviceBuffer)` - Row digests and GPU-resident buffer
    pub fn ntt_xfield_fused_coset_table(
        &self,
        table_data: DeviceBuffer,
        num_rows: usize,
        num_columns: usize,
        offset: BFieldElement,
        twiddle_factors: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(Vec<crate::tip5::Digest>, DeviceBuffer), Box<dyn std::error::Error>> {
        self.execute_fused_coset_ntt_table_xfield(
            table_data,
            num_rows,
            num_columns,
            offset,
            twiddle_factors,
            phase_name,
        )
    }

    /// Perform GPU INTT (inverse NTT) on row-major BField table using strided access
    /// **Maximum performance** - eliminates all column extraction overhead
    ///
    /// **Performance Benefits:**
    /// - No column extraction (works with table directly)
    /// - Zero-copy optimization (direct u64 pointer casting)
    /// - Strided GPU kernels access columns directly
    /// - Single upload/download cycle
    ///
    /// # Arguments
    /// * `table_data` - Mutable slice of BFieldElements in row-major layout
    /// * `num_rows` - Number of rows in the table (must be power of 2)
    /// * `num_columns` - Number of columns in the table
    /// * `twiddle_factors_inv` - Precomputed inverse twiddle factors for INTT
    /// * `phase_name` - Name for logging
    ///
    /// # Note
    /// Caller must still divide by len (unscaling) after INTT
    pub fn intt_bfield_table(
        &self,
        table_data: &mut [BFieldElement],
        num_rows: usize,
        num_columns: usize,
        twiddle_factors_inv: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.execute_intt_table_bfield(
            table_data,
            num_rows,
            num_columns,
            twiddle_factors_inv,
            phase_name,
        )
    }

    /// Perform GPU INTT (inverse NTT) on row-major XField table using strided access
    /// **Maximum performance** - eliminates all column extraction overhead
    ///
    /// **Performance Benefits:**
    /// - No column extraction (works with table directly)
    /// - Zero-copy optimization (direct u64 pointer casting)
    /// - Strided GPU kernels access columns directly
    /// - Single upload/download cycle
    ///
    /// # Arguments
    /// * `table_data` - Mutable slice of XFieldElements in row-major layout
    /// * `num_rows` - Number of rows in the table (must be power of 2)
    /// * `num_columns` - Number of columns in the table
    /// * `twiddle_factors_inv` - Precomputed inverse twiddle factors for INTT
    /// * `phase_name` - Name for logging
    ///
    /// # Note
    /// Caller must still divide by len (unscaling) after INTT
    pub fn intt_xfield_table(
        &self,
        table_data: &mut [XFieldElement],
        num_rows: usize,
        num_columns: usize,
        twiddle_factors_inv: &[Vec<BFieldElement>],
        phase_name: &str,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.execute_intt_table_xfield(
            table_data,
            num_rows,
            num_columns,
            twiddle_factors_inv,
            phase_name,
        )
    }
}
