use std::simd::{LaneCount, Mask, Simd, SupportedLaneCount, cmp::SimdPartialOrd};

use std::num::NonZeroUsize;
use std::ops::MulAssign;
use std::sync::OnceLock;
use num_traits::ConstOne;

use super::b_field_element::BFieldElement;
use super::traits::Inverse;
use super::traits::ModPowU32;
use super::traits::PrimitiveRootOfUnity;
use super::traits::{FieldKind, FiniteField};
use super::x_field_element::XFieldElement;

// GPU NTT module (conditional compilation based on gpu feature)
#[cfg(feature = "gpu")]
pub mod gpu_ntt;

#[cfg(feature = "gpu")]
use gpu_ntt::{should_use_gpu, get_gpu_context};

// NTT call pattern tracking for batching analysis
#[cfg(feature = "gpu")]
use std::sync::Mutex;

#[cfg(feature = "gpu")]
static NTT_CALL_TRACKER: std::sync::OnceLock<Mutex<NttCallTracker>> = std::sync::OnceLock::new();

#[cfg(feature = "gpu")]
struct NttCallTracker {
    calls: Vec<NttCallRecord>,
    start_time: std::time::Instant,
}

#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
struct NttCallRecord {
    timestamp_us: u64,
    size: usize,
    is_base_field: bool,
    is_forward: bool, // true for NTT, false for INTT
}

#[cfg(feature = "gpu")]
impl NttCallTracker {
    fn new() -> Self {
        Self {
            calls: Vec::new(),
            start_time: std::time::Instant::now(),
        }
    }

    fn record(&mut self, size: usize, is_base_field: bool, is_forward: bool) {
        let timestamp_us = self.start_time.elapsed().as_micros() as u64;
        self.calls.push(NttCallRecord {
            timestamp_us,
            size,
            is_base_field,
            is_forward,
        });
    }

    fn analyze_and_print(&self) {
        if self.calls.is_empty() {
            return;
        }

        eprintln!("\n=== NTT CALL PATTERN ANALYSIS ===");
        eprintln!("Total calls: {}", self.calls.len());

        // Group by size and type
        let mut bfield_ntt_sizes = std::collections::HashMap::new();
        let mut bfield_intt_sizes = std::collections::HashMap::new();
        let mut xfield_ntt_sizes = std::collections::HashMap::new();
        let mut xfield_intt_sizes = std::collections::HashMap::new();

        for call in &self.calls {
            let counter = match (call.is_base_field, call.is_forward) {
                (true, true) => &mut bfield_ntt_sizes,
                (true, false) => &mut bfield_intt_sizes,
                (false, true) => &mut xfield_ntt_sizes,
                (false, false) => &mut xfield_intt_sizes,
            };
            *counter.entry(call.size).or_insert(0) += 1;
        }

        eprintln!("\nBField NTT calls by size:");
        for (size, count) in bfield_ntt_sizes.iter() {
            eprintln!("  2^{}: {} calls", size.trailing_zeros(), count);
        }

        eprintln!("\nBField INTT calls by size:");
        for (size, count) in bfield_intt_sizes.iter() {
            eprintln!("  2^{}: {} calls", size.trailing_zeros(), count);
        }

        eprintln!("\nXField NTT calls by size:");
        for (size, count) in xfield_ntt_sizes.iter() {
            eprintln!("  2^{}: {} calls", size.trailing_zeros(), count);
        }

        eprintln!("\nXField INTT calls by size:");
        for (size, count) in xfield_intt_sizes.iter() {
            eprintln!("  2^{}: {} calls", size.trailing_zeros(), count);
        }

        // Analyze consecutive same-size batches
        eprintln!("\n=== BATCHING OPPORTUNITIES ===");
        let mut i = 0;
        let mut total_batchable = 0;
        let mut max_batch_size = 0;
        while i < self.calls.len() {
            let current = &self.calls[i];
            let mut batch_size = 1;
            let mut j = i + 1;

            while j < self.calls.len() {
                let next = &self.calls[j];
                if next.size == current.size
                    && next.is_base_field == current.is_base_field
                    && next.is_forward == current.is_forward {
                    batch_size += 1;
                    j += 1;
                } else {
                    break;
                }
            }

            if batch_size > 1 {
                let time_span_us = if j < self.calls.len() {
                    self.calls[j - 1].timestamp_us - current.timestamp_us
                } else {
                    self.calls.last().unwrap().timestamp_us - current.timestamp_us
                };

                let field_type = if current.is_base_field { "BField" } else { "XField" };
                let op_type = if current.is_forward { "NTT" } else { "INTT" };

                eprintln!(
                    "Batch of {} consecutive {} {} (2^{}) operations over {}us (avg gap: {}us)",
                    batch_size,
                    field_type,
                    op_type,
                    current.size.trailing_zeros(),
                    time_span_us,
                    time_span_us / batch_size.max(2) as u64
                );

                total_batchable += batch_size;
                max_batch_size = max_batch_size.max(batch_size);
            }

            i = j;
        }

        eprintln!("\nTotal batchable operations: {} ({:.1}% of total)",
            total_batchable,
            100.0 * total_batchable as f64 / self.calls.len() as f64
        );
        eprintln!("Max consecutive batch size: {}", max_batch_size);

        // Print detailed sequence for large batches
        eprintln!("\n=== DETAILED CALL SEQUENCE (first 50 GPU-eligible calls) ===");
        let mut count = 0;
        for call in &self.calls {
            if call.size >= gpu_ntt::GPU_THRESHOLD {
                let field_type = if call.is_base_field { "BF" } else { "XF" };
                let op_type = if call.is_forward { "NTT " } else { "INTT" };
                eprintln!(
                    "NTT_PATTERN: {:8}us  {} {}  2^{}  (size: {})",
                    call.timestamp_us,
                    field_type,
                    op_type,
                    call.size.trailing_zeros(),
                    call.size
                );
                count += 1;
                if count >= 50 {
                    break;
                }
            }
        }
        eprintln!("=== END NTT PATTERN ===\n");
    }
}

#[cfg(feature = "gpu")]
fn track_ntt_call(size: usize, is_base_field: bool, is_forward: bool) {
    if size >= gpu_ntt::GPU_THRESHOLD {
        let tracker = NTT_CALL_TRACKER.get_or_init(|| Mutex::new(NttCallTracker::new()));
        if let Ok(mut tracker) = tracker.lock() {
            tracker.record(size, is_base_field, is_forward);
        }
    }
}

#[cfg(feature = "gpu")]
pub fn print_ntt_analysis() {
    if let Some(tracker) = NTT_CALL_TRACKER.get() {
        if let Ok(tracker) = tracker.lock() {
            tracker.analyze_and_print();
        }
    }
}

#[cfg(feature = "gpu")]
pub fn print_gpu_timing_summary() {
    gpu_ntt::print_gpu_timing_summary();
}

#[cfg(not(feature = "gpu"))]
pub fn print_gpu_timing_summary() {
    // No-op when GPU is not enabled
}

/// Perform chunked batched NTT on multiple BFieldElement arrays
///
/// This function processes arrays in chunks to manage GPU memory while
/// eliminating thread contention. Use GPU_BATCH_CHUNK_SIZE or a custom
/// chunk size based on available GPU memory.
///
/// # Example
/// ```ignore
/// use twenty_first::math::ntt::ntt_bfield_batch_chunked;
/// use twenty_first::math::ntt::gpu_ntt::GPU_BATCH_CHUNK_SIZE;
///
/// let mut columns: Vec<&mut [BFieldElement]> = ...; // Your columns
/// ntt_bfield_batch_chunked(&mut columns, GPU_BATCH_CHUNK_SIZE)?;
/// ```
#[cfg(feature = "gpu")]
pub fn ntt_bfield_batch_chunked(
    data_arrays: &mut [&mut [BFieldElement]],
    chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ctx) = get_gpu_context() {
        // Get twiddle factors
        if data_arrays.is_empty() {
            return Ok(());
        }
        let len = data_arrays[0].len();
        let log2_len = len.checked_ilog2().ok_or("Invalid array length")?;
        static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
            [const { OnceLock::new() }; NUM_DOMAINS];
        let twiddle_factors = ALL_TWIDDLE_FACTORS[log2_len as usize]
            .get_or_init(|| {
                let omega = BFieldElement::primitive_root_of_unity(len as u64).unwrap();
                twiddle_factors(len as u32, omega)
            });

        ctx.ntt_bfield_chunked(data_arrays, twiddle_factors, chunk_size, "BField NTT")
    } else {
        Err("GPU context not available".into())
    }
}

#[cfg(not(feature = "gpu"))]
pub fn ntt_bfield_batch_chunked(
    _data_arrays: &mut [&mut [BFieldElement]],
    _chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("GPU feature not enabled".into())
}

/// Perform chunked batched NTT on multiple XFieldElement arrays
#[cfg(feature = "gpu")]
pub fn ntt_xfield_batch_chunked(
    data_arrays: &mut [&mut [XFieldElement]],
    chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ctx) = get_gpu_context() {
        // Get twiddle factors
        if data_arrays.is_empty() {
            return Ok(());
        }
        let len = data_arrays[0].len();
        let log2_len = len.checked_ilog2().ok_or("Invalid array length")?;
        static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
            [const { OnceLock::new() }; NUM_DOMAINS];
        let twiddle_factors = ALL_TWIDDLE_FACTORS[log2_len as usize]
            .get_or_init(|| {
                let omega = BFieldElement::primitive_root_of_unity(len as u64).unwrap();
                twiddle_factors(len as u32, omega)
            });

        ctx.ntt_xfield_chunked(data_arrays, twiddle_factors, chunk_size, "XField NTT")
    } else {
        Err("GPU context not available".into())
    }
}

#[cfg(not(feature = "gpu"))]
pub fn ntt_xfield_batch_chunked(
    _data_arrays: &mut [&mut [XFieldElement]],
    _chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("GPU feature not enabled".into())
}

/// Perform chunked batched INTT on multiple BFieldElement arrays
#[cfg(feature = "gpu")]
pub fn intt_bfield_batch_chunked(
    data_arrays: &mut [&mut [BFieldElement]],
    chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ctx) = get_gpu_context() {
        if data_arrays.is_empty() {
            return Ok(());
        }
        let len = data_arrays[0].len();
        let log2_len = len.checked_ilog2().ok_or("Invalid array length")?;
        static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
            [const { OnceLock::new() }; NUM_DOMAINS];
        let twiddle_factors = ALL_TWIDDLE_FACTORS[log2_len as usize]
            .get_or_init(|| {
                let omega = BFieldElement::primitive_root_of_unity(len as u64).unwrap();
                twiddle_factors(len as u32, omega.inverse())
            });

        ctx.intt_bfield_chunked(data_arrays, twiddle_factors, chunk_size, "BField INTT")
    } else {
        Err("GPU context not available".into())
    }
}

/// Perform unchunked batched INTT on multiple BFieldElement arrays
/// This processes all arrays in a single GPU call, eliminating chunking overhead
#[cfg(feature = "gpu")]
pub fn intt_bfield_batch_unchunked(
    data_arrays: &mut [&mut [BFieldElement]],
    phase_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ctx) = get_gpu_context() {
        if data_arrays.is_empty() {
            return Ok(());
        }
        let len = data_arrays[0].len();
        let log2_len = len.checked_ilog2().ok_or("Invalid array length")?;
        static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
            [const { OnceLock::new() }; NUM_DOMAINS];
        let twiddle_factors = ALL_TWIDDLE_FACTORS[log2_len as usize]
            .get_or_init(|| {
                let omega = BFieldElement::primitive_root_of_unity(len as u64).unwrap();
                twiddle_factors(len as u32, omega.inverse())
            });

        ctx.intt_bfield_unchunked(data_arrays, twiddle_factors, phase_name)
    } else {
        Err("GPU context not available".into())
    }
}

#[cfg(not(feature = "gpu"))]
pub fn intt_bfield_batch_chunked(
    _data_arrays: &mut [&mut [BFieldElement]],
    _chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("GPU feature not enabled".into())
}

/// Perform chunked batched INTT on multiple XFieldElement arrays
#[cfg(feature = "gpu")]
pub fn intt_xfield_batch_chunked(
    data_arrays: &mut [&mut [XFieldElement]],
    chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ctx) = get_gpu_context() {
        if data_arrays.is_empty() {
            return Ok(());
        }
        let len = data_arrays[0].len();
        let log2_len = len.checked_ilog2().ok_or("Invalid array length")?;
        static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
            [const { OnceLock::new() }; NUM_DOMAINS];
        let twiddle_factors = ALL_TWIDDLE_FACTORS[log2_len as usize]
            .get_or_init(|| {
                let omega = BFieldElement::primitive_root_of_unity(len as u64).unwrap();
                twiddle_factors(len as u32, omega.inverse())
            });

        ctx.intt_xfield_chunked(data_arrays, twiddle_factors, chunk_size, "XField INTT")
    } else {
        Err("GPU context not available".into())
    }
}

/// Perform unchunked batched INTT on multiple XFieldElement arrays
/// This processes all arrays in a single GPU call, eliminating chunking overhead
#[cfg(feature = "gpu")]
pub fn intt_xfield_batch_unchunked(
    data_arrays: &mut [&mut [XFieldElement]],
    phase_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ctx) = get_gpu_context() {
        if data_arrays.is_empty() {
            return Ok(());
        }
        let len = data_arrays[0].len();
        let log2_len = len.checked_ilog2().ok_or("Invalid array length")?;
        static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
            [const { OnceLock::new() }; NUM_DOMAINS];
        let twiddle_factors = ALL_TWIDDLE_FACTORS[log2_len as usize]
            .get_or_init(|| {
                let omega = BFieldElement::primitive_root_of_unity(len as u64).unwrap();
                twiddle_factors(len as u32, omega.inverse())
            });

        ctx.intt_xfield_unchunked(data_arrays, twiddle_factors, phase_name)
    } else {
        Err("GPU context not available".into())
    }
}

/// Perform unchunked batched INTT with fused unscaling on multiple BFieldElement arrays
/// This combines INTT and unscaling (multiply by n_inv) in a single GPU kernel
/// **Eliminates CPU postprocessing overhead**
#[cfg(feature = "gpu")]
pub fn intt_bfield_fused_unscale(
    data_arrays: &mut [&mut [BFieldElement]],
    phase_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ctx) = get_gpu_context() {
        if data_arrays.is_empty() {
            return Ok(());
        }
        let len = data_arrays[0].len();
        let log2_len = len.checked_ilog2().ok_or("Invalid array length")?;
        static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
            [const { OnceLock::new() }; NUM_DOMAINS];
        let twiddle_factors = ALL_TWIDDLE_FACTORS[log2_len as usize]
            .get_or_init(|| {
                let omega = BFieldElement::primitive_root_of_unity(len as u64).unwrap();
                twiddle_factors(len as u32, omega.inverse())
            });

        // Compute n_inv for unscaling
        let n_inv = BFieldElement::from(len as u64).inverse_or_zero();

        ctx.intt_bfield_fused_unscale(data_arrays, twiddle_factors, n_inv, phase_name)
    } else {
        Err("GPU context not available".into())
    }
}

/// Perform unchunked batched INTT with fused unscaling on multiple XFieldElement arrays
/// This combines INTT and unscaling (multiply by n_inv) in a single GPU kernel
/// **Eliminates CPU postprocessing overhead**
#[cfg(feature = "gpu")]
pub fn intt_xfield_fused_unscale(
    data_arrays: &mut [&mut [XFieldElement]],
    phase_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ctx) = get_gpu_context() {
        if data_arrays.is_empty() {
            return Ok(());
        }
        let len = data_arrays[0].len();
        let log2_len = len.checked_ilog2().ok_or("Invalid array length")?;
        static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
            [const { OnceLock::new() }; NUM_DOMAINS];
        let twiddle_factors = ALL_TWIDDLE_FACTORS[log2_len as usize]
            .get_or_init(|| {
                let omega = BFieldElement::primitive_root_of_unity(len as u64).unwrap();
                twiddle_factors(len as u32, omega.inverse())
            });

        // Compute n_inv for unscaling
        let n_inv = BFieldElement::from(len as u64).inverse_or_zero();

        ctx.intt_xfield_fused_unscale(data_arrays, twiddle_factors, n_inv, phase_name)
    } else {
        Err("GPU context not available".into())
    }
}

/// Perform unchunked batched INTT with fused unscaling and randomizer on BFieldElement arrays
/// This combines INTT, unscaling, and randomizer addition in a single GPU kernel
/// **Eliminates ALL CPU postprocessing overhead**
#[cfg(feature = "gpu")]
pub fn intt_bfield_fused_unscale_randomize(
    data_arrays: &mut [Vec<BFieldElement>],
    randomizers: &[Vec<BFieldElement>],
    offset_power_n: BFieldElement,
    phase_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ctx) = get_gpu_context() {
        if data_arrays.is_empty() {
            return Ok(());
        }
        let len = data_arrays[0].len();
        let log2_len = len.checked_ilog2().ok_or("Invalid array length")?;
        static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
            [const { OnceLock::new() }; NUM_DOMAINS];
        let twiddle_factors = ALL_TWIDDLE_FACTORS[log2_len as usize]
            .get_or_init(|| {
                let omega = BFieldElement::primitive_root_of_unity(len as u64).unwrap();
                twiddle_factors(len as u32, omega.inverse())
            });

        // Compute n_inv for unscaling
        let n_inv = BFieldElement::from(len as u64).inverse_or_zero();

        ctx.intt_bfield_fused_unscale_randomize(
            data_arrays,
            twiddle_factors,
            n_inv,
            randomizers,
            offset_power_n,
            phase_name,
        )
    } else {
        Err("GPU context not available".into())
    }
}

/// Perform unchunked batched INTT with fused unscaling and randomizer on XFieldElement arrays
/// This combines INTT, unscaling, and randomizer addition in a single GPU kernel
/// **Eliminates ALL CPU postprocessing overhead**
#[cfg(feature = "gpu")]
pub fn intt_xfield_fused_unscale_randomize(
    data_arrays: &mut [Vec<XFieldElement>],
    randomizers: &[Vec<XFieldElement>],
    offset_power_n: BFieldElement,
    phase_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(ctx) = get_gpu_context() {
        if data_arrays.is_empty() {
            return Ok(());
        }
        let len = data_arrays[0].len();
        let log2_len = len.checked_ilog2().ok_or("Invalid array length")?;
        static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
            [const { OnceLock::new() }; NUM_DOMAINS];
        let twiddle_factors = ALL_TWIDDLE_FACTORS[log2_len as usize]
            .get_or_init(|| {
                let omega = BFieldElement::primitive_root_of_unity(len as u64).unwrap();
                twiddle_factors(len as u32, omega.inverse())
            });

        // Compute n_inv for unscaling
        let n_inv = BFieldElement::from(len as u64).inverse_or_zero();

        ctx.intt_xfield_fused_unscale_randomize(
            data_arrays,
            twiddle_factors,
            n_inv,
            randomizers,
            offset_power_n,
            phase_name,
        )
    } else {
        Err("GPU context not available".into())
    }
}

#[cfg(not(feature = "gpu"))]
pub fn intt_xfield_batch_chunked(
    _data_arrays: &mut [&mut [XFieldElement]],
    _chunk_size: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    Err("GPU feature not enabled".into())
}

/// The number of different domains over which this library can compute (i)NTT.
///
/// In particular, the maximum slice length for both [NTT][ntt] and [iNTT][intt]
/// supported by this library is 2^31. All domains of length some power of 2
/// smaller than this, plus the empty domain, are supported as well.
const NUM_DOMAINS: usize = 32;

/// ## Perform NTT on slices of prime-field elements
///
/// NTTs are Number Theoretic Transforms, which are Discrete Fourier Transforms
/// (DFTs) over finite fields. This implementation specifically aims at being
/// used to compute polynomial multiplication over finite fields. NTT reduces
/// the complexity of such multiplication.
///
/// For a brief introduction to the math, see:
///
/// * <https://cgyurgyik.github.io/posts/2021/04/brief-introduction-to-ntt/>
/// * <https://www.nayuki.io/page/number-theoretic-transform-integer-dft>
///
/// The implementation is adapted from:
///
/// <pre>
/// Speeding up the Number Theoretic Transform
/// for Faster Ideal Lattice-Based Cryptography
/// Longa and Naehrig
/// https://eprint.iacr.org/2016/504.pdf
/// </pre>
///
/// as well as inspired by <https://github.com/dusk-network/plonk>
///
/// The transform is performed in-place.
/// If called on an empty array, returns an empty array.
///
/// For the inverse, see [iNTT][self::intt].
///
/// # Panics
///
/// Panics if the length of the input slice is
/// - not a power of two
/// - larger than [`u32::MAX`]
pub fn ntt<FF>(x: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement> + FieldKind,
{
    static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    let slice_len = slice_len(x);
    let twiddle_factors = ALL_TWIDDLE_FACTORS[slice_len.checked_ilog2().unwrap_or(0) as usize]
        .get_or_init(|| {
            let omega = BFieldElement::primitive_root_of_unity(u64::from(slice_len)).unwrap();
            twiddle_factors(slice_len, omega)
        });

    // Try GPU acceleration if available and beneficial
    #[cfg(feature = "gpu")]
    {
        if should_use_gpu(x.len()) {
            track_ntt_call(x.len(), FF::IS_BASE, true);
            if FF::IS_BASE {
                // BFieldElement: cast and use GPU
                let x_bfield = unsafe {
                    std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut BFieldElement, x.len())
                };
                if let Some(ctx) = get_gpu_context() {
                    if ctx.ntt_bfield(x_bfield, twiddle_factors).is_ok() {
                        return; // GPU succeeded
                    }
                }
                // GPU failed, fall through to CPU
            } else {
                // XFieldElement: cast and use GPU
                let x_xfield = unsafe {
                    std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut XFieldElement, x.len())
                };
                if let Some(ctx) = get_gpu_context() {
                    if ctx.ntt_xfield(x_xfield, twiddle_factors).is_ok() {
                        return; // GPU succeeded
                    }
                }
                // GPU failed, fall through to CPU
            }
        }
    }

    // CPU fallback
    ntt_unchecked(x, twiddle_factors);
}

pub fn ntt_scalar<FF>(x: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement> + FieldKind,
{
    static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    let slice_len = slice_len(x);
    let twiddle_factors = ALL_TWIDDLE_FACTORS[slice_len.checked_ilog2().unwrap_or(0) as usize]
        .get_or_init(|| {
            let omega = BFieldElement::primitive_root_of_unity(u64::from(slice_len)).unwrap();
            twiddle_factors(slice_len, omega)
        });

    ntt_unchecked_scalar(x, twiddle_factors);
}

/// ## Perform INTT on slices of prime-field elements
///
/// INTT is the inverse [NTT][self::ntt], so abstractly,
/// *intt(values) = ntt(values) / n*.
///
/// This transform is performed in-place.
///
/// # Example
///
/// ```
/// # use twenty_first::prelude::*;
/// # use twenty_first::math::ntt::ntt;
/// # use twenty_first::math::ntt::intt;
/// let original_values = bfe_vec![0, 1, 1, 2, 3, 5, 8, 13];
/// let mut transformed_values = original_values.clone();
/// ntt(&mut transformed_values);
/// intt(&mut transformed_values);
/// assert_eq!(original_values, transformed_values);
/// ```
///
/// # Panics
///
/// Panics if the length of the input slice is
/// - not a power of two
/// - larger than [`u32::MAX`]
pub fn intt<FF>(x: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement> + FieldKind,
{
    static ALL_TWIDDLE_FACTORS: [OnceLock<Vec<Vec<BFieldElement>>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    let slice_len = slice_len(x);
    let twiddle_factors = ALL_TWIDDLE_FACTORS[slice_len.checked_ilog2().unwrap_or(0) as usize]
        .get_or_init(|| {
            let omega = BFieldElement::primitive_root_of_unity(u64::from(slice_len)).unwrap();
            twiddle_factors(slice_len, omega.inverse())
        });

    // Try GPU acceleration if available and beneficial
    #[cfg(feature = "gpu")]
    {
        if should_use_gpu(x.len()) {
            track_ntt_call(x.len(), FF::IS_BASE, false);
            if FF::IS_BASE {
                // BFieldElement: cast and use GPU
                let x_bfield = unsafe {
                    std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut BFieldElement, x.len())
                };
                if let Some(ctx) = get_gpu_context() {
                    if ctx.intt_bfield(x_bfield, twiddle_factors).is_ok() {
                        unscale(x);
                        return; // GPU succeeded
                    }
                }
                // GPU failed, fall through to CPU
            } else {
                // XFieldElement: cast and use GPU
                let x_xfield = unsafe {
                    std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut XFieldElement, x.len())
                };
                if let Some(ctx) = get_gpu_context() {
                    if ctx.intt_xfield(x_xfield, twiddle_factors).is_ok() {
                        unscale(x);
                        return; // GPU succeeded
                    }
                }
                // GPU failed, fall through to CPU
            }
        }
    }

    // CPU fallback
    ntt_unchecked(x, twiddle_factors);
    unscale(x);
}

/// Internal helper function to assert that the slice for [NTT][self::ntt] or
/// [iNTT][self::intt] is of a correct length.
///
/// # Panics
///
/// Panics if the slice length is
/// - neither 0 nor a power of two, or
/// - larger than [`u32::MAX`].
fn slice_len<FF>(x: &[FF]) -> u32 {
    let slice_len = u32::try_from(x.len()).expect("slice should be no longer than u32::MAX");
    assert!(slice_len == 0 || slice_len.is_power_of_two());

    slice_len
}


#[inline(always)]
pub fn umulh_sq<const N: usize>(x: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let xl = x & Simd::<u64, N>::splat(0xFFFFFFFF);
    let xh = x >> 32;

    let t00 = xl * xl;
    let t01_10 = xh * xl;
    let mut t11 = xh * xh;

    t11 += (t01_10 >> 32) << 1;
    let tc = ((t01_10 & Simd::<u64, N>::splat(0xFFFFFFFF)) << 1) + (t00 >> 32);
    t11 += tc >> 32;
    t11
}

#[inline(always)]
pub fn umulh<const N: usize>(x: Simd<u64, N>, y: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let xl = x & Simd::<u64, N>::splat(0xFFFFFFFF);
    let xh = x >> 32;
    let yl = y & Simd::<u64, N>::splat(0xFFFFFFFF);
    let yh = y >> 32;

    let t00 = xl * yl;
    let t01 = xh * yl;
    let t10 = xl * yh;
    let mut t11 = xh * yh;

    t11 += (t10 >> 32) + (t01 >> 32);
    let tc = (t10 & Simd::<u64, N>::splat(0xFFFFFFFF))
        + (t01 & Simd::<u64, N>::splat(0xFFFFFFFF))
        + (t00 >> 32);
    t11 += tc >> 32;
    t11
}

#[inline(always)]
pub fn check_overflowing_add<const N: usize>(a: Simd<u64, N>, b: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let msk = b.simd_gt(Simd::<u64, N>::splat(u64::MAX) - a);

    msk.select(Simd::<u64, N>::splat(1), Simd::<u64, N>::splat(0))
}

#[inline(always)]
pub fn check_overflowing_sub<const N: usize>(a: Simd<u64, N>, b: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let msk = b.simd_gt(a);

    msk.select(Simd::<u64, N>::splat(1), Simd::<u64, N>::splat(0))
}

#[inline(always)]
pub fn check_overflowing_sub_msk<const N: usize>(a: Simd<u64, N>, b: Simd<u64, N>) -> Mask<i64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    b.simd_gt(a)
}

#[inline(always)]
pub fn bfield_mul<const N: usize>(x: Simd<u64, N>, y: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let xl = x * y;
    let xh = umulh(x, y);

    let a = xl + (xl << 32);
    let e = check_overflowing_add(xl, xl << 32);

    let b = a - (a >> 32) - e;

    let r = xh - b;
    let c = check_overflowing_sub(xh, b);

    r - (Simd::<u64, N>::splat(1 + !18446744069414584321) * c)
}

#[inline(always)]
pub fn bfield_add<const N: usize>(x: Simd<u64, N>, y: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let x1 = x - (Simd::<u64, N>::splat(BFieldElement::P) - y);
    let c1 = check_overflowing_sub_msk(x, Simd::<u64, N>::splat(BFieldElement::P) - y);

    c1.select(x1 + Simd::<u64, N>::splat(BFieldElement::P), x1)
}

#[inline(always)]
pub fn bfield_sub<const N: usize>(x: Simd<u64, N>, y: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    // result = x - y (wrapping)
    let res = x - y;

    // detect borrow: if x < y then we underflowed
    let borrow = check_overflowing_sub_msk(x, y);

    // if borrow, add modulus P back
    borrow.select(res + Simd::<u64, N>::splat(BFieldElement::P), res)
}

/// Internal helper function for [NTT][self::ntt] and [iNTT][self::intt].
///
/// Assumes that
/// - the passed-in twiddle factors are correct for the length of the slice,
/// - the length of the slice is a power of two, and
/// - the length of the slice is smaller than [`u32::MAX`].
///
/// If any of the above assumptions are violated, the function may panic or
/// produce incorrect results.
#[expect(clippy::many_single_char_names)]
#[inline]
fn ntt_unchecked<FF>(x: &mut [FF], twiddle_factors: &[Vec<BFieldElement>])
where
    FF: FiniteField + MulAssign<BFieldElement> + FieldKind,
{
    // It is possible to pre-compute all swap indices at compile time, but that
    // would incur a big compile time penalty.
    //
    // The type here is quite the mouthful. A short explainer is in order.
    // - `OnceLock` is used to ensure that the swap indices are computed only
    //   once per slice length, and that the computation is thread-safe. This
    //   cache significantly speeds up the computation.
    // - For the remaining `Vec<Option<NonZeroUsize>>`, see the documentation of
    //   `swap_indices`.

    static ALL_SWAP_INDICES: [OnceLock<Vec<Option<NonZeroUsize>>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    let slice_len = x.len();
    let log2_slice_len = slice_len.checked_ilog2().unwrap_or(0);
    let swap_indices =
        ALL_SWAP_INDICES[log2_slice_len as usize].get_or_init(|| swap_indices(slice_len));
    debug_assert_eq!(swap_indices.len(), slice_len);

    let _log2_slice_len = log2_slice_len; // Keep variable for potential future use

    // This is the most performant version of the code I can produce.
    // Things I've tried:
    // - swap_indices: Vec<(usize, usize)>, where each element in the vector
    //   is a pair of indices to swap. This vector is shorter than x, and the
    //   body of the loop is branch-free (at least on our end) so it seems like
    //   it should be faster, but I couldn't measure any difference.
    // - swap_indices: Vec<usize>, where the element equals its index for those
    //   indices that do not need to be swapped. Since core::slice::swap
    //   guarantees that elements don't get swapped if its two arguments are
    //   equal, the behavior is unchanged and removes the branching in the loop
    //   body, but resulted in a slowdown.
    for (k, maybe_rev_k) in swap_indices.iter().enumerate() {
        if let Some(rev_k) = maybe_rev_k {
            x.swap(k, rev_k.get());
        }
    }

    let slice_len = slice_len as u32;
    let mut m = 1;

    //let x_u64 = as_u64_mut_ptr(x);


    for twiddles in twiddle_factors {
        let mut k = 0;
        if m >= 8 && FF::IS_BASE {
            //simd
            while k < slice_len {
                for j in (0..m).step_by(8) {
                    let idx1 = (k + j) as usize;
                    let idx2 = (k + j + m) as usize;

                    let u_x: Simd<u64, 8> = unsafe {
                        std::ptr::read_unaligned(x.as_ptr().add(idx1 as usize) as *const _)
                    };
                    let v_x: Simd<u64, 8> = unsafe {
                        std::ptr::read_unaligned(x.as_ptr().add(idx2 as usize) as *const _)
                    };

                    let twd: Simd<u64, 8> = unsafe {
                        std::ptr::read_unaligned(twiddles.as_ptr().add(j as usize) as *const _)
                    };

                    let s = bfield_mul(v_x, twd);

                    let xx: &mut [u64] = unsafe {
                        std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u64, x.len())
                    };
                    xx[idx1..idx1 + 8].copy_from_slice(&bfield_add(u_x, s).to_array());
                    xx[idx2..idx2 + 8].copy_from_slice(&bfield_sub(u_x, s).to_array());
                }

                k += 2 * m;
            }
        } else if m >= 4 && FF::IS_BASE {
            //simd
            while k < slice_len {
                for j in (0..m).step_by(4) {
                    let idx1 = (k + j) as usize;
                    let idx2 = (k + j + m) as usize;

                    let u_x: Simd<u64, 4> = unsafe {
                        std::ptr::read_unaligned(x.as_ptr().add(idx1 as usize) as *const _)
                    };
                    let v_x: Simd<u64, 4> = unsafe {
                        std::ptr::read_unaligned(x.as_ptr().add(idx2 as usize) as *const _)
                    };

                    let twd: Simd<u64, 4> = unsafe {
                        std::ptr::read_unaligned(twiddles.as_ptr().add(j as usize) as *const _)
                    };

                    let s = bfield_mul(v_x, twd);

                    let xx: &mut [u64] = unsafe {
                        std::slice::from_raw_parts_mut(x.as_mut_ptr() as *mut u64, x.len())
                    };
                    xx[idx1..idx1 + 4].copy_from_slice(&bfield_add(u_x, s).to_array());
                    xx[idx2..idx2 + 4].copy_from_slice(&bfield_sub(u_x, s).to_array());


                    

                    //let u = x[idx1];
                    //let mut v = x[idx2];
                    //v *= twiddles[j as usize];

                    //x[idx1] = u + v;
                    //x[idx2] = u - v;
                }

                k += 2 * m;
            }
        } else {
            //scalar
            while k < slice_len {
                for j in (0..m).step_by(1) {
                    let idx1 = (k + j) as usize;
                    let idx2 = (k + j + m) as usize;
                    let u = x[idx1];
                    let mut v = x[idx2];
                    v *= twiddles[j as usize];
                    x[idx1] = u + v;
                    x[idx2] = u - v;
                }

                k += 2 * m;
            }
        }

        m *= 2;
    }
}



#[expect(clippy::many_single_char_names)]
#[inline]
fn ntt_unchecked_scalar<FF>(x: &mut [FF], twiddle_factors: &[Vec<BFieldElement>])
where
    FF: FiniteField + MulAssign<BFieldElement> + FieldKind,
{
    // It is possible to pre-compute all swap indices at compile time, but that
    // would incur a big compile time penalty.
    //
    // The type here is quite the mouthful. A short explainer is in order.
    // - `OnceLock` is used to ensure that the swap indices are computed only
    //   once per slice length, and that the computation is thread-safe. This
    //   cache significantly speeds up the computation.
    // - For the remaining `Vec<Option<NonZeroUsize>>`, see the documentation of
    //   `swap_indices`.

    static ALL_SWAP_INDICES: [OnceLock<Vec<Option<NonZeroUsize>>>; NUM_DOMAINS] =
        [const { OnceLock::new() }; NUM_DOMAINS];

    let slice_len = x.len();
    let log2_slice_len = slice_len.checked_ilog2().unwrap_or(0);
    let swap_indices =
        ALL_SWAP_INDICES[log2_slice_len as usize].get_or_init(|| swap_indices(slice_len));
    debug_assert_eq!(swap_indices.len(), slice_len);

    // This is the most performant version of the code I can produce.
    // Things I've tried:
    // - swap_indices: Vec<(usize, usize)>, where each element in the vector
    //   is a pair of indices to swap. This vector is shorter than x, and the
    //   body of the loop is branch-free (at least on our end) so it seems like
    //   it should be faster, but I couldn't measure any difference.
    // - swap_indices: Vec<usize>, where the element equals its index for those
    //   indices that do not need to be swapped. Since core::slice::swap
    //   guarantees that elements don't get swapped if its two arguments are
    //   equal, the behavior is unchanged and removes the branching in the loop
    //   body, but resulted in a slowdown.
    for (k, maybe_rev_k) in swap_indices.iter().enumerate() {
        if let Some(rev_k) = maybe_rev_k {
            x.swap(k, rev_k.get());
        }
    }

    let slice_len = slice_len as u32;
    let mut m = 1;


    for twiddles in twiddle_factors {
        let mut k = 0;

        //scalar
        while k < slice_len {
            for j in (0..m).step_by(1) {
                let idx1 = (k + j) as usize;
                let idx2 = (k + j + m) as usize;
                let u = x[idx1];
                let mut v = x[idx2];
                v *= twiddles[j as usize];
                x[idx1] = u + v;
                x[idx2] = u - v;
            }

            k += 2 * m;
        }
        m *= 2;
    }
}

/// Unscale the array by multiplying every element by the
/// inverse of the array's length. Useful for following up intt.
#[inline]
fn unscale<FF>(array: &mut [FF])
where
    FF: FiniteField + MulAssign<BFieldElement> + FieldKind,
{
    let n_inv = BFieldElement::from(array.len()).inverse_or_zero();
    for elem in array {
        *elem *= n_inv;
    }
}

/// A list of options, where the `i`-th element is `Some(j)` if and only if
/// `i` and `j` are indices that should be swapped in the NTT.
//
// `Option<NonZeroUsize>` makes use of niche optimization, which means that
// the return value takes the same amount of space as a `Vec<usize>`, but
// allows us to use `None` as a marker for the case where no swap is needed.
//
// Only public for benchmarking purposes.
#[doc(hidden)]
pub fn swap_indices(len: usize) -> Vec<Option<NonZeroUsize>> {
    #[inline(always)]
    const fn bitreverse(mut k: u32, log2_n: u32) -> u32 {
        k = ((k & 0x55555555) << 1) | ((k & 0xaaaaaaaa) >> 1);
        k = ((k & 0x33333333) << 2) | ((k & 0xcccccccc) >> 2);
        k = ((k & 0x0f0f0f0f) << 4) | ((k & 0xf0f0f0f0) >> 4);
        k = ((k & 0x00ff00ff) << 8) | ((k & 0xff00ff00) >> 8);
        k = k.rotate_right(16);
        k >> ((32 - log2_n) & 0x1f)
    }

    // For large enough `len`, the computation benefits from parallelization.
    // However, if NTT is also being called from within a rayon-parallel
    // context, the potential parallelization here can lead to a deadlock.
    // The relevant issue is <https://github.com/rayon-rs/rayon/issues/592>.
    //
    // As a short summary, consider the following scenario.
    // 1. Some task on some rayon thread calls NTT's OnceLock::get_or_init.
    // 2. The initialization task, i.e., execution of swap_indices, is also done
    //    in parallel. Some of that work is stolen by other rayon threads.
    // 3. The task that originally called OnceLock::get_or_init finishes its
    //    work and starts looking for more work.
    // 4. It steals part of the _outer_ parallelization effort, which just so
    //    happens to be a call to an NTT with the same slice length.
    // 5. It calls OnceLock::get_or_init on the _same_ OnceLock.
    // 6. This, implicitly, is re-entrant initialization of the OnceLock, which
    //    is documented as resulting in a deadlock.
    //
    // While parallel initialization would benefit runtime, a deadlock clearly
    // does not. Because it's a reasonable assumption that NTT is being called
    // in a rayon-parallelized context, we avoid parallelization here for now.
    // Potential ways forward are:
    // - use <https://github.com/rayon-rs/rayon/pull/1175> once that is merged
    // - use a parallelization approach that does not perform or allow
    //   work-stealing, like <https://crates.io/crates/chili> (though this
    //   particular crate might not be the best fit – do some research first 🙂)
    let log_2_len = len.checked_ilog2().unwrap_or(0);
    (0..len)
        .map(|k| {
            let rev_k = bitreverse(k as u32, log_2_len);

            // 0 >= bitreverse(0, log_2_len) == 0 => unwrap is fine
            ((k as u32) < rev_k).then(|| NonZeroUsize::new(rev_k as usize).unwrap())
        })
        .collect()
}

/// Internal helper function to (pre-) compute the twiddle factors for use in
/// [NTT][ntt] and [iNTT][intt].
///
/// Assumes that the given root of unity and the slice length match.
//
// The runtime of this function, especially when seen in the larger context,
// could potentially still be improved. Since this function is run at most twice
// per slice length (once for NTT, once for iNTT), any runtime savings are
// amortized pretty quickly. Saving RAM might be more interesting.
//
// One difference to the Longa+Naehrig paper [0] is the return value of
// Vec<Vec<_>> instead of a single Vec<_>.
// Also note that the twiddle factors for smaller domains are a subset of those
// for larger domains. In order to save both space and time, what can be shared,
// should be shared. I think the engineering work to get this working with the
// current OnceLock-based lazy-initialization is non-trivial, considering that
// OnceLocks must not be re-entrantly initialized. I could be wrong and it's
// actually easy.
//
// [0] <https://eprint.iacr.org/2016/504.pdf>
//
// Only public for benchmarking purposes.
#[doc(hidden)]
pub fn twiddle_factors(slice_len: u32, root_of_unity: BFieldElement) -> Vec<Vec<BFieldElement>> {
    // For an explanation of why this is not parallelized, see `swap_indices`.
    (0..slice_len.checked_ilog2().unwrap_or(0))
        .map(|i| {
            let m = 1 << i;
            let exponent = slice_len / (2 * m);
            let w_m = root_of_unity.mod_pow_u32(exponent);
            let mut w_powers = vec![BFieldElement::ONE; m as usize];
            for j in 1..m as usize {
                w_powers[j] = w_powers[j - 1] * w_m;
            }

            w_powers
        })
        .collect()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use itertools::Itertools;
    use num_traits::ConstZero;
    use num_traits::Zero;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;
    use crate::math::other::random_elements;
    use crate::math::traits::PrimitiveRootOfUnity;
    use crate::math::x_field_element::EXTENSION_DEGREE;
    use crate::prelude::*;
    use crate::xfe;

    #[test]
    fn chu_ntt_b_field_prop_test() {
        for log_2_n in 1..10 {
            let n = 1 << log_2_n;
            for _ in 0..10 {
                let mut values = random_elements(n);
                let original_values = values.clone();
                ntt::<BFieldElement>(&mut values);
                assert_ne!(original_values, values);
                intt::<BFieldElement>(&mut values);
                assert_eq!(original_values, values);

                values[0] = bfe!(BFieldElement::MAX);
                let original_values_with_max_element = values.clone();
                ntt::<BFieldElement>(&mut values);
                assert_ne!(original_values, values);
                intt::<BFieldElement>(&mut values);
                assert_eq!(original_values_with_max_element, values);
            }
        }
    }

    #[test]
    fn chu_ntt_x_field_prop_test() {
        for log_2_n in 1..10 {
            let n = 1 << log_2_n;
            for _ in 0..10 {
                let mut values = random_elements(n);
                let original_values = values.clone();
                ntt::<XFieldElement>(&mut values);
                assert_ne!(original_values, values);
                intt::<XFieldElement>(&mut values);
                assert_eq!(original_values, values);

                // Verify that we are not just operating in the B-field
                // statistically this should hold except one out of
                // ~ (2^64)^2 times this test runs
                assert!(
                    !original_values[1].coefficients[1].is_zero()
                        || !original_values[1].coefficients[2].is_zero()
                );

                values[0] = xfe!([BFieldElement::MAX; EXTENSION_DEGREE]);
                let original_values_with_max_element = values.clone();
                ntt::<XFieldElement>(&mut values);
                assert_ne!(original_values, values);
                intt::<XFieldElement>(&mut values);
                assert_eq!(original_values_with_max_element, values);
            }
        }
    }

    #[test]
    fn xfield_basic_test_of_chu_ntt() {
        let mut input_output = vec![
            XFieldElement::new_const(BFieldElement::ONE),
            XFieldElement::new_const(BFieldElement::ZERO),
            XFieldElement::new_const(BFieldElement::ZERO),
            XFieldElement::new_const(BFieldElement::ZERO),
        ];
        let original_input = input_output.clone();
        let expected = vec![
            XFieldElement::new_const(BFieldElement::ONE),
            XFieldElement::new_const(BFieldElement::ONE),
            XFieldElement::new_const(BFieldElement::ONE),
            XFieldElement::new_const(BFieldElement::ONE),
        ];

        println!("input_output = {input_output:?}");
        ntt::<XFieldElement>(&mut input_output);
        assert_eq!(expected, input_output);
        println!("input_output = {input_output:?}");

        // Verify that INTT(NTT(x)) = x
        intt::<XFieldElement>(&mut input_output);
        assert_eq!(original_input, input_output);
    }

    #[test]
    fn bfield_basic_test_of_chu_ntt() {
        let mut input_output = vec![
            BFieldElement::new(1),
            BFieldElement::new(4),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ];
        let original_input = input_output.clone();
        let expected = vec![
            BFieldElement::new(5),
            BFieldElement::new(1125899906842625),
            BFieldElement::new(18446744069414584318),
            BFieldElement::new(18445618169507741698),
        ];

        ntt::<BFieldElement>(&mut input_output);
        assert_eq!(expected, input_output);

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output);
        assert_eq!(original_input, input_output);
    }

    #[test]
    fn bfield_max_value_test_of_chu_ntt() {
        let mut input_output = vec![
            BFieldElement::new(BFieldElement::MAX),
            BFieldElement::new(0),
            BFieldElement::new(0),
            BFieldElement::new(0),
        ];
        let original_input = input_output.clone();
        let expected = vec![
            BFieldElement::new(BFieldElement::MAX),
            BFieldElement::new(BFieldElement::MAX),
            BFieldElement::new(BFieldElement::MAX),
            BFieldElement::new(BFieldElement::MAX),
        ];

        ntt::<BFieldElement>(&mut input_output);
        assert_eq!(expected, input_output);

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output);
        assert_eq!(original_input, input_output);
    }

    #[test]
    fn ntt_on_empty_input() {
        let mut input_output = vec![];
        let original_input = input_output.clone();

        ntt::<BFieldElement>(&mut input_output);
        assert_eq!(0, input_output.len());

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output);
        assert_eq!(original_input, input_output);
    }

    #[proptest]
    fn ntt_on_input_of_length_one(bfe: BFieldElement) {
        let mut test_vector = vec![bfe];
        ntt(&mut test_vector);
        assert_eq!(vec![bfe], test_vector);
    }

    #[proptest(cases = 10)]
    fn ntt_then_intt_is_identity_operation(
        #[strategy((0_usize..18).prop_map(|l| 1 << l))] _vector_length: usize,
        #[strategy(vec(arb(), #_vector_length))] mut input: Vec<BFieldElement>,
    ) {
        let original_input = input.clone();
        ntt::<BFieldElement>(&mut input);
        intt::<BFieldElement>(&mut input);
        assert_eq!(original_input, input);
    }

    #[test]
    fn b_field_ntt_with_length_32() {
        let mut input_output = bfe_vec![
            1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0, 0, 0, 0, 1, 4, 0, 0, 0,
            0, 0, 0,
        ];
        let original_input = input_output.clone();
        ntt::<BFieldElement>(&mut input_output);
        // let actual_output = ntt(&mut input_output, &omega, 5);
        println!("actual_output = {input_output:?}");
        let expected = bfe_vec![
            20,
            0,
            0,
            0,
            18446744069146148869_u64,
            0,
            0,
            0,
            4503599627370500_u64,
            0,
            0,
            0,
            18446726477228544005_u64,
            0,
            0,
            0,
            18446744069414584309_u64,
            0,
            0,
            0,
            268435460,
            0,
            0,
            0,
            18442240469787213829_u64,
            0,
            0,
            0,
            17592186040324_u64,
            0,
            0,
            0,
        ];
        assert_eq!(expected, input_output);

        // Verify that INTT(NTT(x)) = x
        intt::<BFieldElement>(&mut input_output);
        assert_eq!(original_input, input_output);
    }

    #[test]
    fn test_compare_ntt_to_eval() {
        for log_size in 1..10 {
            let size = 1 << log_size;
            let mut coefficients = random_elements(size);
            let polynomial = Polynomial::new(coefficients.clone());

            let omega = BFieldElement::primitive_root_of_unity(size.try_into().unwrap()).unwrap();
            ntt(&mut coefficients);

            let evals = (0..size)
                .map(|i| omega.mod_pow(i.try_into().unwrap()))
                .map(|p| polynomial.evaluate_in_same_field(p))
                .collect_vec();

            assert_eq!(evals, coefficients);
        }
    }

    #[test]
    fn swap_indices_can_be_computed() {
        // exponential growth is powerful; cap the number of domains
        for log_size in 0..NUM_DOMAINS - 2 {
            swap_indices(1 << log_size);
        }
    }

    #[test]
    fn twiddle_factors_can_be_computed() {
        // exponential growth is powerful; cap the number of domains
        for log_size in 0..NUM_DOMAINS - 5 {
            let size = 1 << log_size;
            let root = BFieldElement::primitive_root_of_unity(size.into()).unwrap();
            twiddle_factors(size, root);
        }
    }
}
