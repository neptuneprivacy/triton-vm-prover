use super::super::types::{GpuError, GpuFieldElement, GpuNttContext};
use crate::math::b_field_element::BFieldElement;
use super::super::cuda_driver::CudaFunction;
use std::sync::Mutex;
use std::time::Instant;

impl GpuNttContext {
    /// Generic chunked operation handler for NTT/INTT on any field type
    /// Multi-stream version that enables concurrent kernel execution across chunks
    pub(crate) fn execute_chunked<T: GpuFieldElement>(
        &self,
        data_arrays: &mut [&mut [T]],
        twiddle_factors: &[Vec<BFieldElement>],
        kernel_fn: &CudaFunction,
        chunk_size: usize,
        phase_name: &str,
        operation_name: &str, // "NTT" or "INTT"
    ) -> Result<(), Box<dyn std::error::Error>> {
        if data_arrays.is_empty() {
            return Ok(());
        }

        let operation_start = Instant::now();
        let num_gpus = self.devices.len();

        // Calculate configured streams per GPU
        let streams_per_gpu = self.devices.first().map(|d| d.num_streams).unwrap_or(0);
        let total_configured_streams = streams_per_gpu * num_gpus;

        eprintln!(
            "\n[GPU Batch {} {} - {}] Processing {} {} arrays in chunks of {} across {} GPU(s)",
            T::field_name(),
            operation_name,
            phase_name,
            data_arrays.len(),
            T::field_name(),
            chunk_size,
            num_gpus
        );
        if streams_per_gpu > 1 {
            eprintln!("  Multi-stream mode:     {} streams per GPU ({} total)",
                streams_per_gpu, total_configured_streams);
        }

        // Track per-chunk timing with detailed breakdown
        let chunk_times = Mutex::new(Vec::new());

        // Create a thread pool with exactly num_gpus threads
        // Each thread can process multiple chunks on different streams
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_gpus)
            .build()
            .map_err(|e| {
                Box::new(GpuError(format!("Failed to create thread pool: {}", e)))
                    as Box<dyn std::error::Error>
            })?;

        // Process chunks using multi-stream async execution
        // Each thread creates its own streams for the GPU it's assigned to
        pool.install(|| {
            use std::cell::RefCell;
            use std::collections::HashMap;

            // Thread-local storage for streams per device
            thread_local! {
                static STREAM_CACHE: RefCell<HashMap<usize, Vec<super::super::cuda_driver::CudaStream>>> = RefCell::new(HashMap::new());
            }

            rayon::scope(|s| {
                for (chunk_idx, chunk) in data_arrays.chunks_mut(chunk_size).enumerate() {
                    let chunk_times_ref = &chunk_times;

                    // Assign chunk to specific GPU (round-robin)
                    let device_idx = chunk_idx % num_gpus;
                    let dev_ctx = &self.devices[device_idx];

                    s.spawn(move |_| {
                        let chunk_start = Instant::now();
                        let chunk_len = chunk.len();

                        // Get or create streams for this device in this thread
                        let result = if dev_ctx.num_streams > 0 {
                            STREAM_CACHE.with(|cache| {
                                let mut cache_map = cache.borrow_mut();
                                let streams = cache_map.entry(device_idx).or_insert_with(|| {
                                    // Create streams for this device (once per thread)
                                    let mut new_streams = Vec::new();
                                    for stream_idx in 0..dev_ctx.num_streams {
                                        match dev_ctx.device.create_stream() {
                                            Ok(stream) => new_streams.push(stream),
                                            Err(e) => {
                                                eprintln!("Warning: Failed to create stream {} for GPU {}: {}",
                                                    stream_idx, device_idx, e);
                                                break;
                                            }
                                        }
                                    }
                                    new_streams
                                });

                                if !streams.is_empty() {
                                    // Round-robin stream selection within this thread's streams
                                    let stream_idx = chunk_idx % streams.len();
                                    let stream = &streams[stream_idx];
                                    Self::execute_batch_on_stream(dev_ctx, stream, chunk, twiddle_factors, kernel_fn)
                                } else {
                                    // Fallback if stream creation failed
                                    self.execute_batch(chunk, twiddle_factors, kernel_fn)
                                }
                            })
                        } else {
                            // Streams disabled, use default execution
                            self.execute_batch(chunk, twiddle_factors, kernel_fn)
                        };

                        let chunk_time = chunk_start.elapsed();

                        match result {
                            Ok((prep, upload, kernel, download)) => {
                                let stream_info = if dev_ctx.num_streams > 0 {
                                    format!(" Stream {}", chunk_idx % dev_ctx.num_streams)
                                } else {
                                    String::new()
                                };
                                eprintln!("    [Chunk {} GPU {}{}] {} arrays: total={:.1}ms (prep={:.1}ms, upload={:.1}ms, kernel={:.1}ms, download={:.1}ms)",
                                    chunk_idx,
                                    device_idx,
                                    stream_info,
                                    chunk_len,
                                    chunk_time.as_secs_f64() * 1000.0,
                                    prep.as_secs_f64() * 1000.0,
                                    upload.as_secs_f64() * 1000.0,
                                    kernel.as_secs_f64() * 1000.0,
                                    download.as_secs_f64() * 1000.0
                                );
                                if let Ok(mut times) = chunk_times_ref.lock() {
                                    times.push((chunk_idx, chunk_len, chunk_time, prep, upload, kernel, download));
                                }
                            }
                            Err(e) => {
                                eprintln!("    [Chunk {} GPU {}] ERROR: GPU {} error: {}", chunk_idx, device_idx, operation_name, e);
                            }
                        }
                    });
                }
            });
        });

        let total_time = operation_start.elapsed();

        // Print summary
        super::super::timing::print_chunked_summary(phase_name, data_arrays.len(), total_time, &chunk_times);

        Ok(())
    }
}
