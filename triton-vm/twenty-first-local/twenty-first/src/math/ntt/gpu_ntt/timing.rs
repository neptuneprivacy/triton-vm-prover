use super::types::GpuTimingStats;
use std::sync::{Mutex, OnceLock};

pub(crate) static GPU_TIMING_BFIELD_NTT: OnceLock<Mutex<GpuTimingStats>> = OnceLock::new();
pub(crate) static GPU_TIMING_BFIELD_INTT: OnceLock<Mutex<GpuTimingStats>> = OnceLock::new();
pub(crate) static GPU_TIMING_XFIELD_NTT: OnceLock<Mutex<GpuTimingStats>> = OnceLock::new();
pub(crate) static GPU_TIMING_XFIELD_INTT: OnceLock<Mutex<GpuTimingStats>> = OnceLock::new();

/// Print summary for chunked operations
pub(crate) fn print_chunked_summary(
    phase_name: &str,
    total_arrays: usize,
    total_time: std::time::Duration,
    chunk_times: &Mutex<Vec<(usize, usize, std::time::Duration, std::time::Duration, std::time::Duration, std::time::Duration, std::time::Duration)>>,
) {
    eprintln!("\n  [Summary - {}]", phase_name);
    eprintln!("    Total arrays:     {}", total_arrays);
    eprintln!("    Total time:       {:.3}s", total_time.as_secs_f64());
    if let Ok(times) = chunk_times.lock() {
        if !times.is_empty() {
            let total_prep: f64 = times
                .iter()
                .map(|(_, _, _, p, _, _, _)| p.as_secs_f64())
                .sum();
            let total_upload: f64 = times
                .iter()
                .map(|(_, _, _, _, u, _, _)| u.as_secs_f64())
                .sum();
            let total_kernel: f64 = times
                .iter()
                .map(|(_, _, _, _, _, k, _)| k.as_secs_f64())
                .sum();
            let total_download: f64 = times
                .iter()
                .map(|(_, _, _, _, _, _, d)| d.as_secs_f64())
                .sum();

            eprintln!("    Chunks processed: {}", times.len());
            eprintln!("    Time breakdown:");
            eprintln!(
                "      Prep:     {:.3}s ({:.1}%)",
                total_prep,
                100.0 * total_prep / total_time.as_secs_f64()
            );
            eprintln!(
                "      Upload:   {:.3}s ({:.1}%)",
                total_upload,
                100.0 * total_upload / total_time.as_secs_f64()
            );
            eprintln!(
                "      Kernel:   {:.3}s ({:.1}%)",
                total_kernel,
                100.0 * total_kernel / total_time.as_secs_f64()
            );
            eprintln!(
                "      Download: {:.3}s ({:.1}%)",
                total_download,
                100.0 * total_download / total_time.as_secs_f64()
            );
        }
    }
    eprintln!();
}

pub fn print_gpu_timing_summary() {
    eprintln!("\n=== GPU NTT TIMING SUMMARY ===\n");

    let format_bytes = |bytes: usize| -> String {
        if bytes >= 1_000_000_000 {
            format!("{:.2} GB", bytes as f64 / 1_000_000_000.0)
        } else if bytes >= 1_000_000 {
            format!("{:.2} MB", bytes as f64 / 1_000_000.0)
        } else if bytes >= 1_000 {
            format!("{:.2} KB", bytes as f64 / 1_000.0)
        } else {
            format!("{} B", bytes)
        }
    };

    let print_stats = |name: &str, stats_lock: &OnceLock<Mutex<GpuTimingStats>>| {
        if let Some(stats_mutex) = stats_lock.get() {
            if let Ok(stats) = stats_mutex.lock() {
                if stats.count > 0 {
                    let total_transfer_bytes =
                        stats.total_upload_bytes + stats.total_download_bytes;
                    eprintln!("{} ({} calls):", name, stats.count);
                    eprintln!("  Total time:        {:?}", stats.total_time);
                    eprintln!(
                        "  Avg per call:      {:?}",
                        stats.total_time / stats.count as u32
                    );
                    eprintln!(
                        "  Data transferred:  {} (up: {}, down: {})",
                        format_bytes(total_transfer_bytes),
                        format_bytes(stats.total_upload_bytes),
                        format_bytes(stats.total_download_bytes)
                    );
                    eprintln!(
                        "  Avg per call:      {} (up: {}, down: {})",
                        format_bytes(total_transfer_bytes / stats.count),
                        format_bytes(stats.total_upload_bytes / stats.count),
                        format_bytes(stats.total_download_bytes / stats.count)
                    );
                    eprintln!("  Time breakdown:");
                    eprintln!(
                        "    Prep:            {:?} ({:.1}%)",
                        stats.total_prep / stats.count as u32,
                        100.0 * stats.total_prep.as_secs_f64() / stats.total_time.as_secs_f64()
                    );
                    eprintln!(
                        "    Upload:          {:?} ({:.1}%)",
                        stats.total_upload / stats.count as u32,
                        100.0 * stats.total_upload.as_secs_f64() / stats.total_time.as_secs_f64()
                    );
                    eprintln!(
                        "    Kernel:          {:?} ({:.1}%)",
                        stats.total_kernel / stats.count as u32,
                        100.0 * stats.total_kernel.as_secs_f64() / stats.total_time.as_secs_f64()
                    );
                    eprintln!(
                        "    Download:        {:?} ({:.1}%)",
                        stats.total_download / stats.count as u32,
                        100.0 * stats.total_download.as_secs_f64() / stats.total_time.as_secs_f64()
                    );
                    eprintln!(
                        "    Postproc:        {:?} ({:.1}%)",
                        stats.total_postproc / stats.count as u32,
                        100.0 * stats.total_postproc.as_secs_f64() / stats.total_time.as_secs_f64()
                    );

                    // Calculate effective bandwidth
                    let upload_bandwidth =
                        stats.total_upload_bytes as f64 / stats.total_upload.as_secs_f64();
                    let download_bandwidth =
                        stats.total_download_bytes as f64 / stats.total_download.as_secs_f64();
                    eprintln!("  PCIe Bandwidth:");
                    eprintln!(
                        "    Upload:          {}/s",
                        format_bytes(upload_bandwidth as usize)
                    );
                    eprintln!(
                        "    Download:        {}/s\n",
                        format_bytes(download_bandwidth as usize)
                    );
                }
            }
        }
    };

    print_stats("BField NTT ", &GPU_TIMING_BFIELD_NTT);
    print_stats("BField INTT", &GPU_TIMING_BFIELD_INTT);
    print_stats("XField NTT ", &GPU_TIMING_XFIELD_NTT);
    print_stats("XField INTT", &GPU_TIMING_XFIELD_INTT);

    // Calculate total PCIe traffic
    let mut total_upload = 0usize;
    let mut total_download = 0usize;
    let mut total_calls = 0usize;

    for stats_lock in [
        &GPU_TIMING_BFIELD_NTT,
        &GPU_TIMING_BFIELD_INTT,
        &GPU_TIMING_XFIELD_NTT,
        &GPU_TIMING_XFIELD_INTT,
    ] {
        if let Some(stats_mutex) = stats_lock.get() {
            if let Ok(stats) = stats_mutex.lock() {
                total_upload += stats.total_upload_bytes;
                total_download += stats.total_download_bytes;
                total_calls += stats.count;
            }
        }
    }

    if total_calls > 0 {
        eprintln!("=== TOTAL PCIe TRAFFIC ===");
        eprintln!("Total calls:       {}", total_calls);
        eprintln!("Total uploaded:    {}", format_bytes(total_upload));
        eprintln!("Total downloaded:  {}", format_bytes(total_download));
        eprintln!(
            "Total transferred: {}",
            format_bytes(total_upload + total_download)
        );
        eprintln!(
            "Avg per call:      {}\n",
            format_bytes((total_upload + total_download) / total_calls)
        );
    }

    eprintln!("=== END GPU TIMING ===\n");
}
