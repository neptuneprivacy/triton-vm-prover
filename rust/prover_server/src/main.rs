//! GPU Prover Server binary
//!
//! Usage:
//!   prover-server [OPTIONS]
//!
//! Options:
//!   --tcp <ADDR>      TCP address to listen on (default: 127.0.0.1:5555)
//!   --unix <PATH>     Unix socket path to listen on
//!   --max-jobs <N>    Maximum concurrent jobs (default: matches num-gpus)
//!   --num-gpus <N>    Number of GPU devices available (default: 2, or TRITON_GPU_COUNT env var)
//!   --omp-threads <N> OpenMP thread count (OMP_NUM_THREADS, default: from env or system default)
//!   --omp-init <0|1>  Enable/disable OpenMP init parallelization (TRITON_OMP_INIT, default: from env or disabled)

use std::env;
use tracing::info;
use tracing_subscriber::{fmt, prelude::*, EnvFilter};

use prover_server::{ProverServer, ServerConfig};

fn parse_args() -> ServerConfig {
    let args: Vec<String> = env::args().collect();
    let mut config = ServerConfig::default();
    
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tcp" => {
                i += 1;
                if i < args.len() {
                    config.tcp_addr = Some(args[i].clone());
                }
            }
            "--unix" => {
                i += 1;
                if i < args.len() {
                    config.unix_socket = Some(args[i].clone());
                    config.tcp_addr = None; // Prefer Unix socket if specified
                }
            }
            "--max-jobs" => {
                i += 1;
                if i < args.len() {
                    config.max_concurrent_jobs = args[i].parse().unwrap_or(1);
                }
            }
            "--num-gpus" => {
                i += 1;
                if i < args.len() {
                    config.num_gpus = args[i].parse().unwrap_or(2);
                    // Update max_concurrent_jobs to match num_gpus if not explicitly set
                    if config.max_concurrent_jobs == 1 {
                        config.max_concurrent_jobs = config.num_gpus;
                    }
                }
            }
            "--omp-threads" => {
                i += 1;
                if i < args.len() {
                    config.omp_num_threads = args[i].parse().ok();
                }
            }
            "--omp-init" => {
                i += 1;
                if i < args.len() {
                    config.triton_omp_init = match args[i].as_str() {
                        "1" | "true" | "yes" | "enabled" => Some(true),
                        "0" | "false" | "no" | "disabled" => Some(false),
                        _ => None,
                    };
                }
            }
            "--help" | "-h" => {
                println!("GPU Prover Server for Neptune/XNT-Core Integration");
                println!();
                println!("Usage: prover-server [OPTIONS]");
                println!();
                println!("Options:");
                println!("  --tcp <ADDR>      TCP address to listen on (default: 127.0.0.1:5555)");
                println!("  --unix <PATH>     Unix socket path to listen on");
                println!("  --max-jobs <N>    Maximum concurrent jobs (default: matches num-gpus)");
                println!("  --num-gpus <N>    Number of GPU devices available (default: 2, or TRITON_GPU_COUNT env var)");
                println!("  --omp-threads <N> OpenMP thread count (OMP_NUM_THREADS, default: from env or system default)");
                println!("  --omp-init <0|1>  Enable/disable OpenMP init parallelization (TRITON_OMP_INIT, default: from env or disabled)");
                println!("  --help, -h        Show this help message");
                std::process::exit(0);
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }
    
    config
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::registry()
        .with(fmt::layer())
        .with(EnvFilter::from_default_env().add_directive("prover_server=info".parse()?))
        .init();
    
    let config = parse_args();
    
    info!("Starting GPU Prover Server");
    info!("  TCP: {:?}", config.tcp_addr);
    info!("  Unix: {:?}", config.unix_socket);
    info!("  Number of GPUs: {}", config.num_gpus);
    info!("  Max concurrent jobs: {} (one per GPU)", config.max_concurrent_jobs);
    if let Some(threads) = config.omp_num_threads {
        info!("  OMP_NUM_THREADS: {}", threads);
    }
    if let Some(init) = config.triton_omp_init {
        info!("  TRITON_OMP_INIT: {}", if init { "enabled" } else { "disabled" });
    }
    
    let server = ProverServer::new(config);
    server.run().await?;
    
    info!("Server shutdown complete");
    Ok(())
}

