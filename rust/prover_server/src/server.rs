//! Socket server for GPU prover

use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::Arc;
use tokio::io::{BufReader, BufWriter};
use tokio::net::{TcpListener, UnixListener};
use tokio::sync::Semaphore;
use tracing::{error, info, warn};

use crate::protocol::{ProverRequest, ProverResponse};
use crate::prover::{prove_request_async, ProveOutcome};

/// Server configuration
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// TCP address to listen on (e.g., "127.0.0.1:5555")
    pub tcp_addr: Option<String>,
    
    /// Unix socket path to listen on
    pub unix_socket: Option<String>,
    
    /// Maximum concurrent jobs (one per GPU device)
    pub max_concurrent_jobs: usize,
    
    /// Number of GPU devices available (for round-robin assignment)
    pub num_gpus: usize,
    
    /// OpenMP thread count (OMP_NUM_THREADS)
    /// If None, uses environment variable or system default
    pub omp_num_threads: Option<usize>,
    
    /// Enable/disable OpenMP init parallelization (TRITON_OMP_INIT)
    /// If None, uses environment variable or default (disabled)
    pub triton_omp_init: Option<bool>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        // Default to 2 GPUs (H200 setup)
        let num_gpus = std::env::var("TRITON_GPU_COUNT")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(2);
        
        // Read OpenMP settings from environment if available
        let omp_num_threads = std::env::var("OMP_NUM_THREADS")
            .ok()
            .and_then(|s| s.parse().ok());
        
        let triton_omp_init = std::env::var("TRITON_OMP_INIT")
            .ok()
            .and_then(|s| match s.as_str() {
                "1" | "true" | "yes" => Some(true),
                "0" | "false" | "no" => Some(false),
                _ => None,
            });
        
        Self {
            tcp_addr: Some("127.0.0.1:5555".to_string()),
            unix_socket: None,
            max_concurrent_jobs: num_gpus, // One job per GPU by default
            num_gpus,
            omp_num_threads,
            triton_omp_init,
        }
    }
}

/// Prover server state
pub struct ProverServer {
    config: ServerConfig,
    job_semaphore: Arc<Semaphore>,
    shutdown: Arc<AtomicBool>,
    active_jobs: Arc<AtomicU32>,
    next_gpu_id: Arc<AtomicU32>, // For round-robin GPU assignment
}

impl ProverServer {
    pub fn new(config: ServerConfig) -> Self {
        let max_jobs = config.max_concurrent_jobs;
        Self {
            config: config.clone(),
            job_semaphore: Arc::new(Semaphore::new(max_jobs)),
            shutdown: Arc::new(AtomicBool::new(false)),
            active_jobs: Arc::new(AtomicU32::new(0)),
            next_gpu_id: Arc::new(AtomicU32::new(0)),
        }
    }
    
    /// Get next GPU device ID in round-robin fashion
    fn next_gpu_device(&self) -> usize {
        let current = self.next_gpu_id.fetch_add(1, Ordering::Relaxed);
        (current as usize) % self.config.num_gpus
    }
    
    /// Run the server until shutdown
    pub async fn run(&self) -> anyhow::Result<()> {
        // Start TCP listener if configured
        if let Some(addr) = &self.config.tcp_addr {
            let listener = TcpListener::bind(addr).await?;
            info!("Listening on TCP {}", addr);
            
            loop {
                if self.shutdown.load(Ordering::Relaxed) {
                    info!("Shutdown requested");
                    break;
                }
                
                tokio::select! {
                    result = listener.accept() => {
                        match result {
                            Ok((stream, peer_addr)) => {
                                info!("Accepted connection from {}", peer_addr);
                                let permit = self.job_semaphore.clone().acquire_owned().await?;
                                let active_jobs = self.active_jobs.clone();
                                let gpu_device_id = self.next_gpu_device();
                                
                                info!("Assigned GPU device {} to connection from {}", gpu_device_id, peer_addr);
                                
                                let config = self.config.clone();
                                tokio::spawn(async move {
                                    active_jobs.fetch_add(1, Ordering::Relaxed);
                                    
                                    let (read_half, write_half) = stream.into_split();
                                    let mut reader = BufReader::new(read_half);
                                    let mut writer = BufWriter::new(write_half);
                                    
                                    if let Err(e) = handle_connection(&mut reader, &mut writer, gpu_device_id, &config).await {
                                        error!("Connection error: {:?}", e);
                                    }
                                    
                                    active_jobs.fetch_sub(1, Ordering::Relaxed);
                                    drop(permit);
                                });
                            }
                            Err(e) => {
                                error!("Accept error: {:?}", e);
                            }
                        }
                    }
                    _ = tokio::signal::ctrl_c() => {
                        info!("Ctrl-C received, shutting down");
                        self.shutdown.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
        }
        
        // Start Unix socket listener if configured
        if let Some(path) = &self.config.unix_socket {
            // Remove existing socket file
            let _ = std::fs::remove_file(path);
            
            let listener = UnixListener::bind(path)?;
            info!("Listening on Unix socket {}", path);
            
            loop {
                if self.shutdown.load(Ordering::Relaxed) {
                    info!("Shutdown requested");
                    break;
                }
                
                tokio::select! {
                    result = listener.accept() => {
                        match result {
                            Ok((stream, _)) => {
                                info!("Accepted Unix connection");
                                let permit = self.job_semaphore.clone().acquire_owned().await?;
                                let active_jobs = self.active_jobs.clone();
                                let gpu_device_id = self.next_gpu_device();
                                
                                info!("Assigned GPU device {} to Unix connection", gpu_device_id);
                                
                                let config = self.config.clone();
                                tokio::spawn(async move {
                                    active_jobs.fetch_add(1, Ordering::Relaxed);
                                    
                                    let (read_half, write_half) = stream.into_split();
                                    let mut reader = BufReader::new(read_half);
                                    let mut writer = BufWriter::new(write_half);
                                    
                                    if let Err(e) = handle_connection(&mut reader, &mut writer, gpu_device_id, &config).await {
                                        error!("Connection error: {:?}", e);
                                    }
                                    
                                    active_jobs.fetch_sub(1, Ordering::Relaxed);
                                    drop(permit);
                                });
                            }
                            Err(e) => {
                                error!("Accept error: {:?}", e);
                            }
                        }
                    }
                    _ = tokio::signal::ctrl_c() => {
                        info!("Ctrl-C received, shutting down");
                        self.shutdown.store(true, Ordering::Relaxed);
                        break;
                    }
                }
            }
            
            // Clean up socket file
            let _ = std::fs::remove_file(path);
        }
        
        Ok(())
    }
    
    pub fn shutdown(&self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
    
    pub fn active_jobs(&self) -> u32 {
        self.active_jobs.load(Ordering::Relaxed)
    }
}

/// Handle a single connection: read request, prove, send response
async fn handle_connection<R, W>(reader: &mut R, writer: &mut W, gpu_device_id: usize, config: &ServerConfig) -> anyhow::Result<()>
where
    R: tokio::io::AsyncRead + Unpin,
    W: tokio::io::AsyncWrite + Unpin,
{
    // Read request
    info!("[SERVER] Waiting for request from client...");
    let request = match ProverRequest::read_from(reader).await {
        Ok(req) => {
            info!("[SERVER] =========================================");
            info!("[SERVER] Request received: job_id={}", req.job_id);
            info!("[SERVER] Claim JSON length: {} bytes", req.claim_json.len());
            info!("[SERVER] Program JSON length: {} bytes", req.program_json.len());
            info!("[SERVER] =========================================");
            req
        },
        Err(e) => {
            error!("[SERVER] Failed to read request: {:?}", e);
            let response = ProverResponse::Error {
                job_id: 0,
                message: format!("Failed to parse request: {:?}", e),
            };
            response.write_to(writer).await?;
            return Ok(());
        }
    };
    
    info!("[SERVER] Starting proof generation pipeline on GPU device {}...", gpu_device_id);
    
    // Run prover with GPU device assignment
    info!("[SERVER] Calling prover pipeline on GPU device {}...", gpu_device_id);
    let response = match prove_request_async(request.clone(), gpu_device_id, config.omp_num_threads, config.triton_omp_init).await {
        Ok(ProveOutcome::Success(result)) => {
            info!("[SERVER] =========================================");
            info!(
                job_id = request.job_id,
                proof_size = result.proof_bincode.len(),
                padded_height = result.padded_height,
                "[SERVER] Proof generation SUCCESS"
            );
            info!("[SERVER] Sending proof back to client...");
            info!("[SERVER] =========================================");
            ProverResponse::Ok {
                job_id: request.job_id,
                proof_bincode: result.proof_bincode,
            }
        }
        Ok(ProveOutcome::PaddedHeightTooBig { observed_log2 }) => {
            warn!(
                job_id = request.job_id,
                observed_log2 = observed_log2,
                "[SERVER] Padded height too big - sending error response"
            );
            ProverResponse::PaddedHeightTooBig {
                job_id: request.job_id,
                observed_log2,
            }
        }
        Err(e) => {
            error!(job_id = request.job_id, error = %e, "[SERVER] Proving FAILED - sending error response");
            ProverResponse::Error {
                job_id: request.job_id,
                message: e.to_string(),
            }
        }
    };
    
    // Send response
    info!("[SERVER] Writing response to socket...");
    response.write_to(writer).await?;
    info!("[SERVER] Response sent, connection complete");
    
    Ok(())
}

