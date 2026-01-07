//! GPU Prover Server for Neptune/XNT-Core Integration
//!
//! This crate provides a socket-based server that accepts proving requests
//! from Neptune's or XNT-Core's triton-vm-prover and runs them on the GPU.
//!
//! ## Architecture
//!
//! ```text
//! [Neptune/XNT-Core node]
//!       |
//!       | spawn triton-vm-prover
//!       v
//! [triton-vm-prover] ---- socket ----> [prover-server]
//!       ^                                    |
//!       |                                    | GPU proving
//!       | bincode Proof                      v
//!       +-------------------------------- [GPU]
//! ```
//!
//! ## Protocol
//!
//! The socket protocol uses length-prefixed binary messages.
//! See `protocol.rs` for details.

pub mod protocol;
pub mod prover;
pub mod server;

pub use protocol::{ProverRequest, ProverResponse};
pub use prover::{prove_request, ProveOutcome, ProveResult, ProverError};
pub use server::{ProverServer, ServerConfig};

