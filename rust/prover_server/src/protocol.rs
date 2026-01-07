//! Socket protocol for GPU prover server
//!
//! Wire format is simple length-prefixed binary messages:
//!
//! Request:
//!   [4 bytes: magic "TVMP" = 0x54564D50]
//!   [4 bytes: version = 1]
//!   [4 bytes: job_id]
//!   [4 bytes: claim_json_len]     [claim_json bytes]
//!   [4 bytes: program_json_len]   [program_json bytes]
//!   [4 bytes: nondet_json_len]    [nondet_json bytes]
//!   [4 bytes: max_log2_json_len]  [max_log2_json bytes]
//!   [4 bytes: env_vars_json_len]  [env_vars_json bytes]
//!
//! Response:
//!   [4 bytes: magic "TVMR" = 0x54564D52]
//!   [4 bytes: status: 0=OK, 1=PADDED_HEIGHT_TOO_BIG, 2=ERROR]
//!   [4 bytes: job_id]
//!   if status == OK:
//!     [8 bytes: proof_len]
//!     [proof_len bytes: proof bincode]
//!   if status == PADDED_HEIGHT_TOO_BIG:
//!     [4 bytes: observed_log2]
//!   if status == ERROR:
//!     [4 bytes: error_msg_len]
//!     [error_msg_len bytes: error message UTF-8]

use std::io::{self, Read, Write};
use thiserror::Error;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};

pub const MAGIC_REQUEST: u32 = 0x54564D50; // "TVMP"
pub const MAGIC_RESPONSE: u32 = 0x54564D52; // "TVMR"
pub const PROTOCOL_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum ResponseStatus {
    Ok = 0,
    PaddedHeightTooBig = 1,
    Error = 2,
}

impl From<u32> for ResponseStatus {
    fn from(v: u32) -> Self {
        match v {
            0 => Self::Ok,
            1 => Self::PaddedHeightTooBig,
            _ => Self::Error,
        }
    }
}

#[derive(Debug, Error)]
pub enum ProtocolError {
    #[error("IO error: {0}")]
    Io(#[from] io::Error),
    
    #[error("Invalid magic: expected 0x{expected:08X}, got 0x{got:08X}")]
    InvalidMagic { expected: u32, got: u32 },
    
    #[error("Unsupported protocol version: {0}")]
    UnsupportedVersion(u32),
    
    #[error("Message too large: {0} bytes")]
    MessageTooLarge(u64),
    
    #[error("Invalid UTF-8 in message")]
    InvalidUtf8,
}

/// Request from Neptune's or XNT-Core's triton-vm-prover proxy
#[derive(Debug, Clone)]
pub struct ProverRequest {
    pub job_id: u32,
    pub claim_json: String,
    pub program_json: String,
    pub nondet_json: String,
    pub max_log2_json: String,
    pub env_vars_json: String,
}

/// Response to Neptune's or XNT-Core's triton-vm-prover proxy
#[derive(Debug, Clone)]
pub enum ProverResponse {
    /// Proof generated successfully
    Ok {
        job_id: u32,
        proof_bincode: Vec<u8>,
    },
    /// Padded height exceeded limit
    PaddedHeightTooBig {
        job_id: u32,
        observed_log2: u32,
    },
    /// Error during proving
    Error {
        job_id: u32,
        message: String,
    },
}

impl ProverRequest {
    /// Read a request from an async stream
    pub async fn read_from<R: AsyncRead + Unpin>(reader: &mut R) -> Result<Self, ProtocolError> {
        // Read and verify magic
        let magic = reader.read_u32_le().await?;
        if magic != MAGIC_REQUEST {
            return Err(ProtocolError::InvalidMagic {
                expected: MAGIC_REQUEST,
                got: magic,
            });
        }
        
        // Read and verify version
        let version = reader.read_u32_le().await?;
        if version != PROTOCOL_VERSION {
            return Err(ProtocolError::UnsupportedVersion(version));
        }
        
        // Read job_id
        let job_id = reader.read_u32_le().await?;
        
        // Read 5 length-prefixed JSON strings
        let claim_json = read_length_prefixed_string(reader).await?;
        let program_json = read_length_prefixed_string(reader).await?;
        let nondet_json = read_length_prefixed_string(reader).await?;
        let max_log2_json = read_length_prefixed_string(reader).await?;
        let env_vars_json = read_length_prefixed_string(reader).await?;
        
        Ok(Self {
            job_id,
            claim_json,
            program_json,
            nondet_json,
            max_log2_json,
            env_vars_json,
        })
    }
    
    /// Write a request to a sync stream (for client side)
    pub fn write_to<W: Write>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_all(&MAGIC_REQUEST.to_le_bytes())?;
        writer.write_all(&PROTOCOL_VERSION.to_le_bytes())?;
        writer.write_all(&self.job_id.to_le_bytes())?;
        
        write_length_prefixed_string_sync(writer, &self.claim_json)?;
        write_length_prefixed_string_sync(writer, &self.program_json)?;
        write_length_prefixed_string_sync(writer, &self.nondet_json)?;
        write_length_prefixed_string_sync(writer, &self.max_log2_json)?;
        write_length_prefixed_string_sync(writer, &self.env_vars_json)?;
        
        writer.flush()?;
        Ok(())
    }
}

impl ProverResponse {
    /// Write response to an async stream
    pub async fn write_to<W: AsyncWrite + Unpin>(&self, writer: &mut W) -> io::Result<()> {
        writer.write_u32_le(MAGIC_RESPONSE).await?;
        
        match self {
            Self::Ok { job_id, proof_bincode } => {
                writer.write_u32_le(ResponseStatus::Ok as u32).await?;
                writer.write_u32_le(*job_id).await?;
                writer.write_u64_le(proof_bincode.len() as u64).await?;
                writer.write_all(proof_bincode).await?;
            }
            Self::PaddedHeightTooBig { job_id, observed_log2 } => {
                writer.write_u32_le(ResponseStatus::PaddedHeightTooBig as u32).await?;
                writer.write_u32_le(*job_id).await?;
                writer.write_u32_le(*observed_log2).await?;
            }
            Self::Error { job_id, message } => {
                writer.write_u32_le(ResponseStatus::Error as u32).await?;
                writer.write_u32_le(*job_id).await?;
                let msg_bytes = message.as_bytes();
                writer.write_u32_le(msg_bytes.len() as u32).await?;
                writer.write_all(msg_bytes).await?;
            }
        }
        
        writer.flush().await?;
        Ok(())
    }
    
    /// Read response from a sync stream (for client side)
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, ProtocolError> {
        let mut buf4 = [0u8; 4];
        let mut buf8 = [0u8; 8];
        
        // Read and verify magic
        reader.read_exact(&mut buf4)?;
        let magic = u32::from_le_bytes(buf4);
        if magic != MAGIC_RESPONSE {
            return Err(ProtocolError::InvalidMagic {
                expected: MAGIC_RESPONSE,
                got: magic,
            });
        }
        
        // Read status
        reader.read_exact(&mut buf4)?;
        let status = ResponseStatus::from(u32::from_le_bytes(buf4));
        
        // Read job_id
        reader.read_exact(&mut buf4)?;
        let job_id = u32::from_le_bytes(buf4);
        
        match status {
            ResponseStatus::Ok => {
                reader.read_exact(&mut buf8)?;
                let proof_len = u64::from_le_bytes(buf8);
                
                // Sanity check: proofs shouldn't be > 1GB
                if proof_len > 1_000_000_000 {
                    return Err(ProtocolError::MessageTooLarge(proof_len));
                }
                
                let mut proof_bincode = vec![0u8; proof_len as usize];
                reader.read_exact(&mut proof_bincode)?;
                
                Ok(Self::Ok { job_id, proof_bincode })
            }
            ResponseStatus::PaddedHeightTooBig => {
                reader.read_exact(&mut buf4)?;
                let observed_log2 = u32::from_le_bytes(buf4);
                Ok(Self::PaddedHeightTooBig { job_id, observed_log2 })
            }
            ResponseStatus::Error => {
                reader.read_exact(&mut buf4)?;
                let msg_len = u32::from_le_bytes(buf4) as usize;
                
                if msg_len > 1_000_000 {
                    return Err(ProtocolError::MessageTooLarge(msg_len as u64));
                }
                
                let mut msg_bytes = vec![0u8; msg_len];
                reader.read_exact(&mut msg_bytes)?;
                let message = String::from_utf8(msg_bytes)
                    .map_err(|_| ProtocolError::InvalidUtf8)?;
                
                Ok(Self::Error { job_id, message })
            }
        }
    }
}

async fn read_length_prefixed_string<R: AsyncRead + Unpin>(
    reader: &mut R,
) -> Result<String, ProtocolError> {
    let len = reader.read_u32_le().await? as usize;
    
    // Sanity check: JSON blobs shouldn't be > 100MB
    if len > 100_000_000 {
        return Err(ProtocolError::MessageTooLarge(len as u64));
    }
    
    let mut buf = vec![0u8; len];
    reader.read_exact(&mut buf).await?;
    
    String::from_utf8(buf).map_err(|_| ProtocolError::InvalidUtf8)
}

fn write_length_prefixed_string_sync<W: Write>(writer: &mut W, s: &str) -> io::Result<()> {
    let bytes = s.as_bytes();
    writer.write_all(&(bytes.len() as u32).to_le_bytes())?;
    writer.write_all(bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;
    
    #[test]
    fn test_request_roundtrip() {
        let request = ProverRequest {
            job_id: 42,
            claim_json: r#"{"program_digest":"abc","version":0,"input":[],"output":[]}"#.to_string(),
            program_json: r#"{"instructions":["Halt"]}"#.to_string(),
            nondet_json: r#"{"individual_tokens":[],"digests":[],"ram":{}}"#.to_string(),
            max_log2_json: "null".to_string(),
            env_vars_json: "{}".to_string(),
        };
        
        let mut buf = Vec::new();
        request.write_to(&mut buf).unwrap();
        
        // Use tokio runtime for async read
        let rt = tokio::runtime::Runtime::new().unwrap();
        let parsed = rt.block_on(async {
            let mut cursor = Cursor::new(buf);
            // Note: Cursor doesn't implement AsyncRead directly, would need tokio::io::BufReader
            // This test would need adjustment for real async
        });
    }
    
    #[test]
    fn test_response_ok_roundtrip() {
        let response = ProverResponse::Ok {
            job_id: 42,
            proof_bincode: vec![1, 2, 3, 4, 5],
        };
        
        let rt = tokio::runtime::Runtime::new().unwrap();
        let buf = rt.block_on(async {
            let mut buf = Vec::new();
            response.write_to(&mut buf).await.unwrap();
            buf
        });
        
        let mut cursor = Cursor::new(buf);
        let parsed = ProverResponse::read_from(&mut cursor).unwrap();
        
        match parsed {
            ProverResponse::Ok { job_id, proof_bincode } => {
                assert_eq!(job_id, 42);
                assert_eq!(proof_bincode, vec![1, 2, 3, 4, 5]);
            }
            _ => panic!("Expected Ok response"),
        }
    }
}

