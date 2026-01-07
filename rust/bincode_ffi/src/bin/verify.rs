use anyhow::{Context, Result};
use std::fs::File;
use triton_vm::prelude::*;
use twenty_first::prelude::BFieldCodec;

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let claim_path = args.next().context("usage: verify <claim.json> <proof.bincode>")?;
    let proof_path = args.next().context("usage: verify <claim.json> <proof.bincode>")?;

    let claim_file = File::open(&claim_path).with_context(|| format!("open claim: {claim_path}"))?;
    let claim: Claim =
        serde_json::from_reader(claim_file).with_context(|| format!("parse claim json: {claim_path}"))?;

    let proof_file = File::open(&proof_path).with_context(|| format!("open proof: {proof_path}"))?;
    let proof: Proof =
        bincode::deserialize_from(proof_file).with_context(|| format!("deserialize proof bincode: {proof_path}"))?;

    if std::env::var("TVM_DEBUG_CLAIM_ENCODING").is_ok() {
        let enc = claim.encode();
        eprintln!("[DBG] rust_claim_encoding_len={} first16={:?}", enc.len(), &enc.iter().take(16).collect::<Vec<_>>());
    }

    // Use Triton VM's native verifier and print a detailed error on failure.
    let stark = Stark::default();
    match stark.verify(&claim, &proof) {
        Ok(()) => {
            println!("OK");
            Ok(())
        }
        Err(e) => {
            eprintln!("VERIFY FAILED: {e:?}");
            std::process::exit(1);
        }
    }
}


