use std::collections::VecDeque;
use std::path::PathBuf;

use anyhow::bail;
use anyhow::Context;
use anyhow::Result;
use triton_vm::prelude::BFieldElement;
use triton_vm::prelude::NonDeterminism;
use triton_vm::prelude::Program;
use serde_json::json;

fn parse_public_input_csv(s: &str) -> Result<VecDeque<BFieldElement>> {
    if s.trim().is_empty() {
        return Ok(VecDeque::new());
    }
    let mut v = VecDeque::new();
    for part in s.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        let x: u64 = part.parse().with_context(|| format!("invalid u64 in public_input: {part}"))?;
        v.push_back(BFieldElement::new(x));
    }
    Ok(v)
}

fn main() -> Result<()> {
    let mut args = std::env::args().skip(1);
    let dump_dir = args.next().map(PathBuf::from).context("usage: dump-to-vmstate <dump_dir> <out_vmstate_json>")?;
    let out_path = args.next().map(PathBuf::from).context("usage: dump-to-vmstate <dump_dir> <out_vmstate_json>")?;

    if args.next().is_some() {
        bail!("usage: dump-to-vmstate <dump_dir> <out_vmstate_json>");
    }

    let program_json_path = dump_dir.join("program.json");
    let nondet_json_path = dump_dir.join("nondet.json");
    let public_input_path = dump_dir.join("public_input.txt");

    let program_json = std::fs::read_to_string(&program_json_path)
        .with_context(|| format!("failed reading {}", program_json_path.display()))?;
    let nondet_json = std::fs::read_to_string(&nondet_json_path)
        .with_context(|| format!("failed reading {}", nondet_json_path.display()))?;
    let public_input_str = std::fs::read_to_string(&public_input_path)
        .unwrap_or_else(|_| String::new());

    let program: Program = serde_json::from_str(&program_json)
        .context("failed to parse program.json as triton_vm::Program")?;
    let nondet: NonDeterminism = serde_json::from_str(&nondet_json)
        .context("failed to parse nondet.json as triton_vm::NonDeterminism")?;
    let public_input = parse_public_input_csv(&public_input_str)?;

    // Build a JSON object matching `triton_vm::vm::VMState`.
    // We intentionally do NOT construct `VMState` directly because it has some private fields.
    // triton-cli will deserialize this JSON into `VMState` and then discard custom IP/sponge/stack,
    // only extracting: program, public_input, and non-determinism (secret_* and ram).
    let public_input_vec: Vec<BFieldElement> = public_input.into_iter().collect();
    let secret_individual_tokens: Vec<BFieldElement> = nondet.individual_tokens.into_iter().collect();
    let secret_digests: Vec<triton_vm::prelude::Digest> = nondet.digests.into_iter().collect();

    let state_json = json!({
        "program": program,
        "public_input": public_input_vec,
        "public_output": [],
        "secret_individual_tokens": secret_individual_tokens,
        "secret_digests": secret_digests,
        "ram": nondet.ram,
        "ram_calls": [],
        "op_stack": {
            "stack": vec![BFieldElement::new(0); 16],
            "underflow_io_sequence": [],
        },
        "jump_stack": [],
        "cycle_count": 0,
        "instruction_pointer": 0,
        "sponge": null,
        "halting": false,
    });

    let out_file = std::fs::File::create(&out_path)
        .with_context(|| format!("failed creating {}", out_path.display()))?;
    serde_json::to_writer_pretty(out_file, &state_json)?;

    eprintln!(
        "[dump-to-vmstate] wrote VMState JSON to {}",
        out_path.display()
    );

    Ok(())
}


