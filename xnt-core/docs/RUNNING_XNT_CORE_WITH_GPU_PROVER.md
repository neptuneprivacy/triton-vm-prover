# Running XNT-Core with GPU Prover Server

## Quick Start

### Step 1: Build Everything

```bash
# Build XNT-Core
cd xnt-core
cargo build --release

# Build Prover Server (if not already built)
cd ../triton-vm-cpp/rust/prover_server
cargo build --release

# Build GPU Prover (optional, for GPU acceleration)
cd ../..
# Follow GPU prover build instructions in triton-vm-cpp
```

### Step 2: Start Prover Server

```bash
cd xnt-core/scripts/linux
./run-gpu-prover-server.sh --tcp 127.0.0.1:5555 --num-gpus 2
```

**What this does:**
- Starts the prover server on port 5555
- Configures for 2 GPUs (adjust `--num-gpus` for your setup)
- Uses GPU prover if available, falls back to Rust proving otherwise

### Step 3: Run XNT-Core

```bash
cd xnt-core
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555
export RUST_LOG=info

./target/release/xnt-core \
    --network testnet \
    --peer 51.15.139.238:19798 \
    --peer-port=19798 \
    --rpc-port=19799 \
    --compose \
    --guess
```

XNT-core will automatically use the prover server for parallel proof generation.

---

## What You Need to Know

### Environment Variables

**Required:**
- `TRITON_VM_PROVER_SOCKET=127.0.0.1:5555` - Tells xnt-core where the prover server is

**Optional:**
- `TRITON_GPU_PROVER_PATH=/path/to/triton_vm_prove_gpu_full` - Path to GPU prover binary (if using GPU)
- `TRITON_VM_MAX_CONCURRENT_JOBS=2` - Max parallel jobs (defaults to 2 when socket is set)
- `RUST_LOG=info` - Logging level
- `TRITON_GPU_COUNT=2` - Number of GPUs (default: 2)
- `OMP_NUM_THREADS=60` - OpenMP thread count (optional)
- `TRITON_OMP_INIT=0` - Enable/disable OpenMP init parallelization (optional)

### Prover Server Options

```bash
./run-gpu-prover-server.sh [options]
```

**Options:**
- `--tcp <ADDR>` - Listen address (default: `127.0.0.1:5555`)
- `--unix <PATH>` - Unix socket path (alternative to TCP)
- `--num-gpus <N>` - Number of GPUs (default: 2)
- `--max-jobs <N>` - Max concurrent jobs (default: matches num-gpus)
- `--omp-threads <N>` - OpenMP thread count
- `--omp-init <0|1>` - Enable/disable OpenMP init parallelization

### How It Works

1. **XNT-Core** generates proof requests
2. **Job Queue** routes them to the prover server (up to `max_concurrent_jobs` at a time for parallel execution)
3. **Prover Server** assigns each request to a different GPU (round-robin)
4. **Proofs run in parallel** on different GPUs

**Result:** Large proofs (2^21 padded height) run on GPU, small proofs (proof collections) run on CPU, and multiple large proofs can run simultaneously on different GPUs.

### Routing Logic

- **Large proofs (2^21 padded height)**: Routed to GPU prover server
- **Small proofs (proof collections)**: Routed to CPU (via `force_cpu` flag)
- **Concurrent execution**: Multiple GPU proofs can run in parallel (one per GPU)

---

## Verification

### Check Prover Server

```bash
# Check if server is listening
ss -tlnp | grep 5555

# Check server logs for:
# - "Starting GPU Prover Server"
# - "Number of GPUs: 2"
```

### Check XNT-Core Logs

Look for:
- `JobQueue: starting with max_concurrent_jobs=2 (GPU prover: true)`
- `[proxy] Forwarding to GPU prover server at 127.0.0.1:5555`
- `Using GPU prover for job` (for large proofs)
- `Using CPU prover for job` (for proof collections)

### Parallel Execution

Both coinbase and NOP single proofs should start within milliseconds of each other (not 30+ seconds apart) when using multiple GPUs.

---

## Troubleshooting

### Server Won't Start

```bash
# Check if port is in use
ss -tlnp | grep 5555

# Check if binary exists
ls -la triton-vm-cpp/rust/prover_server/target/release/prover-server

# Build if missing
cd triton-vm-cpp/rust/prover_server && cargo build --release
```

### XNT-Core Can't Connect

```bash
# Check server is running
ss -tlnp | grep 5555

# Verify environment variable
echo $TRITON_VM_PROVER_SOCKET  # Should show: 127.0.0.1:5555

# Test connection manually
nc -zv 127.0.0.1 5555
```

### Parallel Execution Not Working

**Symptoms:** NOP proof waits 30+ seconds after coinbase starts

**Fix:**
```bash
# Make sure socket is set BEFORE starting xnt-core
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555
export TRITON_VM_MAX_CONCURRENT_JOBS=2  # Explicit override

# Then start xnt-core
./target/release/xnt-core --compose --guess
```

### GPU Not Being Used

**Check:**
```bash
# Verify GPU prover path
echo $TRITON_GPU_PROVER_PATH

# Check if binary exists
ls -la $TRITON_GPU_PROVER_PATH
```

**Note:** If GPU prover isn't available, the server uses Rust proving (CPU-only). Parallel execution still works.

### Proof Collections Not Using CPU

**Check logs for:**
- `force_cpu=true` in proof collection generation
- `Using CPU prover for job` messages

If proof collections are still going to GPU, verify that `proof_collection.rs` is setting `force_cpu(true)`.

---

## Complete Example

**Terminal 1 - Prover Server:**
```bash
cd xnt-core/scripts/linux
export TRITON_GPU_PROVER_PATH=$(cd ../../.. && pwd)/triton-vm-cpp/build/triton_vm_prove_gpu_full
./run-gpu-prover-server.sh --tcp 127.0.0.1:5555 --num-gpus 2
```

**Terminal 2 - XNT-Core:**
```bash
cd xnt-core
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555
export RUST_LOG=info
./target/release/xnt-core \
    --network testnet \
    --peer 51.15.139.238:19798 \
    --peer-port=19798 \
    --rpc-port=19799 \
    --compose \
    --guess
```

---

## Architecture

```
┌─────────────┐
│  XNT-Core   │
│             │
│  Job Queue  │─── spawns ───> [triton-vm-prover] ── socket ──> [Prover Server]
│             │                                              │
│             │                                              │
│  Composer   │                                              v
└─────────────┘                                        ┌──────────────┐
                                                      │ GPU Prover   │
                                                      │ (parallel)   │
                                                      └──────────────┘
```

- **triton-vm-prover**: Acts as a proxy, forwarding requests to the GPU prover server
- **Prover Server**: Manages multiple GPU devices and routes requests round-robin
- **Job Queue**: Manages concurrent proof generation jobs (up to `max_concurrent_jobs`)

---

## Performance Notes

- **Large proofs (2^21)**: ~30-60 seconds on GPU (vs. much longer on CPU)
- **Proof collections**: Faster on CPU (small proofs, better CPU utilization)
- **Parallel execution**: With 2 GPUs, two large proofs can run simultaneously, cutting total time in half
- **Concurrent jobs**: Controlled by `TRITON_VM_MAX_CONCURRENT_JOBS` (default: 2 when GPU prover is available)

