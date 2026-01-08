# GPU Upgrader Implementation

## Overview

The GPU upgrader enhances the proof upgrader to leverage GPU acceleration when available, significantly improving upgrade performance for proof collection → single proof conversions and other upgrade operations.

## Architecture

### Current State

The codebase already has GPU infrastructure in place:
- **GPU Prover Server**: Communicates via `TRITON_VM_PROVER_SOCKET` environment variable
- **Concurrent Job Execution**: Job queue supports multiple concurrent jobs when GPU is available (default: 2 concurrent jobs)
- **Hybrid CPU/GPU Routing**: `force_cpu` flag allows forcing CPU execution for specific operations (e.g., proof collections which are faster on CPU)

### GPU Upgrader Enhancements

1. **Explicit GPU Usage**: Proof upgrader explicitly uses GPU when available
2. **Optimized Job Scheduling**: Better prioritization and scheduling for GPU-bound upgrade operations
3. **Enhanced Monitoring**: GPU-specific logging and metrics for upgrade operations
4. **Performance Optimization**: Ensures upgrades leverage GPU acceleration effectively

## Configuration

### Environment Variables

- `TRITON_VM_PROVER_SOCKET`: Set to GPU prover server address (e.g., `127.0.0.1:5555` or `/tmp/gpu-prover.sock`)
- `TRITON_VM_MAX_CONCURRENT_JOBS`: Override default concurrent job limit (default: 2 when GPU available, 1 for CPU-only)

### CLI Arguments

The proof upgrader uses existing CLI arguments:
- `--tx-proof-upgrading`: Enable proof upgrading
- `--tx-upgrade-filter <divisor>:<remainder>`: Filter transactions to upgrade based on txid modulo (default: `1:0` - matches all transactions)
  - Only upgrade transactions where `txid % divisor == remainder`
  - Example: `--tx-upgrade-filter 4:2` means this node upgrades ~1/4 of transactions (those with txid % 4 == 2)
  - Useful for coordinating multiple upgraders to avoid duplicate work
- `--gobbling-fraction`: Fraction of fee collected by upgrader (default: 0.6)
- `--min-gobbling-fee`: Minimum fee threshold for upgrading (default: 0.01)

## Performance Characteristics

### GPU vs CPU Performance

- **Single Proof Generation**: GPU is ~2.3x faster than CPU
- **Proof Collections**: CPU is ~4x faster than GPU (uses `force_cpu=true`)
- **Merge Operations**: GPU is ~2.3x faster than CPU

### Upgrade Operations

1. **ProofCollectionToSingleProof**: Uses GPU (if available) for single proof generation
2. **Merge Operations**: Uses GPU (if available) for merge proof generation
3. **UpdateMutatorSetData**: Uses GPU (if available) for proof updates

## Implementation Details

### Job Queue Integration

The proof upgrader integrates with `TritonVmJobQueue` which:
- Detects GPU availability via `TRITON_VM_PROVER_SOCKET`
- Supports concurrent execution (2 jobs by default with GPU, 1 job for CPU-only)
- Manages job priorities (High for Critical upgrades, Low for others)
- **Composer and Upgrader can run together**: With GPU available, both can execute concurrently (2 jobs max)

### Proof Job Options

Upgrade operations use `TritonVmProofJobOptions` with:
- **Priority**: `High` for Critical upgrades, `Low` for others
- **GPU Usage**: Automatically uses GPU if `TRITON_VM_PROVER_SOCKET` is set
- **Force CPU**: Only set for proof collections (which are faster on CPU)

## How to Run GPU Upgrader

### Prerequisites

1. **GPU Hardware**: NVIDIA GPU with CUDA support (recommended: H200, A100, or similar)
2. **GPU Prover Binary**: Built `triton_vm_prove_gpu_full` binary
3. **Prover Server**: Rust prover server built and ready to run

### Step 1: Build GPU Prover Server

The GPU prover server is a separate process that xnt-core connects to. Build it first:

```bash
# Navigate to triton-vm-cpp directory
cd triton-vm-cpp

# Build the Rust prover server
cd rust/prover_server
cargo build --release
cd ../..

# Build the GPU prover binary (if not already built)
cd build
cmake ..
make triton_vm_prove_gpu_full -j$(nproc)
cd ..
```

### Step 2: Start GPU Prover Server

Start the GPU prover server **before** starting xnt-core. The server listens for proof requests from xnt-core.

**Option A: Using the provided script (recommended)**

```bash
# From triton-vm-cpp directory
./start_prover_server.sh
```

Or with custom configuration:
```bash
# TCP socket (default: 127.0.0.1:5555)
./start_prover_server.sh --tcp 127.0.0.1:5555 --num-gpus 2

# Unix socket
./start_prover_server.sh --unix /tmp/gpu-prover.sock --num-gpus 2
```

**Option B: Manual start**

```bash
# Set GPU prover path
export TRITON_GPU_PROVER_PATH=/path/to/triton_vm_prove_gpu_full

# Set number of GPUs (default: 2)
export TRITON_GPU_COUNT=2

# Start server
cd rust/prover_server
cargo run --release -- --tcp 127.0.0.1:5555 --num-gpus 2
```

**Verify server is running:**

```bash
# Check if server process is running
pgrep -f prover-server

# Check if port is listening
ss -tlnp | grep 5555
# or
netstat -tlnp | grep 5555
```

### Step 3: Configure xnt-core Environment

Set the environment variable to point xnt-core to the GPU prover server:

```bash
# For TCP socket
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555

# For Unix socket
export TRITON_VM_PROVER_SOCKET=/tmp/gpu-prover.sock

# Optional: Override concurrent job limit (default: 2 when GPU available)
export TRITON_VM_MAX_CONCURRENT_JOBS=2
```

### Step 4: Run xnt-core with GPU Upgrader

Start xnt-core with proof upgrading enabled:

```bash
# Basic usage
xnt-core --tx-proof-upgrading

# With custom gobbling settings
xnt-core --tx-proof-upgrading \
  --gobbling-fraction 0.6 \
  --min-gobbling-fee 0.01

# With transaction filter (for coordinating multiple upgraders)
xnt-core --tx-proof-upgrading \
  --tx-upgrade-filter 4:0
```

### Step 5: Verify GPU Usage

Check the logs to confirm GPU is being used:

**Look for these log messages:**

```
[GPU-UPGRADER] GPU acceleration available - upgrades will use GPU for single proofs and merges
[GPU-UPGRADER] Starting ProofCollection→SingleProof upgrade (GPU-accelerated)
[HYBRID] Using GPU execution (TRITON_VM_PROVER_SOCKET is set) - GPU is 2.3x faster for single proofs
[GPU-UPGRADER] Successfully upgraded transaction <txid> (GPU-accelerated)
```

**If GPU is not available, you'll see:**

```
[GPU-UPGRADER] CPU-only mode - set TRITON_VM_PROVER_SOCKET to enable GPU acceleration
[GPU-UPGRADER] Starting ProofCollection→SingleProof upgrade (CPU-only)
[HYBRID] Using CPU execution (TRITON_VM_PROVER_SOCKET not set)
```

### Complete Example

Here's a complete example of running the GPU upgrader:

```bash
# Terminal 1: Start GPU prover server
cd triton-vm-cpp
export TRITON_GPU_PROVER_PATH=./build/triton_vm_prove_gpu_full
export TRITON_GPU_COUNT=2
./start_prover_server.sh --tcp 127.0.0.1:5555 --num-gpus 2

# Terminal 2: Run xnt-core with GPU upgrader
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555
cd xnt-core
cargo run --release -- --tx-proof-upgrading \
  --gobbling-fraction 0.6 \
  --min-gobbling-fee 0.01
```

### Troubleshooting

**Problem: GPU not being used**

1. **Check environment variable:**
   ```bash
   echo $TRITON_VM_PROVER_SOCKET
   # Should output: 127.0.0.1:5555 (or your socket path)
   ```

2. **Verify prover server is running:**
   ```bash
   pgrep -f prover-server
   ss -tlnp | grep 5555
   ```

3. **Check server logs** for connection errors

4. **Verify GPU prover binary exists:**
   ```bash
   ls -lh $TRITON_GPU_PROVER_PATH
   ```

**Problem: "Connection refused" errors**

- Ensure prover server is started **before** xnt-core
- Check socket address matches: `TRITON_VM_PROVER_SOCKET` must match server's `--tcp` or `--unix` argument
- Check firewall settings if using TCP

**Problem: Slow performance**

- Verify GPU is actually being used (check logs for `[HYBRID] Using GPU execution`)
- Check GPU utilization: `nvidia-smi`
- Ensure `TRITON_VM_MAX_CONCURRENT_JOBS` is set appropriately (default: 2 for dual-GPU)
- Verify GPU prover binary is built with GPU support

### Coordinating Multiple GPU Upgraders

To avoid duplicate work when running multiple GPU upgraders, use `--tx-upgrade-filter` to split the workload:

**Example: 4 GPU upgraders working in parallel**

Each upgrader connects to the same (or different) GPU prover server:

```bash
# Upgrader 1 (processes transactions where txid % 4 == 0)
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555
xnt-core --tx-proof-upgrading --tx-upgrade-filter 4:0

# Upgrader 2 (processes transactions where txid % 4 == 1)
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555
xnt-core --tx-proof-upgrading --tx-upgrade-filter 4:1

# Upgrader 3 (processes transactions where txid % 4 == 2)
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555
xnt-core --tx-proof-upgrading --tx-upgrade-filter 4:2

# Upgrader 4 (processes transactions where txid % 4 == 3)
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555
xnt-core --tx-proof-upgrading --tx-upgrade-filter 4:3
```

Each upgrader will process approximately 1/4 of the transactions, avoiding duplicate upgrades and maximizing GPU utilization across multiple nodes.

**Note:** All upgraders can connect to the same GPU prover server. The server handles concurrent requests and distributes work across available GPUs.

### Monitoring

The proof upgrader logs GPU usage:
- `[HYBRID] Using GPU execution` - GPU is being used
- `[HYBRID] Forcing CPU execution` - CPU is forced (e.g., for proof collections)
- `[HYBRID] Using CPU execution` - CPU-only mode (GPU not available)

## Composer and Upgrader Together

### Can They Run Simultaneously?

**YES, with GPU acceleration!**

Both the composer (block composition) and proof upgrader use the same `TritonVmJobQueue`, but they can work together:

**With GPU Available (`TRITON_VM_PROVER_SOCKET` set):**
- **Concurrent execution**: Up to 2 jobs can run simultaneously (default: `max_concurrent_jobs=2`)
- **Priority-based scheduling**: 
  - Composer uses `High` priority
  - Critical upgrades use `High` priority  
  - Gobbling upgrades use `Low` priority
- **Result**: Composer and upgrader can run **at the same time** when GPU is available
  - Example: Composer (High) + Critical upgrader (High) = both run concurrently
  - Example: Composer (High) + Gobbling upgrader (Low) = both run concurrently, composer gets priority if resources are limited

**Without GPU (CPU-only mode):**
- **Serial execution**: Only 1 job runs at a time (`max_concurrent_jobs=1`)
- **Result**: Composer and upgrader **compete** for the prover
  - Higher priority jobs run first
  - Composer (High) will run before Low-priority upgrades
  - Critical upgrades (High) compete equally with composer

### Running Both Together

To run both composer and upgrader simultaneously:

```bash
# Start GPU prover server (supports 2+ concurrent jobs)
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555
./start_prover_server.sh --tcp 127.0.0.1:5555 --num-gpus 2

# Run xnt-core with both composer and upgrader
export TRITON_VM_PROVER_SOCKET=127.0.0.1:5555
xnt-core --compose --tx-proof-upgrading
```

**Benefits:**
- Composer creates blocks while upgrader improves transaction quality
- Both leverage GPU acceleration simultaneously
- Priority system ensures critical operations (composer, critical upgrades) get resources first

**Monitoring:**
- Check logs for both `[GPU-UPGRADER]` and composer messages
- Job queue logs show concurrent execution: `JobQueue: job running: #1 - <id1>, #2 - <id2>`
- GPU utilization should show both jobs using GPU resources

## Related Documentation

- [Upgrader Flow](upgrader.md) - Complete flow of upgraded transactions
- [Mempool Analysis](mempool.md) - Mempool and upgrading discussion

