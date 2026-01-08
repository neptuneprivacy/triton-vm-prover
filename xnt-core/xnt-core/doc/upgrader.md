# How Upgraded Transactions (Proof Collection → Single Proof) Are Added to Blocks in xnt-core

## Overview

This document explains the complete flow of how transactions upgraded via the proof upgrader (from proof collection to single proof) are eventually included in blocks.

## Important Note: Proof Upgrader vs Composer

**The proof upgrader and composer are separate, independent components:**

- **Proof Upgrader**: Runs in the main loop, independently upgrades proof collections → single proofs in the mempool
- **Composer**: Runs in the mine loop, creates block proposals by composing transactions together

They are **not directly related** - the proof upgrader doesn't call the composer, and the composer doesn't call the proof upgrader's main functionality. They work together **indirectly** through the mempool:
1. Proof upgrader upgrades transactions and puts them in the mempool
2. Composer later reads from the mempool when creating blocks

The only connection is that the composer can use `UpgradeJob::UpdateMutatorSetData` as a fallback when no synced transactions are available, but this is a different use case than the main proof upgrader functionality.

## Flow Diagram

```
1. Proof Upgrader Task (main_loop.rs)
   ↓
2. Find Upgrade Candidate (get_upgrade_task_from_mempool)
   ↓
3. Execute Upgrade (handle_upgrade)
   ├─ Convert ProofCollection → SingleProof
   ├─ Insert into Mempool
   └─ Notify Peers
   ↓
4. Mempool Storage
   ↓
5. Block Composition (mine_loop.rs)
   ├─ Get transactions from mempool
   ├─ Filter: Only SingleProof + Synced to tip
   └─ Merge into block transaction
   ↓
6. Block Created
```

## Detailed Flow

### Step 1: Proof Upgrader Task Scheduling

**Location**: `xnt-core/src/application/loops/main_loop.rs:1492-1556`

The main loop periodically runs the `proof_upgrader` function which:
- Checks if proof upgrading is enabled (`tx_proof_upgrading`)
- Verifies the node has `SingleProof` capability
- Ensures no previous upgrade task is still running
- Finds upgrade candidates from the mempool

```rust
async fn proof_upgrader(&mut self, main_loop_state: &mut MutableMainLoopState) -> Result<()> {
    // Find upgrade candidate
    let upgrade_candidate = get_upgrade_task_from_mempool(&mut global_state).await?;
    
    // Spawn upgrade task
    let proof_upgrader_task = tokio::task::spawn(async move {
        upgrade_candidate.handle_upgrade(...).await
    });
}
```

### Step 2: Finding Upgrade Candidates

**Location**: `xnt-core/src/application/loops/main_loop/proof_upgrader.rs:856-962`

The `get_upgrade_task_from_mempool` function searches for:
1. **Proof Collections** that can be upgraded to single proofs
2. **Unsynced single proofs** that need mutator set updates
3. **Single proof pairs** that can be merged

For proof collection upgrades, it:
- Finds proof collections in mempool via `preferred_proof_collection()`
- Calculates gobbling potential (fee * gobbling_fraction)
- Creates `UpgradeJob::ProofCollectionToSingleProof` if worth it

```rust
let proof_collection_job = if let Some((kernel, proof, upgrade_priority)) = 
    global_state.mempool.preferred_proof_collection(num_proofs_threshold, upgrade_filter) {
    // Calculate if upgrade is worth it
    let gobbling_potential = kernel.fee.lossy_f64_fraction_mul(gobbling_fraction);
    let upgrade_incentive = upgrade_priority.incentive_given_gobble_potential(gobbling_potential);
    
    if upgrade_incentive.upgrade_is_worth_it(min_gobbling_fee) {
        UpgradeJob::ProofCollectionToSingleProof(...)
    }
}
```

### Step 3: Executing the Upgrade

**Location**: `xnt-core/src/application/loops/main_loop/proof_upgrader.rs:413-595`

The `handle_upgrade` method performs the actual upgrade:

#### 3.1 Upgrade Proof Collection to Single Proof

**Location**: `proof_upgrader.rs:741-780`

```rust
UpgradeJob::ProofCollectionToSingleProof(ProofCollectionToSingleProof {
    kernel,
    proof,  // This is a ProofCollection
    ..
}) => {
    // Build single proof from proof collection
    let single_proof = TransactionProofBuilder::new()
        .consensus_rule_set(consensus_rule_set)
        .proof_collection(proof)  // Convert ProofCollection → SingleProof
        .job_queue(triton_vm_job_queue.clone())
        .proof_job_options(proof_job_options.clone())
        .build()
        .await?;
    
    let upgraded_tx = Transaction {
        kernel,
        proof: single_proof,  // Now a SingleProof
    };
    
    // Optionally merge with gobbler transaction (for fee collection)
    let tx = if let Some(gobbler) = maybe_gobbler {
        gobbler.merge_with(upgraded_tx, ...).await?
    } else {
        upgraded_tx
    };
}
```

#### 3.2 Insert into Mempool

**Location**: `proof_upgrader.rs:509-514`

After successful upgrade, the transaction is inserted into the mempool:

```rust
/* Handle successful upgrade */
// Insert tx into mempool before notifying peers, so we're
// sure to have it when they ask.
global_state
    .mempool_insert(upgraded.clone(), upgrade_incentive.into())
    .await;
```

**Key Points**:
- The upgraded transaction now has a `SingleProof` instead of `ProofCollection`
- It's stored in the mempool with the appropriate upgrade priority
- The transaction is synced to the current tip's mutator set

### Step 4: Mempool Storage

**Location**: `xnt-core/src/state/mempool.rs`

The mempool stores transactions sorted by fee density. The upgraded transaction:
- Has `TransactionProof::SingleProof` type
- Is synced to the current tip (mutator_set_hash matches tip)
- Can be retrieved via `get_transactions_for_block_composition()`

### Step 5: Block Composition

**Location**: `xnt-core/src/application/loops/mine_loop.rs:524-637`

When composing a block, the mine loop:

#### 5.1 Get Transactions from Mempool

```rust
let mut transactions_to_merge = global_state_lock
    .lock_guard()
    .await
    .mempool
    .get_transactions_for_block_composition(
        block_capacity_for_transactions,
        Some(max_num_mergers),
    );
```

#### 5.2 Filtering Criteria

**Location**: `xnt-core/src/state/mempool.rs:815-853`

The `get_transactions_for_block_composition` method only returns transactions that:

1. **Are synced to tip**: `tx_is_synced(&transaction_ptr.kernel)` must be true
2. **Have SingleProof**: `matches!(transaction_ptr.proof, TransactionProof::SingleProof(_))`
3. **Fit in block**: Transaction size ≤ remaining storage
4. **Sorted by fee density**: Highest fee density first

```rust
pub(crate) fn get_transactions_for_block_composition(
    &self,
    mut remaining_storage: usize,
    max_num_txs: Option<usize>,
) -> Vec<Transaction> {
    for (transaction_digest, _fee_density) in self.fee_density_iter() {
        if let Some(transaction_ptr) = self.get(transaction_digest) {
            // Only return transaction synced to tip
            if !self.tx_is_synced(&transaction_ptr.kernel) {
                continue;
            }
            
            // Only SingleProof transactions
            if !matches!(transaction_ptr.proof, TransactionProof::SingleProof(_)) {
                continue;
            }
            
            // Check size constraints
            if transaction_size > remaining_storage {
                continue;
            }
            
            transactions.push(transaction_copy)
        }
    }
}
```

**Important**: This is why proof collection → single proof upgrade is necessary! Only `SingleProof` transactions are eligible for block inclusion.

#### 5.3 Merge Transactions into Block

**Location**: `mine_loop.rs:607-627`

Selected transactions are merged into the block transaction in a sequential loop:

```rust
let num_merges = transactions_to_merge.len();
let mut block_transaction = BlockOrRegularTransaction::from(coinbase_transaction);
for (i, tx_to_include) in transactions_to_merge.into_iter().enumerate() {
    info!("Merging transaction {} / {}", i + 1, num_merges);
    block_transaction = BlockTransaction::merge(
        block_transaction,  // Accumulated block transaction (coinbase + previous merges)
        tx_to_include,      // Next transaction to merge (includes upgraded transactions)
        rng.random(),       // Random seed for shuffling
        vm_job_queue.clone(),
        job_options.clone(),
        consensus_rule_set,
    )
    .await?
    .into();
}
```

**Merge Process Details**:

The merge happens in several steps:

1. **Create MergeWitness** (`block_transaction.rs:174`)
   - Combines the accumulated block transaction (left) with the new transaction (right)
   - Both transactions must have `SingleProof` proofs
   - Uses a random `shuffle_seed` for privacy

2. **Create New Kernel** (`merge_branch.rs:204-247`)
   - **Concatenates and shuffles inputs**: `[left.inputs, right.inputs].concat().shuffle()`
   - **Concatenates and shuffles outputs**: `[left.outputs, right.outputs].concat().shuffle()`
   - **Concatenates and shuffles announcements**: `[left.announcements, right.announcements].concat().shuffle()`
   - **Combines fees**: `left.fee + right.fee`
   - **Takes maximum timestamp**: `max(left.timestamp, right.timestamp)`
   - **Preserves coinbase**: Only from left transaction (coinbase must be in LHS)
   - **Sets merge_bit**: `true` (indicates this is a merged transaction)
   - **Preserves mutator_set_hash**: Must match between both transactions

3. **Generate New Proof** (`merge_branch.rs:144-169`)
   - Creates a `SingleProofWitness` from the merge witness
   - Proves the merged transaction using TritonVM
   - This is computationally expensive and can take minutes
   - The proof verifies that:
     - Both original transactions were valid (verifies their proofs)
     - The new kernel correctly combines the inputs/outputs/announcements
     - The shuffling preserves all elements (no data loss)

4. **Pack Removal Records** (`merge_branch.rs:192-195`)
   - For block transactions, removal records are packed to save space
   - Removes redundant information that can be reconstructed

**Important Constraints**:
- Both transactions must have the same `mutator_set_hash`
- Only `SingleProof` transactions can be merged
- Coinbase transaction must be on the left side (LHS)
- Right-hand transaction cannot have negative fee
- The merge bit is set to `true` after the first merge

**Result**:
After all merges complete, the final `block_transaction` contains:
- All inputs from coinbase + all merged transactions (shuffled)
- All outputs from coinbase + all merged transactions (shuffled)
- All announcements from coinbase + all merged transactions (shuffled)
- Combined fee (sum of all transaction fees)
- A single `SingleProof` that proves the entire merged transaction is valid

### Step 6: Block Created

The final block contains:
- Coinbase transaction
- All merged transactions (including upgraded ones)
- All transactions are merged into a single block transaction

## Key Files

1. **Proof Upgrader**: `xnt-core/src/application/loops/main_loop/proof_upgrader.rs`
   - `UpgradeJob::ProofCollectionToSingleProof` - Upgrade job type
   - `handle_upgrade()` - Main upgrade execution
   - `get_upgrade_task_from_mempool()` - Find upgrade candidates

2. **Main Loop**: `xnt-core/src/application/loops/main_loop.rs`
   - `proof_upgrader()` - Scheduled upgrade task

3. **Mine Loop**: `xnt-core/src/application/loops/mine_loop.rs`
   - `compose_task()` - Block composition logic
   - Gets transactions from mempool and merges them

4. **Mempool**: `xnt-core/src/state/mempool.rs`
   - `get_transactions_for_block_composition()` - Transaction selection
   - Only returns `SingleProof` transactions synced to tip

## Why Proof Collection → Single Proof Upgrade is Needed

1. **Block Inclusion Requirement**: Only `SingleProof` transactions can be included in blocks (see `mempool.rs:834`)
2. **Mutator Set Sync**: Single proofs can be updated to sync with new blocks, proof collections cannot
3. **Miner Preference**: Miners prefer single proofs as they're more efficient to verify and merge

## Summary

The flow ensures that:
1. Proof collections in the mempool are upgraded to single proofs
2. Upgraded transactions are inserted back into the mempool
3. During block composition, only single-proof transactions synced to tip are selected
4. Selected transactions (including upgraded ones) are merged into the block

This creates a pipeline: **ProofCollection → Upgrade → SingleProof → Mempool → Block**


