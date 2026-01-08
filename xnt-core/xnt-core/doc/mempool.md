# Analysis: Mempool Issue and Upgrading Discussion vs Implementation

Based on the discussion thread at [Neptune Talk](https://talk.neptune.cash/t/thoughts-on-the-mempool-issue-and-upgrading/52/26) and the current codebase implementation.

## Summary of Discussion Points

From the web search results and codebase analysis, the discussion covered several key proposals:

1. **Incentivizing Upgraders** - Make it profitable for nodes to upgrade transactions
2. **Mempool Redesign** - Prevent frequent clearing and ensure transactions aren't forgotten
3. **Coordination Among Upgraders** - Avoid redundant work when multiple nodes upgrade
4. **Fee Structures** - Adjust fees to encourage transaction inclusion in blocks
5. **Zero-Fee Transaction Handling** - Zero-fee transactions might not add positive weight to block proposals
6. **Resource Contention** - Balance resources between upgrading and composing

---

## Implementation Status Analysis

### ✅ **IMPLEMENTED**

#### 1. **Fee-Based Incentives for Upgraders** ✅

**Proposal:** Introduce mechanisms to make it profitable for nodes to upgrade transactions.

**Implementation:**
- **Location**: `cli_args.rs:217-224`, `proof_upgrader.rs:274-286`, `upgrade_incentive.rs`
- **Features**:
  - `gobbling_fraction` (default: 0.6) - Fraction of transaction fee collected by upgrader
  - `min_gobbling_fee` (default: 0.01 coins) - Minimum fee threshold for upgrading
  - `UpgradeIncentive` enum with three levels:
    - `Gobble(amount)` - Fee-based incentive
    - `BalanceAffecting(amount)` - After fees are gobbled
    - `Critical` - For own transactions

**Code Evidence:**
```rust
// proof_upgrader.rs:880-883
let gobbling_potential = kernel.fee.lossy_f64_fraction_mul(gobbling_fraction);
let upgrade_incentive = upgrade_priority.incentive_given_gobble_potential(gobbling_potential);
if upgrade_incentive.upgrade_is_worth_it(min_gobbling_fee) {
    // Upgrade transaction
}
```

**Status:** ✅ **FULLY IMPLEMENTED** - Upgraders can collect fees through "gobbling" mechanism.

---

#### 2. **Coordination Among Upgraders** ✅

**Proposal:** Implement mechanism to coordinate upgraders to avoid redundant work.

**Implementation:**
- **Location**: `tx_upgrade_filter.rs`, `proof_upgrader.rs:868`
- **Feature**: `tx_upgrade_filter` - Filter based on `txid % divisor == remainder`
- **Default**: `1:0` (matches all transactions)
- **Usage**: Nodes can set filters like `4:0`, `4:1`, `4:2`, `4:3` to split work

**Code Evidence:**
```rust
// tx_upgrade_filter.rs:60-65
pub(crate) fn matches(&self, txid: TransactionKernelId) -> bool {
    let txid: Digest = txid.into();
    let txid: [u8; Digest::BYTES] = txid.into();
    txid.last().unwrap() % self.divisor == self.remainder
}
```

**Status:** ✅ **IMPLEMENTED** - Basic coordination via TXID-based filtering exists, but requires manual configuration.

**Limitation:** No automatic coordination protocol - nodes must manually configure filters.

---

#### 3. **Fee Density-Based Transaction Selection** ✅

**Proposal:** Use fee structures to encourage transaction inclusion in blocks.

**Implementation:**
- **Location**: `mempool.rs:27-44`, `transaction/mod.rs:211-218`
- **Feature**: Transactions sorted by `FeeDensity = Fee / Size`
- **Usage**: `get_transactions_for_block_composition()` iterates by descending fee density

**Code Evidence:**
```rust
// mempool.rs:822
for (transaction_digest, _fee_density) in self.fee_density_iter() {
    // Selects transactions in order of highest fee density first
}
```

**Status:** ✅ **FULLY IMPLEMENTED** - Fee density sorting is used for block composition.

---

### ⚠️ **PARTIALLY IMPLEMENTED**

#### 4. **Zero-Fee Transaction Handling** ⚠️

**Proposal (from sword-smith):** Zero-fee transactions might not add positive weight to block proposals.

**Current Implementation:**
- **Location**: `mempool.rs:427-433`, `get_transactions_for_block_composition()`
- **For Merging**: Zero-fee transactions are **skipped** if `upgrade_priority.is_irrelevant()`
- **For Block Composition**: Zero-fee transactions are **NOT explicitly filtered** - they can be included if they have high fee density (which would be 0)

**Code Evidence:**
```rust
// mempool.rs:427-433 (for merging)
if candidate.upgrade_priority.is_irrelevant()
    && candidate.transaction.kernel.fee.is_zero()
{
    continue; // Skip zero-fee transactions for merging
}

// mempool.rs:822-848 (for block composition)
// No explicit zero-fee check - transactions sorted by fee_density
// Zero-fee transactions would have fee_density = 0, so they'd be last
```

**Status:** ⚠️ **PARTIALLY IMPLEMENTED**
- ✅ Zero-fee transactions skipped for merging
- ❌ Zero-fee transactions can still be included in blocks (but would be lowest priority)
- **Gap**: No explicit exclusion of zero-fee transactions from block composition

**Recommendation:** Add explicit zero-fee check in `get_transactions_for_block_composition()` if desired.

---

### ❌ **REJECTED / NOT IMPLEMENTED**

#### 5. **Mempool Redesign** ❌

**Proposal:** Redesign mempool to prevent frequent clearing and ensure transactions aren't forgotten.

**Status:** ❌ **REJECTED / RE-EVALUATED**

According to the search results:
- Work began on mempool redesign ([dev-update-6th-may-2025](https://talk.neptune.cash/t/dev-update-6th-may-2025/50))
- Faced challenges and was re-evaluated ([dev-update-20th-may-2025](https://talk.neptune.cash/t/dev-update-20th-may-2025/64))
- Shifted towards incremental patches instead of full redesign

**Current State:**
- Mempool still uses basic size-based eviction (`pop_min()` removes lowest fee density)
- No persistent mempool storage
- Transactions can still be lost on restart

---

#### 6. **Automatic Coordination Protocol** ❌

**Proposal:** Implement automatic coordination mechanism (not just manual filters).

**Status:** ❌ **NOT IMPLEMENTED**

**Current State:**
- Only manual `tx_upgrade_filter` configuration exists
- No protocol for nodes to discover and coordinate with each other
- No consensus on which node should upgrade which transaction
- Risk of duplicate work if filters aren't properly configured

**Gap:** No automatic discovery or negotiation between upgraders.

---

#### 7. **Resource Contention Management** ⚠️

**Proposal:** Develop strategies to balance resources between upgrading and composing.

**Status:** ⚠️ **PARTIALLY ADDRESSED**

**Current Implementation:**
- **Job Queue**: `TritonVmJobQueue` ensures only one proof job runs at a time
- **Priority System**: `TritonVmJobPriority` (High for Critical, Low for Gobble)
- **No Explicit Balancing**: No automatic resource allocation between upgrading and composing

**Code Evidence:**
```rust
// proof_upgrader.rs:422-425
let priority = match upgrade_incentive {
    UpgradeIncentive::Critical => TritonVmJobPriority::High,
    _ => TritonVmJobPriority::Low,
};
```

**Gap:** No explicit policy to prioritize composing over upgrading or vice versa. Both compete for the same prover resource.

---

## Detailed Code References

### Fee-Based Incentives
- **Gobbling Fraction**: `cli_args.rs:217` - `gobbling_fraction: f64` (default: 0.6)
- **Min Gobbling Fee**: `cli_args.rs:223` - `min_gobbling_fee: NativeCurrencyAmount` (default: 0.01)
- **Upgrade Incentive Calculation**: `proof_upgrader.rs:880-883`
- **Gobbling Fee Collection**: `proof_upgrader.rs:274-286`, `upgrade_incentive.rs:22-27`

### Coordination
- **TX Upgrade Filter**: `tx_upgrade_filter.rs` - `TxUpgradeFilter { divisor, remainder }`
- **Filter Usage**: `proof_upgrader.rs:868`, `mempool.rs:355` (in `preferred_proof_collection`)

### Zero-Fee Handling
- **Merging Skip**: `mempool.rs:427-433`
- **Block Composition**: `mempool.rs:815-853` (no explicit zero-fee check)

### Fee Density
- **Calculation**: `transaction/mod.rs:211-218`
- **Sorting**: `mempool.rs:1174-1179` (`fee_density_iter()`)
- **Block Selection**: `mempool.rs:822` (iterates by fee density)

---

## Recommendations

### High Priority
1. **Explicit Zero-Fee Exclusion**: Add zero-fee check in `get_transactions_for_block_composition()` if zero-fee transactions should be excluded from blocks.

2. **Resource Balancing**: Consider adding explicit priority rules (e.g., composing always takes priority over non-critical upgrades).

### Medium Priority
3. **Automatic Coordination**: Design a protocol for nodes to automatically coordinate upgrades (e.g., via peer-to-peer negotiation).

4. **Mempool Persistence**: Revisit mempool redesign with incremental approach (e.g., optional persistent storage).

### Low Priority
5. **Enhanced Filtering**: Extend `tx_upgrade_filter` to support more sophisticated coordination strategies.

---

## Conclusion

Most of the core proposals from the discussion thread have been **implemented**, particularly:
- ✅ Fee-based incentives (gobbling mechanism)
- ✅ Basic coordination (TXID-based filters)
- ✅ Fee density sorting

However, some proposals were **rejected or deferred**:
- ❌ Full mempool redesign (too complex, shifted to incremental patches)
- ❌ Automatic coordination protocol (not implemented, requires manual config)

The system is functional but could benefit from:
- Explicit zero-fee transaction handling in block composition
- Better resource contention management
- Automatic coordination protocols

