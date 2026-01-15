use std::fmt::Display;
use std::time::Duration;
use std::time::SystemTime;

use serde::Deserialize;
use serde::Serialize;

use crate::protocol::consensus::block::Block;
use crate::protocol::consensus::type_scripts::native_currency_amount::NativeCurrencyAmount;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct GuessingWorkInfo {
    work_start: SystemTime,
    num_inputs: usize,
    num_outputs: usize,
    total_coinbase: NativeCurrencyAmount,
    pub(crate) total_guesser_fee: NativeCurrencyAmount,
}

impl GuessingWorkInfo {
    pub(crate) fn new(work_start: SystemTime, block: &Block) -> Self {
        Self {
            work_start,
            num_inputs: block.body().transaction_kernel.inputs.len(),
            num_outputs: block.body().transaction_kernel.outputs.len(),
            total_coinbase: block.body().transaction_kernel.coinbase.unwrap_or_default(),
            total_guesser_fee: block.body().transaction_kernel.fee,
        }
    }
}

/// Tracks the stage of a binary merge operation.
///
/// Binary merge processes multiple transactions in parallel:
/// - Level 1: Parallel merges of transaction pairs
/// - Level 2: Final merge of intermediate results
///
/// This tracking allows smarter cancellation decisions - once Level 1
/// is complete, it's usually worth letting Level 2 finish since it's
/// much faster than restarting the entire merge process.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryMergeStage {
    /// No binary merge is active (sequential merge or not merging)
    #[default]
    None,
    /// Binary merge Level 1 is in progress (parallel transaction merges)
    Level1InProgress {
        /// Number of transactions being merged
        num_transactions: usize,
    },
    /// Binary merge Level 1 complete, Level 2 in progress (final merge)
    Level2InProgress {
        /// Number of transactions that were merged
        num_transactions: usize,
    },
}

impl BinaryMergeStage {
    /// Returns true if a binary merge is currently active
    pub fn is_active(&self) -> bool {
        !matches!(self, BinaryMergeStage::None)
    }

    /// Returns true if Level 1 has completed (Level 2 is in progress or completed)
    pub fn level1_complete(&self) -> bool {
        matches!(self, BinaryMergeStage::Level2InProgress { .. })
    }

    /// Returns true if Level 1 is currently in progress
    pub fn level1_in_progress(&self) -> bool {
        matches!(self, BinaryMergeStage::Level1InProgress { .. })
    }
}

impl Display for BinaryMergeStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryMergeStage::None => write!(f, "none"),
            BinaryMergeStage::Level1InProgress { num_transactions } => {
                write!(f, "level1 ({} txs)", num_transactions)
            }
            BinaryMergeStage::Level2InProgress { num_transactions } => {
                write!(f, "level2 ({} txs)", num_transactions)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ComposingWorkInfo {
    // Only this info is available at the beginning of the composition work.
    // The rest of the information will have to be read from the log.
    work_start: SystemTime,
    /// Tracks the current stage of binary merge, if active.
    /// This allows the main loop to make smarter decisions about
    /// whether to cancel the composing task.
    binary_merge_stage: BinaryMergeStage,
}

impl ComposingWorkInfo {
    pub(crate) fn new(work_start: SystemTime) -> Self {
        Self {
            work_start,
            binary_merge_stage: BinaryMergeStage::None,
        }
    }

    /// Returns the current binary merge stage
    pub fn binary_merge_stage(&self) -> BinaryMergeStage {
        self.binary_merge_stage
    }

    /// Creates a new ComposingWorkInfo with the binary merge stage set
    pub(crate) fn with_binary_merge_stage(mut self, stage: BinaryMergeStage) -> Self {
        self.binary_merge_stage = stage;
        self
    }
}

#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize)]
pub enum MiningStatus {
    Guessing(GuessingWorkInfo),
    Composing(ComposingWorkInfo),

    #[default]
    Inactive,
}

impl MiningStatus {
    /// Returns the binary merge stage if composing, None otherwise
    pub fn binary_merge_stage(&self) -> Option<BinaryMergeStage> {
        match self {
            MiningStatus::Composing(info) => Some(info.binary_merge_stage()),
            _ => None,
        }
    }

    /// Returns true if a binary merge is currently active
    pub fn is_binary_merge_active(&self) -> bool {
        self.binary_merge_stage()
            .map(|s| s.is_active())
            .unwrap_or(false)
    }

    /// Returns true if binary merge Level 1 has completed
    pub fn is_binary_merge_level1_complete(&self) -> bool {
        self.binary_merge_stage()
            .map(|s| s.level1_complete())
            .unwrap_or(false)
    }
}

impl Display for MiningStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let elapsed_time_exact = match self {
            MiningStatus::Guessing(guessing_work_info) => Some(
                guessing_work_info
                    .work_start
                    .elapsed()
                    .unwrap_or(Duration::ZERO),
            ),
            MiningStatus::Composing(composing_work_info) => Some(
                composing_work_info
                    .work_start
                    .elapsed()
                    .unwrap_or(Duration::ZERO),
            ),
            MiningStatus::Inactive => None,
        };
        // remove sub-second component, so humantime ends with seconds.
        let elapsed_time = elapsed_time_exact.map(|v| {
            v.checked_sub(Duration::from_nanos(v.subsec_nanos().into()))
                .unwrap()
        });
        let input_output_info = match self {
            MiningStatus::Guessing(info) => {
                format!(" {}/{}", info.num_inputs, info.num_outputs)
            }
            _ => String::default(),
        };

        let work_type_and_duration = match self {
            MiningStatus::Guessing(_) => {
                format!(
                    "guessing for {}",
                    humantime::format_duration(elapsed_time.unwrap())
                )
            }
            MiningStatus::Composing(info) => {
                let merge_info = if info.binary_merge_stage.is_active() {
                    format!(" [binary merge: {}]", info.binary_merge_stage)
                } else {
                    String::default()
                };
                format!(
                    "composing for {}{}",
                    humantime::format_duration(elapsed_time.unwrap()),
                    merge_info
                )
            }
            MiningStatus::Inactive => "inactive".to_owned(),
        };
        let reward = match self {
            MiningStatus::Guessing(block_work_info) => format!(
                "; total guesser reward: {}",
                block_work_info.total_guesser_fee
            ),
            _ => String::default(),
        };

        write!(f, "{work_type_and_duration}{input_output_info}{reward}",)
    }
}

#[cfg(feature = "mock-rpc")]
impl rand::distr::Distribution<MiningStatus> for rand::distr::StandardUniform {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> MiningStatus {
        let random_time = SystemTime::UNIX_EPOCH + Duration::from_millis(rng.next_u64() >> 20);
        match rng.random_range(0..3) {
            0 => MiningStatus::Inactive,
            1 => {
                let composing_work_info = ComposingWorkInfo {
                    work_start: random_time,
                    binary_merge_stage: BinaryMergeStage::None,
                };
                MiningStatus::Composing(composing_work_info)
            }
            2 => {
                let guessing_work_info = GuessingWorkInfo {
                    work_start: random_time,
                    num_inputs: rng.random_range(0..10000),
                    num_outputs: rng.random_range(0..10000),
                    total_coinbase: rng.random(),
                    total_guesser_fee: rng.random(),
                };
                MiningStatus::Guessing(guessing_work_info)
            }
            _ => unreachable!(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn binary_merge_stage_is_active() {
        assert!(!BinaryMergeStage::None.is_active());
        assert!(BinaryMergeStage::Level1InProgress {
            num_transactions: 3
        }
        .is_active());
        assert!(BinaryMergeStage::Level2InProgress {
            num_transactions: 3
        }
        .is_active());
    }

    #[test]
    fn binary_merge_stage_level1_complete() {
        assert!(!BinaryMergeStage::None.level1_complete());
        assert!(!BinaryMergeStage::Level1InProgress {
            num_transactions: 3
        }
        .level1_complete());
        assert!(BinaryMergeStage::Level2InProgress {
            num_transactions: 3
        }
        .level1_complete());
    }

    #[test]
    fn mining_status_binary_merge_helpers() {
        let inactive = MiningStatus::Inactive;
        assert!(!inactive.is_binary_merge_active());
        assert!(!inactive.is_binary_merge_level1_complete());

        let composing_no_merge = MiningStatus::Composing(ComposingWorkInfo::new(SystemTime::now()));
        assert!(!composing_no_merge.is_binary_merge_active());
        assert!(!composing_no_merge.is_binary_merge_level1_complete());

        let composing_level1 = MiningStatus::Composing(
            ComposingWorkInfo::new(SystemTime::now()).with_binary_merge_stage(
                BinaryMergeStage::Level1InProgress {
                    num_transactions: 3,
                },
            ),
        );
        assert!(composing_level1.is_binary_merge_active());
        assert!(!composing_level1.is_binary_merge_level1_complete());

        let composing_level2 = MiningStatus::Composing(
            ComposingWorkInfo::new(SystemTime::now()).with_binary_merge_stage(
                BinaryMergeStage::Level2InProgress {
                    num_transactions: 3,
                },
            ),
        );
        assert!(composing_level2.is_binary_merge_active());
        assert!(composing_level2.is_binary_merge_level1_complete());
    }
}
