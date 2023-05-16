//! Implements BinaryFuse16 filters.

use crate::{bfusep_contains_impl, bfusep_from_impl, Filter};
use alloc::{boxed::Box, vec::Vec};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A `BinaryFuseP32` filter is an Xor-like filter with 32-bit fingerprints arranged in a binary-partitioned [fuse graph].
/// `BinaryFuseP32`s are similar to [`Fuse32`]s, but their construction is faster, uses less
/// memory, and is more likely to succeed.
///
/// A `BinaryFuseP32` filter uses ≈36 bits per entry of the set is it constructed from, and has a false
/// positive rate of effectively zero (1/2^32 =~ 1/4 billion). As with other
/// probabilistic filters, a higher number of entries decreases the bits per
/// entry but increases the false positive rate.
///
/// A `BinaryFuseP32` is constructed from a set of 64-bit unsigned integers and is immutable.
/// Construction may fail, but usually only if there are duplicate keys.
///
/// ```
/// # extern crate alloc;
/// use xorf::{Filter, BinaryFuseP32};
/// use core::convert::TryFrom;
/// # use alloc::vec::Vec;
/// # use rand::Rng;
///
/// # let mut rng = rand::thread_rng();
/// const SAMPLE_SIZE: usize = 1_000_000;
/// const PTXT_MOD: u64 = 1_024;
/// let keys: Vec<u64> = (0..SAMPLE_SIZE).map(|_| rng.gen()).collect();
/// let filter = BinaryFuseP32::from_slice(&keys, PTXT_MOD).unwrap();
///
/// // no false negatives
/// for key in keys {
///     assert!(filter.contains(&key));
/// }
///
/// // bits per entry
/// let bpe = (filter.len() as f64) * 32.0 / (SAMPLE_SIZE as f64);
/// assert!(bpe < (PTXT_MOD as f64).log(2.0) + 2.0, "Bits per entry is {}", bpe);
///
/// // false positive rate
/// let false_positives: usize = (0..SAMPLE_SIZE)
///     .map(|_| rng.gen())
///     .filter(|n| filter.contains(n))
///     .count();
/// let fp_rate: f64 = (false_positives * 100) as f64 / SAMPLE_SIZE as f64;
/// assert!(fp_rate < (0.426 * 256.0) / (PTXT_MOD as f64), "False positive rate is {}", fp_rate);
/// ```
///
/// Serializing and deserializing `BinaryFuseP32` filters can be enabled with the [`serde`] feature.
///
/// [fuse graph]: https://arxiv.org/abs/1907.04749
/// [`Fuse32`]: crate::Fuse32
/// [`serde`]: http://serde.rs
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Debug)]
pub struct BinaryFuseP32 {
    seed: u64,
    segment_length: u32,
    segment_length_mask: u32,
    segment_count_length: u32,
    /// The fingerprints for the filter
    pub fingerprints: Box<[u32]>,
    ptxt_mod: u64,
}

impl Filter<u64> for BinaryFuseP32 {
    /// Returns `true` if the filter contains the specified key.
    /// Has a false positive rate of <0.4%.
    /// Has no false negatives.
    fn contains(&self, key: &u64) -> bool {
        bfusep_contains_impl!(*key, self)
    }

    fn len(&self) -> usize {
        self.fingerprints.len()
    }
}

impl BinaryFuseP32 {
    pub fn from_slice(keys: &[u64], ptxt_mod: u64) -> Result<Self, &'static str> {
        bfusep_from_impl!(keys, ptxt_mod, max iter 1_000)
    }

    pub fn from_vec(keys: Vec<u64>, ptxt_mod: u64) -> Result<Self, &'static str> {
        let slice = keys.as_slice();
        bfusep_from_impl!(slice, ptxt_mod, max iter 1_000)
    }
}

#[cfg(test)]
mod test {
    use crate::{BinaryFuseP32, Filter};
    use core::convert::TryFrom;

    use alloc::vec::Vec;
    use rand::Rng;

    #[test]
    fn test_initialization() {
        const SAMPLE_SIZE: usize = 1_000_000;
        const PTXT_MOD: u64 = 1024;
        let mut rng = rand::thread_rng();
        let keys: Vec<u64> = (0..SAMPLE_SIZE).map(|_| rng.gen()).collect();

        let filter = BinaryFuseP32::from_slice(&keys, PTXT_MOD).unwrap();

        for key in keys {
            assert!(filter.contains(&key));
        }
    }

    #[test]
    fn test_bits_per_entry() {
        const SAMPLE_SIZE: usize = 1_000_000;
        const PTXT_MOD: u64 = 1024;
        let mut rng = rand::thread_rng();
        let keys: Vec<u64> = (0..SAMPLE_SIZE).map(|_| rng.gen()).collect();

        let filter = BinaryFuseP32::from_slice(&keys, PTXT_MOD).unwrap();
        let bpe = (filter.len() as f64) * (PTXT_MOD as f64).log(2.0) / (SAMPLE_SIZE as f64);

        assert!(bpe < (PTXT_MOD as f64).log(2.0) + 2.0, "Bits per entry is {}", bpe);
    }

    #[test]
    fn test_false_positives() {
        const SAMPLE_SIZE: usize = 1_000_000;
        const PTXT_MOD: u64 = 1024;
        let mut rng = rand::thread_rng();
        let keys: Vec<u64> = (0..SAMPLE_SIZE).map(|_| rng.gen()).collect();

        let filter = BinaryFuseP32::from_slice(&keys, PTXT_MOD).unwrap();

        let false_positives: usize = (0..SAMPLE_SIZE)
            .map(|_| rng.gen())
            .filter(|n| filter.contains(n))
            .count();
        let fp_rate: f64 = (false_positives * 100) as f64 / SAMPLE_SIZE as f64;
        assert!(
            fp_rate < (0.426 * 256.0) / (PTXT_MOD as f64),
            "False positive rate is {}",
            fp_rate
        );
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(
        expected = "Binary Fuse filters must be constructed from a collection containing all distinct keys."
    )]
    fn test_debug_assert_duplicates() {
        let _ = BinaryFuseP32::from_vec(vec![1, 2, 1], 1024);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(
        expected = "Binary Fuse filters must be constructed using a plaintext modulus >= 256."
    )]
    fn test_debug_assert_ptxt_mod() {
        let _ = BinaryFuseP32::from_vec(vec![1, 2], 128);
    }
}
