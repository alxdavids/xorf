//! Implements BinaryFuse16 filters.

use crate::{bfusep_retrieve_impl, bfusep_from_impl, bfusep_hash_eval_impl, Filter};
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
/// # use alloc::vec::Vec;
/// # use rand::Rng;
///
/// # let mut rng = rand::thread_rng();
/// const SAMPLE_SIZE: usize = 1_000_000;
/// const PTXT_MOD: u64 = 1_024;
/// let keys: Vec<[u64; 4]> = (0..SAMPLE_SIZE).map(|_| [rng.gen(); 4]).collect();
/// let data: Vec<u32> = (0..SAMPLE_SIZE).map(|i| (i as u32) % (PTXT_MOD as u32)).collect();
/// let seed: u128 = rng.gen_range(0..u128::MAX);
/// let filter = BinaryFuseP32::from_slice(seed, &keys, &data, PTXT_MOD).unwrap();
///
/// // no false negatives
/// for i in 0..keys.len() {
///     assert_eq!(data[i], filter.retrieve(&keys[i]));
/// }
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
    seed: u128,
    segment_length: u32,
    segment_length_mask: u32,
    segment_count_length: u32,
    /// The fingerprints for the filter
    pub fingerprints: Box<[u32]>,
    ptxt_mod: u64,
}

impl Filter<u64> for BinaryFuseP32 {
    /// unimplemented
    fn contains(&self, _: &u64) -> bool {
        unimplemented!();
    }

    fn len(&self) -> usize {
        self.fingerprints.len()
    }
}

impl BinaryFuseP32 {
    /// Creates a new `BinaryFuseP32` filter from the specified `keys` (as a slice), `data`, `ptxt_mod`
    pub fn from_slice(seed: u128, keys: &[[u64; 4]], data: &[u32], ptxt_mod: u64) -> Result<Self, &'static str> {
        if data.len() != keys.len() {
            return Err("The data should correspond to the number of keys");
        }
        bfusep_from_impl!(seed, keys, data, ptxt_mod, max iter 1_000)
    }

    /// Creates a new `BinaryFuseP32` filter from the specified `keys` (as a vector), `data`, `ptxt_mod`
    pub fn from_vec(seed: u128, keys: Vec<[u64; 4]>, data: &[u32], ptxt_mod: u64) -> Result<Self, &'static str> {
        let slice = keys.as_slice();
        bfusep_from_impl!(seed, slice, data, ptxt_mod, max iter 1_000)
    }

    /// Retrieves the `data` modulo the plaintext modulus for a given `key`
    pub fn retrieve(&self, key: &[u64; 4]) -> u32 {
        bfusep_retrieve_impl!(key, self)
    }

    /// Retrieves the hash function evaluations for a given key
    pub fn hash_eval(&self, key: &[u64; 4]) -> Vec<usize> {
        bfusep_hash_eval_impl!(key, self)
    }
}

#[cfg(test)]
mod test {
    use crate::{BinaryFuseP32, Filter};

    use alloc::vec::Vec;
    use rand::{Rng, RngCore};

    #[test]
    fn test_initialization() {
        const SAMPLE_SIZE: usize = 1_000_000;
        const PTXT_MOD: u64 = 1024;
        let mut rng = rand::thread_rng();
        let keys: Vec<[u64; 4]> = (0..SAMPLE_SIZE).map(|_| [rng.gen(); 4]).collect();
        let data: Vec<u32> = (0..SAMPLE_SIZE).map(|i| (i as u32) % (PTXT_MOD as u32)).collect();
        let seed: u128 = rng.gen_range(0..u128::MAX);

        let filter = BinaryFuseP32::from_slice(seed, &keys, &data, PTXT_MOD).unwrap();

        for i in 0..keys.len() {
            assert_eq!(data[i], filter.retrieve(&keys[i]));
        }
    }

    #[test]
    fn test_bits_per_entry() {
        const SAMPLE_SIZE: usize = 1_000_000;
        const PTXT_MOD: u64 = 1024;
        let mut rng = rand::thread_rng();
        let keys: Vec<[u64; 4]> = (0..SAMPLE_SIZE).map(|_| [rng.gen(); 4]).collect();
        let data: Vec<u32> = (0..SAMPLE_SIZE).map(|i| (i as u32) % (PTXT_MOD as u32)).collect();
        let seed: u128 = rng.gen_range(0..u128::MAX);

        let filter = BinaryFuseP32::from_slice(seed, &keys, &data, PTXT_MOD).unwrap();
        let bpe = (filter.len() as f64) * (PTXT_MOD as f64).log(2.0) / (SAMPLE_SIZE as f64);

        assert!(bpe < (PTXT_MOD as f64).log(2.0) + 2.0, "Bits per entry is {}", bpe);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(
        expected = "Binary Fuse filters must be constructed from a collection containing all distinct keys."
    )]
    fn test_debug_assert_duplicates() {
        let _ = BinaryFuseP32::from_vec(1u128, vec![[1; 4], [2; 4], [1; 4]], &[0, 0, 0], 1024);
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(
        expected = "Binary Fuse filters must be constructed using a plaintext modulus >= 256."
    )]
    fn test_debug_assert_ptxt_mod() {
        let _ = BinaryFuseP32::from_vec(1u128, vec![[1; 4], [2; 4]], &[0, 0], 128);
    }
}
