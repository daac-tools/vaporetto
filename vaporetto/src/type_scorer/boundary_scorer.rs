use alloc::vec::Vec;

use daachorse::DoubleArrayAhoCorasick;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::predictor::{BoundaryWeights, PatternValue, PositionalWeight};
use crate::sentence::Sentence;
use crate::type_scorer::TypeWeightMerger;
use crate::utils::Bytes;

pub struct TypeScorerBoundary {
    pma: DoubleArrayAhoCorasick<PatternValue>,
    weights: BoundaryWeights,
}

impl<'de> Deserialize<'de> for TypeScorerBoundary {
    /// WARNING: This function is inherently unsafe. Do not publish this function outside this
    /// crate.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (pma_data, weights): (&[u8], _) = Deserialize::deserialize(deserializer)?;
        let (pma, _) = unsafe { DoubleArrayAhoCorasick::deserialize_unchecked(pma_data) };
        Ok(Self { pma, weights })
    }
}

impl Serialize for TypeScorerBoundary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (Bytes(&self.pma.serialize()), &self.weights).serialize(serializer)
    }
}

impl TypeScorerBoundary {
    pub fn new(ngram_model: NgramModel<Vec<u8>>, window_size: u8) -> Result<Self> {
        let mut merger = TypeWeightMerger::default();
        for d in ngram_model.0 {
            let weight = PositionalWeight::new(-i16::from(window_size), d.weights);
            merger.add(d.ngram, weight);
        }
        let mut patvals = vec![];
        let mut weights = BoundaryWeights::new(window_size);
        for (ngram, weight) in merger.merge() {
            patvals.push((ngram, weights.push(weight)?));
        }
        let pma = DoubleArrayAhoCorasick::with_values(patvals)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        Ok(Self { pma, weights })
    }

    #[inline(always)]
    pub fn add_scores(&self, sentence: &mut Sentence) {
        for m in self
            .pma
            .find_overlapping_no_suffix_iter(&sentence.char_types)
        {
            debug_assert!(m.end() != 0 && m.end() <= sentence.char_types.len());
            // The value is always returned by `BoundaryWeights::push()`.
            unsafe {
                self.weights.add_score(
                    &m.value(),
                    m.end() + sentence.score_padding - 1,
                    &mut sentence.boundary_scores,
                );
            }
        }
    }
}
