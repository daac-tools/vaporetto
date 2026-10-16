use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "charwise-pma")]
use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
#[cfg(not(feature = "charwise-pma"))]
use daachorse::DoubleArrayAhoCorasick;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::char_scorer::CharWeightMerger;
use crate::dict_model::DictModel;
use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::predictor::{PositionalWeight, WeightVector};
use crate::sentence::Sentence;
use crate::utils::Bytes;

pub struct CharScorerBoundary {
    #[cfg(not(feature = "charwise-pma"))]
    pma: DoubleArrayAhoCorasick<u32>,
    #[cfg(feature = "charwise-pma")]
    pma: CharwiseDoubleArrayAhoCorasick<u32>,
    weights: Vec<PositionalWeight<WeightVector>>,
}

impl<'de> Deserialize<'de> for CharScorerBoundary {
    /// WARNING: This function is inherently unsafe. Do not publish this function outside this
    /// crate.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (pma_data, weights): (&[u8], _) = Deserialize::deserialize(deserializer)?;
        #[cfg(not(feature = "charwise-pma"))]
        let (pma, _) = unsafe { DoubleArrayAhoCorasick::deserialize_unchecked(pma_data) };
        #[cfg(feature = "charwise-pma")]
        let (pma, _) = unsafe { CharwiseDoubleArrayAhoCorasick::deserialize_unchecked(pma_data) };
        Ok(Self { pma, weights })
    }
}

impl Serialize for CharScorerBoundary {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (Bytes(&self.pma.serialize()), &self.weights).serialize(serializer)
    }
}

impl CharScorerBoundary {
    pub fn new(
        ngram_model: NgramModel<String>,
        dict_model: DictModel,
        window_size: u8,
    ) -> Result<Self> {
        let mut merger = CharWeightMerger::default();
        for d in ngram_model.0 {
            let weight = PositionalWeight::new(-i16::from(window_size), d.weights);
            merger.add(d.ngram, weight);
        }
        for d in dict_model.0 {
            let word_len = d.word.chars().count();
            let word_len = i16::try_from(word_len).map_err(|_| {
                VaporettoError::invalid_model(
                    "words must be shorter than or equal to 32767 characters",
                )
            })?;
            let weight = PositionalWeight::new(-word_len, d.weights);
            merger.add(d.word, weight);
        }
        let mut ngrams = vec![];
        let mut weights = vec![];
        for (ngram, weight) in merger.merge() {
            ngrams.push(ngram);
            weights.push(weight.into());
        }
        #[cfg(not(feature = "charwise-pma"))]
        let pma = DoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        #[cfg(feature = "charwise-pma")]
        let pma = CharwiseDoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        Ok(Self { pma, weights })
    }

    #[allow(clippy::cast_possible_wrap)]
    #[inline(always)]
    pub fn add_scores(&self, sentence: &mut Sentence) {
        #[cfg(not(feature = "charwise-pma"))]
        let it = self
            .pma
            .find_overlapping_no_suffix_iter(sentence.text.as_bytes());
        #[cfg(feature = "charwise-pma")]
        let it = self.pma.find_overlapping_no_suffix_iter(&sentence.text);
        for m in it {
            debug_assert!(m.end() != 0 && sentence.text.is_char_boundary(m.end()));
            let end = unsafe { sentence.str_to_char_pos(m.end()) };
            debug_assert!(usize::try_from(m.value()).unwrap() < self.weights.len());
            let weight = unsafe {
                self.weights
                    .get_unchecked(usize::try_from(m.value()).unwrap())
            };
            weight.add_score(
                (end + sentence.score_padding - 1) as isize,
                &mut sentence.boundary_scores,
            );
        }
    }
}
