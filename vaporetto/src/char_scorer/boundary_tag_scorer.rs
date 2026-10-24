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
use crate::ngram_model::{NgramModel, TagNgramModel};
use crate::predictor::{
    BoundaryWeights, PatternValue, PositionalWeightWithTag, TagWeights, TAG_WEIGHT_FLAG,
};
use crate::sentence::Sentence;
use crate::utils::Bytes;

pub struct CharScorerBoundaryTag {
    #[cfg(not(feature = "charwise-pma"))]
    pma: DoubleArrayAhoCorasick<PatternValue>,
    #[cfg(feature = "charwise-pma")]
    pma: CharwiseDoubleArrayAhoCorasick<PatternValue>,
    weights: BoundaryWeights,
    tag_weight: TagWeights,
}

impl<'de> Deserialize<'de> for CharScorerBoundaryTag {
    /// WARNING: This function is inherently unsafe. Do not publish this function outside this
    /// crate.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let (pma_data, weights, tag_weight): (&[u8], _, _) =
            Deserialize::deserialize(deserializer)?;
        #[cfg(not(feature = "charwise-pma"))]
        let (pma, _) = unsafe { DoubleArrayAhoCorasick::deserialize_unchecked(pma_data) };
        #[cfg(feature = "charwise-pma")]
        let (pma, _) = unsafe { CharwiseDoubleArrayAhoCorasick::deserialize_unchecked(pma_data) };
        Ok(Self {
            pma,
            weights,
            tag_weight,
        })
    }
}

impl Serialize for CharScorerBoundaryTag {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        (
            Bytes(&self.pma.serialize()),
            &self.weights,
            &self.tag_weight,
        )
            .serialize(serializer)
    }
}

impl CharScorerBoundaryTag {
    pub fn new(
        ngram_model: NgramModel<String>,
        dict_model: DictModel,
        window_size: u8,
        tag_ngram_model: Vec<TagNgramModel<String>>,
    ) -> Result<Self> {
        let mut merger = CharWeightMerger::default();
        for d in ngram_model.0 {
            let weight = PositionalWeightWithTag::with_boundary(-i16::from(window_size), d.weights);
            merger.add(d.ngram, weight);
        }
        for d in dict_model.0 {
            let word_len = d.word.chars().count();
            let word_len = i16::try_from(word_len).map_err(|_| {
                VaporettoError::invalid_model(
                    "words must be shorter than or equal to 32767 characters",
                )
            })?;
            let weight = PositionalWeightWithTag::with_boundary(-word_len, d.weights);
            merger.add(d.word, weight);
        }
        let mut tag_weight = TagWeights::new(tag_ngram_model.len(), window_size)?;
        for (i, tag_model) in tag_ngram_model.into_iter().enumerate() {
            for d in tag_model.0 {
                for w in d.weights {
                    let weight = PositionalWeightWithTag::with_tag(i, w.rel_position, w.weights);
                    merger.add(&d.ngram, weight);
                }
            }
        }
        let mut patvals = vec![];
        let mut weights = BoundaryWeights::new(window_size);
        for (i, (ngram, weight)) in merger.merge().into_iter().enumerate() {
            let value = if let Some(w) = weight.weight {
                weights.push(w)?
            } else {
                BoundaryWeights::no_weight_value(i)?
            };
            let mut value = value;
            if !weight.tag_info.is_empty() {
                value.id |= TAG_WEIGHT_FLAG;
            }
            for ((token_id, rel_position), weight) in weight.tag_info {
                tag_weight.insert(token_id, rel_position, value.id, weight)?;
            }
            patvals.push((ngram, value));
        }
        #[cfg(not(feature = "charwise-pma"))]
        let pma = DoubleArrayAhoCorasick::with_values(patvals)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        #[cfg(feature = "charwise-pma")]
        let pma = CharwiseDoubleArrayAhoCorasick::with_values(patvals)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        Ok(Self {
            pma,
            weights,
            tag_weight,
        })
    }

    #[inline(always)]
    pub fn add_scores(&self, sentence: &mut Sentence) {
        sentence.char_pma_states.clear();
        sentence.char_pma_states.resize(sentence.len(), u32::MAX);
        #[cfg(not(feature = "charwise-pma"))]
        let it = self
            .pma
            .find_overlapping_no_suffix_iter(sentence.text.as_bytes());
        #[cfg(feature = "charwise-pma")]
        let it = self.pma.find_overlapping_no_suffix_iter(&sentence.text);
        for m in it {
            debug_assert!(m.end() != 0 && sentence.text.is_char_boundary(m.end()));
            let end = unsafe { sentence.str_to_char_pos(m.end()) };
            // The value is always returned by `BoundaryWeights::push()` or
            // `BoundaryWeights::no_weight_value()`.
            let value = m.value();
            unsafe {
                self.weights.add_score(
                    &value,
                    end + sentence.score_padding - 1,
                    &mut sentence.boundary_scores,
                );
            }
            debug_assert!(end <= sentence.char_pma_states.len());
            unsafe { *sentence.char_pma_states.get_unchecked_mut(end - 1) = value.id };
        }
    }

    /// # Satety
    ///
    /// `pos` must be smaller than `sentence.char_pma_states.len()`.
    #[inline(always)]
    pub unsafe fn add_tag_scores(
        &self,
        token_id: u32,
        pos: usize,
        sentence: &Sentence,
        scores: &mut [i32],
    ) {
        self.tag_weight.add_scores(
            token_id,
            sentence.char_pma_states.get_unchecked(pos..),
            scores,
        );
    }
}
