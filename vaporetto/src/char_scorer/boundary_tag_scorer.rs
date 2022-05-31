use alloc::string::String;
use alloc::vec::Vec;

use bincode::{
    de::BorrowDecoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
    BorrowDecode, Decode, Encode,
};
#[cfg(feature = "charwise-pma")]
use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
#[cfg(not(feature = "charwise-pma"))]
use daachorse::DoubleArrayAhoCorasick;
use hashbrown::HashMap;

use crate::char_scorer::CharWeightMerger;
use crate::dict_model::DictModel;
use crate::errors::{Result, VaporettoError};
use crate::ngram_model::{NgramModel, TagNgramModel};
use crate::predictor::{PositionalWeight, PositionalWeightWithTag, WeightVector};
use crate::sentence::Sentence;
use crate::utils::SplitMix64Builder;

pub struct CharScorerBoundaryTag {
    #[cfg(not(feature = "charwise-pma"))]
    pma: DoubleArrayAhoCorasick,
    #[cfg(feature = "charwise-pma")]
    pma: CharwiseDoubleArrayAhoCorasick,
    weights: Vec<Option<PositionalWeight<WeightVector>>>,
    tag_weight: Vec<Vec<HashMap<u32, WeightVector, SplitMix64Builder>>>,
}

impl<'de> BorrowDecode<'de> for CharScorerBoundaryTag {
    /// WARNING: This function is inherently unsafe. Do not publish this function outside this
    /// crate.
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let pma_data: &[u8] = BorrowDecode::borrow_decode(decoder)?;
        #[cfg(not(feature = "charwise-pma"))]
        let (pma, _) =
            unsafe { DoubleArrayAhoCorasick::deserialize_from_slice_unchecked(pma_data) };
        #[cfg(feature = "charwise-pma")]
        let (pma, _) =
            unsafe { CharwiseDoubleArrayAhoCorasick::deserialize_from_slice_unchecked(pma_data) };
        let tag_weight: Vec<Vec<Vec<(u32, WeightVector)>>> = Decode::decode(decoder)?;
        let tag_weight = tag_weight
            .iter()
            .map(|x| {
                x.iter()
                    .map(|x| x.iter().map(|(a, b)| (*a, b.clone())).collect())
                    .collect()
            })
            .collect();
        Ok(Self {
            pma,
            weights: Decode::decode(decoder)?,
            tag_weight,
        })
    }
}

impl Encode for CharScorerBoundaryTag {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let pma_data = self.pma.serialize_to_vec();
        Encode::encode(&pma_data, encoder)?;
        Encode::encode(&self.weights, encoder)?;
        let tag_weight: Vec<Vec<Vec<_>>> = self
            .tag_weight
            .iter()
            .map(|x| x.iter().map(|x| x.iter().collect()).collect())
            .collect();
        Encode::encode(&tag_weight, encoder)?;
        Ok(())
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
        let mut tag_weight =
            vec![
                vec![HashMap::with_hasher(SplitMix64Builder); usize::from(window_size) + 1];
                tag_ngram_model.len()
            ];
        for (i, tag_model) in tag_ngram_model.into_iter().enumerate() {
            for d in tag_model.0 {
                for (rel_position, weights) in d.weights {
                    let weight = PositionalWeightWithTag::with_tag(i, rel_position, weights);
                    merger.add(d.ngram.clone(), weight);
                }
            }
        }
        let mut ngrams = vec![];
        let mut weights = vec![];
        for (i, (ngram, weight)) in merger.merge().into_iter().enumerate() {
            ngrams.push(ngram);
            weights.push(weight.weight.map(|w| w.into()));
            for ((token_id, rel_position), weight) in weight.tag_info {
                tag_weight[token_id][usize::from(rel_position)]
                    .insert(u32::try_from(i).unwrap(), weight.into());
            }
        }
        #[cfg(not(feature = "charwise-pma"))]
        let pma = DoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        #[cfg(feature = "charwise-pma")]
        let pma = CharwiseDoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        Ok(Self {
            pma,
            weights,
            tag_weight,
        })
    }

    #[allow(clippy::cast_possible_truncation)]
    #[allow(clippy::cast_possible_wrap)]
    #[inline(always)]
    pub fn add_scores<'a, 'b>(&self, sentence: &mut Sentence<'a, 'b>) {
        sentence.char_pma_states.clear();
        sentence.char_pma_states.resize(sentence.len(), u32::MAX);
        #[cfg(not(feature = "charwise-pma"))]
        let it = self
            .pma
            .find_overlapping_no_suffix_iter(sentence.text.as_bytes());
        #[cfg(feature = "charwise-pma")]
        let it = self.pma.find_overlapping_no_suffix_iter(&sentence.text);
        for m in it {
            debug_assert!(m.end() != 0 && m.end() <= sentence.text.len());
            let end = unsafe { sentence.str_to_char_pos(m.end()) };
            debug_assert!(m.value() < self.weights.len());
            if let Some(weight) = unsafe { self.weights.get_unchecked(m.value()).as_ref() } {
                weight.add_score(
                    (end + sentence.score_padding - 1) as isize,
                    &mut sentence.boundary_scores,
                );
            }
            debug_assert!(end as usize <= sentence.char_pma_states.len());
            unsafe {
                *sentence.char_pma_states.get_unchecked_mut(end as usize - 1) = m.value() as u32
            };
        }
    }

    /// # Satety
    ///
    /// `token_id` must be smaller than `scorer.tag_weight.len()`.
    /// `pos` must be smaller than `sentence.char_pma_states.len()`.
    #[inline(always)]
    pub unsafe fn add_tag_scores(
        &self,
        token_id: u32,
        pos: usize,
        sentence: &Sentence,
        scores: &mut [i32],
    ) {
        let tag_weight = self
            .tag_weight
            .get_unchecked(usize::try_from(token_id).unwrap());
        for (state_id, tag_weights) in sentence
            .char_pma_states
            .get_unchecked(pos..)
            .iter()
            .zip(tag_weight)
        {
            if let Some(weight) = tag_weights.get(state_id) {
                weight.add_scores(scores);
            }
        }
    }
}
