use alloc::vec::Vec;

use bincode::{
    de::BorrowDecoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
    BorrowDecode, Decode, Encode,
};
use daachorse::DoubleArrayAhoCorasick;
use hashbrown::HashMap;

use crate::errors::{Result, VaporettoError};
use crate::ngram_model::{NgramModel, TagNgramModel};
use crate::predictor::{PositionalWeight, PositionalWeightWithTag, WeightVector};
use crate::sentence::Sentence;
use crate::type_scorer::TypeWeightMerger;
use crate::utils::SplitMix64Builder;

pub struct TypeScorerBoundaryTag {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<Option<PositionalWeight<WeightVector>>>,
    tag_weight: Vec<Vec<HashMap<u32, WeightVector, SplitMix64Builder>>>,
}

impl<'de> BorrowDecode<'de> for TypeScorerBoundaryTag {
    /// WARNING: This function is inherently unsafe. Do not publish this function outside this
    /// crate.
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let pma_data: &[u8] = BorrowDecode::borrow_decode(decoder)?;
        let (pma, _) =
            unsafe { DoubleArrayAhoCorasick::deserialize_from_slice_unchecked(pma_data) };
        let tag_weight: Vec<Vec<Vec<(u32, WeightVector)>>> = Decode::decode(decoder)?;
        let tag_weight = tag_weight
            .into_iter()
            .map(|x| x.into_iter().map(|x| x.iter().collect()).collect())
            .collect();
        Ok(Self {
            pma,
            weights: Decode::decode(decoder)?,
            tag_weight,
        })
    }
}

impl Encode for TypeScorerBoundaryTag {
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

impl TypeScorerBoundaryTag {
    pub fn new(
        ngram_model: NgramModel<Vec<u8>>,
        window_size: u8,
        tag_ngram_model: Vec<TagNgramModel<Vec<u8>>>,
    ) -> Result<Self> {
        let mut merger = TypeWeightMerger::default();
        for d in ngram_model.0 {
            let weight = PositionalWeightWithTag::with_boundary(-i16::from(window_size), d.weights);
            merger.add(d.ngram, weight);
        }
        let mut tag_weight =
            vec![
                vec![HashMap::with_hasher(SplitMix64Builder); usize::from(window_size) + 1];
                tag_ngram_model.len()
            ];
        for (i, tag_model) in tag_ngram_model.into_iter().enumerate() {
            for d in tag_model.0 {
                for w in d.weights {
                    let weight = PositionalWeightWithTag::with_tag(i, w.rel_position, w.weights);
                    merger.add(&d.ngram, weight);
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
        let pma = DoubleArrayAhoCorasick::new(ngrams)
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
        sentence.type_pma_states.clear();
        sentence.type_pma_states.resize(sentence.len(), u32::MAX);
        for m in self
            .pma
            .find_overlapping_no_suffix_iter(&sentence.char_types)
        {
            debug_assert!(m.end() != 0 && m.end() <= sentence.char_types.len());
            debug_assert!(m.value() < self.weights.len());
            if let Some(weight) = unsafe { self.weights.get_unchecked(m.value()) } {
                weight.add_score(
                    (m.end() + sentence.score_padding - 1) as isize,
                    &mut sentence.boundary_scores,
                );
            }
            debug_assert!(m.end() <= sentence.type_pma_states.len());
            unsafe { *sentence.type_pma_states.get_unchecked_mut(m.end() - 1) = m.value() as u32 };
        }
    }

    /// # Satety
    ///
    /// `token_id` must be smaller than `scorer.tag_weight.len()`.
    /// `pos` must be smaller than `sentence.type_pma_states.len()`.
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
            .type_pma_states
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
