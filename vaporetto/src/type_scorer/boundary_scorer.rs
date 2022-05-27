use alloc::vec::Vec;

use bincode::{
    de::BorrowDecoder,
    enc::Encoder,
    error::{DecodeError, EncodeError},
    BorrowDecode, Decode, Encode,
};
use daachorse::DoubleArrayAhoCorasick;

use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::predictor::{PositionalWeight, WeightVector};
use crate::sentence::Sentence;
use crate::type_scorer::TypeWeightMerger;

pub struct TypeScorerBoundary {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<PositionalWeight<WeightVector>>,
}

impl<'de> BorrowDecode<'de> for TypeScorerBoundary {
    /// WARNING: This function is inherently unsafe. Do not publish this function outside this
    /// crate.
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let pma_data: &[u8] = BorrowDecode::borrow_decode(decoder)?;
        let (pma, _) =
            unsafe { DoubleArrayAhoCorasick::deserialize_from_slice_unchecked(pma_data) };
        Ok(Self {
            pma,
            weights: Decode::decode(decoder)?,
        })
    }
}

impl Encode for TypeScorerBoundary {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let pma_data = self.pma.serialize_to_vec();
        Encode::encode(&pma_data, encoder)?;
        Encode::encode(&self.weights, encoder)?;
        Ok(())
    }
}

impl TypeScorerBoundary {
    pub fn new(ngram_model: NgramModel<Vec<u8>>, window_size: u8) -> Result<Self> {
        let mut merger = TypeWeightMerger::default();
        for d in ngram_model.0 {
            let weight = PositionalWeight::new(-i16::from(window_size), d.weights);
            merger.add(d.ngram, weight);
        }
        let mut ngrams = vec![];
        let mut weights = vec![];
        for (ngram, weight) in merger.merge() {
            ngrams.push(ngram);
            weights.push(weight.into());
        }
        let pma = DoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        Ok(Self { pma, weights })
    }

    #[allow(clippy::cast_possible_wrap)]
    #[inline(always)]
    pub fn add_scores<'a, 'b>(&self, sentence: &mut Sentence<'a, 'b>) {
        sentence.type_pma_states.clear();
        for m in self
            .pma
            .find_overlapping_no_suffix_iter(&sentence.char_types)
        {
            debug_assert!(m.end() != 0 && m.end() <= sentence.text.len());
            debug_assert!(m.value() < self.weights.len());
            let weight = unsafe { self.weights.get_unchecked(m.value()) };
            weight.add_score(
                (m.end() + sentence.score_padding - 1) as isize,
                &mut sentence.boundary_scores,
            );
        }
    }
}
