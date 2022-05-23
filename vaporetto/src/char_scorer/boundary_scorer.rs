use alloc::string::String;
use alloc::vec::Vec;

use bincode::{
    de::BorrowDecoder, enc::Encoder, error::DecodeError, error::EncodeError, BorrowDecode, Decode,
    Encode,
};
use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;

use crate::char_scorer::CharWeightMerger;
use crate::dict_model::DictModel;
use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::predictor::{PositionalWeight, WeightVector};
use crate::sentence::Sentence;

pub struct CharScorerBoundary {
    pma: CharwiseDoubleArrayAhoCorasick,
    weights: Vec<PositionalWeight<WeightVector>>,
}

impl<'de> BorrowDecode<'de> for CharScorerBoundary {
    /// WARNING: This function is inherently unsafe. Do not publish this function outside this
    /// crate.
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let pma_data: &[u8] = BorrowDecode::borrow_decode(decoder)?;
        let (pma, _) =
            unsafe { CharwiseDoubleArrayAhoCorasick::deserialize_from_slice_unchecked(pma_data) };
        Ok(Self {
            pma,
            weights: Decode::decode(decoder)?,
        })
    }
}

impl Encode for CharScorerBoundary {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let pma_data = self.pma.serialize_to_vec();
        Encode::encode(&pma_data, encoder)?;
        Encode::encode(&self.weights, encoder)?;
        Ok(())
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
        let pma = CharwiseDoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        Ok(Self { pma, weights })
    }

    #[inline(always)]
    pub fn add_scores<'a, 'b>(&self, sentence: &mut Sentence<'a, 'b>) {
        sentence.char_pma_states.clear();
        for m in self.pma.find_overlapping_no_suffix_iter(&sentence.text) {
            let end = unsafe { *sentence.str_to_char_pos().get_unchecked(m.end()) };
            let weight = unsafe { self.weights.get_unchecked(m.value()) };
            weight.add_score(
                (end + sentence.score_padding - 1) as isize,
                &mut sentence.boundary_scores,
            );
        }
    }
}
