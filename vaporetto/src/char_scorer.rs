use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "tag-prediction")]
use core::iter;

#[cfg(feature = "tag-prediction")]
use alloc::sync::Arc;

use bincode::{
    de::{BorrowDecoder, Decoder},
    enc::Encoder,
    error::{DecodeError, EncodeError},
    BorrowDecode, Decode, Encode,
};

#[cfg(feature = "charwise-daachorse")]
use daachorse::charwise::CharwiseDoubleArrayAhoCorasick;
#[cfg(not(feature = "charwise-daachorse"))]
use daachorse::DoubleArrayAhoCorasick;

use crate::dict_model::DictModel;
use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::sentence::Sentence;
use crate::utils::{AddWeight, MergableWeight, WeightMerger};

#[cfg(feature = "tag-prediction")]
use crate::sentence::{TagRangeScore, TagRangeScores, TagScores};
#[cfg(feature = "tag-prediction")]
use crate::utils;

#[cfg(feature = "portable-simd")]
use core::simd::i32x8;

pub const SIMD_SIZE: usize = 8;
#[cfg(feature = "portable-simd")]
type I32Vec = i32x8;

#[derive(Clone, Decode, Encode)]
struct PositionalWeight<W> {
    pub weight: W,
    pub offset: i16,
}

type NaivePositionalWeight = PositionalWeight<Vec<i32>>;

impl NaivePositionalWeight {
    fn new(offset: i16, weight: Vec<i32>) -> Self {
        Self { offset, weight }
    }
}

impl MergableWeight for NaivePositionalWeight {
    fn from_two_weights(weight1: &Self, weight2: &Self, n_classes: usize) -> Self {
        debug_assert!(n_classes != 0);
        let (weight1, weight2) = if weight1.offset > weight2.offset {
            (weight2, weight1)
        } else {
            (weight1, weight2)
        };
        let shift = (weight2.offset - weight1.offset) as usize * n_classes;
        let mut weight = vec![0; weight1.weight.len().max(shift + weight2.weight.len())];
        weight[..weight1.weight.len()].copy_from_slice(&weight1.weight);
        for (r, w2) in weight[shift..].iter_mut().zip(&weight2.weight) {
            *r += w2;
        }
        Self {
            offset: weight1.offset,
            weight,
        }
    }
}

#[derive(Clone)]
enum WeightVector {
    Variable(Vec<i32>),

    #[cfg(all(feature = "fix-weight-length", not(feature = "portable-simd")))]
    Fixed([i32; SIMD_SIZE]),
    #[cfg(all(feature = "fix-weight-length", feature = "portable-simd"))]
    Fixed(I32Vec),
}

impl Decode for WeightVector {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let v: Vec<i32> = Decode::decode(decoder)?;
        #[cfg(feature = "fix-weight-length")]
        let result = if v.len() <= SIMD_SIZE {
            let mut arr = [0; SIMD_SIZE];
            arr[..v.len()].copy_from_slice(&v);

            #[cfg(feature = "portable-simd")]
            let arr = I32Vec::from_array(arr);

            Self::Fixed(arr)
        } else {
            Self::Variable(v)
        };

        #[cfg(not(feature = "fix-weight-length"))]
        let result = Self::Variable(v);

        Ok(result)
    }
}

impl Encode for WeightVector {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        match self {
            Self::Variable(v) => {
                Encode::encode(v, encoder)?;
            }

            #[cfg(feature = "fix-weight-length")]
            Self::Fixed(v) => {
                #[cfg(feature = "portable-simd")]
                let v = &v.as_array();

                let mut len = v.len();
                for (i, w) in v.iter().enumerate().rev() {
                    if *w != 0 {
                        break;
                    }
                    len = i;
                }
                Encode::encode(&v[..len].to_vec(), encoder)?;
            }
        }
        Ok(())
    }
}

impl WeightVector {
    pub fn new(weight: Vec<i32>) -> Self {
        #[cfg(feature = "fix-weight-length")]
        let v = if weight.len() <= SIMD_SIZE {
            let mut arr = [0i32; SIMD_SIZE];
            arr[..weight.len()].copy_from_slice(weight.as_slice());

            #[cfg(feature = "portable-simd")]
            let arr = I32Vec::from_array(arr);

            Self::Fixed(arr)
        } else {
            Self::Variable(weight)
        };

        #[cfg(not(feature = "fix-weight-length"))]
        let v = Self::Variable(weight);

        v
    }

    fn add_weight(&self, ys: &mut [i32], offset: usize) {
        match self {
            WeightVector::Variable(weight) => {
                weight.add_weight(ys, offset);
            }

            #[cfg(feature = "fix-weight-length")]
            WeightVector::Fixed(weight) => {
                let ys_slice = &mut ys[offset..offset + SIMD_SIZE];
                #[cfg(feature = "portable-simd")]
                {
                    let mut target = I32Vec::from_slice(ys_slice);
                    target += weight;
                    ys_slice.copy_from_slice(target.as_array());
                }
                #[cfg(not(feature = "portable-simd"))]
                for (y, w) in ys_slice.iter_mut().zip(weight) {
                    *y += w;
                }
            }
        }
    }
}

#[cfg(feature = "tag-prediction")]
#[derive(Decode, Encode)]
pub struct WeightSet<W> {
    boundary: Option<PositionalWeight<W>>,
    tag_left: Option<PositionalWeight<Vec<i32>>>,
    tag_right: Option<PositionalWeight<Vec<i32>>>,
    tag_self: Option<TagRangeScores>,
}

#[cfg(feature = "tag-prediction")]
type NaiveWeightSet = WeightSet<Vec<i32>>;

#[cfg(feature = "tag-prediction")]
impl NaiveWeightSet {
    fn boundary_weight(offset: i16, weight: Vec<i32>) -> Self {
        Self {
            boundary: Some(PositionalWeight::new(offset, weight)),
            tag_left: None,
            tag_right: None,
            tag_self: None,
        }
    }

    fn tag_left_weight(offset: i16, weight: Vec<i32>) -> Self {
        Self {
            boundary: None,
            tag_left: Some(PositionalWeight::new(offset, weight)),
            tag_right: None,
            tag_self: None,
        }
    }

    fn tag_right_weight(offset: i16, weight: Vec<i32>) -> Self {
        Self {
            boundary: None,
            tag_left: None,
            tag_right: Some(PositionalWeight::new(offset, weight)),
            tag_self: None,
        }
    }

    fn tag_self_weight(start_rel_position: i16, weight: Vec<i32>) -> Self {
        Self {
            boundary: None,
            tag_left: None,
            tag_right: None,
            tag_self: Some(Arc::new(vec![TagRangeScore::new(
                start_rel_position,
                weight,
            )])),
        }
    }
}

#[cfg(feature = "tag-prediction")]
impl MergableWeight for NaiveWeightSet {
    fn from_two_weights(weight1: &Self, weight2: &Self, n_classes: usize) -> Self {
        Self {
            boundary: utils::xor_or_zip_with(&weight1.boundary, &weight2.boundary, |w1, w2| {
                PositionalWeight::from_two_weights(w1, w2, 1)
            }),
            tag_left: utils::xor_or_zip_with(&weight1.tag_left, &weight2.tag_left, |w1, w2| {
                PositionalWeight::from_two_weights(w1, w2, n_classes)
            }),
            tag_right: utils::xor_or_zip_with(&weight1.tag_right, &weight2.tag_right, |w1, w2| {
                PositionalWeight::from_two_weights(w1, w2, n_classes)
            }),
            tag_self: utils::xor_or_zip_with(&weight1.tag_self, &weight2.tag_self, |w1, w2| {
                let mut w = w1.to_vec();
                w.append(&mut w2.to_vec());
                Arc::new(w)
            }),
        }
    }
}

pub struct CharScorer {
    #[cfg(feature = "charwise-daachorse")]
    pma: CharwiseDoubleArrayAhoCorasick,
    #[cfg(not(feature = "charwise-daachorse"))]
    pma: DoubleArrayAhoCorasick,
    weights: Vec<PositionalWeight<WeightVector>>,
}

impl CharScorer {
    pub fn new(model: NgramModel<String>, window_size: u8, dict: DictModel) -> Result<Self> {
        let mut weight_merger = WeightMerger::new(1);

        for d in model.data {
            let weight = PositionalWeight::new(-i16::from(window_size) - 1, d.weights);
            weight_merger.add(&d.ngram, weight);
        }
        for d in dict.dict {
            let word_len = d.word.chars().count();
            let mut weight = Vec::with_capacity(word_len + 1);
            weight.push(d.weights.right);
            weight.resize(word_len, d.weights.inside);
            weight.push(d.weights.left);
            let word_len = i16::try_from(word_len).map_err(|_| {
                VaporettoError::invalid_model(
                    "words must be shorter than or equal to 32767 characters",
                )
            })?;
            let weight = PositionalWeight::new(-word_len - 1, weight);
            weight_merger.add(&d.word, weight);
        }

        let mut ngrams = vec![];
        let mut weights = vec![];
        for (ngram, data) in weight_merger.merge() {
            ngrams.push(ngram);
            let PositionalWeight { offset, weight } = data;
            weights.push(PositionalWeight {
                offset,
                weight: WeightVector::new(weight),
            });
        }
        #[cfg(feature = "charwise-daachorse")]
        let pma = CharwiseDoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        #[cfg(not(feature = "charwise-daachorse"))]
        let pma = DoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        Ok(Self { pma, weights })
    }

    #[allow(clippy::cast_possible_wrap)]
    pub fn add_scores(&self, sentence: &Sentence, padding: u8, ys: &mut [i32]) {
        // If the following assertion fails, Vaporetto has a bug.
        assert_eq!(sentence.str_to_char_pos.len(), sentence.text.len() + 1);

        for m in self.pma.find_overlapping_no_suffix_iter(&sentence.text) {
            // This was checked outside of the iteration.
            let m_end = unsafe { *sentence.str_to_char_pos.get_unchecked(m.end()) };
            // Both the weights and the PMA always have the same number of items.
            // Therefore, the following code is safe.
            let pos_weights = unsafe { self.weights.get_unchecked(m.value()) };

            let offset = isize::from(padding) + m_end as isize + isize::from(pos_weights.offset);
            pos_weights.weight.add_weight(ys, offset as usize);
        }
    }
}

impl<'de> BorrowDecode<'de> for CharScorer {
    /// WARNING: This function is inherently unsafe. Do not publish this function outside this
    /// crate.
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let pma_data: &[u8] = BorrowDecode::borrow_decode(decoder)?;
        #[cfg(feature = "charwise-daachorse")]
        let (pma, _) =
            unsafe { CharwiseDoubleArrayAhoCorasick::deserialize_from_slice_unchecked(pma_data) };
        #[cfg(not(feature = "charwise-daachorse"))]
        let (pma, _) =
            unsafe { DoubleArrayAhoCorasick::deserialize_from_slice_unchecked(pma_data) };
        Ok(Self {
            pma,
            weights: Decode::decode(decoder)?,
        })
    }
}

impl Encode for CharScorer {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let pma_data = self.pma.serialize_to_vec();
        Encode::encode(&pma_data, encoder)?;
        Encode::encode(&self.weights, encoder)?;
        Ok(())
    }
}

#[cfg(feature = "tag-prediction")]
pub struct CharScorerWithTags {
    #[cfg(feature = "charwise-daachorse")]
    pma: CharwiseDoubleArrayAhoCorasick,
    #[cfg(not(feature = "charwise-daachorse"))]
    pma: DoubleArrayAhoCorasick,
    weights: Vec<WeightSet<WeightVector>>,
    n_tags: usize,
}

#[cfg(feature = "tag-prediction")]
impl CharScorerWithTags {
    pub fn new(
        model: NgramModel<String>,
        window_size: u8,
        dict: DictModel,
        n_tags: usize,
        tag_left_model: NgramModel<String>,
        tag_right_model: NgramModel<String>,
        tag_self_model: NgramModel<String>,
    ) -> Result<Self> {
        let mut weight_merger = WeightMerger::new(n_tags);

        for d in model.data {
            let weight = WeightSet::boundary_weight(-i16::from(window_size), d.weights);
            weight_merger.add(&d.ngram, weight);
        }
        for d in dict.dict {
            let word_len = d.word.chars().count();
            let mut weight = Vec::with_capacity(word_len + 1);
            weight.push(d.weights.right);
            weight.resize(word_len, d.weights.inside);
            weight.push(d.weights.left);
            let word_len = i16::try_from(word_len).map_err(|_| {
                VaporettoError::invalid_model(
                    "words must be shorter than or equal to 32767 characters",
                )
            })?;
            let weight = WeightSet::boundary_weight(-word_len, weight);
            weight_merger.add(&d.word, weight);
        }
        for d in tag_left_model.data {
            let ngram_len = i16::try_from(d.ngram.chars().count()).map_err(|_| {
                VaporettoError::invalid_model(
                    "character n-grams must be shorter than or equal to 32767 characters",
                )
            })?;
            let weight = WeightSet::tag_left_weight(-ngram_len + 1, d.weights);
            weight_merger.add(&d.ngram, weight);
        }
        for d in tag_right_model.data {
            let weight = WeightSet::tag_right_weight(-i16::from(window_size) - 1, d.weights);
            weight_merger.add(&d.ngram, weight);
        }
        for d in tag_self_model.data {
            let ngram_len = i16::try_from(d.ngram.chars().count()).map_err(|_| {
                VaporettoError::invalid_model(
                    "character n-grams must be shorter than or equal to 32767 characters",
                )
            })?;
            let weight = WeightSet::tag_self_weight(-ngram_len, d.weights);
            weight_merger.add(&d.ngram, weight);
        }

        let mut ngrams = vec![];
        let mut weights = vec![];
        for (ngram, data) in weight_merger.merge() {
            ngrams.push(ngram);
            let WeightSet {
                boundary,
                tag_left,
                tag_right,
                tag_self,
            } = data;
            weights.push(WeightSet {
                boundary: boundary.map(|PositionalWeight { offset, weight }| PositionalWeight {
                    offset,
                    weight: WeightVector::new(weight),
                }),
                tag_left,
                tag_right,
                tag_self,
            });
        }
        #[cfg(feature = "charwise-daachorse")]
        let pma = CharwiseDoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        #[cfg(not(feature = "charwise-daachorse"))]
        let pma = DoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("failed to build the automaton"))?;
        Ok(Self {
            pma,
            weights,
            n_tags,
        })
    }

    #[allow(clippy::cast_possible_wrap)]
    pub fn add_scores(
        &self,
        sentence: &Sentence,
        padding: u8,
        ys: &mut [i32],
        tag_ys: &mut TagScores,
    ) {
        #[cfg(not(feature = "charwise-daachorse"))]
        let no_suffix_iter = self.pma.find_overlapping_no_suffix_iter_from_iter(
            iter::once(0)
                .chain(sentence.text.as_bytes().iter().cloned())
                .chain(iter::once(0)),
        );
        // Since `sentence.text` is a valid UTF-8 string ensured by type `String`,
        // the following code is safe.
        #[cfg(feature = "charwise-daachorse")]
        let no_suffix_iter = unsafe {
            self.pma.find_overlapping_no_suffix_iter_from_iter(
                iter::once(0)
                    .chain(sentence.text.as_bytes().iter().cloned())
                    .chain(iter::once(0)),
            )
        };
        for m in no_suffix_iter {
            let m_end = sentence
                .str_to_char_pos
                .get(m.end() - 1)
                .copied()
                .unwrap_or(sentence.chars.len() + 1);

            // Both the weights and the PMA always have the same number of items.
            // Therefore, the following code is safe.
            let weight_set = unsafe { self.weights.get_unchecked(m.value()) };

            if let Some(pos_weights) = weight_set.boundary.as_ref() {
                let offset =
                    isize::from(padding) + m_end as isize + isize::from(pos_weights.offset) - 1;
                pos_weights.weight.add_weight(ys, offset as usize);
            }
            if let Some(pos_weights) = weight_set.tag_left.as_ref() {
                let offset =
                    (m_end as isize + isize::from(pos_weights.offset)) * self.n_tags as isize;
                pos_weights
                    .weight
                    .add_weight_signed(&mut tag_ys.left_scores, offset);
            }
            if let Some(pos_weights) = weight_set.tag_right.as_ref() {
                let offset =
                    (m_end as isize + isize::from(pos_weights.offset)) * self.n_tags as isize;
                pos_weights
                    .weight
                    .add_weight_signed(&mut tag_ys.right_scores, offset);
            }
            if let Some(weight) = weight_set.tag_self.as_ref() {
                tag_ys.self_scores[m_end - 1].replace(Arc::clone(weight));
            }
        }
    }
}

#[cfg(feature = "tag-prediction")]
impl<'de> BorrowDecode<'de> for CharScorerWithTags {
    /// WARNING: This function is inherently unsafe. Do not publish this function outside this
    /// crate.
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let pma_data: &[u8] = BorrowDecode::borrow_decode(decoder)?;
        #[cfg(feature = "charwise-daachorse")]
        let (pma, _) =
            unsafe { CharwiseDoubleArrayAhoCorasick::deserialize_from_slice_unchecked(pma_data) };
        #[cfg(not(feature = "charwise-daachorse"))]
        let (pma, _) =
            unsafe { DoubleArrayAhoCorasick::deserialize_from_slice_unchecked(pma_data) };
        Ok(Self {
            pma,
            weights: Decode::decode(decoder)?,
            n_tags: Decode::decode(decoder)?,
        })
    }
}

#[cfg(feature = "tag-prediction")]
impl Encode for CharScorerWithTags {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let pma_data = self.pma.serialize_to_vec();
        Encode::encode(&pma_data, encoder)?;
        Encode::encode(&self.weights, encoder)?;
        Encode::encode(&self.n_tags, encoder)?;
        Ok(())
    }
}
