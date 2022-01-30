use daachorse::DoubleArrayAhoCorasick;

use crate::dict_model::DictModel;
use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::sentence::Sentence;
use crate::utils::{AddWeight, MergableWeight, WeightMerger};

#[cfg(feature = "portable-simd")]
use std::simd::i32x8;

pub const SIMD_SIZE: usize = 8;
#[cfg(feature = "portable-simd")]
type I32Vec = i32x8;

struct PositionalWeight<W> {
    pub offset: i32,
    pub weight: W,
}

type NaivePositionalWeight = PositionalWeight<Vec<i32>>;

impl NaivePositionalWeight {
    fn new(offset: i32, weight: Vec<i32>) -> Self {
        Self { offset, weight }
    }
}

impl MergableWeight for NaivePositionalWeight {
    fn from_two_weights(weight1: &Self, weight2: &Self) -> Self {
        let (weight1, weight2) = if weight1.offset > weight2.offset {
            (weight2, weight1)
        } else {
            (weight1, weight2)
        };
        let shift = (weight2.offset - weight1.offset) as usize;
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

enum WeightVector {
    Array(Vec<i32>),

    #[cfg(not(feature = "portable-simd"))]
    Simd([i32; SIMD_SIZE]),
    #[cfg(feature = "portable-simd")]
    Simd(I32Vec),
}

impl WeightVector {
    pub fn new(weight: Vec<i32>) -> Self {
        if weight.len() <= SIMD_SIZE {
            let mut s = [0i32; SIMD_SIZE];
            s[..weight.len()].copy_from_slice(weight.as_slice());
            #[cfg(not(feature = "portable-simd"))]
            {
                Self::Simd(s)
            }
            #[cfg(feature = "portable-simd")]
            {
                Self::Simd(I32Vec::from_array(s))
            }
        } else {
            Self::Array(weight)
        }
    }
}

impl AddWeight for WeightVector {
    fn add_weight(&self, ys: &mut [i32], offset: isize) {
        match self {
            WeightVector::Array(weight) => {
                weight.add_weight(ys, offset);
            }
            WeightVector::Simd(weight) => {
                let ys_slice = &mut ys[offset as usize..offset as usize + SIMD_SIZE];
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

pub struct CharScorer {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<PositionalWeight<WeightVector>>,
}

impl CharScorer {
    pub fn new(model: NgramModel<String>, window_size: usize, dict: DictModel) -> Result<Self> {
        let mut weight_merger = WeightMerger::new();

        for d in model.data {
            let weight = PositionalWeight::new(-(window_size as i32) - 1, d.weights);
            weight_merger.add(&d.ngram, weight);
        }
        for d in dict.dict {
            let word_len = d.word.chars().count();
            let mut weight = Vec::with_capacity(word_len + 1);
            weight.push(d.weights.right);
            weight.resize(word_len, d.weights.inside);
            weight.push(d.weights.left);
            let weight = PositionalWeight::new(-(word_len as i32) - 1, weight);
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
        let pma = DoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("invalid character n-grams"))?;
        Ok(Self { pma, weights })
    }

    pub fn add_scores(&self, sentence: &Sentence, padding: usize, ys: &mut [i32]) {
        // If the following assertion fails, Vaporetto has a bug.
        assert_eq!(sentence.str_to_char_pos.len(), sentence.text.len() + 1);

        for m in self.pma.find_overlapping_no_suffix_iter(&sentence.text) {
            // This was checked outside of the iteration.
            let m_end = unsafe { *sentence.str_to_char_pos.get_unchecked(m.end()) };
            // Both the weights and the PMA always have the same number of items.
            // Therefore, the following code is safe.
            let pos_weights = unsafe { self.weights.get_unchecked(m.value()) };

            let offset = padding as isize + m_end as isize + pos_weights.offset as isize;
            pos_weights.weight.add_weight(ys, offset);
        }
    }
}
