use std::cell::RefCell;
use std::collections::BTreeMap;

use daachorse::DoubleArrayAhoCorasick;

use crate::dict_model::DictModel;
use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::sentence::Sentence;

#[cfg(all(feature = "simd", feature = "portable-simd"))]
use std::simd::i32x8;

#[cfg(feature = "simd")]
pub const SIMD_SIZE: usize = 8;
#[cfg(all(feature = "simd", feature = "portable-simd"))]
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

    #[cfg(all(feature = "simd", not(feature = "portable-simd")))]
    Simd([i32; SIMD_SIZE]),
    #[cfg(all(feature = "simd", feature = "portable-simd"))]
    Simd(I32Vec),
}

pub struct CharScorer {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<PositionalWeight<WeightVector>>,
}

impl CharScorer {
    pub fn new(model: NgramModel<String>, window_size: usize, dict: DictModel) -> Result<Self> {
        // key: ngram, value: (weight, check)
        let mut weights_map: BTreeMap<String, RefCell<(NaivePositionalWeight, bool)>> =
            BTreeMap::new();

        for d in model.data {
            let weight = PositionalWeight::new(-(window_size as i32), d.weights);
            if let Some(data) = weights_map.get_mut(&d.ngram) {
                let (prev_weight, _) = &mut *data.borrow_mut();
                *prev_weight = PositionalWeight::from_two_weights(&weight, prev_weight);
            } else {
                weights_map.insert(d.ngram, RefCell::new((weight, false)));
            }
        }
        for d in dict.dict {
            let word_len = d.word.chars().count();
            let mut weight = Vec::with_capacity(word_len + 1);
            weight.push(d.weights.right);
            weight.resize(word_len, d.weights.inside);
            weight.push(d.weights.left);
            let weight = PositionalWeight::new(-(word_len as i32), weight);
            if let Some(data) = weights_map.get_mut(&d.word) {
                let (prev_weight, _) = &mut *data.borrow_mut();
                *prev_weight = PositionalWeight::from_two_weights(&weight, prev_weight);
            } else {
                weights_map.insert(d.word, RefCell::new((weight, false)));
            }
        }

        let mut stack = vec![];
        for (ngram, data) in &weights_map {
            if data.borrow().1 {
                continue;
            }
            stack.push(data);
            for (j, _) in ngram.char_indices().skip(1) {
                if let Some(data) = weights_map.get(&ngram[j..]) {
                    stack.push(data);
                    if data.borrow().1 {
                        break;
                    }
                }
            }
            let mut data_from = stack.pop().unwrap();
            data_from.borrow_mut().1 = true;
            while let Some(data_to) = stack.pop() {
                let new_data = (
                    PositionalWeight::from_two_weights(&data_from.borrow().0, &data_to.borrow().0),
                    true,
                );
                *data_to.borrow_mut() = new_data;
                data_from = data_to;
            }
        }
        let mut ngrams = vec![];
        let mut weights = vec![];
        for (ngram, data) in weights_map {
            ngrams.push(ngram);
            let PositionalWeight { offset, weight } = data.into_inner().0;

            let weight = {
                #[cfg(not(feature = "simd"))]
                {
                    WeightVector::Array(weight)
                }

                #[cfg(feature = "simd")]
                if weight.len() <= SIMD_SIZE {
                    let mut s = [0i32; SIMD_SIZE];
                    s[..weight.len()].copy_from_slice(weight.as_slice());
                    #[cfg(not(feature = "portable-simd"))]
                    {
                        WeightVector::Simd(s)
                    }
                    #[cfg(feature = "portable-simd")]
                    {
                        WeightVector::Simd(I32Vec::from_array(s))
                    }
                } else {
                    WeightVector::Array(weight)
                }
            };
            weights.push(PositionalWeight { offset, weight });
        }
        let pma = DoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("invalid character n-grams"))?;
        Ok(Self { pma, weights })
    }

    pub fn add_scores(&self, sentence: &Sentence, padding: usize, ys: &mut [i32]) {
        for m in self.pma.find_overlapping_no_suffix_iter(&sentence.text) {
            let m_end = sentence.str_to_char_pos[m.end()];
            // Both the weights and the PMA always have the same number of items.
            // Therefore, the following code is safe.
            let pos_weights = unsafe { self.weights.get_unchecked(m.value()) };

            match &pos_weights.weight {
                WeightVector::Array(weight) => {
                    let offset = m_end as isize + pos_weights.offset as isize - 1;
                    if offset >= 0 {
                        for (w, y) in weight.iter().zip(&mut ys[padding + offset as usize..]) {
                            *y += w;
                        }
                    } else {
                        for (w, y) in weight[-offset as usize..]
                            .iter()
                            .zip(ys[padding..].iter_mut())
                        {
                            *y += w;
                        }
                    }
                }

                #[cfg(feature = "simd")]
                WeightVector::Simd(weight) => {
                    let offset =
                        padding as isize + m_end as isize + pos_weights.offset as isize - 1;
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
}
