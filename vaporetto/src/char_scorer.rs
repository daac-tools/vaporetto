use daachorse::DoubleArrayAhoCorasick;

use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::sentence::Sentence;

#[cfg(feature = "simd")]
use std::simd::i32x8;

pub enum CharScorer {
    Naive(CharScorerNaive),

    #[cfg(feature = "simd")]
    Simd(CharScorerSimd),
}

impl CharScorer {
    pub fn new(model: NgramModel<String>, window_size: usize) -> Result<Self> {
        #[cfg(not(feature = "simd"))]
        {
            Ok(Self::Naive(CharScorerNaive::new(model, window_size)?))
        }

        #[cfg(feature = "simd")]
        Ok(if window_size <= 4 {
            Self::Simd(CharScorerSimd::new(model, window_size)?)
        } else {
            Self::Naive(CharScorerNaive::new(model, window_size)?)
        })
    }

    pub fn add_scores(&self, sentence: &Sentence, padding: usize, ys: &mut [i32]) {
        match self {
            CharScorer::Naive(naive) => naive.add_scores(sentence, &mut ys[padding..]),

            #[cfg(feature = "simd")]
            CharScorer::Simd(simd) => simd.add_scores(sentence, padding, ys),
        }
    }
}

pub struct CharScorerNaive {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<Vec<i32>>,
    window_size: usize,
}

impl CharScorerNaive {
    pub fn new(mut model: NgramModel<String>, window_size: usize) -> Result<Self> {
        model.merge_weights();
        let pma = DoubleArrayAhoCorasick::new(model.data.iter().map(|d| &d.ngram))
            .map_err(|_| VaporettoError::invalid_model("invalid character n-grams"))?;
        let mut weights = vec![];
        for d in model.data {
            if d.weights.len() <= 2 * window_size - d.ngram.chars().count() {
                return Err(VaporettoError::invalid_model(
                    "invalid size of weight vector",
                ));
            }
            weights.push(d.weights);
        }
        Ok(Self {
            pma,
            weights,
            window_size,
        })
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [i32]) {
        for m in self.pma.find_overlapping_no_suffix_iter(&sentence.text) {
            let m_end = sentence.str_to_char_pos[m.end()];
            let offset = m_end as isize - self.window_size as isize - 1;
            let weights = &self.weights[m.pattern()];
            if offset >= 0 {
                for (w, y) in weights.iter().zip(&mut ys[offset as usize..]) {
                    *y += w;
                }
            } else {
                for (w, y) in weights[-offset as usize..].iter().zip(ys.iter_mut()) {
                    *y += w;
                }
            }
        }
    }
}

#[cfg(feature = "simd")]
pub struct CharScorerSimd {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<i32x8>,
    window_size: usize,
}

#[cfg(feature = "simd")]
impl CharScorerSimd {
    pub fn new(mut model: NgramModel<String>, window_size: usize) -> Result<Self> {
        model.merge_weights();
        let pma = DoubleArrayAhoCorasick::new(model.data.iter().map(|d| &d.ngram))
            .map_err(|_| VaporettoError::invalid_model("invalid character n-grams"))?;
        let mut weights = vec![];
        for d in model.data {
            let mut s = [0i32; 8];
            if let Some(s) = s.get_mut(..d.weights.len()) {
                s.copy_from_slice(&d.weights);
            } else {
                return Err(VaporettoError::invalid_model(
                    "invalid size of weight vector",
                ));
            }
            weights.push(i32x8::from_array(s));
        }
        Ok(Self {
            pma,
            weights,
            window_size,
        })
    }

    pub fn add_scores(&self, sentence: &Sentence, padding: usize, ys: &mut [i32]) {
        for m in self.pma.find_overlapping_no_suffix_iter(&sentence.text) {
            let m_end = sentence.str_to_char_pos[m.end()];
            let offset = padding as isize + m_end as isize - self.window_size as isize - 1;
            let weights = &self.weights[m.pattern()];
            let ys_slice = &mut ys[offset as usize..offset as usize + 8];
            let mut target = i32x8::from_slice(ys_slice);
            target += weights;
            ys_slice.copy_from_slice(target.as_array());
        }
    }

    pub const fn simd_len() -> usize {
        8
    }
}
