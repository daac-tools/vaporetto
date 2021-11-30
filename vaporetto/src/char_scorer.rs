use crate::model::ScoreValue;
use crate::sentence::Sentence;
use daachorse::DoubleArrayAhoCorasick;

#[cfg(feature = "simd")]
use std::simd::i32x8;

pub enum CharScorer {
    Naive(CharScorerNaive),

    #[cfg(feature = "simd")]
    Simd(CharScorerSimd),
}

impl CharScorer {
    /// # Panics
    ///
    /// `ngrams` and `weights` must have same number of entries.
    pub fn new(ngrams: &[String], weights: Vec<Vec<ScoreValue>>, window_size: usize) -> Self {
        #[cfg(not(feature = "simd"))]
        {
            Self::Naive(CharScorerNaive::new(ngrams, weights, window_size))
        }

        #[cfg(feature = "simd")]
        if window_size <= 4 {
            Self::Simd(CharScorerSimd::new(ngrams, weights, window_size))
        } else {
            Self::Naive(CharScorerNaive::new(ngrams, weights, window_size))
        }
    }

    pub fn add_scores(&self, sentence: &Sentence, padding: usize, ys: &mut [ScoreValue]) {
        match self {
            CharScorer::Naive(naive) => naive.add_scores(sentence, &mut ys[padding..]),

            #[cfg(feature = "simd")]
            CharScorer::Simd(simd) => simd.add_scores(sentence, padding, ys),
        }
    }
}

pub struct CharScorerNaive {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<Vec<ScoreValue>>,
    window_size: usize,
}

impl CharScorerNaive {
    /// # Panics
    ///
    /// `ngrams` and `weights` must have same number of entries.
    pub fn new(ngrams: &[String], weights: Vec<Vec<ScoreValue>>, window_size: usize) -> Self {
        if ngrams.len() != weights.len() {
            panic!("ngrams.len() != weights.len()");
        }
        Self {
            pma: DoubleArrayAhoCorasick::new(ngrams).unwrap(),
            weights,
            window_size,
        }
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [ScoreValue]) {
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
    /// # Panics
    ///
    /// `ngrams` and `weights` must have same number of entries.
    pub fn new(ngrams: &[String], weights: Vec<Vec<i32>>, window_size: usize) -> Self {
        if ngrams.len() != weights.len() {
            panic!("ngrams.len() != weights.len()");
        }
        let weights: Vec<_> = weights
            .iter()
            .map(|w| {
                let mut s = [0i32; 8];
                s[..w.len()].copy_from_slice(&w);
                i32x8::from_array(s)
            })
            .collect();
        Self {
            pma: DoubleArrayAhoCorasick::new(ngrams).unwrap(),
            weights,
            window_size,
        }
    }

    pub fn add_scores(&self, sentence: &Sentence, padding: usize, ys: &mut [ScoreValue]) {
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
