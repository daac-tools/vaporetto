use crate::model::ScoreValue;
use crate::sentence::Sentence;
use daachorse::DoubleArrayAhoCorasick;

pub enum TypePredictor {
    Pma(TypePredictorPma),
}

impl TypePredictor {
    pub fn new(
        pma: DoubleArrayAhoCorasick,
        weights: Vec<Vec<ScoreValue>>,
        window_size: usize,
    ) -> Self {
        Self::Pma(TypePredictorPma::new(pma, weights, window_size))
    }

    pub fn add_scores(&self, sentence: &Sentence, start: usize, ys: &mut [ScoreValue]) {
        match self {
            TypePredictor::Pma(pma) => pma.add_scores(sentence, start, ys),
        }
    }
}

pub struct TypePredictorPma {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<Vec<ScoreValue>>,
    window_size: usize,
}

impl TypePredictorPma {
    pub fn new(
        pma: DoubleArrayAhoCorasick,
        weights: Vec<Vec<ScoreValue>>,
        window_size: usize,
    ) -> Self {
        Self {
            pma,
            weights,
            window_size,
        }
    }

    pub fn add_scores(&self, sentence: &Sentence, start: usize, ys: &mut [ScoreValue]) {
        let type_start = if start >= self.window_size {
            start + 1 - self.window_size
        } else {
            0
        };
        let type_end = std::cmp::min(
            start + ys.len() + self.window_size,
            sentence.char_type.len(),
        );
        let char_type = &sentence.char_type[type_start..type_end];
        let padding = start - type_start + 1;
        for m in self.pma.find_overlapping_no_suffix_iter(&char_type) {
            let offset = m.end() as isize - self.window_size as isize - padding as isize;
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
