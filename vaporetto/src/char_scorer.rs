use crate::model::ScoreValue;
use crate::sentence::Sentence;
use daachorse::DoubleArrayAhoCorasick;

pub struct CharScorer {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<Vec<ScoreValue>>,
    window_size: usize,
}

impl CharScorer {
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
