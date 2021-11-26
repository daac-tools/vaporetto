use crate::model::{DictWeight, ScoreValue};
use crate::sentence::Sentence;
use daachorse::DoubleArrayAhoCorasick;

pub struct DictScorer {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<DictWeight>,
    word_wise_score: bool,
}

impl DictScorer {
    pub fn new(
        pma: DoubleArrayAhoCorasick,
        weights: Vec<DictWeight>,
        word_wise_score: bool,
    ) -> Self {
        Self {
            pma,
            weights,
            word_wise_score,
        }
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [ScoreValue]) {
        for m in self.pma.find_overlapping_iter(&sentence.text) {
            let m_start = sentence.str_to_char_pos[m.start()];
            let m_end = sentence.str_to_char_pos[m.end()];
            let idx = if self.word_wise_score {
                m.pattern()
            } else {
                std::cmp::min(m_end - m_start, self.weights.len()) - 1
            };
            let dict_weight = self.weights[idx];
            if m_start != 0 {
                ys[m_start - 1] += dict_weight.right;
            }
            for y in &mut ys[m_start..m_end - 1] {
                *y += dict_weight.inner;
            }
            if m_end <= ys.len() {
                ys[m_end - 1] += dict_weight.left;
            }
        }
    }
}
