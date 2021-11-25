use crate::model::{DictWeight, ScoreValue};
use crate::sentence::Sentence;
use daachorse::DoubleArrayAhoCorasick;

pub struct DictScorer {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<DictWeight>,
    window_size: usize,
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
            window_size: 1,
            word_wise_score,
        }
    }

    pub fn add_scores(&self, sentence: &Sentence, start: usize, ys: &mut [ScoreValue]) {
        let char_start = if start >= self.window_size {
            start + 1 - self.window_size
        } else {
            0
        };
        let text_start = sentence.char_to_str_pos[char_start];
        let char_end = std::cmp::min(
            start + ys.len() + self.window_size,
            sentence.char_to_str_pos.len() - 1,
        );
        let text_end = sentence.char_to_str_pos[char_end];
        let text = &sentence.text[text_start..text_end];
        let padding = start - char_start + 1;
        for m in self.pma.find_overlapping_iter(&text) {
            let m_start = sentence.str_to_char_pos[m.start() + text_start] - char_start;
            let m_end = sentence.str_to_char_pos[m.end() + text_start] - char_start;
            let idx = if self.word_wise_score {
                m.pattern()
            } else {
                std::cmp::min(m_end - m_start, self.weights.len()) - 1
            };
            let dict_weight = self.weights[idx];
            if m_start >= padding && m_start < padding + ys.len() {
                ys[m_start - padding] += dict_weight.right;
            }
            let range_start = std::cmp::max(0, m_start as isize - padding as isize + 1);
            let range_end = std::cmp::min(m_end as isize - padding as isize, ys.len() as isize);
            if range_start < range_end {
                for y in &mut ys[range_start as usize..range_end as usize] {
                    *y += dict_weight.inner;
                }
            }
            if m_end >= padding && m_end < ys.len() + padding {
                ys[m_end - padding] += dict_weight.left;
            }
        }
    }

    pub fn window_size(&mut self, size: usize) {
        self.window_size = std::cmp::max(size, 1);
    }
}
