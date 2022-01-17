use daachorse::DoubleArrayAhoCorasick;

use crate::dict_model::{DictModel, DictWeight};
use crate::errors::{Result, VaporettoError};
use crate::sentence::Sentence;

pub struct DictScorer {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<DictWeight>,
}

impl DictScorer {
    pub fn new(model: DictModel) -> Result<Self> {
        let mut words = vec![];
        let mut weights = vec![];
        for pair in model.dict {
            words.push(pair.word);
            weights.push(pair.weights);
        }
        let pma = DoubleArrayAhoCorasick::new(words)
            .map_err(|_| VaporettoError::invalid_model("invalid dictionary"))?;
        Ok(Self { pma, weights })
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [i32]) {
        for m in self.pma.find_overlapping_iter(&sentence.text) {
            let m_start = sentence.str_to_char_pos[m.start()];
            let m_end = sentence.str_to_char_pos[m.end()];
            // Both the weights and the PMA always have the same number of items.
            // Therefore, the following code is safe.
            let dict_weight = unsafe { self.weights.get_unchecked(m.value()) };
            if m_start != 0 {
                ys[m_start - 1] += dict_weight.right;
            }
            for y in &mut ys[m_start..m_end - 1] {
                *y += dict_weight.inside;
            }
            if m_end <= ys.len() {
                ys[m_end - 1] += dict_weight.left;
            }
        }
    }
}
