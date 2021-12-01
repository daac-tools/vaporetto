use daachorse::DoubleArrayAhoCorasick;

use crate::dict_model::{DictModel, DictModelLengthwise, DictModelWordwise, DictWeight};
use crate::errors::{Result, VaporettoError};
use crate::sentence::Sentence;

pub enum DictScorer {
    Wordwise(DictScorerWordwise),
    Lengthwise(DictScorerLengthwise),
}

impl DictScorer {
    pub fn new(model: DictModel) -> Result<Self> {
        Ok(match model {
            DictModel::Wordwise(model) => Self::Wordwise(DictScorerWordwise::new(model)),
            DictModel::Lengthwise(model) => Self::Lengthwise(DictScorerLengthwise::new(model)?),
        })
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [i32]) {
        match self {
            Self::Wordwise(model) => model.add_scores(sentence, ys),
            Self::Lengthwise(model) => model.add_scores(sentence, ys),
        }
    }
}

pub struct DictScorerWordwise {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<DictWeight>,
}

impl DictScorerWordwise {
    pub fn new(model: DictModelWordwise) -> Self {
        let mut words = vec![];
        let mut weights = vec![];
        for pair in model.data {
            words.push(pair.word);
            weights.push(pair.weights);
        }
        Self {
            pma: DoubleArrayAhoCorasick::new(words).unwrap(),
            weights,
        }
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [i32]) {
        for m in self.pma.find_overlapping_iter(&sentence.text) {
            let m_start = sentence.str_to_char_pos[m.start()];
            let m_end = sentence.str_to_char_pos[m.end()];
            let idx = m.pattern();
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

pub struct DictScorerLengthwise {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<DictWeight>,
}

impl DictScorerLengthwise {
    pub fn new(model: DictModelLengthwise) -> Result<Self> {
        if model.weights.is_empty() {
            return Err(VaporettoError::invalid_model(
                "dict_word_max_size must be >= 1",
            ));
        }
        Ok(Self {
            pma: DoubleArrayAhoCorasick::new(model.words).unwrap(),
            weights: model.weights,
        })
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [i32]) {
        for m in self.pma.find_overlapping_iter(&sentence.text) {
            let m_start = sentence.str_to_char_pos[m.start()];
            let m_end = sentence.str_to_char_pos[m.end()];
            let idx = (m_end - m_start).min(self.weights.len()) - 1;
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
