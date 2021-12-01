use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ngram_model::NgramModel;

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct DictWeight {
    pub right: i32,
    pub inner: i32,
    pub left: i32,
}

#[derive(Serialize, Deserialize)]
pub enum DictModel {
    Wordwise(DictModelWordwise),
    Lengthwise(DictModelLengthwise),
}

impl DictModel {
    pub fn merge_dict_weights(
        &mut self,
        char_ngram_model: &mut NgramModel<String>,
        char_window_size: usize,
    ) {
        match self {
            Self::Wordwise(model) => model.merge_dict_weights(char_ngram_model, char_window_size),
            Self::Lengthwise(model) => model.merge_dict_weights(char_ngram_model, char_window_size),
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Wordwise(model) => model.is_empty(),
            Self::Lengthwise(model) => model.is_empty(),
        }
    }
}

#[derive(Clone, Serialize, Deserialize)]
pub struct WordwiseDictData {
    pub(crate) word: String,
    pub(crate) weights: DictWeight,
}

#[derive(Serialize, Deserialize)]
pub struct DictModelWordwise {
    pub(crate) data: Vec<WordwiseDictData>,
}

impl DictModelWordwise {
    pub fn merge_dict_weights(
        &mut self,
        char_ngram_model: &mut NgramModel<String>,
        char_window_size: usize,
    ) {
        let mut word_map = HashMap::new();
        for (i, word) in char_ngram_model
            .data
            .iter()
            .map(|d| d.ngram.clone())
            .enumerate()
        {
            word_map.insert(word, i);
        }
        let mut new_data = vec![];
        for data in self.data.drain(..) {
            let word_size = data.word.chars().count();
            match word_map.get(&data.word) {
                Some(&idx) if char_window_size >= word_size => {
                    let start = char_window_size - word_size;
                    let end = start + word_size;
                    char_ngram_model.data[idx].weights[start] += data.weights.right;
                    for i in start + 1..end {
                        char_ngram_model.data[idx].weights[i] += data.weights.inner;
                    }
                    char_ngram_model.data[idx].weights[end] += data.weights.left;
                }
                _ => {
                    new_data.push(data);
                }
            }
        }
        self.data = new_data;
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

#[derive(Serialize, Deserialize)]
pub struct DictModelLengthwise {
    pub(crate) words: Vec<String>,
    pub(crate) weights: Vec<DictWeight>,
}

impl DictModelLengthwise {
    pub fn merge_dict_weights(
        &mut self,
        char_ngram_model: &mut NgramModel<String>,
        char_window_size: usize,
    ) {
        let mut word_map = HashMap::new();
        for (i, word) in char_ngram_model
            .data
            .iter()
            .map(|d| d.ngram.clone())
            .enumerate()
        {
            word_map.insert(word, i);
        }
        let mut new_words = vec![];
        for word in self.words.drain(..) {
            let word_size = word.chars().count();
            match word_map.get(&word) {
                Some(&idx) if char_window_size >= word_size => {
                    let start = char_window_size - word_size;
                    let end = start + word_size;
                    let word_size_idx = word_size.min(self.weights.len()) - 1;
                    let weight = &self.weights[word_size_idx];
                    char_ngram_model.data[idx].weights[start] += weight.right;
                    for i in start + 1..end {
                        char_ngram_model.data[idx].weights[i] += weight.inner;
                    }
                    char_ngram_model.data[idx].weights[end] += weight.left;
                }
                _ => new_words.push(word),
            }
        }
        self.words = new_words;
    }

    pub fn is_empty(&self) -> bool {
        self.words.is_empty()
    }
}
