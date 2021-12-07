use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::ngram_model::NgramModel;

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct DictWeight {
    pub right: i32,
    pub inside: i32,
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

    pub fn dump_dictionary(&self) -> Vec<WordWeightRecord> {
        match self {
            Self::Wordwise(model) => model.dump_dictionary(),
            Self::Lengthwise(model) => model.dump_dictionary(),
        }
    }
}

/// Record of weights for each word.
#[derive(Clone, Serialize, Deserialize)]
pub struct WordWeightRecord {
    pub(crate) word: String,
    pub(crate) weights: DictWeight,
}

impl WordWeightRecord {
    /// Creates a new word weight record.
    ///
    /// # Arguments
    ///
    /// * `word` - A word.
    /// * `right` - A weight of the boundary when the word is found at right.
    /// * `inside` - A weight of the boundary when the word is overlapped on the boundary.
    /// * `left` - A weight of the boundary when the word is found at left.
    ///
    /// # Returns
    ///
    /// A new record.
    pub const fn new(word: String, right: i32, inside: i32, left: i32) -> Self {
        Self {
            word,
            weights: DictWeight {
                right,
                inside,
                left,
            },
        }
    }

    /// Gets a reference to the word.
    pub fn get_word(&self) -> &str {
        &self.word
    }

    /// Gets a `right` weight.
    pub const fn get_right_weight(&self) -> i32 {
        self.weights.right
    }

    /// Gets a `inside` weight.
    pub const fn get_inside_weight(&self) -> i32 {
        self.weights.inside
    }

    /// Gets a `left` weight.
    pub const fn get_left_weight(&self) -> i32 {
        self.weights.left
    }
}

#[derive(Serialize, Deserialize)]
pub struct DictModelWordwise {
    pub(crate) dict: Vec<WordWeightRecord>,
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
        let mut new_dict = vec![];
        for data in self.dict.drain(..) {
            let word_size = data.word.chars().count();
            match word_map.get(&data.word) {
                Some(&idx) if char_window_size >= word_size => {
                    let start = char_window_size - word_size;
                    let end = start + word_size;
                    char_ngram_model.data[idx].weights[start] += data.weights.right;
                    for i in start + 1..end {
                        char_ngram_model.data[idx].weights[i] += data.weights.inside;
                    }
                    char_ngram_model.data[idx].weights[end] += data.weights.left;
                }
                _ => {
                    new_dict.push(data);
                }
            }
        }
        self.dict = new_dict;
    }

    pub fn is_empty(&self) -> bool {
        self.dict.is_empty()
    }

    pub fn dump_dictionary(&self) -> Vec<WordWeightRecord> {
        self.dict.clone()
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
                        char_ngram_model.data[idx].weights[i] += weight.inside;
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

    pub fn dump_dictionary(&self) -> Vec<WordWeightRecord> {
        let mut result = vec![];
        for word in &self.words {
            let word = word.clone();
            let word_size = word.chars().count();
            let word_size_idx = word_size.min(self.weights.len()) - 1;
            let weights = self.weights[word_size_idx];
            result.push(WordWeightRecord { word, weights });
        }
        result
    }
}
