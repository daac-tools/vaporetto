use alloc::string::String;
use alloc::vec::Vec;

use bincode::{Decode, Encode};

use crate::errors::{Result, VaporettoError};

#[cfg(feature = "kytea")]
#[derive(Clone, Copy, Default)]
pub struct DictWeight {
    pub right: i32,
    pub inside: i32,
    pub left: i32,
}

/// Record of weights for each word.
#[derive(Clone, Debug, Decode, Encode)]
pub struct WordWeightRecord {
    pub(crate) word: String,
    pub(crate) weights: Vec<i32>,
    pub(crate) comment: String,
}

impl WordWeightRecord {
    /// Creates a new word weight record.
    ///
    /// # Arguments
    ///
    /// * `word` - A word.
    /// * `weights` - Weights of each character boundary.
    /// * `comment` - A comment that does not affect the behaviour.
    ///
    /// # Errors
    ///
    /// If `weights.len() != word.chars().count() + 1`,
    /// an error variant will be returned.
    pub fn new(word: String, weights: Vec<i32>, comment: String) -> Result<Self> {
        if weights.len() != word.chars().count() + 1 {
            return Err(VaporettoError::invalid_argument(
                "weights",
                "does not match the length of the `word`",
            ));
        }
        Ok(Self {
            word,
            weights,
            comment,
        })
    }

    /// Gets a reference to the word.
    pub fn get_word(&self) -> &str {
        &self.word
    }

    /// Gets weights.
    pub fn get_weights(&self) -> &[i32] {
        &self.weights
    }

    /// Gets a reference to the comment.
    pub fn get_comment(&self) -> &str {
        &self.comment
    }
}

#[derive(Debug, Decode, Encode)]
pub struct DictModel(pub(crate) Vec<WordWeightRecord>);

impl DictModel {
    pub const fn new(dict: Vec<WordWeightRecord>) -> Self {
        Self(dict)
    }

    pub fn dictionary(&self) -> &[WordWeightRecord] {
        &self.0
    }
}
