use alloc::string::String;
use alloc::vec::Vec;

use bincode::{Decode, Encode};

#[derive(Clone, Copy, Default, Decode, Encode)]
pub struct DictWeight {
    pub right: i32,
    pub inside: i32,
    pub left: i32,
}

/// Record of weights for each word.
#[derive(Clone, Decode, Encode)]
pub struct WordWeightRecord {
    pub(crate) word: String,
    pub(crate) weights: DictWeight,
    pub(crate) comment: String,
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
    /// * `comment` - A comment that does not affect the behaviour.
    ///
    /// # Returns
    ///
    /// A new record.
    pub const fn new(word: String, right: i32, inside: i32, left: i32, comment: String) -> Self {
        Self {
            word,
            weights: DictWeight {
                right,
                inside,
                left,
            },
            comment,
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

    /// Gets a reference to the comment.
    pub fn get_comment(&self) -> &str {
        &self.comment
    }
}

#[derive(Decode, Encode)]
pub struct DictModel {
    pub(crate) dict: Vec<WordWeightRecord>,
}

impl DictModel {
    pub fn new(dict: Vec<WordWeightRecord>) -> Self {
        Self { dict }
    }

    pub fn dictionary(&self) -> &[WordWeightRecord] {
        &self.dict
    }
}
