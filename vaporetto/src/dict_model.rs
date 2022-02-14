use std::io::{Read, Write};
use std::mem;

use crate::errors::Result;
use crate::utils;

#[derive(Clone, Copy, Default)]
pub struct DictWeight {
    pub right: i32,
    pub inside: i32,
    pub left: i32,
}

impl DictWeight {
    pub fn serialize<W>(&self, mut wtr: W) -> Result<usize>
    where
        W: Write,
    {
        utils::write_i32(&mut wtr, self.right)?;
        utils::write_i32(&mut wtr, self.inside)?;
        utils::write_i32(&mut wtr, self.left)?;
        Ok(mem::size_of::<i32>() * 3)
    }

    pub fn deserialize<R>(mut rdr: R) -> Result<Self>
    where
        R: Read,
    {
        Ok(Self {
            right: utils::read_i32(&mut rdr)?,
            inside: utils::read_i32(&mut rdr)?,
            left: utils::read_i32(&mut rdr)?,
        })
    }
}

/// Record of weights for each word.
#[derive(Clone)]
pub struct WordWeightRecord {
    pub(crate) word: String,
    pub(crate) weights: DictWeight,
    pub(crate) comment: String,
}

impl WordWeightRecord {
    pub fn serialize<W>(&self, mut wtr: W) -> Result<usize>
    where
        W: Write,
    {
        let word_size = self.word.len();
        let comment_size = self.comment.len();
        utils::write_u32(&mut wtr, u32::try_from(word_size).unwrap())?;
        utils::write_u32(&mut wtr, u32::try_from(comment_size).unwrap())?;
        wtr.write_all(self.word.as_bytes())?;
        wtr.write_all(self.comment.as_bytes())?;
        let weights_size = self.weights.serialize(&mut wtr)?;
        Ok(mem::size_of::<u32>() * 2 + word_size + weights_size + comment_size)
    }

    pub fn deserialize<R>(mut rdr: R) -> Result<Self>
    where
        R: Read,
    {
        let word_size = utils::read_u32(&mut rdr)?;
        let comment_size = utils::read_u32(&mut rdr)?;
        let mut word_bytes = vec![0; word_size.try_into().unwrap()];
        rdr.read_exact(&mut word_bytes)?;
        let mut comment_bytes = vec![0; comment_size.try_into().unwrap()];
        rdr.read_exact(&mut comment_bytes)?;
        Ok(Self {
            word: String::from_utf8(word_bytes)?,
            weights: DictWeight::deserialize(&mut rdr)?,
            comment: String::from_utf8(comment_bytes)?,
        })
    }
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

    pub fn serialize<W>(&self, mut wtr: W) -> Result<usize>
    where
        W: Write,
    {
        let dict_size = self.dict.len();
        utils::write_u32(&mut wtr, dict_size.try_into().unwrap())?;
        let mut total_size = mem::size_of::<u32>();
        for entry in &self.dict {
            total_size += entry.serialize(&mut wtr)?;
        }
        Ok(total_size)
    }

    pub fn deserialize<R>(mut rdr: R) -> Result<Self>
    where
        R: Read,
    {
        let dict_size = utils::read_u32(&mut rdr)?;
        let mut dict = Vec::with_capacity(dict_size.try_into().unwrap());
        for _ in 0..dict_size {
            dict.push(WordWeightRecord::deserialize(&mut rdr)?);
        }
        Ok(Self { dict })
    }
}
