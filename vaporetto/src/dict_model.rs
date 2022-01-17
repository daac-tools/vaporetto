use std::collections::HashMap;
use std::io::{Read, Write};
use std::mem;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::errors::Result;
use crate::ngram_model::NgramModel;

#[derive(Clone, Copy, Default)]
pub struct DictWeight {
    pub right: i32,
    pub inside: i32,
    pub left: i32,
}

impl DictWeight {
    pub fn serialize<W>(&self, mut buf: W) -> Result<usize>
    where
        W: Write,
    {
        buf.write_i32::<LittleEndian>(self.right)?;
        buf.write_i32::<LittleEndian>(self.inside)?;
        buf.write_i32::<LittleEndian>(self.left)?;
        Ok(mem::size_of::<i32>() * 3)
    }

    pub fn deserialize<R>(mut buf: R) -> Result<Self>
    where
        R: Read,
    {
        Ok(Self {
            right: buf.read_i32::<LittleEndian>()?,
            inside: buf.read_i32::<LittleEndian>()?,
            left: buf.read_i32::<LittleEndian>()?,
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
    pub fn serialize<W>(&self, mut buf: W) -> Result<usize>
    where
        W: Write,
    {
        let word_size = self.word.len();
        let comment_size = self.comment.len();
        buf.write_u32::<LittleEndian>(word_size.try_into().unwrap())?;
        buf.write_u32::<LittleEndian>(comment_size.try_into().unwrap())?;
        buf.write_all(self.word.as_bytes())?;
        buf.write_all(self.comment.as_bytes())?;
        let weights_size = self.weights.serialize(&mut buf)?;
        Ok(mem::size_of::<u32>() * 2 + word_size + weights_size + comment_size)
    }

    pub fn deserialize<R>(mut buf: R) -> Result<Self>
    where
        R: Read,
    {
        let word_size = buf.read_u32::<LittleEndian>()?;
        let comment_size = buf.read_u32::<LittleEndian>()?;
        let mut word_bytes = vec![0; word_size.try_into().unwrap()];
        buf.read_exact(&mut word_bytes)?;
        let mut comment_bytes = vec![0; comment_size.try_into().unwrap()];
        buf.read_exact(&mut comment_bytes)?;
        Ok(Self {
            word: String::from_utf8(word_bytes)?,
            weights: DictWeight::deserialize(&mut buf)?,
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

    pub fn dictionary(&self) -> &[WordWeightRecord] {
        &self.dict
    }

    pub fn serialize<W>(&self, mut buf: W) -> Result<usize>
    where
        W: Write,
    {
        let dict_size = self.dict.len();
        buf.write_u32::<LittleEndian>(dict_size.try_into().unwrap())?;
        let mut total_size = mem::size_of::<u32>();
        for entry in &self.dict {
            total_size += entry.serialize(&mut buf)?;
        }
        Ok(total_size)
    }

    pub fn deserialize<R>(mut buf: R) -> Result<Self>
    where
        R: Read,
    {
        let dict_size = buf.read_u32::<LittleEndian>()?;
        let mut dict = Vec::with_capacity(dict_size.try_into().unwrap());
        for _ in 0..dict_size {
            dict.push(WordWeightRecord::deserialize(&mut buf)?);
        }
        Ok(Self { dict })
    }
}
