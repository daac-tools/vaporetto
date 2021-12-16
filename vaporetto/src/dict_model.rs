use std::collections::HashMap;
use std::io::{Read, Write};
use std::mem;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::errors::{Result, VaporettoError};
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

pub enum DictModel {
    Wordwise(DictModelWordwise),
    Lengthwise(DictModelLengthwise),
}

impl DictModel {
    const TYPE_ID_WORDWISE: u8 = 0;
    const TYPE_ID_LENGTHWISE: u8 = 1;

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

    pub fn serialize<W>(&self, mut buf: W) -> Result<usize>
    where
        W: Write,
    {
        let size = match self {
            Self::Wordwise(model) => {
                buf.write_u8(Self::TYPE_ID_WORDWISE)?;
                model.serialize(buf)?
            }
            Self::Lengthwise(model) => {
                buf.write_u8(Self::TYPE_ID_LENGTHWISE)?;
                model.serialize(buf)?
            }
        };
        Ok(mem::size_of::<u8>() + size)
    }

    pub fn deserialize<R>(mut buf: R) -> Result<Self>
    where
        R: Read,
    {
        let type_id = buf.read_u8()?;
        match type_id {
            Self::TYPE_ID_WORDWISE => Ok(Self::Wordwise(DictModelWordwise::deserialize(buf)?)),
            Self::TYPE_ID_LENGTHWISE => {
                Ok(Self::Lengthwise(DictModelLengthwise::deserialize(buf)?))
            }
            _ => Err(VaporettoError::invalid_model(
                "invalid type_id of dict_model",
            )),
        }
    }
}

/// Record of weights for each word.
#[derive(Clone)]
pub struct WordWeightRecord {
    pub(crate) word: String,
    pub(crate) weights: DictWeight,
}

impl WordWeightRecord {
    pub fn serialize<W>(&self, mut buf: W) -> Result<usize>
    where
        W: Write,
    {
        let word_size = self.word.len();
        buf.write_u32::<LittleEndian>(word_size.try_into().unwrap())?;
        buf.write_all(self.word.as_bytes())?;
        let weights_size = self.weights.serialize(&mut buf)?;
        Ok(mem::size_of::<u32>() + word_size + weights_size)
    }

    pub fn deserialize<R>(mut buf: R) -> Result<Self>
    where
        R: Read,
    {
        let word_size = buf.read_u32::<LittleEndian>()?;
        let mut str_bytes = vec![0; word_size.try_into().unwrap()];
        buf.read_exact(&mut str_bytes)?;
        Ok(Self {
            word: String::from_utf8(str_bytes)?,
            weights: DictWeight::deserialize(&mut buf)?,
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

    pub fn serialize<W>(&self, mut buf: W) -> Result<usize>
    where
        W: Write,
    {
        let words_size = self.words.len();
        let weights_size = self.weights.len();
        buf.write_u32::<LittleEndian>(words_size.try_into().unwrap())?;
        buf.write_u32::<LittleEndian>(weights_size.try_into().unwrap())?;
        let mut total_size = mem::size_of::<u32>() * 2;
        for word in &self.words {
            let word_size = word.len();
            buf.write_u32::<LittleEndian>(word_size.try_into().unwrap())?;
            buf.write_all(word.as_bytes())?;
            total_size += mem::size_of::<u32>() + word_size;
        }
        for weight in &self.weights {
            total_size += weight.serialize(&mut buf)?;
        }
        Ok(total_size)
    }

    pub fn deserialize<R>(mut buf: R) -> Result<Self>
    where
        R: Read,
    {
        let words_size = buf.read_u32::<LittleEndian>()?;
        let weights_size = buf.read_u32::<LittleEndian>()?;
        let mut words = Vec::with_capacity(words_size.try_into().unwrap());
        for _ in 0..words_size {
            let word_size = buf.read_u32::<LittleEndian>()?;
            let mut word_bytes = vec![0; word_size.try_into().unwrap()];
            buf.read_exact(&mut word_bytes)?;
            words.push(String::from_utf8(word_bytes)?);
        }
        let mut weights = Vec::with_capacity(weights_size.try_into().unwrap());
        for _ in 0..weights_size {
            weights.push(DictWeight::deserialize(&mut buf)?);
        }
        Ok(Self { words, weights })
    }
}
