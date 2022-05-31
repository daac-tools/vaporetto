use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::io::{Read, Write};

use bincode::{Decode, Encode};

use crate::dict_model::{DictModel, WordWeightRecord};
use crate::errors::{Result, VaporettoError};
use crate::ngram_model::{NgramModel, TagNgramModel};
use crate::utils::VecWriter;

/// Magic number.
const MODEL_MAGIC: &[u8] = b"VaporettoTokenizer 0.5.0\n";

// For each token, a model is trained for every tag, but the scores of all tags are calculated in
// parallel during prediction.
// Thus, the score array is a concatenation of all classes of all tags.
//
// For example, the following token has 3 POS tags and 3 pronunciation tags, so the score array
// contains 6 items. The predictor picks the tag with the largest score.
//
//   token:   "君"
//   tags:    [["名詞", "代名詞", "接尾辞"], ["クン", "キミ", "ギミ"]]
//   scores:  [    176,     3647,       39,      518,   9346,    126 ]
//
//   results: ["代名詞", "キミ"]
//
// If there is only one tag candidate, the model is not trained.
// In the following example, the predictor determines the first tag without prediction, so the
// score array only contains scores for the second tag.
//
//   token:   "犬"
//   tags:    [["名詞"], ["イヌ", "ケン"]]
//   scores:  [              475,   1563 ]
//
//   results: ["名詞", "ケン"]
#[derive(Debug, Decode, Encode)]
pub struct TagModel {
    pub(crate) token: String,
    pub(crate) tags: Vec<Vec<String>>,
    pub(crate) char_ngram_model: TagNgramModel<String>,
    pub(crate) type_ngram_model: TagNgramModel<Vec<u8>>,
    pub(crate) bias: Vec<i32>,
}

/// Model data.
#[derive(Debug)]
pub struct Model(pub(crate) ModelData);

#[derive(Debug, Decode, Encode)]
pub struct ModelData {
    pub(crate) char_ngram_model: NgramModel<String>,
    pub(crate) type_ngram_model: NgramModel<Vec<u8>>,
    pub(crate) dict_model: DictModel,
    pub(crate) bias: i32,
    pub(crate) char_window_size: u8,
    pub(crate) type_window_size: u8,
    pub(crate) tag_models: Vec<TagModel>,
}

impl Model {
    #[cfg(any(feature = "train", feature = "kytea", test))]
    pub(crate) const fn new(
        char_ngram_model: NgramModel<String>,
        type_ngram_model: NgramModel<Vec<u8>>,
        dict_model: DictModel,
        bias: i32,
        char_window_size: u8,
        type_window_size: u8,
        tag_models: Vec<TagModel>,
    ) -> Self {
        Self(ModelData {
            char_ngram_model,
            type_ngram_model,
            dict_model,
            bias,
            char_window_size,
            type_window_size,
            tag_models,
        })
    }

    /// Exports the model data into a [`Vec`].
    ///
    /// # Errors
    ///
    /// When bincode generates an error, it will be returned as is.
    pub fn to_vec(&self) -> Result<Vec<u8>> {
        let mut wtr = VecWriter(MODEL_MAGIC.to_vec());
        let config = bincode::config::standard();
        bincode::encode_into_writer(&self.0, &mut wtr, config)?;
        Ok(wtr.0)
    }

    /// Exports the model data.
    ///
    /// # Errors
    ///
    /// When bincode generates an error, it will be returned as is.
    #[cfg(feature = "std")]
    pub fn write<W>(&self, mut wtr: W) -> Result<()>
    where
        W: Write,
    {
        wtr.write_all(MODEL_MAGIC)?;
        let config = bincode::config::standard();
        bincode::encode_into_std_write(&self.0, &mut wtr, config)?;
        Ok(())
    }

    /// Creates a model from a slice and returns a tuple of the model and the remaining slice.
    ///
    /// # Errors
    ///
    /// When bincode generates an error, it will be returned as is.
    pub fn read_slice(slice: &[u8]) -> Result<(Self, &[u8])> {
        if &slice[..MODEL_MAGIC.len()] != MODEL_MAGIC {
            return Err(VaporettoError::invalid_model("model version mismatch"));
        }
        let config = bincode::config::standard();
        let (data, size) = bincode::decode_from_slice(&slice[MODEL_MAGIC.len()..], config)?;
        Ok((Self(data), &slice[MODEL_MAGIC.len() + size..]))
    }

    /// Creates a model from a reader.
    ///
    /// # Errors
    ///
    /// When bincode generates an error, it will be returned as is.
    #[cfg(feature = "std")]
    pub fn read<R>(mut rdr: R) -> Result<Self>
    where
        R: Read,
    {
        let mut magic = [0; MODEL_MAGIC.len()];
        rdr.read_exact(&mut magic)?;
        if magic != MODEL_MAGIC {
            return Err(VaporettoError::invalid_model("model version mismatch"));
        }
        let config = bincode::config::standard();
        Ok(Self(bincode::decode_from_std_read(&mut rdr, config)?))
    }

    /// Returns the slice of dictionary words.
    pub fn dictionary(&self) -> &[WordWeightRecord] {
        self.0.dict_model.dictionary()
    }

    /// Replaces the dictionary with the given data.
    pub fn replace_dictionary(&mut self, dict: Vec<WordWeightRecord>) {
        self.0.dict_model = DictModel::new(dict);
    }
}
