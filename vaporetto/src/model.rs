use alloc::string::String;
use alloc::vec::Vec;

#[cfg(feature = "std")]
use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::dict_model::{DictModel, WordWeightRecord};
use crate::errors::{Result, VaporettoError};
use crate::ngram_model::{NgramModel, TagNgramModel};

/// Magic number.
const MODEL_MAGIC: &[u8] = b"VaporettoTokenizer 0.7.0\n";

// For each token, a model is trained for every tag independently, but the scores of all tags are
// calculated in parallel during prediction.
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
/// Internal representation of a tag model.
#[derive(Debug, Deserialize, Serialize)]
pub struct TagModel {
    pub(crate) token: String,
    pub(crate) tags: Vec<Vec<String>>,
    pub(crate) char_ngram_model: TagNgramModel<String>,
    pub(crate) type_ngram_model: TagNgramModel<Vec<u8>>,
    pub(crate) bias: Vec<i32>,
}

impl TagModel {
    /// Returns the token string.
    pub fn token(&self) -> &str {
        &self.token
    }
}

/// Model data.
#[derive(Debug)]
pub struct Model(pub(crate) ModelData);

#[derive(Debug, Deserialize, Serialize)]
pub struct ModelData {
    pub(crate) char_ngram_model: NgramModel<String>,
    pub(crate) type_ngram_model: NgramModel<Vec<u8>>,
    pub(crate) dict_model: DictModel,
    pub(crate) bias: i32,
    pub(crate) char_window_size: u8,
    pub(crate) type_window_size: u8,
    // Instead of using Map, we use Vec to increase compression ratio and performance.
    pub(crate) tag_models: Vec<TagModel>,
}

impl ModelData {
    fn validate(&self) -> Result<()> {
        for record in self.dict_model.dictionary() {
            if record.weights.len() != record.word.chars().count() + 1 {
                return Err(VaporettoError::invalid_model(
                    "weights of a dictionary word do not match the length of the word",
                ));
            }
        }
        Ok(())
    }
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
    /// When postcard generates an error, it will be returned as is.
    pub fn to_vec(&self) -> Result<Vec<u8>> {
        Ok(postcard::to_extend(&self.0, MODEL_MAGIC.to_vec())?)
    }

    /// Exports the model data.
    ///
    /// # Errors
    ///
    /// When postcard generates an error, it will be returned as is.
    #[cfg(feature = "std")]
    pub fn write<W>(&self, mut wtr: W) -> Result<()>
    where
        W: Write,
    {
        wtr.write_all(MODEL_MAGIC)?;
        postcard::to_io(&self.0, wtr)?;
        Ok(())
    }

    /// Creates a model from a slice and returns a tuple of the model and the remaining slice.
    ///
    /// # Errors
    ///
    /// * If postcard generates an error, it will be returned as is.
    /// * If the model is invalid, an error variant will be returned.
    pub fn read_slice(slice: &[u8]) -> Result<(Self, &[u8])> {
        let Some(slice) = slice.strip_prefix(MODEL_MAGIC) else {
            return Err(VaporettoError::invalid_model("model version mismatch"));
        };
        let (data, rest): (ModelData, _) = postcard::take_from_bytes(slice)?;
        data.validate()?;
        Ok((Self(data), rest))
    }

    /// Creates a model from a reader.
    ///
    /// # Errors
    ///
    /// * If postcard generates an error, it will be returned as is.
    /// * If the model is invalid, an error variant will be returned.
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
        // postcard does not support streaming deserialization of owned data, so the whole model
        // is loaded into memory first.
        let mut data = vec![];
        rdr.read_to_end(&mut data)?;
        let data: ModelData = postcard::from_bytes(&data)?;
        data.validate()?;
        Ok(Self(data))
    }

    /// Returns the slice of dictionary words.
    pub fn dictionary(&self) -> &[WordWeightRecord] {
        self.0.dict_model.dictionary()
    }

    /// Replaces the dictionary with the given data.
    pub fn replace_dictionary(&mut self, dict: Vec<WordWeightRecord>) {
        self.0.dict_model = DictModel::new(dict);
    }

    /// Returns the slice of tag models.
    pub fn tag_models(&self) -> &[TagModel] {
        &self.0.tag_models
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MODEL_DATA: &[u8] = include_bytes!("../../resources/model.bin");

    #[test]
    fn test_read_slice_and_to_vec() {
        let mut data = MODEL_DATA.to_vec();
        data.extend_from_slice(b"rest");
        let (model, rest) = Model::read_slice(&data).unwrap();
        assert_eq!(b"rest", rest);
        assert_eq!(MODEL_DATA, model.to_vec().unwrap());
    }

    #[test]
    fn test_read_slice_version_mismatch() {
        let mut data = b"VaporettoTokenizer 0.5.0\n".to_vec();
        data.extend_from_slice(&MODEL_DATA[MODEL_MAGIC.len()..]);
        assert!(matches!(
            Model::read_slice(&data),
            Err(VaporettoError::InvalidModel(_)),
        ));
        assert!(matches!(
            Model::read_slice(b"Vaporetto"),
            Err(VaporettoError::InvalidModel(_)),
        ));
    }

    fn create_model_with_invalid_dictionary() -> Model {
        Model::new(
            NgramModel(vec![]),
            NgramModel(vec![]),
            DictModel(vec![WordWeightRecord {
                word: "火星".into(),
                weights: vec![1],
                comment: "".into(),
            }]),
            0,
            3,
            3,
            vec![],
        )
    }

    #[test]
    fn test_read_slice_invalid_dictionary() {
        let data = create_model_with_invalid_dictionary().to_vec().unwrap();
        assert!(matches!(
            Model::read_slice(&data),
            Err(VaporettoError::InvalidModel(_)),
        ));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_read_invalid_dictionary() {
        let data = create_model_with_invalid_dictionary().to_vec().unwrap();
        assert!(matches!(
            Model::read(data.as_slice()),
            Err(VaporettoError::InvalidModel(_)),
        ));
    }

    #[cfg(feature = "std")]
    #[test]
    fn test_read_and_write() {
        let model = Model::read(MODEL_DATA).unwrap();
        let mut data = vec![];
        model.write(&mut data).unwrap();
        assert_eq!(MODEL_DATA, data);
    }
}
