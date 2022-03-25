use std::io::{Read, Write};

use bincode::{Decode, Encode};

use crate::dict_model::{DictModel, WordWeightRecord};
use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::tag_model::TagModel;

/// Magic number.
const MODEL_MAGIC: &[u8] = b"VaporettoTokenizer 0.4.0\n";

/// Model data.
pub struct Model {
    pub(crate) data: ModelData,
}

#[derive(Decode, Encode)]
pub struct ModelData {
    pub(crate) char_ngram_model: NgramModel<String>,
    pub(crate) type_ngram_model: NgramModel<Vec<u8>>,
    pub(crate) dict_model: DictModel,
    pub(crate) bias: i32,
    pub(crate) char_window_size: u8,
    pub(crate) type_window_size: u8,
    pub(crate) tag_model: TagModel,
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
        tag_model: TagModel,
    ) -> Self {
        Self {
            data: ModelData {
                char_ngram_model,
                type_ngram_model,
                dict_model,
                bias,
                char_window_size,
                type_window_size,
                tag_model,
            },
        }
    }

    /// Exports the model data.
    ///
    /// # Arguments
    ///
    /// * `wtr` - Byte-oriented sink object.
    ///
    /// # Errors
    ///
    /// When `wtr` generates an error, it will be returned as is.
    pub fn write<W>(&self, mut wtr: W) -> Result<()>
    where
        W: Write,
    {
        wtr.write_all(MODEL_MAGIC)?;
        let config = bincode::config::standard();
        bincode::encode_into_std_write(&self.data, &mut wtr, config)?;
        Ok(())
    }

    /// Creates a model from a reader.
    ///
    /// # Arguments
    ///
    /// * `rdr` - A data source.
    ///
    /// # Returns
    ///
    /// A model data read from `rdr`.
    ///
    /// # Errors
    ///
    /// When `rdr` generates an error, it will be returned as is.
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
        Ok(Self {
            data: bincode::decode_from_std_read(&mut rdr, config)?,
        })
    }

    pub fn dictionary(&self) -> &[WordWeightRecord] {
        self.data.dict_model.dictionary()
    }

    pub fn replace_dictionary(&mut self, dict: Vec<WordWeightRecord>) {
        self.data.dict_model = DictModel::new(dict);
    }
}
