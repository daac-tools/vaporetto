use std::io::{Read, Write};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::dict_model::{DictModel, WordWeightRecord};
use crate::errors::Result;
use crate::ngram_model::NgramModel;

/// Model data.
pub struct Model {
    pub(crate) char_ngram_model: NgramModel<String>,
    pub(crate) type_ngram_model: NgramModel<Vec<u8>>,
    pub(crate) dict_model: DictModel,
    pub(crate) bias: i32,
    pub(crate) char_window_size: usize,
    pub(crate) type_window_size: usize,
}

impl Model {
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
        self.char_ngram_model.serialize(&mut wtr)?;
        self.type_ngram_model.serialize(&mut wtr)?;
        self.dict_model.serialize(&mut wtr)?;
        wtr.write_i32::<LittleEndian>(self.bias)?;
        wtr.write_u32::<LittleEndian>(self.char_window_size.try_into().unwrap())?;
        wtr.write_u32::<LittleEndian>(self.type_window_size.try_into().unwrap())?;
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
        Ok(Self {
            char_ngram_model: NgramModel::<String>::deserialize(&mut rdr)?,
            type_ngram_model: NgramModel::<Vec<u8>>::deserialize(&mut rdr)?,
            dict_model: DictModel::deserialize(&mut rdr)?,
            bias: rdr.read_i32::<LittleEndian>()?,
            char_window_size: rdr.read_u32::<LittleEndian>()?.try_into().unwrap(),
            type_window_size: rdr.read_u32::<LittleEndian>()?.try_into().unwrap(),
        })
    }

    pub fn dictionary(&self) -> &[WordWeightRecord] {
        self.dict_model.dictionary()
    }

    pub fn replace_dictionary(&mut self, dict: Vec<WordWeightRecord>) {
        self.dict_model = DictModel::new(dict);
    }
}
