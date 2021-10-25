use std::io::{Read, Write};

use anyhow::Result;
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};

#[cfg(feature = "train")]
use crate::feature::FeatureContent;
#[cfg(feature = "train")]
use crate::sentence::BoundaryType;
#[cfg(feature = "train")]
use crate::utils::{FeatureIDManager, StringIdManager};
#[cfg(feature = "train")]
use liblinear::LibLinearModel;
#[cfg(feature = "train")]
const EPSILON: f64 = 1e-6;

#[cfg(not(feature = "model-quantize"))]
pub type WeightValue = f64;
#[cfg(feature = "model-quantize")]
pub type WeightValue = i16;
#[cfg(not(feature = "model-quantize"))]
pub type ScoreValue = f64;
#[cfg(feature = "model-quantize")]
pub type ScoreValue = i32;

/// Model data.
pub struct Model {
    pub(crate) words: Vec<Vec<u8>>,
    pub(crate) types: Vec<Vec<u8>>,
    pub(crate) dict: Vec<Vec<u8>>,

    pub(crate) word_weights: Vec<Vec<WeightValue>>,
    pub(crate) type_weights: Vec<Vec<WeightValue>>,
    pub(crate) dict_weights: Vec<[ScoreValue; 3]>,

    #[cfg(feature = "model-quantize")]
    pub(crate) quantize_multiplier: f64,

    pub(crate) dict_word_wise: bool,

    pub(crate) bias: WeightValue,
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
    pub fn write<W: Write>(&self, wtr: &mut W) -> Result<()> {
        wtr.write_u64::<BigEndian>(self.words.len() as u64)?;
        for word in &self.words {
            wtr.write_u64::<BigEndian>(word.len() as u64)?;
            wtr.write_all(word)?;
        }

        wtr.write_u64::<BigEndian>(self.types.len() as u64)?;
        for word in &self.types {
            wtr.write_u64::<BigEndian>(word.len() as u64)?;
            wtr.write_all(word)?;
        }

        wtr.write_u64::<BigEndian>(self.dict.len() as u64)?;
        for word in &self.dict {
            wtr.write_u64::<BigEndian>(word.len() as u64)?;
            wtr.write_all(word)?;
        }
        #[cfg(feature = "model-quantize")]
        wtr.write_f64::<BigEndian>(self.quantize_multiplier)?;

        wtr.write_u64::<BigEndian>(self.word_weights.len() as u64)?;
        for ws in &self.word_weights {
            wtr.write_u64::<BigEndian>(ws.len() as u64)?;
            for &w in ws {
                #[cfg(not(feature = "model-quantize"))]
                wtr.write_f64::<BigEndian>(w)?;
                #[cfg(feature = "model-quantize")]
                wtr.write_i16::<BigEndian>(w)?;
            }
        }

        wtr.write_u64::<BigEndian>(self.type_weights.len() as u64)?;
        for ws in &self.type_weights {
            wtr.write_u64::<BigEndian>(ws.len() as u64)?;
            for &w in ws {
                #[cfg(not(feature = "model-quantize"))]
                wtr.write_f64::<BigEndian>(w)?;
                #[cfg(feature = "model-quantize")]
                wtr.write_i16::<BigEndian>(w)?;
            }
        }

        wtr.write_u64::<BigEndian>(self.dict_weights.len() as u64)?;
        for ws in &self.dict_weights {
            for &w in ws {
                #[cfg(not(feature = "model-quantize"))]
                wtr.write_f64::<BigEndian>(w)?;
                #[cfg(feature = "model-quantize")]
                wtr.write_i32::<BigEndian>(w)?;
            }
        }

        wtr.write_u8(self.dict_word_wise as u8)?;

        #[cfg(not(feature = "model-quantize"))]
        wtr.write_f64::<BigEndian>(self.bias)?;
        #[cfg(feature = "model-quantize")]
        wtr.write_i16::<BigEndian>(self.bias)?;

        wtr.write_u64::<BigEndian>(self.char_window_size as u64)?;
        wtr.write_u64::<BigEndian>(self.type_window_size as u64)?;

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
    pub fn read<R: Read>(rdr: &mut R) -> Result<Self> {
        let words_size = rdr.read_u64::<BigEndian>()? as usize;
        let mut words = Vec::with_capacity(words_size);
        for _ in 0..words_size {
            let word_size = rdr.read_u64::<BigEndian>()? as usize;
            let mut word_bytes = vec![0; word_size];
            rdr.read_exact(&mut word_bytes)?;
            words.push(word_bytes);
        }

        let types_size = rdr.read_u64::<BigEndian>()? as usize;
        let mut types = Vec::with_capacity(types_size);
        for _ in 0..types_size {
            let word_size = rdr.read_u64::<BigEndian>()? as usize;
            let mut word_bytes = vec![0; word_size];
            rdr.read_exact(&mut word_bytes)?;
            types.push(word_bytes);
        }

        let dict_size = rdr.read_u64::<BigEndian>()? as usize;
        let mut dict = Vec::with_capacity(dict_size);
        for _ in 0..dict_size {
            let word_size = rdr.read_u64::<BigEndian>()? as usize;
            let mut word_bytes = vec![0; word_size];
            rdr.read_exact(&mut word_bytes)?;
            dict.push(word_bytes);
        }

        #[cfg(feature = "model-quantize")]
        let quantize_multiplier = rdr.read_f64::<BigEndian>()?;

        let word_weights_size = rdr.read_u64::<BigEndian>()? as usize;
        let mut word_weights = Vec::with_capacity(word_weights_size);
        for _ in 0..word_weights_size {
            let weight_size = rdr.read_u64::<BigEndian>()? as usize;
            let mut weights = Vec::with_capacity(weight_size);
            for _ in 0..weight_size {
                #[cfg(not(feature = "model-quantize"))]
                let weight = rdr.read_f64::<BigEndian>()?;
                #[cfg(feature = "model-quantize")]
                let weight = rdr.read_i16::<BigEndian>()?;
                weights.push(weight);
            }
            word_weights.push(weights);
        }

        let type_weights_size = rdr.read_u64::<BigEndian>()? as usize;
        let mut type_weights = Vec::with_capacity(type_weights_size);
        for _ in 0..type_weights_size {
            let weight_size = rdr.read_u64::<BigEndian>()? as usize;
            let mut weights = Vec::with_capacity(weight_size);
            for _ in 0..weight_size {
                #[cfg(not(feature = "model-quantize"))]
                let weight = rdr.read_f64::<BigEndian>()?;
                #[cfg(feature = "model-quantize")]
                let weight = rdr.read_i16::<BigEndian>()?;
                weights.push(weight);
            }
            type_weights.push(weights);
        }

        let dict_weights_size = rdr.read_u64::<BigEndian>()? as usize;
        let mut dict_weights = Vec::with_capacity(dict_weights_size);
        for _ in 0..dict_weights_size {
            #[cfg(not(feature = "model-quantize"))]
            let mut weights = [0.; 3];
            #[cfg(feature = "model-quantize")]
            let mut weights = [0; 3];
            #[cfg(not(feature = "model-quantize"))]
            for weight in &mut weights {
                *weight = rdr.read_f64::<BigEndian>()?;
            }
            #[cfg(feature = "model-quantize")]
            for weight in &mut weights {
                *weight = rdr.read_i32::<BigEndian>()?;
            }
            dict_weights.push(weights);
        }

        let dict_word_wise = rdr.read_u8()? != 0;

        #[cfg(not(feature = "model-quantize"))]
        let bias = rdr.read_f64::<BigEndian>()?;
        #[cfg(feature = "model-quantize")]
        let bias = rdr.read_i16::<BigEndian>()?;

        let char_window_size = rdr.read_u64::<BigEndian>()? as usize;
        let type_window_size = rdr.read_u64::<BigEndian>()? as usize;

        Ok(Self {
            words,
            types,
            dict,

            #[cfg(feature = "model-quantize")]
            quantize_multiplier,

            word_weights,
            type_weights,
            dict_weights,
            dict_word_wise,
            bias,
            char_window_size,
            type_window_size,
        })
    }

    #[cfg(feature = "train")]
    pub(crate) fn from_liblinear_model(
        model: impl LibLinearModel,
        fid_manager: FeatureIDManager,
        dict: Vec<Vec<u8>>,
        char_window_size: usize,
        type_window_size: usize,
        dict_word_max_size: usize,
    ) -> Self {
        let wb_idx = model
            .labels()
            .iter()
            .position(|&cls| BoundaryType::WordBoundary as i32 == cls)
            .unwrap() as i32;

        let bias = model.label_bias(wb_idx);
        let mut words = vec![];
        let mut types = vec![];
        let mut word_weights = vec![];
        let mut type_weights = vec![];
        let mut dict_weights: Vec<[_; 3]> = (0..dict_word_max_size)
            .map(|_| [ScoreValue::default(); 3])
            .collect();
        let mut word_ids = StringIdManager::new();
        let mut type_ids = StringIdManager::new();

        #[cfg(feature = "model-quantize")]
        let mut weight_max = bias.abs();
        #[cfg(feature = "model-quantize")]
        for fid in 0..model.num_features() {
            let weight = model.feature_coefficient(fid as i32, wb_idx).abs();
            if weight > weight_max {
                weight_max = weight;
            }
        }
        #[cfg(feature = "model-quantize")]
        let quantize_multiplier = weight_max / 32767.;

        #[cfg(feature = "model-quantize")]
        let bias = (bias / quantize_multiplier) as i16;

        for (feature, fid) in fid_manager.map {
            let weight = model.feature_coefficient(fid as i32 + 1, wb_idx);
            if weight > -EPSILON && weight < EPSILON {
                continue;
            }
            #[cfg(not(feature = "model-quantize"))]
            match feature.feature {
                FeatureContent::CharacterNgram(word) => {
                    let id = word_ids.get_id(word.as_bytes());
                    if id == word_weights.len() {
                        words.push(word);
                        word_weights
                            .push(vec![0.; char_window_size * 2 - word.chars().count() + 1]);
                    }
                    word_weights[id][feature.rel_position] = weight;
                }
                FeatureContent::CharacterTypeNgram(types) => {
                    let types_u8: Vec<u8> = types.iter().map(|&t| t as u8).collect();
                    let id = type_ids.get_id(&types_u8);
                    if id == type_weights.len() {
                        type_weights.push(vec![0.; type_window_size * 2 - word.len() + 1]);
                    }
                    type_weights[id][feature.rel_position] = weight;
                }
                FeatureContent::DictionaryWord(size) => {
                    dict_weights[size - 1][feature.rel_position] = weight;
                }
            };
            #[cfg(feature = "model-quantize")]
            match feature.feature {
                FeatureContent::CharacterNgram(word) => {
                    let id = word_ids.get_id(word.as_bytes());
                    if id == word_weights.len() {
                        words.push(word.as_bytes().to_vec());
                        word_weights.push(vec![0; char_window_size * 2 - word.chars().count() + 1]);
                    }
                    word_weights[id][feature.rel_position] = (weight / quantize_multiplier) as i16;
                }
                FeatureContent::CharacterTypeNgram(word) => {
                    let id = type_ids.get_id(word) as usize;
                    if id == type_weights.len() {
                        types.push(word.to_vec());
                        type_weights.push(vec![0; type_window_size * 2 - word.len() + 1]);
                    }
                    type_weights[id][feature.rel_position] = (weight / quantize_multiplier) as i16;
                }
                FeatureContent::DictionaryWord(size) => {
                    dict_weights[size - 1][feature.rel_position] =
                        (weight / quantize_multiplier) as i32;
                }
            };
        }
        Self {
            words,
            types,
            dict,

            #[cfg(feature = "model-quantize")]
            quantize_multiplier,

            word_weights,
            type_weights,
            dict_weights,
            dict_word_wise: false,
            bias,
            char_window_size,
            type_window_size,
        }
    }
}
