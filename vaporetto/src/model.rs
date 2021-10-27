use std::io::{Read, Write};

use anyhow::Result;
use serde::{Serialize, Deserialize};

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
#[derive(Serialize, Deserialize)]
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
    pub fn write<W>(&self, wtr: &mut W) -> Result<()> where W: Write {
        bincode::serialize_into(wtr, self)?;
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
    pub fn read<R>(rdr: &mut R) -> Result<Self> where R: Read {
        Ok(bincode::deserialize_from(rdr)?)
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
