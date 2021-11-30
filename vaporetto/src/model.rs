use std::io::{Read, Write};

use anyhow::Result;
use serde::{Deserialize, Serialize};

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

#[derive(Clone, Copy, Default, Serialize, Deserialize)]
pub struct DictWeight {
    pub right: ScoreValue,
    pub inner: ScoreValue,
    pub left: ScoreValue,
}

/// Model data.
#[derive(Serialize, Deserialize)]
pub struct Model {
    pub(crate) char_ngrams: Vec<String>,
    pub(crate) type_ngrams: Vec<Vec<u8>>,
    pub(crate) dict: Vec<String>,

    pub(crate) char_ngram_weights: Vec<Vec<WeightValue>>,
    pub(crate) type_ngram_weights: Vec<Vec<WeightValue>>,
    pub(crate) dict_weights: Vec<DictWeight>,

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
    pub fn write<W>(&self, wtr: &mut W) -> Result<()>
    where
        W: Write,
    {
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
    pub fn read<R>(rdr: &mut R) -> Result<Self>
    where
        R: Read,
    {
        Ok(bincode::deserialize_from(rdr)?)
    }

    #[cfg(feature = "train")]
    pub(crate) fn from_liblinear_model(
        model: impl LibLinearModel,
        fid_manager: FeatureIDManager,
        dict: Vec<String>,
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
        let mut char_ngrams = vec![];
        let mut type_ngrams = vec![];
        let mut char_ngram_weights = vec![];
        let mut type_ngram_weights = vec![];
        let mut dict_weights = vec![DictWeight::default(); dict_word_max_size];
        let mut char_ngram_ids = StringIdManager::new();
        let mut type_ngram_ids = StringIdManager::new();

        #[cfg(feature = "model-quantize")]
        let quantize_multiplier = {
            let mut weight_max = bias.abs();
            for fid in 0..model.num_features() {
                let weight = model.feature_coefficient(fid as i32, wb_idx).abs();
                if weight > weight_max {
                    weight_max = weight;
                }
            }
            weight_max / 32767.
        };

        #[cfg(feature = "model-quantize")]
        let bias = (bias / quantize_multiplier) as i16;

        for (feature, fid) in fid_manager.map {
            let weight = model.feature_coefficient(fid as i32 + 1, wb_idx);
            if weight > -EPSILON && weight < EPSILON {
                continue;
            }

            #[cfg(feature = "model-quantize")]
            let weight = weight / quantize_multiplier;

            match feature.feature {
                FeatureContent::CharacterNgram(char_ngram) => {
                    let id = char_ngram_ids.get_id(&char_ngram);
                    if id == char_ngram_weights.len() {
                        char_ngrams.push(char_ngram.to_string());
                        char_ngram_weights.push(vec![
                            WeightValue::default();
                            char_window_size * 2
                                - char_ngram.chars().count()
                                + 1
                        ]);
                    }
                    char_ngram_weights[id][feature.rel_position] = weight as WeightValue;
                }
                FeatureContent::CharacterTypeNgram(type_ngram) => {
                    let id = type_ngram_ids.get_id(type_ngram) as usize;
                    if id == type_ngram_weights.len() {
                        type_ngrams.push(type_ngram.to_vec());
                        type_ngram_weights.push(vec![
                            WeightValue::default();
                            type_window_size * 2 - type_ngram.len() + 1
                        ]);
                    }
                    type_ngram_weights[id][feature.rel_position] = weight as WeightValue;
                }
                FeatureContent::DictionaryWord(size) => match feature.rel_position {
                    0 => dict_weights[size - 1].right = weight as ScoreValue,
                    1 => dict_weights[size - 1].inner = weight as ScoreValue,
                    2 => dict_weights[size - 1].left = weight as ScoreValue,
                    _ => panic!("Invalid rel_position"),
                },
            };
        }
        Self {
            char_ngrams,
            type_ngrams,
            dict,

            #[cfg(feature = "model-quantize")]
            quantize_multiplier,

            char_ngram_weights,
            type_ngram_weights,
            dict_weights,
            dict_word_wise: false,
            bias,
            char_window_size,
            type_window_size,
        }
    }
}
