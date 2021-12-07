use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::dict_model::{DictModel, DictModelWordwise, WordWeightRecord};
use crate::ngram_model::NgramModel;

#[cfg(feature = "train")]
use crate::dict_model::{DictModelLengthwise, DictWeight};
#[cfg(feature = "train")]
use crate::feature::FeatureContent;
#[cfg(feature = "train")]
use crate::ngram_model::NgramData;
#[cfg(feature = "train")]
use crate::sentence::BoundaryType;
#[cfg(feature = "train")]
use crate::utils::{FeatureIDManager, StringIdManager};
#[cfg(feature = "train")]
use liblinear::LibLinearModel;

#[cfg(feature = "train")]
const EPSILON: f64 = 1e-6;

// Bit depth for weight quantization.
#[cfg(feature = "train")]
const QUANTIZE_BIT_DEPTH: u8 = 16;

/// Model data.
#[derive(Serialize, Deserialize)]
pub struct Model {
    pub(crate) char_ngram_model: NgramModel<String>,
    pub(crate) type_ngram_model: NgramModel<Vec<u8>>,
    pub(crate) dict_model: DictModel,

    pub(crate) quantize_multiplier: f64,

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
    pub fn write<W>(&self, wtr: &mut W) -> Result<(), bincode::Error>
    where
        W: Write,
    {
        bincode::serialize_into(wtr, self)
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
    pub fn read<R>(rdr: &mut R) -> Result<Self, bincode::Error>
    where
        R: Read,
    {
        bincode::deserialize_from(rdr)
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
        let mut dict_weights = vec![DictWeight::default(); dict_word_max_size];
        let mut char_ngram_ids = StringIdManager::new();
        let mut type_ngram_ids = StringIdManager::new();

        let mut weight_max = bias.abs();
        for fid in 0..model.num_features() {
            let weight = model.feature_coefficient(fid as i32, wb_idx).abs();
            if weight > weight_max {
                weight_max = weight;
            }
        }
        let quantize_multiplier = weight_max / ((1 << (QUANTIZE_BIT_DEPTH - 1)) - 1) as f64;

        let bias = (bias / quantize_multiplier) as i32;

        for (feature, fid) in fid_manager.map {
            let weight = model.feature_coefficient(fid as i32 + 1, wb_idx);
            if weight > -EPSILON && weight < EPSILON {
                continue;
            }

            let weight = weight / quantize_multiplier;

            match feature.feature {
                FeatureContent::CharacterNgram(char_ngram) => {
                    let id = char_ngram_ids.get_id(&char_ngram);
                    if id == char_ngrams.len() {
                        char_ngrams.push(NgramData {
                            ngram: char_ngram.to_string(),
                            weights: vec![0; char_window_size * 2 - char_ngram.chars().count() + 1],
                        });
                    }
                    char_ngrams[id].weights[feature.rel_position] = weight as i32;
                }
                FeatureContent::CharacterTypeNgram(type_ngram) => {
                    let id = type_ngram_ids.get_id(type_ngram) as usize;
                    if id == type_ngrams.len() {
                        type_ngrams.push(NgramData {
                            ngram: type_ngram.to_vec(),
                            weights: vec![0; type_window_size * 2 - type_ngram.len() + 1],
                        });
                    }
                    type_ngrams[id].weights[feature.rel_position] = weight as i32;
                }
                FeatureContent::DictionaryWord(size) => match feature.rel_position {
                    0 => dict_weights[size - 1].right = weight as i32,
                    1 => dict_weights[size - 1].inside = weight as i32,
                    2 => dict_weights[size - 1].left = weight as i32,
                    _ => panic!("Invalid rel_position"),
                },
            };
        }
        Self {
            char_ngram_model: NgramModel::new(char_ngrams),
            type_ngram_model: NgramModel::new(type_ngrams),
            dict_model: DictModel::Lengthwise(DictModelLengthwise {
                words: dict,
                weights: dict_weights,
            }),

            quantize_multiplier,

            bias,
            char_window_size,
            type_window_size,
        }
    }

    pub fn dump_dictionary(&self) -> Vec<WordWeightRecord> {
        self.dict_model.dump_dictionary()
    }

    pub fn replace_dictionary(&mut self, dict: Vec<WordWeightRecord>) {
        self.dict_model = DictModel::Wordwise(DictModelWordwise { dict });
    }
}
