use std::io::{Read, Write};

use anyhow::Result;
use byteorder::{BigEndian, ReadBytesExt, WriteBytesExt};
use fst::raw::Fst;

#[cfg(feature = "train")]
use crate::feature::FeatureContent;
#[cfg(feature = "train")]
use crate::sentence::BoundaryType;
#[cfg(feature = "train")]
use crate::utils::{FeatureIDManager, LazyIndexSort};
#[cfg(feature = "train")]
use liblinear::LibLinearModel;
#[cfg(feature = "train")]
const EPSILON: f64 = 1e-6;

#[cfg(not(feature = "model-quantize"))]
pub(crate) type WeightValue = f64;
#[cfg(feature = "model-quantize")]
pub(crate) type WeightValue = i16;
#[cfg(not(feature = "model-quantize"))]
pub(crate) type ScoreValue = f64;
#[cfg(feature = "model-quantize")]
pub(crate) type ScoreValue = i32;

pub struct Model {
    pub(crate) word_fst: Fst<Vec<u8>>,
    pub(crate) type_fst: Fst<Vec<u8>>,
    pub(crate) dict_fst: Fst<Vec<u8>>,

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
    pub fn write<W: Write>(&self, wtr: &mut W) -> Result<()> {
        wtr.write_u64::<BigEndian>(self.word_fst.as_bytes().len() as u64)?;
        wtr.write_all(self.word_fst.as_bytes())?;

        wtr.write_u64::<BigEndian>(self.type_fst.as_bytes().len() as u64)?;
        wtr.write_all(self.type_fst.as_bytes())?;

        wtr.write_u64::<BigEndian>(self.dict_fst.as_bytes().len() as u64)?;
        wtr.write_all(self.dict_fst.as_bytes())?;

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

    pub fn read<R: Read>(rdr: &mut R) -> Result<Self> {
        let word_fst_size = rdr.read_u64::<BigEndian>()? as usize;
        let mut word_fst_bytes = vec![0; word_fst_size];
        rdr.read_exact(&mut word_fst_bytes)?;
        let word_fst = Fst::new(word_fst_bytes)?;

        let type_fst_size = rdr.read_u64::<BigEndian>()? as usize;
        let mut type_fst_bytes = vec![0; type_fst_size];
        rdr.read_exact(&mut type_fst_bytes)?;
        let type_fst = Fst::new(type_fst_bytes)?;

        let dict_fst_size = rdr.read_u64::<BigEndian>()? as usize;
        let mut dict_fst_bytes = vec![0; dict_fst_size];
        rdr.read_exact(&mut dict_fst_bytes)?;
        let dict_fst = Fst::new(dict_fst_bytes)?;

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
            word_fst,
            type_fst,
            dict_fst,

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
    pub fn from_liblinear_model(
        model: impl LibLinearModel,
        fid_manager: FeatureIDManager,
        dict_fst: Fst<Vec<u8>>,
        char_window_size: usize,
        type_window_size: usize,
        dict_word_max_size: usize,
    ) -> Self {
        let bias = model.label_bias(BoundaryType::WordBoundary as i32);
        let mut word_sorter = LazyIndexSort::new();
        let mut type_sorter = LazyIndexSort::new();
        let mut word_weights_tmp = vec![];
        let mut type_weights_tmp = vec![];

        let mut dict_weights: Vec<[_; 3]> = (0..dict_word_max_size)
            .map(|_| [ScoreValue::default(); 3])
            .collect();

        #[cfg(feature = "model-quantize")]
        let mut weight_max = bias.abs();
        #[cfg(feature = "model-quantize")]
        for fid in 0..model.num_features() {
            let weight = model
                .feature_coefficient(fid as i32, BoundaryType::WordBoundary as i32)
                .abs();
            if weight > weight_max {
                weight_max = weight;
            }
        }
        #[cfg(feature = "model-quantize")]
        let quantize_multiplier = weight_max / 32767.;

        #[cfg(feature = "model-quantize")]
        let bias = (bias / quantize_multiplier) as i16;

        for (feature, fid) in fid_manager.map {
            let weight =
                model.feature_coefficient(fid as i32 + 1, BoundaryType::WordBoundary as i32);
            if weight > -EPSILON && weight < EPSILON {
                continue;
            }
            #[cfg(not(feature = "model-quantize"))]
            match feature.feature {
                FeatureContent::CharacterNgram(word) => {
                    let id = word_sorter.get_id(word.as_bytes()) as usize;
                    if id == word_weights_tmp.len() {
                        word_weights_tmp
                            .push(vec![0.; char_window_size * 2 - word.chars().count() + 1]);
                    }
                    word_weights_tmp[id][feature.rel_position] = weight;
                }
                FeatureContent::CharacterTypeNgram(types) => {
                    let types_u8: Vec<u8> = types.iter().map(|&t| t as u8).collect();
                    let id = type_sorter.get_id(&types_u8) as usize;
                    if id == type_weights_tmp.len() {
                        type_weights_tmp.push(vec![0.; type_window_size * 2 - types.len() + 1]);
                    }
                    type_weights_tmp[id][feature.rel_position] = weight;
                }
                FeatureContent::DictionaryWord(size) => {
                    dict_weights[size][feature.rel_position] = weight;
                }
            };
            #[cfg(feature = "model-quantize")]
            match feature.feature {
                FeatureContent::CharacterNgram(word) => {
                    let id = word_sorter.get_id(word.as_bytes()) as usize;
                    if id == word_weights_tmp.len() {
                        word_weights_tmp
                            .push(vec![0; char_window_size * 2 - word.chars().count() + 1]);
                    }
                    word_weights_tmp[id][feature.rel_position] =
                        (weight / quantize_multiplier) as i16;
                }
                FeatureContent::CharacterTypeNgram(types) => {
                    let types_u8: Vec<u8> = types.iter().map(|&t| t as u8).collect();
                    let id = type_sorter.get_id(&types_u8) as usize;
                    if id == type_weights_tmp.len() {
                        type_weights_tmp.push(vec![0; type_window_size * 2 - types.len() + 1]);
                    }
                    type_weights_tmp[id][feature.rel_position] =
                        (weight / quantize_multiplier) as i16;
                }
                FeatureContent::DictionaryWord(size) => {
                    dict_weights[size - 1][feature.rel_position] =
                        (weight / quantize_multiplier) as i32;
                }
            };
        }
        let word_id_sort_map = word_sorter.sort();
        let type_id_sort_map = type_sorter.sort();
        let mut word_weights = vec![];
        let mut type_weights = vec![];
        for id in word_id_sort_map {
            word_weights.push(word_weights_tmp[id].clone());
        }
        for id in type_id_sort_map {
            type_weights.push(type_weights_tmp[id].clone());
        }
        let word_fst = Fst::from_iter_map(word_sorter.map).unwrap();
        let type_fst = Fst::from_iter_map(type_sorter.map).unwrap();
        Self {
            word_fst,
            type_fst,
            dict_fst,

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
