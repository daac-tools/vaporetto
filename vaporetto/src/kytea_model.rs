use std::collections::BTreeMap;
use std::convert::TryFrom;
use std::io::BufRead;

use anyhow::{anyhow, Result};
use byteorder::{LittleEndian, ReadBytesExt};
use fst::raw::Fst;

use crate::model::Model;

struct KyteaConfig {
    _model_tag: String,
    _do_ws: bool,
    _do_tags: bool,
    n_tags: u32,
    char_w: u8,
    _char_n: u8,
    type_w: u8,
    _type_n: u8,
    dict_n: u8,
    _bias: bool,
    _epsilon: f64,
    _solver_type: u8,
    char_map: Vec<char>,
}

impl KyteaConfig {
    fn read<R: BufRead>(rdr: &mut R) -> Result<Self> {
        let mut model_tag = String::new();
        rdr.read_line(&mut model_tag)?;
        let do_ws = rdr.read_u8()? != 0;
        let do_tags = rdr.read_u8()? != 0;
        let n_tags = rdr.read_u32::<LittleEndian>()?;
        let char_w = rdr.read_u8()?;
        let char_n = rdr.read_u8()?;
        let type_w = rdr.read_u8()?;
        let type_n = rdr.read_u8()?;
        let dict_n = rdr.read_u8()?;
        let bias = rdr.read_u8()? != 0;
        let epsilon = rdr.read_f64::<LittleEndian>()?;
        let solver_type = rdr.read_u8()?;
        let mut char_map = vec![];
        rdr.read_until(0, &mut char_map)?;
        let char_map: Vec<char> = String::from_utf8(char_map)?.chars().collect();
        Ok(Self {
            _model_tag: model_tag,
            _do_ws: do_ws,
            _do_tags: do_tags,
            n_tags,
            char_w,
            _char_n: char_n,
            type_w,
            _type_n: type_n,
            dict_n,
            _bias: bias,
            _epsilon: epsilon,
            _solver_type: solver_type,
            char_map,
        })
    }
}

trait Readable: Sized {
    fn read<R: BufRead>(config: &KyteaConfig, rdr: &mut R) -> Result<Self>;
}

impl Readable for i16 {
    fn read<R: BufRead>(_config: &KyteaConfig, rdr: &mut R) -> Result<Self> {
        Ok(rdr.read_i16::<LittleEndian>()?)
    }
}

impl Readable for f64 {
    fn read<R: BufRead>(_config: &KyteaConfig, rdr: &mut R) -> Result<Self> {
        Ok(rdr.read_f64::<LittleEndian>()?)
    }
}

impl Readable for char {
    fn read<R: BufRead>(config: &KyteaConfig, rdr: &mut R) -> Result<Self> {
        let cidx = rdr.read_u16::<LittleEndian>()? as usize;
        Ok(config.char_map[cidx - 1])
    }
}

impl<T> Readable for Vec<T>
where
    T: Readable,
{
    fn read<R: BufRead>(config: &KyteaConfig, rdr: &mut R) -> Result<Self> {
        let size = rdr.read_u32::<LittleEndian>()?;
        let mut result = Self::with_capacity(size as usize);
        for _ in 0..size {
            result.push(T::read(config, rdr)?);
        }
        Ok(result)
    }
}

impl Readable for String {
    fn read<R: BufRead>(config: &KyteaConfig, rdr: &mut R) -> Result<Self> {
        let size = rdr.read_u32::<LittleEndian>()?;
        let mut result = Self::new();
        for _ in 0..size {
            let cidx = rdr.read_u16::<LittleEndian>()? as usize;
            result.push(config.char_map[cidx - 1]);
        }
        Ok(result)
    }
}

struct State {
    _failure: u32,
    gotos: BTreeMap<char, u32>,
    outputs: Vec<u32>,
    is_branch: bool,
}

struct Dictionary<T>
where
    T: Readable,
{
    n_dicts: u8,
    states: Vec<State>,
    entries: Vec<T>,
}

impl<T> Dictionary<T>
where
    T: Readable,
{
    fn dump_items(&self) -> Vec<(String, &T)> {
        let mut result = vec![];
        let mut stack = vec![(0, vec![])];
        while let Some((idx, word)) = stack.pop() {
            let state = &self.states[idx];
            if state.is_branch {
                result.push((
                    word.iter().collect(),
                    &self.entries[state.outputs[0] as usize],
                ));
            }
            for (&c, &next_idx) in state.gotos.iter().rev() {
                let mut word = word.clone();
                word.push(c);
                stack.push((next_idx as usize, word));
            }
        }
        result
    }
}

impl<T> Dictionary<T>
where
    T: Readable,
{
    fn read<R: BufRead>(config: &KyteaConfig, rdr: &mut R) -> Result<Option<Self>> {
        let n_dicts = rdr.read_u8()?;
        let n_states = rdr.read_u32::<LittleEndian>()? as usize;
        if n_states == 0 {
            return Ok(None);
        }
        let mut states = Vec::with_capacity(n_states);
        for _ in 0..n_states {
            let failure = rdr.read_u32::<LittleEndian>()?;
            let n_gotos = rdr.read_u32::<LittleEndian>()?;
            let mut gotos = BTreeMap::new();
            for _ in 0..n_gotos {
                let k = char::read(config, rdr)?;
                let v = rdr.read_u32::<LittleEndian>()?;
                gotos.insert(k, v);
            }
            let n_outputs = rdr.read_u32::<LittleEndian>()? as usize;
            let mut outputs = Vec::with_capacity(n_outputs);
            for _ in 0..n_outputs {
                outputs.push(rdr.read_u32::<LittleEndian>()?);
            }
            let is_branch = rdr.read_u8()? != 0;
            states.push(State {
                _failure: failure,
                gotos,
                outputs,
                is_branch,
            });
        }
        let n_entries = rdr.read_u32::<LittleEndian>()? as usize;
        let mut entries = Vec::with_capacity(n_entries);
        for _ in 0..n_entries {
            entries.push(T::read(config, rdr)?);
        }
        Ok(Some(Self {
            n_dicts,
            states,
            entries,
        }))
    }
}

struct FeatureLookup<T>
where
    T: Readable,
{
    char_dict: Option<Dictionary<Vec<T>>>,
    type_dict: Option<Dictionary<Vec<T>>>,
    _self_dict: Option<Dictionary<Vec<T>>>,
    dict_vec: Vec<T>,
    biases: Vec<T>,
    _tag_dict_vec: Vec<T>,
    _tag_unk_vec: Vec<T>,
}

impl<T> FeatureLookup<T>
where
    T: Readable,
{
    fn read<R: BufRead>(config: &KyteaConfig, rdr: &mut R) -> Result<Option<Self>> {
        let active = rdr.read_u8()?;
        if active == 0 {
            return Ok(None);
        }
        let char_dict = Dictionary::read(config, rdr)?;
        let type_dict = Dictionary::read(config, rdr)?;
        let self_dict = Dictionary::read(config, rdr)?;
        let dict_vec = Vec::<T>::read(config, rdr)?;
        let biases = Vec::<T>::read(config, rdr)?;
        let tag_dict_vec = Vec::<T>::read(config, rdr)?;
        let tag_unk_vec = Vec::<T>::read(config, rdr)?;
        Ok(Some(Self {
            char_dict,
            type_dict,
            _self_dict: self_dict,
            dict_vec,
            biases,
            _tag_dict_vec: tag_dict_vec,
            _tag_unk_vec: tag_unk_vec,
        }))
    }
}

struct LinearModel {
    _add_features: bool,
    _solver_type: u8,
    _labels: Vec<i32>,
    _bias: bool,
    multiplier: f64,
    feature_lookup: Option<FeatureLookup<i16>>,
}

impl Readable for Option<LinearModel> {
    fn read<R: BufRead>(config: &KyteaConfig, rdr: &mut R) -> Result<Self> {
        let n_classes = rdr.read_u32::<LittleEndian>()?;
        if n_classes == 0 {
            return Ok(None);
        }
        let add_features = false;
        let solver_type = rdr.read_u8()?;
        let mut labels = vec![];
        for _ in 0..n_classes {
            labels.push(rdr.read_i32::<LittleEndian>()?);
        }
        let bias = rdr.read_u8()? != 0;
        let multiplier = rdr.read_f64::<LittleEndian>()?;
        let feature_lookup = FeatureLookup::read(config, rdr)?;
        Ok(Some(LinearModel {
            _add_features: add_features,
            _solver_type: solver_type,
            _labels: labels,
            _bias: bias,
            multiplier,
            feature_lookup,
        }))
    }
}

struct ModelTagEntry {
    _word: String,
    _tags: Vec<Vec<String>>,
    _tags_in_dicts: Vec<Vec<u8>>,
    in_dict: u8,
    _tag_models: Vec<Option<LinearModel>>,
}

impl Readable for ModelTagEntry {
    fn read<R: BufRead>(config: &KyteaConfig, rdr: &mut R) -> Result<Self> {
        let word = String::read(config, rdr)?;
        let mut tags = Vec::with_capacity(config.n_tags as usize);
        let mut tags_in_dicts = Vec::with_capacity(config.n_tags as usize);
        for _ in 0..config.n_tags {
            let size = rdr.read_u32::<LittleEndian>()? as usize;
            let mut t = Vec::with_capacity(size);
            let mut td = Vec::with_capacity(size);
            for _ in 0..size {
                t.push(String::read(config, rdr)?);
                td.push(rdr.read_u8()?);
            }
            tags.push(t);
            tags_in_dicts.push(td);
        }
        let in_dict = rdr.read_u8()?;
        let mut tag_models = Vec::with_capacity(config.n_tags as usize);
        for _ in 0..config.n_tags {
            tag_models.push(Option::<LinearModel>::read(config, rdr)?);
        }
        Ok(Self {
            _word: word,
            _tags: tags,
            _tags_in_dicts: tags_in_dicts,
            in_dict,
            _tag_models: tag_models,
        })
    }
}

struct ProbTagEntry {
    _word: String,
    _tags: Vec<Vec<String>>,
    _probs: Vec<Vec<f64>>,
}

impl Readable for ProbTagEntry {
    fn read<R: BufRead>(config: &KyteaConfig, rdr: &mut R) -> Result<Self> {
        let word = String::read(config, rdr)?;
        let mut tags = Vec::with_capacity(config.n_tags as usize);
        let mut probs = Vec::with_capacity(config.n_tags as usize);
        for _ in 0..config.n_tags {
            let size = rdr.read_u32::<LittleEndian>()? as usize;
            let mut t = Vec::with_capacity(size);
            let mut p = Vec::with_capacity(size);
            for _ in 0..size {
                t.push(String::read(config, rdr)?);
                p.push(rdr.read_f64::<LittleEndian>()?);
            }
            tags.push(t);
            probs.push(p);
        }
        Ok(Self {
            _word: word,
            _tags: tags,
            _probs: probs,
        })
    }
}

/// Model data created by KyTea.
pub struct KyteaModel {
    config: KyteaConfig,
    wordseg_model: Option<LinearModel>,
    _global_tags: Vec<Vec<String>>,
    _global_models: Vec<Option<LinearModel>>,
    dict: Option<Dictionary<ModelTagEntry>>,
    _subword_dict: Option<Dictionary<ProbTagEntry>>,
}

impl KyteaModel {
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
    pub fn read<R: BufRead>(rdr: &mut R) -> Result<Self> {
        let config = KyteaConfig::read(rdr)?;

        let wordseg_model = Option::<LinearModel>::read(&config, rdr)?;

        let mut global_tags = Vec::with_capacity(config.n_tags as usize);
        let mut global_models = Vec::with_capacity(config.n_tags as usize);

        for _ in 0..config.n_tags {
            global_tags.push(Vec::<String>::read(&config, rdr)?);
            global_models.push(Option::<LinearModel>::read(&config, rdr)?);
        }

        let dict = Dictionary::<ModelTagEntry>::read(&config, rdr)?;
        let subword_dict = Dictionary::<ProbTagEntry>::read(&config, rdr)?;

        Ok(Self {
            config,
            wordseg_model,
            _global_tags: global_tags,
            _global_models: global_models,
            dict,
            _subword_dict: subword_dict,
        })
    }
}

impl TryFrom<KyteaModel> for Model {
    type Error = anyhow::Error;

    fn try_from(model: KyteaModel) -> Result<Self> {
        let config = &model.config;
        let wordseg_model = model
            .wordseg_model
            .ok_or_else(|| anyhow!("no word segmentation model."))?;
        let quantize_multiplier = wordseg_model.multiplier;
        let feature_lookup = wordseg_model
            .feature_lookup
            .ok_or_else(|| anyhow!("no lookup data."))?;
        let bias = feature_lookup.biases[0];
        let char_dict = feature_lookup
            .char_dict
            .ok_or_else(|| anyhow!("no character dictionary."))?;
        let type_dict = feature_lookup
            .type_dict
            .ok_or_else(|| anyhow!("no type dictionary."))?;

        let mut word_map = vec![];
        let mut word_weights = vec![];
        for (i, (word, v)) in char_dict.dump_items().into_iter().enumerate() {
            word_map.push((word, i as u64));
            word_weights.push(v.clone());
        }
        let word_fst = Fst::from_iter_map(word_map)?;

        let mut type_map = vec![];
        let mut type_weights = vec![];
        for (i, (word, v)) in type_dict.dump_items().into_iter().enumerate() {
            type_map.push((word, i as u64));
            type_weights.push(v.clone());
        }
        let type_fst = Fst::from_iter_map(type_map)?;

        let mut dict_map = vec![];
        let mut dict_weights = vec![];
        if let Some(dict) = model.dict {
            for (i, (w, data)) in dict.dump_items().into_iter().enumerate() {
                let word_len = std::cmp::min(w.chars().count(), config.dict_n as usize) - 1;
                let mut weights = [0i32; 3];
                for j in 0..dict.n_dicts as usize {
                    if data.in_dict >> j & 1 == 1 {
                        let offset = 3 * config.dict_n as usize * j + 3 * word_len;
                        weights[0] += feature_lookup.dict_vec[offset] as i32;
                        weights[1] += feature_lookup.dict_vec[offset + 1] as i32;
                        weights[2] += feature_lookup.dict_vec[offset + 2] as i32;
                    }
                }
                dict_weights.push(weights);
                dict_map.push((w, i as u64));
            }
        }
        let dict_fst = Fst::from_iter_map(dict_map)?;

        Ok(Self {
            word_fst,
            type_fst,
            dict_fst,

            #[cfg(not(feature = "model-quantize"))]
            word_weights,
            #[cfg(not(feature = "model-quantize"))]
            type_weights,
            #[cfg(not(feature = "model-quantize"))]
            dict_weights,

            #[cfg(feature = "model-quantize")]
            quantize_multiplier,
            #[cfg(feature = "model-quantize")]
            word_weights,
            #[cfg(feature = "model-quantize")]
            type_weights,
            #[cfg(feature = "model-quantize")]
            dict_weights,

            dict_word_wise: true,
            bias,
            char_window_size: config.char_w as usize,
            type_window_size: config.type_w as usize,
        })
    }
}
