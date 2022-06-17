use std::convert::TryFrom;
use std::io::BufRead;

use crate::dict_model::{DictModel, DictWeight, WordWeightRecord};
use crate::errors::{Result, VaporettoError};
use crate::model::Model;
use crate::ngram_model::{NgramData, NgramModel};
use crate::sentence::CharacterType;
use crate::utils;

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
    fn read<R>(mut rdr: R) -> Result<Self>
    where
        R: BufRead,
    {
        let mut model_tag = String::new();
        rdr.read_line(&mut model_tag)?;
        let do_ws = utils::read_u8(&mut rdr)? != 0;
        let do_tags = utils::read_u8(&mut rdr)? != 0;
        let n_tags = utils::read_u32(&mut rdr)?;
        let char_w = utils::read_u8(&mut rdr)?;
        let char_n = utils::read_u8(&mut rdr)?;
        let type_w = utils::read_u8(&mut rdr)?;
        let type_n = utils::read_u8(&mut rdr)?;
        let dict_n = utils::read_u8(&mut rdr)?;
        let bias = utils::read_u8(&mut rdr)? != 0;
        let epsilon = utils::read_f64(&mut rdr)?;
        let solver_type = utils::read_u8(&mut rdr)?;
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
    fn read<R>(config: &KyteaConfig, rdr: R) -> Result<Self>
    where
        R: BufRead;
}

impl Readable for i16 {
    fn read<R>(_config: &KyteaConfig, mut rdr: R) -> Result<Self>
    where
        R: BufRead,
    {
        Ok(utils::read_i16(&mut rdr)?)
    }
}

impl Readable for f64 {
    fn read<R>(_config: &KyteaConfig, mut rdr: R) -> Result<Self>
    where
        R: BufRead,
    {
        Ok(utils::read_f64(&mut rdr)?)
    }
}

impl Readable for char {
    fn read<R>(config: &KyteaConfig, mut rdr: R) -> Result<Self>
    where
        R: BufRead,
    {
        let cidx = usize::from(utils::read_u16(&mut rdr)?);
        Ok(config.char_map[cidx - 1])
    }
}

impl<T> Readable for Vec<T>
where
    T: Readable,
{
    fn read<R>(config: &KyteaConfig, mut rdr: R) -> Result<Self>
    where
        R: BufRead,
    {
        let size = utils::read_u32(&mut rdr)?;
        let mut result = Self::with_capacity(size as usize);
        for _ in 0..size {
            result.push(T::read(config, &mut rdr)?);
        }
        Ok(result)
    }
}

impl Readable for String {
    fn read<R>(config: &KyteaConfig, mut rdr: R) -> Result<Self>
    where
        R: BufRead,
    {
        let size = utils::read_u32(&mut rdr)?;
        let mut result = Self::new();
        for _ in 0..size {
            let cidx = usize::from(utils::read_u16(&mut rdr)?);
            result.push(config.char_map[cidx - 1]);
        }
        Ok(result)
    }
}

struct State {
    _failure: u32,
    gotos: Vec<(char, u32)>,
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
    fn dump_items(&self) -> Vec<(Vec<char>, &T)> {
        let mut result = vec![];
        let mut stack = vec![(0, vec![])];
        while let Some((idx, word)) = stack.pop() {
            let state = &self.states[idx];
            if state.is_branch {
                result.push((word.clone(), &self.entries[state.outputs[0] as usize]));
            }
            for &(c, next_idx) in state.gotos.iter().rev() {
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
    fn read<R>(config: &KyteaConfig, mut rdr: R) -> Result<Option<Self>>
    where
        R: BufRead,
    {
        let n_dicts = utils::read_u8(&mut rdr)?;
        let n_states = utils::read_u32(&mut rdr)? as usize;
        if n_states == 0 {
            return Ok(None);
        }
        let mut states = Vec::with_capacity(n_states);
        for _ in 0..n_states {
            let failure = utils::read_u32(&mut rdr)?;
            let n_gotos = utils::read_u32(&mut rdr)?;
            let mut gotos = vec![];
            for _ in 0..n_gotos {
                let k = char::read(config, &mut rdr)?;
                let v = utils::read_u32(&mut rdr)?;
                gotos.push((k, v));
            }
            gotos.sort_unstable();
            let n_outputs = utils::read_u32(&mut rdr)? as usize;
            let mut outputs = Vec::with_capacity(n_outputs);
            for _ in 0..n_outputs {
                outputs.push(utils::read_u32(&mut rdr)?);
            }
            let is_branch = utils::read_u8(&mut rdr)? != 0;
            states.push(State {
                _failure: failure,
                gotos,
                outputs,
                is_branch,
            });
        }
        let n_entries = utils::read_u32(&mut rdr)? as usize;
        let mut entries = Vec::with_capacity(n_entries);
        for _ in 0..n_entries {
            entries.push(T::read(config, &mut rdr)?);
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
    fn read<R>(config: &KyteaConfig, mut rdr: R) -> Result<Option<Self>>
    where
        R: BufRead,
    {
        let active = utils::read_u8(&mut rdr)?;
        if active == 0 {
            return Ok(None);
        }
        let char_dict = Dictionary::read(config, &mut rdr)?;
        let type_dict = Dictionary::read(config, &mut rdr)?;
        let self_dict = Dictionary::read(config, &mut rdr)?;
        let dict_vec = Vec::<T>::read(config, &mut rdr)?;
        let biases = Vec::<T>::read(config, &mut rdr)?;
        let tag_dict_vec = Vec::<T>::read(config, &mut rdr)?;
        let tag_unk_vec = Vec::<T>::read(config, &mut rdr)?;
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
    _multiplier: f64,
    feature_lookup: Option<FeatureLookup<i16>>,
}

impl Readable for Option<LinearModel> {
    fn read<R>(config: &KyteaConfig, mut rdr: R) -> Result<Self>
    where
        R: BufRead,
    {
        let n_classes = utils::read_u32(&mut rdr)?;
        if n_classes == 0 {
            return Ok(None);
        }
        let add_features = false;
        let solver_type = utils::read_u8(&mut rdr)?;
        let mut labels = vec![];
        for _ in 0..n_classes {
            labels.push(utils::read_i32(&mut rdr)?);
        }
        let bias = utils::read_u8(&mut rdr)? != 0;
        let multiplier = utils::read_f64(&mut rdr)?;
        let feature_lookup = FeatureLookup::read(config, &mut rdr)?;
        Ok(Some(LinearModel {
            _add_features: add_features,
            _solver_type: solver_type,
            _labels: labels,
            _bias: bias,
            _multiplier: multiplier,
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
    fn read<R>(config: &KyteaConfig, mut rdr: R) -> Result<Self>
    where
        R: BufRead,
    {
        let word = String::read(config, &mut rdr)?;
        let mut tags = Vec::with_capacity(config.n_tags as usize);
        let mut tags_in_dicts = Vec::with_capacity(config.n_tags as usize);
        for _ in 0..config.n_tags {
            let size = utils::read_u32(&mut rdr)? as usize;
            let mut t = Vec::with_capacity(size);
            let mut td = Vec::with_capacity(size);
            for _ in 0..size {
                t.push(String::read(config, &mut rdr)?);
                td.push(utils::read_u8(&mut rdr)?);
            }
            tags.push(t);
            tags_in_dicts.push(td);
        }
        let in_dict = utils::read_u8(&mut rdr)?;
        let mut tag_models = Vec::with_capacity(config.n_tags as usize);
        for _ in 0..config.n_tags {
            tag_models.push(Option::<LinearModel>::read(config, &mut rdr)?);
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
    fn read<R>(config: &KyteaConfig, mut rdr: R) -> Result<Self>
    where
        R: BufRead,
    {
        let word = String::read(config, &mut rdr)?;
        let mut tags = Vec::with_capacity(config.n_tags as usize);
        let mut probs = Vec::with_capacity(config.n_tags as usize);
        for _ in 0..config.n_tags {
            let size = utils::read_u32(&mut rdr)? as usize;
            let mut t = Vec::with_capacity(size);
            let mut p = Vec::with_capacity(size);
            for _ in 0..size {
                t.push(String::read(config, &mut rdr)?);
                p.push(utils::read_f64(&mut rdr)?);
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
#[cfg_attr(docsrs, doc(cfg(feature = "kytea")))]
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
    pub fn read<R>(mut rdr: R) -> Result<Self>
    where
        R: BufRead,
    {
        let config = KyteaConfig::read(&mut rdr)?;

        let wordseg_model = Option::<LinearModel>::read(&config, &mut rdr)?;

        let mut global_tags = Vec::with_capacity(config.n_tags as usize);
        let mut global_models = Vec::with_capacity(config.n_tags as usize);

        for _ in 0..config.n_tags {
            global_tags.push(Vec::<String>::read(&config, &mut rdr)?);
            global_models.push(Option::<LinearModel>::read(&config, &mut rdr)?);
        }

        let dict = Dictionary::<ModelTagEntry>::read(&config, &mut rdr)?;
        let subword_dict = Dictionary::<ProbTagEntry>::read(&config, &mut rdr)?;

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
    type Error = VaporettoError;

    fn try_from(model: KyteaModel) -> Result<Self> {
        let config = &model.config;
        let wordseg_model = model
            .wordseg_model
            .ok_or_else(|| VaporettoError::invalid_model("no word segmentation model."))?;
        let feature_lookup = wordseg_model
            .feature_lookup
            .ok_or_else(|| VaporettoError::invalid_model("no lookup data."))?;
        let bias = i32::from(feature_lookup.biases[0]);
        let char_dict = feature_lookup
            .char_dict
            .ok_or_else(|| VaporettoError::invalid_model("no character dictionary."))?;
        let type_dict = feature_lookup
            .type_dict
            .ok_or_else(|| VaporettoError::invalid_model("no type dictionary."))?;

        let mut char_ngrams = vec![];
        for (char_ngram, v) in char_dict.dump_items() {
            let weight_size = config.char_w as usize * 2 - char_ngram.len() + 1;
            char_ngrams.push(NgramData {
                ngram: char_ngram.into_iter().collect(),
                weights: v[..weight_size].iter().map(|&w| i32::from(w)).collect(),
            });
        }

        let mut type_ngrams = vec![];
        for (type_ngram, v) in type_dict.dump_items() {
            let weight_size = config.type_w as usize * 2 - type_ngram.len() + 1;
            let mut ngram = type_ngram
                .into_iter()
                .collect::<String>()
                .as_bytes()
                .to_vec();
            for t in &mut ngram {
                *t = match *t {
                    b'D' => CharacterType::Digit as u8,
                    b'R' => CharacterType::Roman as u8,
                    b'H' => CharacterType::Hiragana as u8,
                    b'T' => CharacterType::Katakana as u8,
                    b'K' => CharacterType::Kanji as u8,
                    b'O' => CharacterType::Other as u8,
                    t => t,
                };
            }
            type_ngrams.push(NgramData {
                ngram,
                weights: v[..weight_size].iter().map(|&w| i32::from(w)).collect(),
            });
        }

        let mut dict = vec![];
        if let Some(kytea_dict) = model.dict {
            for (w, data) in kytea_dict.dump_items() {
                let idx = std::cmp::min(w.len(), config.dict_n as usize) - 1;
                let mut dict_weight = DictWeight::default();
                for j in 0..kytea_dict.n_dicts as usize {
                    if data.in_dict >> j & 1 == 1 {
                        let offset = 3 * config.dict_n as usize * j + 3 * idx;
                        dict_weight.left += i32::from(feature_lookup.dict_vec[offset]);
                        dict_weight.inside += i32::from(feature_lookup.dict_vec[offset + 1]);
                        dict_weight.right += i32::from(feature_lookup.dict_vec[offset + 2]);
                    }
                }
                let mut weights = vec![dict_weight.inside; w.len() + 1];
                *weights.first_mut().unwrap() = dict_weight.left;
                *weights.last_mut().unwrap() = dict_weight.right;
                dict.push(WordWeightRecord {
                    word: w.into_iter().collect(),
                    weights,
                    comment: "".to_string(),
                });
            }
        }

        Ok(Self::new(
            NgramModel(char_ngrams),
            NgramModel(type_ngrams),
            DictModel::new(dict),
            bias,
            config.char_w,
            config.type_w,
            vec![],
        ))
    }
}
