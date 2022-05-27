use core::str::FromStr;

use alloc::collections::BTreeMap;

use hashbrown::HashMap;

use daachorse::DoubleArrayAhoCorasick;
use liblinear::LibLinearModel;

use crate::dict_model::{DictModel, WordWeightRecord};
use crate::errors::{Result, VaporettoError};
use crate::model::Model;
use crate::ngram_model::{NgramData, NgramModel};
use crate::sentence::{CharacterBoundary, Sentence};
use crate::tag_trainer::TagTrainer;

// Bit depth for weight quantization.
pub const QUANTIZE_BIT_DEPTH: u8 = 16;

/// Solver type.
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
#[derive(Clone, Copy, Debug)]
pub enum SolverType {
    /// L2-regularized logistic regression (primal).
    L2RegularizedLogistic = 0,

    /// L2-regularized L2-loss support vector classification (dual).
    L2RegularizedL2LossSVCDual = 1,

    /// L2-regularized L2-loss support vector classification (primal).
    L2RegularizedL2LossSVC = 2,

    /// L2-regularized L1-loss support vector classification (dual)
    L2RegularizedL1LossSVCDual = 3,

    /// support vector classification by Crammer and Singer
    CrammerSingerSVC = 4,

    /// L1-regularized L2-loss support vector classification
    L1RegularizedL2LossSVC = 5,

    /// L1-regularized logistic regression
    L1RegularizedLogistic = 6,

    /// L2-regularized logistic regression (dual).
    L2RegularizedLogisticDual = 7,
}

impl FromStr for SolverType {
    type Err = &'static str;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "0" => Ok(Self::L2RegularizedLogistic),
            "1" => Ok(Self::L2RegularizedL2LossSVCDual),
            "2" => Ok(Self::L2RegularizedL2LossSVC),
            "3" => Ok(Self::L2RegularizedL1LossSVCDual),
            "4" => Ok(Self::CrammerSingerSVC),
            "5" => Ok(Self::L1RegularizedL2LossSVC),
            "6" => Ok(Self::L1RegularizedLogistic),
            "7" => Ok(Self::L2RegularizedLogisticDual),
            _ => Err("Unsupported solver type."),
        }
    }
}

impl From<SolverType> for liblinear::SolverType {
    fn from(solver: SolverType) -> Self {
        match solver {
            SolverType::L2RegularizedLogistic => Self::L2R_LR,
            SolverType::L2RegularizedL2LossSVCDual => Self::L2R_L2LOSS_SVC_DUAL,
            SolverType::L2RegularizedL2LossSVC => Self::L2R_L2LOSS_SVC,
            SolverType::L2RegularizedL1LossSVCDual => Self::L2R_L1LOSS_SVC_DUAL,
            SolverType::CrammerSingerSVC => Self::MCSVM_CS,
            SolverType::L1RegularizedL2LossSVC => Self::L1R_L2LOSS_SVC,
            SolverType::L1RegularizedLogistic => Self::L1R_LR,
            SolverType::L2RegularizedLogisticDual => Self::L2R_LR_DUAL,
        }
    }
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub struct NgramFeature<T> {
    pub ngram: T,
    pub rel_position: isize,
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub enum DictionaryWordPosition {
    Left,
    Inside,
    Right,
}

#[derive(Debug, Eq, Hash, PartialEq)]
pub struct DictionaryWordFeature {
    pub(crate) length: usize,
    pub(crate) position: DictionaryWordPosition,
}

#[derive(Debug, Eq, Hash, PartialEq)]
enum BoundaryFeature<'a> {
    CharacterNgram(NgramFeature<&'a str>),
    CharacterTypeNgram(NgramFeature<&'a [u8]>),
    DictionaryWord(DictionaryWordFeature),
}

impl<'a> BoundaryFeature<'a> {
    pub const fn char_ngram(ngram: &'a str, rel_position: isize) -> Self {
        Self::CharacterNgram(NgramFeature {
            ngram,
            rel_position,
        })
    }

    pub const fn type_ngram(ngram: &'a [u8], rel_position: isize) -> Self {
        Self::CharacterTypeNgram(NgramFeature {
            ngram,
            rel_position,
        })
    }

    pub const fn dict_word_left(length: usize) -> Self {
        Self::DictionaryWord(DictionaryWordFeature {
            length,
            position: DictionaryWordPosition::Left,
        })
    }

    pub const fn dict_word_inside(length: usize) -> Self {
        Self::DictionaryWord(DictionaryWordFeature {
            length,
            position: DictionaryWordPosition::Inside,
        })
    }

    pub const fn dict_word_right(length: usize) -> Self {
        Self::DictionaryWord(DictionaryWordFeature {
            length,
            position: DictionaryWordPosition::Right,
        })
    }
}

/// Trainer.
///
/// # Examples
///
/// ```no_run
/// use std::fs::File;
/// use std::io::{prelude::*, BufReader, BufWriter};
///
/// use vaporetto::{Sentence, SolverType, Trainer};
///
/// let mut train_sents = vec![];
/// let f = BufReader::new(File::open("dataset-train.txt").unwrap());
/// for (i, line) in f.lines().enumerate() {
///     train_sents.push(Sentence::from_tokenized(&line.unwrap()).unwrap());
/// }
///
/// let dict: Vec<String> = vec![];
/// let mut trainer = Trainer::new(3, 3, 3, 3, dict, 0).unwrap();
/// for (i, s) in train_sents.iter().enumerate() {
///     trainer.add_example(&s);
/// }
///
/// let model = trainer.train(0.01, 1., SolverType::L1RegularizedL2LossSVC).unwrap();
/// let mut f = BufWriter::new(File::create("model.bin").unwrap());
/// model.write(&mut f).unwrap();
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub struct Trainer<'a> {
    char_window_size: u8,
    char_ngram_size: u8,
    type_window_size: u8,
    type_ngram_size: u8,
    feature_ids: HashMap<BoundaryFeature<'a>, u32>,
    dict_words: Vec<String>,
    dict_pma: Option<DoubleArrayAhoCorasick>,
    dict_word_max_len: u8,
    xs: Vec<Vec<(u32, f64)>>,
    ys: Vec<f64>,

    tag_trainer: TagTrainer<'a>,
}

impl<'a> Trainer<'a> {
    /// Creates a new trainer.
    ///
    /// # Arguments
    ///
    /// * `char_window_size` - The character window size.
    /// * `char_ngram_size` - The character n-gram length.
    /// * `type_window_size` - The character type window size.
    /// * `type_ngram_size` - The character type n-gram length.
    /// * `dict_words` - A word dictionary.
    /// * `dict_word_max_len` - Dictionary words greater than this value will be grouped together.
    ///
    /// # Errors
    ///
    /// If invalid parameters are given, an error variant will be returned.
    pub fn new(
        char_window_size: u8,
        char_ngram_size: u8,
        type_window_size: u8,
        type_ngram_size: u8,
        dict_words: Vec<String>,
        dict_word_max_len: u8,
    ) -> Result<Self> {
        let dict_pma = if dict_words.is_empty() {
            None
        } else {
            Some(
                DoubleArrayAhoCorasick::new(&dict_words)
                    .map_err(|e| VaporettoError::invalid_argument("dict_words", e.to_string()))?,
            )
        };
        Ok(Self {
            char_window_size,
            char_ngram_size,
            type_window_size,
            type_ngram_size,
            feature_ids: HashMap::new(),
            dict_words,
            dict_pma,
            dict_word_max_len,
            xs: vec![],
            ys: vec![],
            tag_trainer: TagTrainer::new(
                char_window_size,
                char_ngram_size,
                type_window_size,
                type_ngram_size,
            ),
        })
    }

    fn gen_features<'b>(
        &self,
        sentence: &'a Sentence<'a, 'b>,
        examples: &mut Vec<(Vec<BoundaryFeature<'a>>, CharacterBoundary)>,
    ) {
        for (i, &b) in sentence.boundaries().iter().enumerate() {
            let mut features = vec![];
            // adds character n-gram features
            for n in 0..self.char_ngram_size {
                for j in (i + 1).saturating_sub(self.char_window_size.into())
                    ..(i + 1 + usize::from(self.char_window_size))
                        .min(sentence.len())
                        .saturating_sub(n.into())
                {
                    features.push(BoundaryFeature::char_ngram(
                        sentence.text_substring(j, j + usize::from(n) + 1),
                        isize::try_from(j).unwrap() - isize::try_from(i).unwrap() - 1,
                    ));
                }
            }
            // adds type n-gram features
            for n in 0..self.type_ngram_size {
                for j in (i + 1).saturating_sub(self.type_window_size.into())
                    ..(i + 1 + usize::from(self.type_window_size))
                        .min(sentence.len())
                        .saturating_sub(n.into())
                {
                    features.push(BoundaryFeature::type_ngram(
                        &sentence.char_types()[j..j + usize::from(n) + 1],
                        isize::try_from(j).unwrap() - isize::try_from(i).unwrap() - 1,
                    ));
                }
            }
            examples.push((features, b));
        }
        // adds dictionary features
        if let Some(pma) = self.dict_pma.as_ref() {
            for m in pma.find_overlapping_iter(sentence.as_raw_text()) {
                let start = unsafe { sentence.str_to_char_pos(m.start()) };
                let end = unsafe { sentence.str_to_char_pos(m.end()) };
                let length = (end - start).min(usize::from(self.dict_word_max_len));
                if start != 0 {
                    examples[start - 1]
                        .0
                        .push(BoundaryFeature::dict_word_left(length));
                }
                for example in &mut examples[start..end - 1] {
                    example.0.push(BoundaryFeature::dict_word_inside(length));
                }
                if end != sentence.len() {
                    examples[end - 1]
                        .0
                        .push(BoundaryFeature::dict_word_right(length));
                }
            }
        }
    }

    /// Adds a sentence to the trainer.
    pub fn add_example<'b>(&mut self, sentence: &'a Sentence<'a, 'b>) {
        let mut examples = vec![];
        self.gen_features(sentence, &mut examples);
        for (features, b) in examples {
            let mut feature_vector = HashMap::new();
            for feature in features {
                let new_id = self.feature_ids.len() + 1;
                let feature_id = *self
                    .feature_ids
                    .entry(feature)
                    .or_insert(new_id.try_into().unwrap());
                *feature_vector.entry(feature_id).or_insert(0f64) += 1f64;
            }
            self.xs.push(feature_vector.into_iter().collect());
            self.ys.push(f64::from(b as u8));
        }

        self.tag_trainer.add_example(sentence);
    }

    /// Trains word boundaries and tags.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The tolerance of the termination criterion.
    /// * `cost` - The parameter C.
    /// * `solver` - Solver type.
    ///
    /// # Errors
    ///
    /// If the solver returns an error, that will be propagated.
    pub fn train(self, epsilon: f64, cost: f64, solver: SolverType) -> Result<Model> {
        let mut builder = liblinear::Builder::new();
        let training_input = liblinear::util::TrainingInput::from_sparse_features(self.ys, self.xs)
            .map_err(|e| VaporettoError::invalid_model(format!("liblinear error: {:?}", e)))?;
        builder.problem().input_data(training_input).bias(1.0);
        builder
            .parameters()
            .solver_type(solver.into())
            .stopping_criterion(epsilon)
            .constraints_violation_cost(cost);
        let model = builder
            .build_model()
            .map_err(|e| VaporettoError::invalid_model(e.to_string()))?;

        let wb_idx = i32::try_from(
            model
                .labels()
                .iter()
                .position(|&cls| CharacterBoundary::WordBoundary as i32 == cls)
                .unwrap(),
        )?;

        let bias = model.label_bias(wb_idx);

        let mut weight_max = bias.abs();
        for fid in 0..model.num_features() {
            let weight = model
                .feature_coefficient(i32::try_from(fid + 1)?, wb_idx)
                .abs();
            weight_max = weight_max.max(weight);
        }
        let quantize_multiplier = weight_max / f64::from((1 << (QUANTIZE_BIT_DEPTH - 1)) - 1);
        if quantize_multiplier == 0. {
            return Err(VaporettoError::invalid_model("all weights are zero"));
        }

        // Uses BTreeMap to increase compression ratio.
        let mut char_ngram_weights: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut type_ngram_weights: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut dict_weights = vec![];
        for i in 0..usize::from(self.dict_word_max_len) {
            dict_weights.push(vec![0; i + 2]);
        }

        let bias = unsafe { (bias / quantize_multiplier).to_int_unchecked::<i32>() };

        for (feature, fid) in self.feature_ids {
            let raw_weight = model.feature_coefficient(i32::try_from(fid)?, wb_idx);
            let weight = unsafe { (raw_weight / quantize_multiplier).to_int_unchecked::<i32>() };

            if weight == 0 {
                continue;
            }

            match feature {
                BoundaryFeature::CharacterNgram(NgramFeature {
                    ngram,
                    rel_position,
                }) => {
                    let len = ngram.chars().count();
                    let pos = usize::try_from(
                        isize::from(self.char_window_size) - isize::try_from(len)? - rel_position,
                    )
                    .unwrap();
                    if let Some(weights) = char_ngram_weights.get_mut(ngram) {
                        weights[pos] = weight;
                    } else {
                        let mut weights = vec![0; usize::from(self.char_window_size) * 2 - len + 1];
                        weights[pos] = weight;
                        char_ngram_weights.insert(ngram.to_string(), weights);
                    }
                }
                BoundaryFeature::CharacterTypeNgram(NgramFeature {
                    ngram,
                    rel_position,
                }) => {
                    let len = ngram.len();
                    let pos = usize::try_from(
                        isize::from(self.char_window_size) - isize::try_from(len)? - rel_position,
                    )
                    .unwrap();
                    if let Some(weights) = type_ngram_weights.get_mut(ngram) {
                        weights[pos] = weight;
                    } else {
                        let mut weights = vec![0; usize::from(self.char_window_size) * 2 - len + 1];
                        weights[pos] = weight;
                        type_ngram_weights.insert(ngram.to_vec(), weights);
                    }
                }
                BoundaryFeature::DictionaryWord(DictionaryWordFeature { length, position }) => {
                    let weights = &mut dict_weights[length - 1];
                    match position {
                        DictionaryWordPosition::Left => *weights.first_mut().unwrap() = weight,
                        DictionaryWordPosition::Inside => weights[1..length - 1].fill(weight),
                        DictionaryWordPosition::Right => *weights.last_mut().unwrap() = weight,
                    }
                }
            }
        }

        let tag_models = self.tag_trainer.train(epsilon, cost, solver)?;

        Ok(Model::new(
            NgramModel(
                char_ngram_weights
                    .into_iter()
                    .map(|(ngram, weights)| NgramData { ngram, weights })
                    .collect(),
            ),
            NgramModel(
                type_ngram_weights
                    .into_iter()
                    .map(|(ngram, weights)| NgramData { ngram, weights })
                    .collect(),
            ),
            DictModel::new(
                self.dict_words
                    .into_iter()
                    .map(|word| {
                        let word_len = word.chars().count();
                        let idx = word_len.min(dict_weights.len()) - 1;
                        WordWeightRecord {
                            word,
                            weights: dict_weights[idx].clone(),
                            comment: "".to_string(),
                        }
                    })
                    .collect(),
            ),
            bias,
            self.char_window_size,
            self.type_window_size,
            tag_models,
        ))
    }

    /// Returns the number of boundary features.
    pub fn n_features(&self) -> usize {
        self.feature_ids.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sentence::CharacterBoundary::*;
    use crate::sentence::CharacterType::*;

    #[test]
    fn check_features_3322() {
        let s = Sentence::from_tokenized("これ は テスト です").unwrap();
        let trainer = Trainer::new(3, 3, 2, 2, vec![], 4).unwrap();
        let mut examples = vec![];
        trainer.gen_features(&s, &mut examples);

        // こ-れ
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("こ", -1),
                BoundaryFeature::char_ngram("れ", 0),
                BoundaryFeature::char_ngram("は", 1),
                BoundaryFeature::char_ngram("テ", 2),
                BoundaryFeature::char_ngram("これ", -1),
                BoundaryFeature::char_ngram("れは", 0),
                BoundaryFeature::char_ngram("はテ", 1),
                BoundaryFeature::char_ngram("これは", -1),
                BoundaryFeature::char_ngram("れはテ", 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], 0),
            ],
            examples[0].0,
        );
        assert_eq!(NotWordBoundary, examples[0].1);

        // れ|は
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("こ", -2),
                BoundaryFeature::char_ngram("れ", -1),
                BoundaryFeature::char_ngram("は", 0),
                BoundaryFeature::char_ngram("テ", 1),
                BoundaryFeature::char_ngram("ス", 2),
                BoundaryFeature::char_ngram("これ", -2),
                BoundaryFeature::char_ngram("れは", -1),
                BoundaryFeature::char_ngram("はテ", 0),
                BoundaryFeature::char_ngram("テス", 1),
                BoundaryFeature::char_ngram("これは", -2),
                BoundaryFeature::char_ngram("れはテ", -1),
                BoundaryFeature::char_ngram("はテス", 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 0),
                BoundaryFeature::type_ngram(&[Katakana as u8], 1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Katakana as u8], 0),
            ],
            examples[1].0,
        );
        assert_eq!(WordBoundary, examples[1].1);

        // は|テ
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("こ", -3),
                BoundaryFeature::char_ngram("れ", -2),
                BoundaryFeature::char_ngram("は", -1),
                BoundaryFeature::char_ngram("テ", 0),
                BoundaryFeature::char_ngram("ス", 1),
                BoundaryFeature::char_ngram("ト", 2),
                BoundaryFeature::char_ngram("これ", -3),
                BoundaryFeature::char_ngram("れは", -2),
                BoundaryFeature::char_ngram("はテ", -1),
                BoundaryFeature::char_ngram("テス", 0),
                BoundaryFeature::char_ngram("スト", 1),
                BoundaryFeature::char_ngram("これは", -3),
                BoundaryFeature::char_ngram("れはテ", -2),
                BoundaryFeature::char_ngram("はテス", -1),
                BoundaryFeature::char_ngram("テスト", 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8], 0),
                BoundaryFeature::type_ngram(&[Katakana as u8], 1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], 0),
            ],
            examples[2].0,
        );
        assert_eq!(WordBoundary, examples[2].1);

        // テ-ス
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("れ", -3),
                BoundaryFeature::char_ngram("は", -2),
                BoundaryFeature::char_ngram("テ", -1),
                BoundaryFeature::char_ngram("ス", 0),
                BoundaryFeature::char_ngram("ト", 1),
                BoundaryFeature::char_ngram("で", 2),
                BoundaryFeature::char_ngram("れは", -3),
                BoundaryFeature::char_ngram("はテ", -2),
                BoundaryFeature::char_ngram("テス", -1),
                BoundaryFeature::char_ngram("スト", 0),
                BoundaryFeature::char_ngram("トで", 1),
                BoundaryFeature::char_ngram("れはテ", -3),
                BoundaryFeature::char_ngram("はテス", -2),
                BoundaryFeature::char_ngram("テスト", -1),
                BoundaryFeature::char_ngram("ストで", 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8], 0),
                BoundaryFeature::type_ngram(&[Katakana as u8], 1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], 0),
            ],
            examples[3].0,
        );
        assert_eq!(NotWordBoundary, examples[3].1);

        // ス-ト
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("は", -3),
                BoundaryFeature::char_ngram("テ", -2),
                BoundaryFeature::char_ngram("ス", -1),
                BoundaryFeature::char_ngram("ト", 0),
                BoundaryFeature::char_ngram("で", 1),
                BoundaryFeature::char_ngram("す", 2),
                BoundaryFeature::char_ngram("はテ", -3),
                BoundaryFeature::char_ngram("テス", -2),
                BoundaryFeature::char_ngram("スト", -1),
                BoundaryFeature::char_ngram("トで", 0),
                BoundaryFeature::char_ngram("です", 1),
                BoundaryFeature::char_ngram("はテス", -3),
                BoundaryFeature::char_ngram("テスト", -2),
                BoundaryFeature::char_ngram("ストで", -1),
                BoundaryFeature::char_ngram("トです", 0),
                BoundaryFeature::type_ngram(&[Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8], 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 1),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8, Hiragana as u8], 0),
            ],
            examples[4].0,
        );
        assert_eq!(NotWordBoundary, examples[4].1);

        // ト|で
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("テ", -3),
                BoundaryFeature::char_ngram("ス", -2),
                BoundaryFeature::char_ngram("ト", -1),
                BoundaryFeature::char_ngram("で", 0),
                BoundaryFeature::char_ngram("す", 1),
                BoundaryFeature::char_ngram("テス", -3),
                BoundaryFeature::char_ngram("スト", -2),
                BoundaryFeature::char_ngram("トで", -1),
                BoundaryFeature::char_ngram("です", 0),
                BoundaryFeature::char_ngram("テスト", -3),
                BoundaryFeature::char_ngram("ストで", -2),
                BoundaryFeature::char_ngram("トです", -1),
                BoundaryFeature::type_ngram(&[Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 1),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8, Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], 0),
            ],
            examples[5].0,
        );
        assert_eq!(WordBoundary, examples[5].1);

        // で-す
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("ス", -3),
                BoundaryFeature::char_ngram("ト", -2),
                BoundaryFeature::char_ngram("で", -1),
                BoundaryFeature::char_ngram("す", 0),
                BoundaryFeature::char_ngram("スト", -3),
                BoundaryFeature::char_ngram("トで", -2),
                BoundaryFeature::char_ngram("です", -1),
                BoundaryFeature::char_ngram("ストで", -3),
                BoundaryFeature::char_ngram("トです", -2),
                BoundaryFeature::type_ngram(&[Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 0),
                BoundaryFeature::type_ngram(&[Katakana as u8, Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], -1),
            ],
            examples[6].0,
        );
        assert_eq!(NotWordBoundary, examples[6].1);
    }

    #[test]
    fn check_features_2222_dict() {
        let s = Sentence::from_tokenized("これ は テスト です").unwrap();
        let trainer = Trainer::new(
            2,
            2,
            2,
            2,
            vec!["これ".into(), "これは".into(), "テスト".into()],
            4,
        )
        .unwrap();
        let mut examples = vec![];
        trainer.gen_features(&s, &mut examples);

        // こ-れ
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("こ", -1),
                BoundaryFeature::char_ngram("れ", 0),
                BoundaryFeature::char_ngram("は", 1),
                BoundaryFeature::char_ngram("これ", -1),
                BoundaryFeature::char_ngram("れは", 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], 0),
                BoundaryFeature::dict_word_inside(2),
                BoundaryFeature::dict_word_inside(3),
            ],
            examples[0].0,
        );
        assert_eq!(NotWordBoundary, examples[0].1);

        // れ|は
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("こ", -2),
                BoundaryFeature::char_ngram("れ", -1),
                BoundaryFeature::char_ngram("は", 0),
                BoundaryFeature::char_ngram("テ", 1),
                BoundaryFeature::char_ngram("これ", -2),
                BoundaryFeature::char_ngram("れは", -1),
                BoundaryFeature::char_ngram("はテ", 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 0),
                BoundaryFeature::type_ngram(&[Katakana as u8], 1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Katakana as u8], 0),
                BoundaryFeature::dict_word_right(2),
                BoundaryFeature::dict_word_inside(3),
            ],
            examples[1].0,
        );
        assert_eq!(WordBoundary, examples[1].1);

        // は|テ
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("れ", -2),
                BoundaryFeature::char_ngram("は", -1),
                BoundaryFeature::char_ngram("テ", 0),
                BoundaryFeature::char_ngram("ス", 1),
                BoundaryFeature::char_ngram("れは", -2),
                BoundaryFeature::char_ngram("はテ", -1),
                BoundaryFeature::char_ngram("テス", 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8], 0),
                BoundaryFeature::type_ngram(&[Katakana as u8], 1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], 0),
                BoundaryFeature::dict_word_right(3),
                BoundaryFeature::dict_word_left(3),
            ],
            examples[2].0,
        );
        assert_eq!(WordBoundary, examples[2].1);

        // テ-ス
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("は", -2),
                BoundaryFeature::char_ngram("テ", -1),
                BoundaryFeature::char_ngram("ス", 0),
                BoundaryFeature::char_ngram("ト", 1),
                BoundaryFeature::char_ngram("はテ", -2),
                BoundaryFeature::char_ngram("テス", -1),
                BoundaryFeature::char_ngram("スト", 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8], 0),
                BoundaryFeature::type_ngram(&[Katakana as u8], 1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], 0),
                BoundaryFeature::dict_word_inside(3),
            ],
            examples[3].0,
        );
        assert_eq!(NotWordBoundary, examples[3].1);

        // ス-ト
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("テ", -2),
                BoundaryFeature::char_ngram("ス", -1),
                BoundaryFeature::char_ngram("ト", 0),
                BoundaryFeature::char_ngram("で", 1),
                BoundaryFeature::char_ngram("テス", -2),
                BoundaryFeature::char_ngram("スト", -1),
                BoundaryFeature::char_ngram("トで", 0),
                BoundaryFeature::type_ngram(&[Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8], 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 1),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Katakana as u8, Hiragana as u8], 0),
                BoundaryFeature::dict_word_inside(3),
            ],
            examples[4].0,
        );
        assert_eq!(NotWordBoundary, examples[4].1);

        // ト|で
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("ス", -2),
                BoundaryFeature::char_ngram("ト", -1),
                BoundaryFeature::char_ngram("で", 0),
                BoundaryFeature::char_ngram("す", 1),
                BoundaryFeature::char_ngram("スト", -2),
                BoundaryFeature::char_ngram("トで", -1),
                BoundaryFeature::char_ngram("です", 0),
                BoundaryFeature::type_ngram(&[Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 0),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 1),
                BoundaryFeature::type_ngram(&[Katakana as u8, Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Katakana as u8, Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], 0),
                BoundaryFeature::dict_word_right(3),
            ],
            examples[5].0,
        );
        assert_eq!(WordBoundary, examples[5].1);

        // で-す
        assert_eq!(
            vec![
                BoundaryFeature::char_ngram("ト", -2),
                BoundaryFeature::char_ngram("で", -1),
                BoundaryFeature::char_ngram("す", 0),
                BoundaryFeature::char_ngram("トで", -2),
                BoundaryFeature::char_ngram("です", -1),
                BoundaryFeature::type_ngram(&[Katakana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8], -1),
                BoundaryFeature::type_ngram(&[Hiragana as u8], 0),
                BoundaryFeature::type_ngram(&[Katakana as u8, Hiragana as u8], -2),
                BoundaryFeature::type_ngram(&[Hiragana as u8, Hiragana as u8], -1),
            ],
            examples[6].0,
        );
        assert_eq!(NotWordBoundary, examples[6].1);
    }
}
