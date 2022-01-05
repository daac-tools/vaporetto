use std::collections::BTreeMap;
use std::collections::HashMap;
use std::hash::Hash;
use std::str::FromStr;

use crate::dict_model::{DictModel, DictModelLengthwise, DictWeight};
use crate::errors::{Result, VaporettoError};
use crate::feature::{
    BoundaryExampleGenerator, BoundaryFeature, BytesNgramFeature, DictionaryWordFeature,
    DictionaryWordPosition, StringNgramFeature,
};
use crate::model::Model;
use crate::ngram_model::{NgramData, NgramModel};
use crate::sentence::{BoundaryType, Sentence};
use liblinear::LibLinearModel;

const EPSILON: f64 = 1e-6;

// Bit depth for weight quantization.
const QUANTIZE_BIT_DEPTH: u8 = 16;

pub struct Indexer<K> {
    ids: HashMap<K, usize>,
    keys: Vec<K>,
}

impl<K> Indexer<K> {
    pub fn new() -> Self {
        Self {
            ids: HashMap::new(),
            keys: vec![],
        }
    }

    pub fn get_id(&mut self, key: &K) -> usize
    where
        K: Clone + Eq + Hash,
    {
        if let Some(&id) = self.ids.get(key) {
            id
        } else {
            let id = self.ids.len();
            self.keys.push(key.clone());
            self.ids.insert(key.clone(), id);
            id
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn keys(&self) -> &[K] {
        &self.keys
    }
}

/// Solver type.
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
///     train_sents.push(Sentence::from_tokenized(line.unwrap()).unwrap());
/// }
///
/// let dict: Vec<String> = vec![];
/// let mut trainer = Trainer::new(3, 3, 3, 3, &dict, 0).unwrap();
/// for (i, s) in train_sents.iter().enumerate() {
///     trainer.push_sentence(s);
/// }
///
/// let model = trainer.train(0.01, 1., SolverType::L1RegularizedL2LossSVC).unwrap();
/// let mut f = BufWriter::new(File::create("model.bin").unwrap());
/// model.write(&mut f).unwrap();
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub struct Trainer<'a> {
    dictionary: Vec<String>,
    example_generator: BoundaryExampleGenerator,
    char_window_size: usize,
    type_window_size: usize,
    dict_max_word_size: usize,
    feature_ids: Indexer<BoundaryFeature<'a>>,
    xs: Vec<Vec<(u32, f64)>>,
    ys: Vec<f64>,
}

impl<'a> Trainer<'a> {
    /// Creates a new dataset manager.
    ///
    /// # Arguments
    ///
    /// * `char_ngram_size` - The character n-gram length.
    /// * `char_window_size` - The character window size.
    /// * `type_ngram_size` - The character type n-gram length.
    /// * `type_window_size` - The character type window size.
    /// * `dictionary` - A word dictionary.
    /// * `dict_max_word_size` - Dictionary words greater than this value will be grouped together.
    ///
    /// # Returns
    ///
    /// A dataset manager.
    ///
    /// # Errors
    ///
    /// If invalid parameters are given, an error variant will be returned.
    pub fn new<D, P>(
        char_ngram_size: usize,
        char_window_size: usize,
        type_ngram_size: usize,
        type_window_size: usize,
        dictionary: D,
        dict_max_word_size: usize,
    ) -> Result<Self>
    where
        D: AsRef<[P]>,
        P: AsRef<[u8]> + AsRef<str>,
    {
        Ok(Self {
            dictionary: dictionary
                .as_ref()
                .iter()
                .map(|word| (word.as_ref() as &str).to_string())
                .collect(),
            example_generator: BoundaryExampleGenerator::new(
                char_ngram_size,
                type_ngram_size,
                char_window_size,
                type_window_size,
                dictionary.as_ref(),
                dict_max_word_size,
            )?,
            char_window_size,
            type_window_size,
            dict_max_word_size,
            feature_ids: Indexer::new(),
            xs: vec![],
            ys: vec![],
        })
    }

    /// Adds a sentence to the dataset.
    ///
    /// # Arguments
    ///
    /// * `s` - A sentence.
    ///
    /// # Errors
    ///
    /// [`VaporettoError::InvalidArgument`] will be returned if the maximum number of feature has
    /// been reached.
    pub fn push_sentence(&mut self, s: &'a Sentence) -> Result<()> {
        let examples = self.example_generator.generate(s);
        for example in examples {
            let mut feature_ids = BTreeMap::new();
            for f in &example.features {
                let fid = self.feature_ids.get_id(f);
                *feature_ids
                    .entry((fid + 1).try_into().unwrap())
                    .or_insert(0.0) += 1.0;
            }
            self.xs.push(feature_ids.into_iter().collect());
            self.ys.push(example.label as u8 as f64);
        }
        Ok(())
    }

    /// Gets the number of features.
    ///
    /// # Returns
    ///
    /// The number of features.
    pub fn n_features(&self) -> usize {
        self.feature_ids.len()
    }

    /// Trains word boundaries.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The tolerance of the termination criterion.
    /// * `cost` - The parameter C.
    /// * `solver` - Solver type.
    ///
    /// # Returns
    ///
    /// A trained model.
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

        let wb_idx = model
            .labels()
            .iter()
            .position(|&cls| BoundaryType::WordBoundary as i32 == cls)
            .unwrap() as i32;

        let bias = model.label_bias(wb_idx);
        let mut char_ngrams = vec![];
        let mut type_ngrams = vec![];
        let mut dict_weights = vec![DictWeight::default(); self.dict_max_word_size];
        let mut char_ngram_ids = Indexer::new();
        let mut type_ngram_ids = Indexer::new();

        let mut weight_max = bias.abs();
        for fid in 0..model.num_features() {
            let weight = model.feature_coefficient(fid as i32, wb_idx).abs();
            if weight > weight_max {
                weight_max = weight;
            }
        }
        let quantize_multiplier = weight_max / ((1 << (QUANTIZE_BIT_DEPTH - 1)) - 1) as f64;

        let bias = (bias / quantize_multiplier) as i32;

        for (fid, feature) in self.feature_ids.keys().iter().enumerate() {
            let weight = model.feature_coefficient(fid as i32 + 1, wb_idx);
            if weight > -EPSILON && weight < EPSILON {
                continue;
            }

            let weight = weight / quantize_multiplier;

            match feature {
                BoundaryFeature::CharacterNgram(StringNgramFeature {
                    rel_position,
                    ngram,
                }) => {
                    let id = char_ngram_ids.get_id(ngram);
                    let len = ngram.chars().count();
                    if id == char_ngrams.len() {
                        char_ngrams.push(NgramData {
                            ngram: ngram.to_string(),
                            weights: vec![0; self.char_window_size * 2 - len + 1],
                        });
                    }
                    let pos = self.char_window_size as isize - len as isize - rel_position;
                    char_ngrams[id].weights[pos as usize] = weight as i32;
                }
                BoundaryFeature::CharacterTypeNgram(BytesNgramFeature {
                    rel_position,
                    ngram,
                }) => {
                    let id = type_ngram_ids.get_id(ngram) as usize;
                    let len = ngram.len();
                    if id == type_ngrams.len() {
                        type_ngrams.push(NgramData {
                            ngram: ngram.to_vec(),
                            weights: vec![0; self.type_window_size * 2 - len + 1],
                        });
                    }
                    let pos = self.type_window_size as isize - len as isize - rel_position;
                    type_ngrams[id].weights[pos as usize] = weight as i32;
                }
                BoundaryFeature::DictionaryWord(DictionaryWordFeature { position, length }) => {
                    match position {
                        DictionaryWordPosition::Right => {
                            dict_weights[length - 1].right = weight as i32
                        }
                        DictionaryWordPosition::Inside => {
                            dict_weights[length - 1].inside = weight as i32
                        }
                        DictionaryWordPosition::Left => {
                            dict_weights[length - 1].left = weight as i32
                        }
                    }
                }
            };
        }
        Ok(Model {
            char_ngram_model: NgramModel::new(char_ngrams),
            type_ngram_model: NgramModel::new(type_ngrams),
            dict_model: DictModel::Lengthwise(DictModelLengthwise {
                words: self.dictionary,
                weights: dict_weights,
            }),
            bias,
            char_window_size: self.char_window_size,
            type_window_size: self.type_window_size,
        })
    }
}
