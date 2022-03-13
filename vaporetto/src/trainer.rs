use std::borrow::Borrow;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::hash::Hash;
use std::str::FromStr;

use liblinear::LibLinearModel;

use crate::dict_model::{DictModel, DictWeight, WordWeightRecord};
use crate::errors::{Result, VaporettoError};
use crate::feature::{
    BoundaryExampleGenerator, BoundaryFeature, BytesNgramFeature, DictionaryWordFeature,
    DictionaryWordPosition, StringNgramFeature,
};
use crate::model::Model;
use crate::ngram_model::{NgramData, NgramModel};
use crate::sentence::{BoundaryType, Sentence};
use crate::tag_trainer::TagTrainer;

// Bit depth for weight quantization.
pub const QUANTIZE_BIT_DEPTH: u8 = 16;

pub struct Indexer<K> {
    ids: HashMap<K, usize>,
    keys: Vec<K>,
}

impl<K> Indexer<K>
where
    K: Eq + Hash,
{
    pub fn new() -> Self {
        Self {
            ids: HashMap::new(),
            keys: vec![],
        }
    }

    pub fn get_id<Q: ?Sized>(&mut self, key: &Q) -> usize
    where
        K: Borrow<Q>,
        Q: ToOwned<Owned = K> + Eq + Hash,
    {
        if let Some(&id) = self.ids.get(key) {
            id
        } else {
            let id = self.ids.len();
            self.keys.push(key.to_owned());
            self.ids.insert(key.to_owned(), id);
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
    char_window_size: u8,
    type_window_size: u8,
    dict_max_word_size: u8,
    feature_ids: Indexer<BoundaryFeature<'a>>,
    xs: Vec<Vec<(u32, f64)>>,
    ys: Vec<f64>,
    tag_trainer: TagTrainer<'a>,
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
        char_ngram_size: u8,
        char_window_size: u8,
        type_ngram_size: u8,
        type_window_size: u8,
        dictionary: D,
        dict_max_word_size: u8,
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
                Some(dictionary.as_ref()).filter(|d| !d.is_empty()),
                dict_max_word_size,
            )?,
            char_window_size,
            type_window_size,
            dict_max_word_size,
            feature_ids: Indexer::new(),
            xs: vec![],
            ys: vec![],
            tag_trainer: TagTrainer::new(char_ngram_size, char_window_size),
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
            self.ys.push(f64::from(example.label as u8));
        }
        self.tag_trainer.push_sentence(s)?;
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

    /// Gets the number of tag features.
    ///
    /// # Returns
    ///
    /// The number of tag features.
    pub fn n_tag_features(&self) -> usize {
        self.tag_trainer.n_features()
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

        let wb_idx = i32::try_from(
            model
                .labels()
                .iter()
                .position(|&cls| BoundaryType::WordBoundary as i32 == cls)
                .unwrap(),
        )?;

        let bias = model.label_bias(wb_idx);

        // Uses BTreeMap to increase compression ratio.
        let mut char_ngram_weights: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut type_ngram_weights: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut dict_weights = vec![DictWeight::default(); self.dict_max_word_size.into()];

        let mut weight_max = bias.abs();
        for fid in 0..model.num_features() {
            let weight = model.feature_coefficient(i32::try_from(fid)?, wb_idx).abs();
            if weight > weight_max {
                weight_max = weight;
            }
        }
        let quantize_multiplier = weight_max / f64::from((1 << (QUANTIZE_BIT_DEPTH - 1)) - 1);

        let bias = unsafe { (bias / quantize_multiplier).to_int_unchecked::<i32>() };

        for (fid, feature) in self.feature_ids.keys().iter().enumerate() {
            let raw_weight = model.feature_coefficient(i32::try_from(fid)? + 1, wb_idx);
            let weight = unsafe { (raw_weight / quantize_multiplier).to_int_unchecked::<i32>() };

            if weight == 0 {
                continue;
            }

            match feature {
                BoundaryFeature::CharacterNgram(StringNgramFeature {
                    rel_position,
                    ngram,
                }) => {
                    let len = ngram.chars().count();
                    let pos = isize::from(self.char_window_size) - len as isize - rel_position;
                    if let Some(weights) = char_ngram_weights.get_mut(*ngram) {
                        weights[pos as usize] = weight;
                    } else {
                        let mut weights = vec![0; usize::from(self.char_window_size) * 2 - len + 1];
                        weights[pos as usize] = weight;
                        char_ngram_weights.insert(ngram.to_string(), weights);
                    }
                }
                BoundaryFeature::CharacterTypeNgram(BytesNgramFeature {
                    rel_position,
                    ngram,
                }) => {
                    let len = ngram.len();
                    let pos = self.char_window_size as isize - len as isize - rel_position;
                    if let Some(weights) = type_ngram_weights.get_mut(*ngram) {
                        weights[pos as usize] = weight;
                    } else {
                        let mut weights = vec![0; usize::from(self.char_window_size) * 2 - len + 1];
                        weights[pos as usize] = weight;
                        type_ngram_weights.insert(ngram.to_vec(), weights);
                    }
                }
                BoundaryFeature::DictionaryWord(DictionaryWordFeature { position, length }) => {
                    match position {
                        DictionaryWordPosition::Right => dict_weights[length - 1].right = weight,
                        DictionaryWordPosition::Inside => dict_weights[length - 1].inside = weight,
                        DictionaryWordPosition::Left => dict_weights[length - 1].left = weight,
                    }
                }
            };
        }
        let tag_model = self.tag_trainer.train(epsilon, cost, solver)?;
        Ok(Model {
            char_ngram_model: NgramModel::new(
                char_ngram_weights
                    .into_iter()
                    .map(|(ngram, weights)| NgramData { ngram, weights })
                    .collect(),
            ),
            type_ngram_model: NgramModel::new(
                type_ngram_weights
                    .into_iter()
                    .map(|(ngram, weights)| NgramData { ngram, weights })
                    .collect(),
            ),
            dict_model: DictModel::new(
                self.dictionary
                    .into_iter()
                    .map(|word| {
                        let idx = word.chars().count().min(dict_weights.len()) - 1;
                        WordWeightRecord {
                            word,
                            weights: dict_weights[idx],
                            comment: "".to_string(),
                        }
                    })
                    .collect(),
            ),
            bias,
            tag_model,
            char_window_size: self.char_window_size,
            type_window_size: self.type_window_size,
        })
    }
}
