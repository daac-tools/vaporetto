use alloc::boxed::Box;
use alloc::collections::BTreeMap;
use alloc::string::ToString;

use hashbrown::HashMap;

use daachorse::DoubleArrayAhoCorasick;

use crate::dict_model::{DictModel, WordWeightRecord};
use crate::errors::{Result, VaporettoError};
use crate::linear_solver::{LinearModel, Problem};
use crate::model::Model;
use crate::ngram_model::{NgramData, NgramModel};
use crate::sentence::{CharacterBoundary, Sentence};
use crate::tag_trainer::TagTrainer;

// Bit depth for weight quantization.
pub const QUANTIZE_BIT_DEPTH: u8 = 16;

/// A solver of LIBLINEAR, used with [`Solver::Liblinear`]. The numbers are LIBLINEAR's.
#[cfg(feature = "liblinear")]
#[cfg_attr(docsrs, doc(cfg(feature = "liblinear")))]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LiblinearSolver {
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

#[cfg(feature = "liblinear")]
impl From<LiblinearSolver> for liblinear::SolverType {
    fn from(solver: LiblinearSolver) -> Self {
        match solver {
            LiblinearSolver::L2RegularizedLogistic => Self::L2R_LR,
            LiblinearSolver::L2RegularizedL2LossSVCDual => Self::L2R_L2LOSS_SVC_DUAL,
            LiblinearSolver::L2RegularizedL2LossSVC => Self::L2R_L2LOSS_SVC,
            LiblinearSolver::L2RegularizedL1LossSVCDual => Self::L2R_L1LOSS_SVC_DUAL,
            LiblinearSolver::CrammerSingerSVC => Self::MCSVM_CS,
            LiblinearSolver::L1RegularizedL2LossSVC => Self::L1R_L2LOSS_SVC,
            LiblinearSolver::L1RegularizedLogistic => Self::L1R_LR,
            LiblinearSolver::L2RegularizedLogisticDual => Self::L2R_LR_DUAL,
        }
    }
}

/// A solver with its costs, used by [`Trainer::train`].
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Solver {
    /// L1-regularized L2-loss support vector classification:
    /// `min_w ||w||_1 + C Σ max(0, 1 - y w^T x)^2`.
    L1RegularizedL2LossSVC {
        /// The cost C.
        cost: f64,
    },

    /// L2-regularized L2-loss support vector classification:
    /// `min_w ||w||_2^2 / 2 + C Σ max(0, 1 - y w^T x)^2`, solved in the dual.
    L2RegularizedL2LossSVCDual {
        /// The cost C.
        cost: f64,
    },

    /// L2-regularized L1-loss support vector classification:
    /// `min_w ||w||_2^2 / 2 + C Σ max(0, 1 - y w^T x)`, solved in the dual.
    L2RegularizedL1LossSVCDual {
        /// The cost C.
        cost: f64,
    },

    /// L2-regularized logistic regression: `min_w ||w||_2^2 / 2 + C Σ log(1 + exp(-y w^T x))`,
    /// solved in the dual.
    L2RegularizedLogisticDual {
        /// The cost C.
        cost: f64,
    },

    /// Elastic-net regularized L2-loss support vector classification:
    /// `min_w ||w||_1 / C1 + ||w||_2^2 / (2 C2) + Σ max(0, 1 - y w^T x)^2`.
    ///
    /// With `C2 = inf`, this is [`Solver::L1RegularizedL2LossSVC`] with the cost C1.
    ElasticNet {
        /// The cost C1 of the L1 term (finite).
        l1_cost: f64,
        /// The cost C2 of the L2 term (may be infinite).
        l2_cost: f64,
    },

    /// A solver of LIBLINEAR.
    #[cfg(feature = "liblinear")]
    #[cfg_attr(docsrs, doc(cfg(feature = "liblinear")))]
    Liblinear {
        /// The solver.
        solver: LiblinearSolver,
        /// The cost C.
        cost: f64,
    },
}

impl Solver {
    fn check(&self) -> Result<()> {
        let valid = match *self {
            Self::L1RegularizedL2LossSVC { cost }
            | Self::L2RegularizedL2LossSVCDual { cost }
            | Self::L2RegularizedL1LossSVCDual { cost }
            | Self::L2RegularizedLogisticDual { cost } => cost.is_finite() && cost > 0.,
            Self::ElasticNet { l1_cost, l2_cost } => {
                l1_cost.is_finite() && l1_cost > 0. && l2_cost > 0.
            }
            #[cfg(feature = "liblinear")]
            Self::Liblinear { cost, .. } => cost.is_finite() && cost > 0.,
        };
        if valid {
            Ok(())
        } else {
            Err(VaporettoError::invalid_argument(
                "solver",
                "costs must be positive (only the cost of the L2 term may be infinite)",
            ))
        }
    }
}

/// The accessors of a trained linear model. The feature IDs start from 0.
pub trait Linear {
    fn labels(&self) -> &[i32];
    fn num_features(&self) -> usize;
    fn feature_coefficient(&self, feature: usize, label_index: usize) -> f64;
    fn label_bias(&self, label_index: usize) -> f64;
}

impl Linear for LinearModel {
    fn labels(&self) -> &[i32] {
        Self::labels(self)
    }
    fn num_features(&self) -> usize {
        Self::num_features(self)
    }
    fn feature_coefficient(&self, feature: usize, label_index: usize) -> f64 {
        Self::feature_coefficient(self, feature, label_index)
    }
    fn label_bias(&self, label_index: usize) -> f64 {
        Self::label_bias(self, label_index)
    }
}

/// A model trained by LIBLINEAR, whose feature IDs start from 1.
#[cfg(feature = "liblinear")]
struct LiblinearModel<M>(M);

#[cfg(feature = "liblinear")]
impl<M: liblinear::LibLinearModel> Linear for LiblinearModel<M> {
    fn labels(&self) -> &[i32] {
        self.0.labels()
    }
    fn num_features(&self) -> usize {
        self.0.num_features()
    }
    fn feature_coefficient(&self, feature: usize, label_index: usize) -> f64 {
        self.0.feature_coefficient(
            i32::try_from(feature + 1).unwrap(),
            i32::try_from(label_index).unwrap(),
        )
    }
    fn label_bias(&self, label_index: usize) -> f64 {
        self.0.label_bias(i32::try_from(label_index).unwrap())
    }
}

/// Trains a classifier with the solver.
///
/// # Arguments
///
/// * `ys` - The labels.
/// * `xs` - The features of each instance.
pub fn fit(
    ys: Vec<f64>,
    xs: Vec<Vec<(u32, f64)>>,
    epsilon: f64,
    solver: Solver,
) -> Result<Box<dyn Linear>> {
    if ys.is_empty() {
        return Err(VaporettoError::invalid_argument(
            "dataset",
            "must contain at least one example",
        ));
    }
    let problem = match solver {
        Solver::L1RegularizedL2LossSVC { cost } => Problem::ElasticNetL2LossSvc { cost, l2: 0. },
        // ||w||_1 / C1 + ||w||^2 / (2 C2) + loss = (||w||_1 + (C1 / C2) ||w||^2 / 2 + C1 loss) / C1
        Solver::ElasticNet { l1_cost, l2_cost } => Problem::ElasticNetL2LossSvc {
            cost: l1_cost,
            l2: l1_cost / l2_cost,
        },
        Solver::L2RegularizedL2LossSVCDual { cost } => Problem::L2RegularizedL2LossSvcDual { cost },
        Solver::L2RegularizedL1LossSVCDual { cost } => Problem::L2RegularizedL1LossSvcDual { cost },
        Solver::L2RegularizedLogisticDual { cost } => Problem::L2RegularizedLogisticDual { cost },
        #[cfg(feature = "liblinear")]
        Solver::Liblinear { solver, cost } => {
            let mut xs = xs;
            // LIBLINEAR's feature IDs start from 1.
            xs.iter_mut().flatten().for_each(|(fid, _)| *fid += 1);
            let mut builder = liblinear::Builder::new();
            let training_input = liblinear::util::TrainingInput::from_sparse_features(ys, xs)
                .map_err(|e| VaporettoError::invalid_model(format!("liblinear error: {e:?}")))?;
            builder.problem().input_data(training_input).bias(1.0);
            builder
                .parameters()
                .solver_type(solver.into())
                .stopping_criterion(epsilon)
                .constraints_violation_cost(cost);
            let model = builder
                .build_model()
                .map_err(|e| VaporettoError::invalid_model(e.to_string()))?;
            return Ok(Box::new(LiblinearModel(model)));
        }
    };
    Ok(Box::new(LinearModel::train(&ys, xs, epsilon, problem)))
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

#[derive(Clone, Copy)]
struct DummyValue;

impl From<usize> for DummyValue {
    fn from(_: usize) -> Self {
        Self
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
/// use vaporetto::{Sentence, Solver, Trainer};
///
/// let mut train_sents = vec![];
/// let f = BufReader::new(File::open("dataset-train.txt").unwrap());
/// for (i, line) in f.lines().enumerate() {
///     train_sents.push(Sentence::from_tokenized(&line.unwrap()).unwrap());
/// }
///
/// let dict: Vec<String> = vec![];
/// let mut trainer = Trainer::new(3, 3, 3, 3, dict, 0, &[]).unwrap();
/// for (i, s) in train_sents.iter().enumerate() {
///     trainer.add_example(&s);
/// }
///
/// let model = trainer
///     .train(0.01, Solver::L1RegularizedL2LossSVC { cost: 1. })
///     .unwrap();
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
    dict_pma: Option<DoubleArrayAhoCorasick<DummyValue>>,
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
    /// * `dict_word_max_len` - Dictionary words longer than this value will be grouped together,
    ///   where the length is in characters.
    /// * `tag_dictionary` - A tag dictionary. Words not included in the corpus are annotated
    ///   with the tag specified here.
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
        tag_dictionary: &'a [Sentence<'a, '_>],
    ) -> Result<Self> {
        if dict_word_max_len == 0 && !dict_words.is_empty() {
            return Err(VaporettoError::invalid_argument(
                "dict_word_max_len",
                "must be greater than 0 if dict_words is not empty",
            ));
        }
        let dict_pma = if dict_words.is_empty() {
            None
        } else {
            Some(
                DoubleArrayAhoCorasick::new(&dict_words)
                    .map_err(|e| VaporettoError::invalid_argument("dict_words", e.to_string()))?,
            )
        };
        let mut default_tags = HashMap::new();
        for s in tag_dictionary {
            for token in s.iter_tokens() {
                if !default_tags.contains_key(token.surface()) {
                    default_tags.insert(token.surface(), token.tags());
                }
            }
        }
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
                default_tags,
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
            for m in pma.find_overlapping_iter(sentence.text.as_ref()) {
                debug_assert!(sentence.text.is_char_boundary(m.start()));
                let start = unsafe { sentence.str_to_char_pos(m.start()) };
                debug_assert!(sentence.text.is_char_boundary(m.end()));
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
            if b == CharacterBoundary::Unknown {
                continue;
            }
            let mut feature_vector = HashMap::new();
            for feature in features {
                let new_id = self.feature_ids.len();
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
    /// * `solver` - The solver and its costs.
    ///
    /// # Errors
    ///
    /// * If the costs are invalid, an error variant will be returned.
    /// * If the solver returns an error, that will be propagated.
    /// * If the dataset contains no word boundaries, an error variant will be returned.
    pub fn train(self, epsilon: f64, solver: Solver) -> Result<Model> {
        solver.check()?;
        let model = fit(self.ys, self.xs, epsilon, solver)?;

        let wb_idx = model
            .labels()
            .iter()
            .position(|&cls| CharacterBoundary::WordBoundary as i32 == cls)
            .ok_or_else(|| {
                VaporettoError::invalid_argument(
                    "dataset",
                    "must contain at least one word boundary",
                )
            })?;

        let bias = model.label_bias(wb_idx);

        let mut weight_max = bias.abs();
        for fid in 0..model.num_features() {
            let weight = model.feature_coefficient(fid, wb_idx).abs();
            weight_max = weight_max.max(weight);
        }
        let quantize_multiplier = weight_max / f64::from((1 << (QUANTIZE_BIT_DEPTH - 1)) - 1);
        if quantize_multiplier == 0. {
            return Err(VaporettoError::invalid_model("all weights are zero"));
        }

        // Uses BTreeMap to improve compression ratio.
        let mut char_ngram_weights: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut type_ngram_weights: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut dict_weights = vec![(0, 0, 0); usize::from(self.dict_word_max_len)];

        let bias = unsafe { (bias / quantize_multiplier).to_int_unchecked::<i32>() };

        for (feature, fid) in self.feature_ids {
            let raw_weight = model.feature_coefficient(fid as usize, wb_idx);
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
                        isize::from(self.type_window_size) - isize::try_from(len)? - rel_position,
                    )
                    .unwrap();
                    if let Some(weights) = type_ngram_weights.get_mut(ngram) {
                        weights[pos] = weight;
                    } else {
                        let mut weights = vec![0; usize::from(self.type_window_size) * 2 - len + 1];
                        weights[pos] = weight;
                        type_ngram_weights.insert(ngram.to_vec(), weights);
                    }
                }
                BoundaryFeature::DictionaryWord(DictionaryWordFeature { length, position }) => {
                    let weights = &mut dict_weights[length - 1];
                    match position {
                        DictionaryWordPosition::Left => weights.0 = weight,
                        DictionaryWordPosition::Inside => weights.1 = weight,
                        DictionaryWordPosition::Right => weights.2 = weight,
                    }
                }
            }
        }

        let tag_models = self.tag_trainer.train(epsilon, solver)?;

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
                        let mut weights = vec![0; word_len + 1];
                        *weights.first_mut().unwrap() = dict_weights[idx].0;
                        weights[1..word_len].fill(dict_weights[idx].1);
                        *weights.last_mut().unwrap() = dict_weights[idx].2;
                        WordWeightRecord {
                            word,
                            weights,
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

    /// The solvers to test: the solvers in Rust, and LIBLINEAR's if enabled.
    fn solvers() -> Vec<Solver> {
        #[cfg_attr(not(feature = "liblinear"), allow(unused_mut))]
        let mut solvers = vec![
            Solver::L1RegularizedL2LossSVC { cost: 1.0 },
            Solver::L2RegularizedL2LossSVCDual { cost: 1.0 },
            Solver::L2RegularizedL1LossSVCDual { cost: 1.0 },
            Solver::L2RegularizedLogisticDual { cost: 1.0 },
            Solver::ElasticNet {
                l1_cost: 1.0,
                l2_cost: 1.0,
            },
        ];
        #[cfg(feature = "liblinear")]
        solvers.extend([
            Solver::Liblinear {
                solver: LiblinearSolver::L2RegularizedLogistic,
                cost: 1.0,
            },
            Solver::Liblinear {
                solver: LiblinearSolver::L1RegularizedL2LossSVC,
                cost: 1.0,
            },
        ]);
        solvers
    }

    fn train_model(
        texts: &[&str],
        char_window_size: u8,
        char_ngram_size: u8,
        type_window_size: u8,
        type_ngram_size: u8,
        solver: Solver,
    ) -> Result<Model> {
        let sentences: Vec<_> = texts
            .iter()
            .map(|text| Sentence::from_tokenized(text).unwrap())
            .collect();
        let mut trainer = Trainer::new(
            char_window_size,
            char_ngram_size,
            type_window_size,
            type_ngram_size,
            vec![],
            4,
            &[],
        )?;
        for _ in 0..10 {
            for s in &sentences {
                trainer.add_example(s);
            }
        }
        trainer.train(0.01, solver)
    }

    #[test]
    fn check_train_type_window_size() {
        // Only character type n-grams are used.
        let texts = [
            "漢字 ひらがな",
            "カタカナ 漢字",
            "ひらがな カタカナ",
            "漢字 カタカナ",
        ];
        for (solver, (char_window_size, type_window_size)) in solvers()
            .into_iter()
            .flat_map(|a| [(1, 1), (3, 1), (1, 3), (2, 3)].map(|w| (a, w)))
        {
            let model =
                train_model(&texts, char_window_size, 0, type_window_size, 2, solver).unwrap();
            for data in &model.0.type_ngram_model.0 {
                assert_eq!(
                    usize::from(type_window_size) * 2 - data.ngram.len() + 1,
                    data.weights.len(),
                );
            }
            let predictor = crate::Predictor::new(model, false).unwrap();
            let mut s = Sentence::from_raw("東京とうきょう").unwrap();
            predictor.predict(&mut s);
            let mut buf = String::new();
            s.write_tokenized_text(&mut buf);
            assert_eq!(
                "東京 とうきょう", buf,
                "{solver:?}, char_window_size={char_window_size}, type_window_size={type_window_size}",
            );
        }
    }

    #[test]
    fn check_train_no_word_boundary() {
        for solver in solvers() {
            let result = train_model(&["あいう", "かきく"], 3, 3, 3, 3, solver);
            assert!(result.is_err(), "{solver:?}");
        }
    }

    #[test]
    fn check_train_no_example() {
        for solver in solvers() {
            // A sentence of one character has no character boundary.
            let result = train_model(&["あ"], 3, 3, 3, 3, solver);
            assert!(result.is_err(), "{solver:?}");
        }
    }

    #[test]
    fn check_train_invalid_cost() {
        for solver in [
            Solver::L1RegularizedL2LossSVC { cost: 0. },
            Solver::L2RegularizedLogisticDual {
                cost: f64::INFINITY,
            },
            Solver::ElasticNet {
                l1_cost: f64::INFINITY,
                l2_cost: 1.,
            },
            Solver::ElasticNet {
                l1_cost: 1.,
                l2_cost: -1.,
            },
        ] {
            let s = Sentence::from_tokenized("あい う").unwrap();
            let mut trainer = Trainer::new(3, 3, 3, 3, vec![], 4, &[]).unwrap();
            trainer.add_example(&s);
            assert!(trainer.train(0.01, solver).is_err(), "{solver:?}");
        }
    }

    #[test]
    fn check_new_dict_word_max_len_zero() {
        assert!(Trainer::new(3, 3, 3, 3, vec!["これ".into()], 0, &[]).is_err());
        assert!(Trainer::new(3, 3, 3, 3, vec![], 0, &[]).is_ok());
    }

    #[cfg(feature = "tag-prediction")]
    #[test]
    fn check_train_tag_rel_position() {
        // "猫" has two tags depending on the context.
        let texts = [
            "あの/連体詞 火星/名詞 猫/名詞 だ/助動詞 よ/助詞 ね/助詞",
            "その/連体詞 地球/名詞 猫/接尾辞 です/助動詞 よ/助詞 ね/助詞",
            "あの/連体詞 火星/名詞 に/助詞 猫/名詞 が/助詞 いる/動詞",
            "その/連体詞 地球/名詞 で/助詞 猫/接尾辞 を/助詞 見る/動詞",
        ];
        for (solver, (char_window_size, char_ngram_size, type_window_size, type_ngram_size)) in
            solvers()
                .into_iter()
                .flat_map(|a| [(2, 3, 2, 2), (1, 3, 1, 1), (3, 3, 0, 3)].map(|w| (a, w)))
        {
            let model = train_model(
                &texts,
                char_window_size,
                char_ngram_size,
                type_window_size,
                type_ngram_size,
                solver,
            )
            .unwrap();
            for tag_model in &model.0.tag_models {
                for data in &tag_model.char_ngram_model.0 {
                    for w in &data.weights {
                        assert!(w.rel_position <= char_window_size);
                    }
                }
                for data in &tag_model.type_ngram_model.0 {
                    for w in &data.weights {
                        assert!(w.rel_position <= type_window_size);
                    }
                }
            }
            assert!(
                crate::Predictor::new(model, true).is_ok(),
                "{solver:?}, window_sizes={:?}",
                (
                    char_window_size,
                    char_ngram_size,
                    type_window_size,
                    type_ngram_size,
                ),
            );
        }
    }

    #[test]
    fn check_features_3322() {
        let s = Sentence::from_tokenized("これ は テスト です").unwrap();
        let trainer = Trainer::new(3, 3, 2, 2, vec![], 4, &[]).unwrap();
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
            &[],
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

    #[test]
    fn check_add_example_partial() {
        let s = Sentence::from_partial_annotation("ホ ー ム ペ ー ジ|を|見-る").unwrap();
        let mut trainer = Trainer::new(3, 3, 3, 3, vec![], 4, &[]).unwrap();
        trainer.add_example(&s);
        assert_eq!(3, trainer.xs.len());
        assert_eq!(vec![1.0, 1.0, 0.0], trainer.ys);
    }
}
