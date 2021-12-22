use std::collections::BTreeMap;
use std::str::FromStr;

use crate::errors::{Result, VaporettoError};
use crate::feature::{ExampleGenerator, FeatureExtractor};
use crate::model::Model;
use crate::sentence::Sentence;
use crate::utils::FeatureIDManager;

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

/// Dataset manager.
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub struct Dataset<'a> {
    dictionary: Vec<String>,
    feature_extractor: FeatureExtractor,
    example_generator: ExampleGenerator,
    char_window_size: usize,
    type_window_size: usize,
    dict_word_max_size: usize,
    fid_manager: FeatureIDManager<'a>,
    xs: Vec<Vec<(u32, f64)>>,
    ys: Vec<f64>,
}

impl<'a> Dataset<'a> {
    /// Creates a new dataset manager.
    ///
    /// # Arguments
    ///
    /// * `char_ngram_size` - The character n-gram length.
    /// * `char_window_size` - The character window size.
    /// * `type_ngram_size` - The character type n-gram length.
    /// * `type_window_size` - The character type window size.
    /// * `dictionary` - A word dictionary.
    /// * `dict_word_max_size` - Dictionary words greater than this value will be grouped together.
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
        dict_word_max_size: usize,
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
            feature_extractor: FeatureExtractor::new(
                char_ngram_size,
                type_ngram_size,
                dictionary,
                dict_word_max_size,
            )?,
            example_generator: ExampleGenerator::new(char_window_size, type_window_size),
            char_window_size,
            type_window_size,
            dict_word_max_size,
            fid_manager: FeatureIDManager::default(),
            xs: vec![],
            ys: vec![],
        })
    }

    /// Adds a sentence to the dataset.
    ///
    /// # Arguments
    ///
    /// * `s` - A sentence.
    pub fn push_sentence(&mut self, s: &'a Sentence) {
        let feature_spans = self.feature_extractor.extract(s);
        let examples = self.example_generator.generate(s, feature_spans, false);
        for example in examples {
            let mut feature_ids = BTreeMap::new();
            for f in example.features {
                let fid = self.fid_manager.get_id(f) + 1;
                *feature_ids.entry(fid).or_insert(0.0) += 1.0;
            }
            self.xs.push(feature_ids.into_iter().collect());
            self.ys.push(example.label as u8 as f64);
        }
    }

    /// Gets the number of features.
    ///
    /// # Returns
    ///
    /// The number of features.
    pub fn n_features(&self) -> usize {
        self.fid_manager.map.len()
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
/// use vaporetto::{Dataset, Sentence, SolverType, Trainer};
///
/// let mut train_sents = vec![];
/// let f = BufReader::new(File::open("dataset-train.txt").unwrap());
/// for (i, line) in f.lines().enumerate() {
///     train_sents.push(Sentence::from_tokenized(line.unwrap()).unwrap());
/// }
///
/// let dict: Vec<String> = vec![];
/// let mut dataset = Dataset::new(3, 3, 3, 3, &dict, 0).unwrap();
/// for (i, s) in train_sents.iter().enumerate() {
///     dataset.push_sentence(s);
/// }
///
/// let trainer = Trainer::new(0.01, 1., 1.);
/// let model = trainer.train(dataset, SolverType::L1RegularizedL2LossSVC).unwrap();
/// let mut f = BufWriter::new(File::create("model.bin").unwrap());
/// model.write(&mut f).unwrap();
/// ```
#[cfg_attr(docsrs, doc(cfg(feature = "train")))]
pub struct Trainer {
    epsilon: f64,
    cost: f64,
    bias: f64,
}

impl Trainer {
    /// Creates a new trainer.
    ///
    /// # Arguments
    ///
    /// * `epsilon` - The tolerance of the termination criterion.
    /// * `cost` - The parameter C.
    /// * `bias` - The bias term.
    ///
    /// # Returns
    ///
    /// A new trainer.
    pub const fn new(epsilon: f64, cost: f64, bias: f64) -> Self {
        Self {
            epsilon,
            cost,
            bias,
        }
    }

    /// Trains a given dataset.
    ///
    /// # Arguments
    ///
    /// * `dataset` - A dataset.
    /// * `solver` - Solver type.
    ///
    /// # Returns
    ///
    /// A trained model.
    pub fn train(&self, dataset: Dataset, solver: SolverType) -> Result<Model> {
        let mut builder = liblinear::Builder::new();
        let training_input =
            liblinear::util::TrainingInput::from_sparse_features(dataset.ys, dataset.xs)
                .map_err(|e| VaporettoError::invalid_model(format!("liblinear error: {:?}", e)))?;
        builder.problem().input_data(training_input).bias(self.bias);
        builder
            .parameters()
            .solver_type(solver.into())
            .stopping_criterion(self.epsilon)
            .constraints_violation_cost(self.cost);
        let model = builder
            .build_model()
            .map_err(|e| VaporettoError::invalid_model(e.to_string()))?;

        Ok(Model::from_liblinear_model(
            model,
            dataset.fid_manager,
            dataset.dictionary,
            dataset.char_window_size,
            dataset.type_window_size,
            dataset.dict_word_max_size,
        ))
    }
}
