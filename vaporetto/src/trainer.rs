use std::collections::BTreeMap;

use anyhow::{anyhow, Result};
use fst::raw::Fst;

use crate::feature::{ExampleGenerator, FeatureExtractor};
use crate::model::Model;
use crate::sentence::Sentence;
use crate::utils::FeatureIDManager;

/// Dataset manager.
#[doc(cfg(feature = "train"))]
pub struct Dataset<'a> {
    dictionary_fst: Fst<Vec<u8>>,
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
    /// * `dictionary` - A word dictionary (must be alphabetical ascending order).
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
            dictionary_fst: Fst::from_iter_map(
                dictionary
                    .as_ref()
                    .iter()
                    .enumerate()
                    .map(|(i, word)| (word, i as u64)),
            )?,
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
                if let Some(v) = feature_ids.get_mut(&fid) {
                    *v += 1.0;
                } else {
                    feature_ids.insert(fid, 1.0);
                }
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
/// use vaporetto::{Dataset, Sentence, Trainer};
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
/// let model = trainer.train(dataset).unwrap();
/// let mut f = BufWriter::new(File::create("model.bin").unwrap());
/// model.write(&mut f).unwrap();
/// ```
#[doc(cfg(feature = "train"))]
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
    ///
    /// # Returns
    ///
    /// A trained model.
    pub fn train(&self, dataset: Dataset) -> Result<Model> {
        let mut builder = liblinear::Builder::new();
        let training_input =
            liblinear::util::TrainingInput::from_sparse_features(dataset.ys, dataset.xs)
                .map_err(|e| anyhow!("liblinear error: {:?}", e))?;
        builder.problem().input_data(training_input).bias(self.bias);
        builder
            .parameters()
            .solver_type(liblinear::SolverType::L1R_L2LOSS_SVC)
            .stopping_criterion(self.epsilon)
            .constraints_violation_cost(self.cost);
        let model = builder.build_model().map_err(|e| anyhow!(e.to_string()))?;

        Ok(Model::from_liblinear_model(
            model,
            dataset.fid_manager,
            dataset.dictionary_fst,
            dataset.char_window_size,
            dataset.type_window_size,
            dataset.dict_word_max_size,
        ))
    }
}
