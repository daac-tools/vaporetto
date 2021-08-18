use std::collections::{BTreeMap, BTreeSet};

use anyhow::{anyhow, Result};
use fst::raw::Fst;

use crate::feature::{ExampleGenerator, FeatureExtractor};
use crate::model::Model;
use crate::sentence::Sentence;
use crate::utils::FeatureIDManager;

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
        let sorted_dict: BTreeSet<Vec<u8>> = dictionary
            .as_ref()
            .iter()
            .map(|w| AsRef::<[u8]>::as_ref(w).into())
            .collect();
        Ok(Self {
            dictionary_fst: Fst::from_iter_set(sorted_dict)?,
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

    pub fn n_features(&self) -> usize {
        self.fid_manager.map.len()
    }
}

pub struct Trainer {
    epsilon: f64,
    cost: f64,
    bias: f64,
}

impl Trainer {
    pub fn new(epsilon: f64, cost: f64, bias: f64) -> Self {
        Self {
            epsilon,
            cost,
            bias,
        }
    }

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
