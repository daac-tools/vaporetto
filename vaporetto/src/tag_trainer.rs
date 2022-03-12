use std::collections::BTreeMap;

use liblinear::LibLinearModel;

use crate::errors::{Result, VaporettoError};
use crate::feature::{StringNgramFeature, TagExampleGenerator, TagFeature};
use crate::ngram_model::{NgramData, NgramModel};
use crate::sentence::Sentence;
use crate::tag_model::{TagClassInfo, TagModel};
use crate::trainer::{Indexer, SolverType, QUANTIZE_BIT_DEPTH};

pub struct TagTrainer<'a> {
    example_generator: TagExampleGenerator,
    char_window_size: u8,
    feature_ids: Indexer<TagFeature<'a>>,
    tag_ids: Indexer<String>,
    xs: Vec<Vec<(u32, f64)>>,
    ys: Vec<f64>,
}

impl<'a> TagTrainer<'a> {
    pub fn new(char_ngram_size: u8, char_window_size: u8) -> Self {
        Self {
            example_generator: TagExampleGenerator::new(char_ngram_size, char_window_size),
            char_window_size,
            feature_ids: Indexer::new(),
            tag_ids: Indexer::new(),
            xs: vec![],
            ys: vec![],
        }
    }

    pub fn push_sentence(&mut self, s: &'a Sentence) -> Result<()> {
        let examples = self.example_generator.generate(s)?;
        for example in examples {
            let mut feature_ids = BTreeMap::new();
            for f in &example.features {
                let fid = self.feature_ids.get_id(f);
                *feature_ids
                    .entry((fid + 1).try_into().unwrap())
                    .or_insert(0.0) += 1.0;
            }
            self.xs.push(feature_ids.into_iter().collect());
            self.ys
                .push(self.tag_ids.get_id(example.tag.as_str()) as f64);
        }
        Ok(())
    }

    pub fn n_features(&self) -> usize {
        self.feature_ids.len()
    }

    pub fn train(self, epsilon: f64, cost: f64, solver: SolverType) -> Result<TagModel> {
        if self.xs.is_empty() {
            // Returns an empty model if there is no training data.
            return Ok(TagModel::default());
        }

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

        // Uses BTreeMap to increase compression ratio.
        let mut left_char_weights: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut right_char_weights: BTreeMap<_, Vec<_>> = BTreeMap::new();
        let mut self_char_weights: BTreeMap<_, Vec<_>> = BTreeMap::new();

        let mut weight_max = 0.;
        for i in 0..self.tag_ids.len() as i32 {
            let weight = model.label_bias(i).abs();
            if weight > weight_max {
                weight_max = weight;
            }
            for fid in 0..model.num_features() {
                let weight = model.feature_coefficient(fid as i32, i).abs();
                if weight > weight_max {
                    weight_max = weight;
                }
            }
        }
        let quantize_multiplier = weight_max / ((1 << (QUANTIZE_BIT_DEPTH - 1)) - 1) as f64;

        let mut class_info = vec![];

        for i in 0..self.tag_ids.len() {
            class_info.push(TagClassInfo {
                name: self.tag_ids.keys()[model.labels()[i] as usize].clone(),
                bias: (model.label_bias(i as i32) / quantize_multiplier) as i32,
            });

            for (fid, feature) in self.feature_ids.keys().iter().enumerate() {
                let raw_weight = model.feature_coefficient(fid as i32 + 1, i as i32);
                let weight = (raw_weight / quantize_multiplier) as i32;

                if weight == 0 {
                    continue;
                }

                match feature {
                    TagFeature::LeftCharacterNgram(StringNgramFeature {
                        rel_position,
                        ngram,
                    }) => {
                        let pos = -rel_position - 1;
                        let idx = i + pos as usize * self.tag_ids.len();
                        if let Some(weights) = left_char_weights.get_mut(*ngram) {
                            weights[idx] = weight;
                        } else {
                            let mut weights =
                                vec![0; usize::from(self.char_window_size) * self.tag_ids.len()];
                            weights[idx] = weight;
                            left_char_weights.insert(ngram.to_string(), weights);
                        }
                    }
                    TagFeature::LeftCharacterNgramBos(StringNgramFeature {
                        rel_position,
                        ngram,
                    }) => {
                        let pos = -rel_position - 1;
                        let idx = i + pos as usize * self.tag_ids.len();
                        let ngram = "\0".to_string() + *ngram;
                        left_char_weights.entry(ngram).or_insert_with(|| {
                            vec![0; usize::from(self.char_window_size) * self.tag_ids.len()]
                        })[idx] = weight;
                    }
                    TagFeature::RightCharacterNgram(StringNgramFeature {
                        rel_position,
                        ngram,
                    }) => {
                        let pos = isize::from(self.char_window_size) - rel_position;
                        let idx = i + pos as usize * self.tag_ids.len();
                        if let Some(weights) = right_char_weights.get_mut(*ngram) {
                            weights[idx] = weight;
                        } else {
                            let mut weights =
                                vec![0; usize::from(self.char_window_size) * self.tag_ids.len()];
                            weights[idx] = weight;
                            right_char_weights.insert(ngram.to_string(), weights);
                        }
                    }
                    TagFeature::RightCharacterNgramEos(StringNgramFeature {
                        rel_position,
                        ngram,
                    }) => {
                        let pos = isize::from(self.char_window_size) - rel_position;
                        let idx = i + pos as usize * self.tag_ids.len();
                        let ngram = ngram.to_string() + "\0";
                        right_char_weights.entry(ngram).or_insert_with(|| {
                            vec![0; usize::from(self.char_window_size) * self.tag_ids.len()]
                        })[idx] = weight;
                    }
                    TagFeature::Character(ngram) => {
                        if let Some(weights) = self_char_weights.get_mut(*ngram) {
                            weights[i] = weight;
                        } else {
                            let mut weights = vec![0; self.tag_ids.len()];
                            weights[i] = weight;
                            self_char_weights.insert(ngram.to_string(), weights);
                        }
                    }
                };
            }
        }
        Ok(TagModel {
            class_info,
            left_char_model: NgramModel::new(
                left_char_weights
                    .into_iter()
                    .map(|(ngram, weights)| NgramData { ngram, weights })
                    .collect(),
            ),
            right_char_model: NgramModel::new(
                right_char_weights
                    .into_iter()
                    .map(|(ngram, weights)| NgramData { ngram, weights })
                    .collect(),
            ),
            self_char_model: NgramModel::new(
                self_char_weights
                    .into_iter()
                    .map(|(ngram, weights)| NgramData { ngram, weights })
                    .collect(),
            ),
        })
    }
}
