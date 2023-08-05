use alloc::borrow::Cow;
use alloc::collections::BTreeMap;
use alloc::string::ToString;

use hashbrown::HashMap;
use liblinear::LibLinearModel;

use crate::errors::{Result, VaporettoError};
use crate::model::TagModel;
use crate::ngram_model::{TagNgramData, TagNgramModel, TagWeight};
use crate::sentence::Sentence;
use crate::trainer::{NgramFeature, SolverType};

use crate::trainer::QUANTIZE_BIT_DEPTH;

#[derive(Debug, Eq, Hash, PartialEq)]
enum TagFeature<'a> {
    CharacterNgram(NgramFeature<&'a str>),
    CharacterTypeNgram(NgramFeature<&'a [u8]>),
}

impl<'a> TagFeature<'a> {
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
}

#[derive(Debug)]
struct TagExample<'a> {
    tags: &'a [Option<Cow<'a, str>>],
    features: Vec<TagFeature<'a>>,
}

pub struct TagTrainer<'a> {
    _char_window_size: u8,
    char_ngram_size: u8,
    _type_window_size: u8,
    type_ngram_size: u8,
    default_tags: HashMap<&'a str, &'a [Option<Cow<'a, str>>]>,
    // Uses BTreeMap to improve compression ratio.
    examples: BTreeMap<&'a str, Vec<TagExample<'a>>>,
}

impl<'a> TagTrainer<'a> {
    pub const fn new(
        char_window_size: u8,
        char_ngram_size: u8,
        type_window_size: u8,
        type_ngram_size: u8,
        default_tags: HashMap<&'a str, &'a [Option<Cow<'a, str>>]>,
    ) -> Self {
        Self {
            _char_window_size: char_window_size,
            char_ngram_size,
            _type_window_size: type_window_size,
            type_ngram_size,
            default_tags,
            examples: BTreeMap::new(),
        }
    }

    pub fn add_example<'b>(&mut self, sentence: &'a Sentence<'a, 'b>) {
        for token in sentence.iter_tokens() {
            if token.tags().is_empty() {
                continue;
            }
            let mut features = vec![];
            let token_len = token.end() - token.start();
            for n in 0..usize::from(self.char_ngram_size) {
                let ngram_len = token_len + n + 1;
                for i in token.end().saturating_sub(ngram_len)
                    ..(token.start() + 1).min(sentence.len().saturating_sub(ngram_len - 1))
                {
                    features.push(TagFeature::char_ngram(
                        sentence.text_substring(i, i + ngram_len),
                        isize::try_from(i + ngram_len - token.end()).unwrap(),
                    ));
                }
            }
            for n in 0..usize::from(self.type_ngram_size) {
                let ngram_len = token_len + n + 1;
                for i in token.end().saturating_sub(ngram_len)
                    ..(token.start() + 1).min(sentence.len().saturating_sub(ngram_len - 1))
                {
                    features.push(TagFeature::type_ngram(
                        &sentence.char_types()[i..i + ngram_len],
                        isize::try_from(i + ngram_len - token.end()).unwrap(),
                    ));
                }
            }
            self.examples
                .entry(token.surface())
                .or_insert_with(Vec::new)
                .push(TagExample {
                    tags: token.tags(),
                    features,
                });
        }
    }

    #[allow(clippy::type_complexity)]
    fn gen_feature_vecs<'b>(
        examples: &'b [TagExample<'a>],
        idx: usize,
        tag_ids: &HashMap<&'a str, usize>,
    ) -> (
        HashMap<&'b TagFeature<'a>, u32>,
        Vec<Vec<(u32, f64)>>,
        Vec<f64>,
    ) {
        let mut feature_ids = HashMap::new();
        let mut xs = vec![];
        let mut ys = vec![];
        for example in examples {
            if let Some(tag) = example.tags.get(idx).and_then(|tag| tag.as_ref()) {
                ys.push(tag_ids[tag.as_ref()] as f64)
            } else {
                continue;
            }
            let mut feature_vec = vec![];
            for feature in &example.features {
                let new_id = u32::try_from(feature_ids.len() + 1).unwrap();
                let feature_id = *feature_ids.entry(feature).or_insert(new_id);
                feature_vec.push((feature_id, 1f64));
            }
            xs.push(feature_vec);
        }
        (feature_ids, xs, ys)
    }

    fn train_tag(
        token: String,
        examples: &[TagExample<'a>],
        epsilon: f64,
        cost: f64,
        solver: SolverType,
    ) -> Result<TagModel> {
        let n_tags = examples.iter().fold(0, |acc, x| acc.max(x.tags.len()));
        let mut tag_ids = vec![HashMap::new(); n_tags];
        let mut tags = vec![vec![]; n_tags];
        for example in examples {
            for ((tag, tag_ids), tags) in example.tags.iter().zip(&mut tag_ids).zip(&mut tags) {
                if let Some(tag) = tag {
                    if !tag_ids.contains_key(tag.as_ref()) {
                        let new_id = tag_ids.len();
                        tag_ids.insert(tag.as_ref(), new_id);
                        tags.push(tag.to_string());
                    }
                }
            }
        }
        let n_class = tags
            .iter()
            .fold(0, |acc, x| acc + if x.len() >= 2 { x.len() } else { 0 });

        let mut bias = vec![0; n_class];

        // Uses BTreeMap to increase compression ratio.
        let mut char_ngram_weights = BTreeMap::new();
        let mut type_ngram_weights = BTreeMap::new();

        let mut class_offset = 0;
        for (i, tag_ids) in tag_ids.iter().enumerate() {
            if tag_ids.len() <= 1 {
                // fixed tag
                continue;
            }

            // train
            let (feature_ids, xs, ys) = Self::gen_feature_vecs(examples, i, tag_ids);

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

            // Calculates the quantize multiplier
            let mut weight_max = 1e-6f64;
            for i in 0..i32::try_from(tag_ids.len()).unwrap() {
                let bias = model.label_bias(i).abs();
                weight_max = weight_max.max(bias);
                for fid in 0..model.num_features() {
                    let weight = model.feature_coefficient(i32::try_from(fid + 1)?, i).abs();
                    weight_max = weight_max.max(weight);
                }
            }
            let quantize_multiplier = weight_max / f64::from((1 << (QUANTIZE_BIT_DEPTH - 1)) - 1);

            for (i, &cls) in model.labels().iter().enumerate() {
                bias[class_offset + usize::try_from(cls).unwrap()] = unsafe {
                    (model.label_bias(i32::try_from(i).unwrap()) / quantize_multiplier)
                        .to_int_unchecked::<i32>()
                };
            }
            for (feature, fid) in feature_ids {
                match feature {
                    TagFeature::CharacterNgram(NgramFeature {
                        ngram,
                        rel_position,
                    }) => {
                        for (i, &cls) in model.labels().iter().enumerate() {
                            let raw_weight = model.feature_coefficient(
                                i32::try_from(fid)?,
                                i32::try_from(i).unwrap(),
                            );
                            let weight = unsafe {
                                (raw_weight / quantize_multiplier).to_int_unchecked::<i32>()
                            };
                            if weight == 0 {
                                continue;
                            }
                            char_ngram_weights
                                .entry((*ngram, u8::try_from(*rel_position).unwrap()))
                                .or_insert_with(|| vec![0; n_class])
                                [class_offset + usize::try_from(cls).unwrap()] = weight;
                        }
                    }
                    TagFeature::CharacterTypeNgram(NgramFeature {
                        ngram,
                        rel_position,
                    }) => {
                        for (i, &cls) in model.labels().iter().enumerate() {
                            let raw_weight = model.feature_coefficient(
                                i32::try_from(fid)?,
                                i32::try_from(i).unwrap(),
                            );
                            let weight = unsafe {
                                (raw_weight / quantize_multiplier).to_int_unchecked::<i32>()
                            };
                            if weight == 0 {
                                continue;
                            }
                            type_ngram_weights
                                .entry((*ngram, u8::try_from(*rel_position).unwrap()))
                                .or_insert_with(|| vec![0; n_class])
                                [class_offset + usize::try_from(cls).unwrap()] = weight;
                        }
                    }
                }
            }
            class_offset += tag_ids.len();
        }

        let mut char_ngram_model = BTreeMap::new();
        for ((ngram, rel_position), weights) in char_ngram_weights {
            char_ngram_model
                .entry(ngram.to_string())
                .or_insert_with(Vec::new)
                .push(TagWeight {
                    rel_position,
                    weights,
                });
        }
        let mut type_ngram_model = BTreeMap::new();
        for ((ngram, rel_position), weights) in type_ngram_weights {
            type_ngram_model
                .entry(ngram.to_vec())
                .or_insert_with(Vec::new)
                .push(TagWeight {
                    rel_position,
                    weights,
                });
        }
        Ok(TagModel {
            token,
            tags,
            char_ngram_model: TagNgramModel(
                char_ngram_model
                    .into_iter()
                    .map(|(ngram, weights)| TagNgramData { ngram, weights })
                    .collect(),
            ),
            type_ngram_model: TagNgramModel(
                type_ngram_model
                    .into_iter()
                    .map(|(ngram, weights)| TagNgramData { ngram, weights })
                    .collect(),
            ),
            bias,
        })
    }

    pub fn train(mut self, epsilon: f64, cost: f64, solver: SolverType) -> Result<Vec<TagModel>> {
        for (token, tags) in self.default_tags {
            if tags.iter().any(|t| t.is_some()) && !self.examples.contains_key(token) {
                self.examples.insert(
                    token,
                    vec![TagExample {
                        tags,
                        features: vec![],
                    }],
                );
            }
        }
        let mut tag_models = vec![];
        liblinear::toggle_liblinear_stdout_output(false);
        let n_tokens = self.examples.len();
        for (i, (token, examples)) in self.examples.into_iter().enumerate() {
            tag_models.push(Self::train_tag(
                token.into(),
                &examples,
                epsilon,
                cost,
                solver,
            )?);
            eprint!("Tags: {i}/{n_tokens}\r");
        }
        eprintln!("Tags: {n_tokens}/{n_tokens}");
        liblinear::toggle_liblinear_stdout_output(true);
        Ok(tag_models)
    }
}
