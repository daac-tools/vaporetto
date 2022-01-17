use crate::char_scorer::CharScorer;
use crate::dict_scorer::DictScorer;
use crate::errors::Result;
use crate::model::Model;
use crate::sentence::{BoundaryType, Sentence};
use crate::type_scorer::TypeScorer;

#[cfg(feature = "simd")]
use crate::char_scorer::CharScorerSimd;

/// Predictor.
pub struct Predictor {
    bias: i32,

    char_scorer: CharScorer,
    type_scorer: TypeScorer,
    dict_scorer: Option<DictScorer>,

    #[cfg(feature = "simd")]
    padding: usize,
}

impl Predictor {
    /// Creates a new predictor.
    ///
    /// # Arguments
    ///
    /// * `model` - A model data.
    ///
    /// # Returns
    ///
    /// A new predictor.
    pub fn new(model: Model) -> Result<Self> {
        let bias = model.bias;

        let mut char_ngram_model = model.char_ngram_model;
        let type_ngram_model = model.type_ngram_model;
        let mut dict_model = model.dict_model;

        dict_model.merge_dict_weights(&mut char_ngram_model, model.char_window_size);

        let char_scorer = CharScorer::new(char_ngram_model, model.char_window_size)?;
        let type_scorer = TypeScorer::new(type_ngram_model, model.type_window_size)?;
        let dict_scorer = if dict_model.is_empty() {
            None
        } else {
            Some(DictScorer::new(dict_model)?)
        };

        Ok(Self {
            bias,

            char_scorer,
            type_scorer,
            dict_scorer,

            #[cfg(feature = "simd")]
            padding: model.char_window_size.max(model.type_window_size),
        })
    }

    fn predict_impl(&self, sentence: &Sentence, padding: usize, ys: &mut [i32]) {
        ys.fill(self.bias);
        self.char_scorer.add_scores(sentence, padding, ys);
        self.type_scorer.add_scores(sentence, &mut ys[padding..]);
        if let Some(dict_scorer) = self.dict_scorer.as_ref() {
            dict_scorer.add_scores(sentence, &mut ys[padding..]);
        }
    }

    /// Predicts word boundaries.
    ///
    /// # Arguments
    ///
    /// * `sentence` - A sentence.
    ///
    /// # Returns
    ///
    /// A sentence with predicted boundary information.
    pub fn predict(&self, mut sentence: Sentence) -> Sentence {
        let boundaries_size = sentence.boundaries.len();

        #[cfg(not(feature = "simd"))]
        if boundaries_size != 0 {
            let mut ys = vec![0; boundaries_size];
            self.predict_impl(&sentence, 0, &mut ys);
            for (y, b) in ys.into_iter().zip(sentence.boundaries.iter_mut()) {
                *b = if y >= 0 {
                    BoundaryType::WordBoundary
                } else {
                    BoundaryType::NotWordBoundary
                };
            }
        }

        #[cfg(feature = "simd")]
        if boundaries_size != 0 {
            let ys_size = boundaries_size + self.padding + CharScorerSimd::simd_len() - 1;
            let mut ys = vec![0; ys_size];
            self.predict_impl(&sentence, self.padding, &mut ys);
            for (&y, b) in ys[self.padding..]
                .iter()
                .zip(sentence.boundaries.iter_mut())
            {
                *b = if y >= 0 {
                    BoundaryType::WordBoundary
                } else {
                    BoundaryType::NotWordBoundary
                };
            }
        }

        sentence
    }

    /// Predicts word boundaries. This function inserts scores.
    ///
    /// # Arguments
    ///
    /// * `sentence` - A sentence.
    ///
    /// # Returns
    ///
    /// A sentence with predicted boundary information.
    pub fn predict_with_score(&self, mut sentence: Sentence) -> Sentence {
        let boundaries_size = sentence.boundaries.len();

        #[cfg(not(feature = "simd"))]
        if boundaries_size != 0 {
            let mut ys = vec![0; boundaries_size];
            self.predict_impl(&sentence, 0, &mut ys);
            for (&y, b) in ys.iter().zip(sentence.boundaries.iter_mut()) {
                *b = if y >= 0 {
                    BoundaryType::WordBoundary
                } else {
                    BoundaryType::NotWordBoundary
                };
            }
            sentence.boundary_scores.replace(ys);
        }

        #[cfg(feature = "simd")]
        if boundaries_size != 0 {
            let ys_size = boundaries_size + self.padding + CharScorerSimd::simd_len() - 1;
            let mut ys = vec![0; ys_size];
            self.predict_impl(&sentence, self.padding, &mut ys);
            let mut scores = sentence
                .boundary_scores
                .take()
                .unwrap_or_else(|| vec![0; boundaries_size]);
            scores.resize(boundaries_size, 0);
            for (&y, (b, s)) in ys[self.padding..]
                .iter()
                .zip(sentence.boundaries.iter_mut().zip(scores.iter_mut()))
            {
                *b = if y >= 0 {
                    BoundaryType::WordBoundary
                } else {
                    BoundaryType::NotWordBoundary
                };

                *s = y;
            }
            sentence.boundary_scores.replace(scores);
        }

        sentence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dict_model::{DictModel, DictWeight, WordWeightRecord};
    use crate::ngram_model::{NgramData, NgramModel};

    /// Input:  我  ら  は  全  世  界  の  国  民
    /// bias:   -200  ..  ..  ..  ..  ..  ..  ..
    /// words:
    ///   我ら:    3   4   5
    ///   全世界:          6   7   8   9
    ///   国民:                       10  11  12
    ///   世界:           15  16  17  18  19
    ///   界:             20  21  22  23  24  25
    /// types:
    ///   H:      27  28  29
    ///           26  27  28  29
    ///                           26  27  28  29
    ///   K:      32  33
    ///               30  31  32  33
    ///                   30  31  32  33
    ///                       30  31  32  33
    ///                               30  31  32
    ///                                   30  31
    ///   KH:     35  36
    ///                           34  35  36
    ///   HK:         37  38  39
    ///                               37  38  39
    /// dict:
    ///   全世界:         43  44  44  45
    ///   世界:               43  44  45
    ///   世:                 40  42
    fn generate_model_1() -> Model {
        Model {
            char_ngram_model: NgramModel::new(vec![
                NgramData {
                    ngram: "我ら".to_string(),
                    weights: vec![1, 2, 3, 4, 5],
                },
                NgramData {
                    ngram: "全世界".to_string(),
                    weights: vec![6, 7, 8, 9],
                },
                NgramData {
                    ngram: "国民".to_string(),
                    weights: vec![10, 11, 12, 13, 14],
                },
                NgramData {
                    ngram: "世界".to_string(),
                    weights: vec![15, 16, 17, 18, 19],
                },
                NgramData {
                    ngram: "界".to_string(),
                    weights: vec![20, 21, 22, 23, 24, 25],
                },
            ]),
            type_ngram_model: NgramModel::new(vec![
                NgramData {
                    ngram: b"H".to_vec(),
                    weights: vec![26, 27, 28, 29],
                },
                NgramData {
                    ngram: b"K".to_vec(),
                    weights: vec![30, 31, 32, 33],
                },
                NgramData {
                    ngram: b"KH".to_vec(),
                    weights: vec![34, 35, 36],
                },
                NgramData {
                    ngram: b"HK".to_vec(),
                    weights: vec![37, 38, 39],
                },
            ]),
            dict_model: DictModel {
                dict: vec![
                    WordWeightRecord {
                        word: "全世界".to_string(),
                        weights: DictWeight {
                            right: 43,
                            inside: 44,
                            left: 45,
                        },
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世界".to_string(),
                        weights: DictWeight {
                            right: 43,
                            inside: 44,
                            left: 45,
                        },
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世".to_string(),
                        weights: DictWeight {
                            right: 40,
                            inside: 41,
                            left: 42,
                        },
                        comment: "".to_string(),
                    },
                ],
            },
            bias: -200,
            char_window_size: 3,
            type_window_size: 2,
        }
    }

    /// Input:  我  ら  は  全  世  界  の  国  民
    /// bias:   -285  ..  ..  ..  ..  ..  ..  ..
    /// words:
    ///   我ら:    2   3
    ///   全世界:              4   5
    ///   国民:                            6   7
    ///   世界:                9  10  11
    ///   界:                 12  13  14  15
    /// types:
    ///   H:      18  19  20  21
    ///           17  18  19  20  21
    ///                       16  17  18  19  20
    ///   K:      25  26  27
    ///           22  23  24  25  26  27
    ///               22  23  24  25  26  27
    ///                   22  23  24  25  26  27
    ///                           22  23  24  25
    ///                               22  23  24
    ///   KH:     30  31  32
    ///                       28  29  30  31  32
    ///   HK:     33  34  35  36  37
    ///                           33  34  35  36
    /// dict:
    ///   全世界:         44  45  45  46
    ///   世界:               41  42  43
    ///   世:                 38  40
    fn generate_model_2() -> Model {
        Model {
            char_ngram_model: NgramModel::new(vec![
                NgramData {
                    ngram: "我ら".to_string(),
                    weights: vec![1, 2, 3],
                },
                NgramData {
                    ngram: "全世界".to_string(),
                    weights: vec![4, 5],
                },
                NgramData {
                    ngram: "国民".to_string(),
                    weights: vec![6, 7, 8],
                },
                NgramData {
                    ngram: "世界".to_string(),
                    weights: vec![9, 10, 11],
                },
                NgramData {
                    ngram: "界".to_string(),
                    weights: vec![12, 13, 14, 15],
                },
            ]),
            type_ngram_model: NgramModel::new(vec![
                NgramData {
                    ngram: b"H".to_vec(),
                    weights: vec![16, 17, 18, 19, 20, 21],
                },
                NgramData {
                    ngram: b"K".to_vec(),
                    weights: vec![22, 23, 24, 25, 26, 27],
                },
                NgramData {
                    ngram: b"KH".to_vec(),
                    weights: vec![28, 29, 30, 31, 32],
                },
                NgramData {
                    ngram: b"HK".to_vec(),
                    weights: vec![33, 34, 35, 36, 37],
                },
            ]),
            dict_model: DictModel {
                dict: vec![
                    WordWeightRecord {
                        word: "全世界".to_string(),
                        weights: DictWeight {
                            right: 44,
                            inside: 45,
                            left: 46,
                        },
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世界".to_string(),
                        weights: DictWeight {
                            right: 41,
                            inside: 42,
                            left: 43,
                        },
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世".to_string(),
                        weights: DictWeight {
                            right: 38,
                            inside: 39,
                            left: 40,
                        },
                        comment: "".to_string(),
                    },
                ],
            },
            bias: -285,
            char_window_size: 2,
            type_window_size: 3,
        }
    }

    /// Input:  我  ら  は  全  世  界  の  国  民
    /// bias:   -285  ..  ..  ..  ..  ..  ..  ..
    /// words:
    ///   我ら:    2   3
    ///   全世界:              4   5
    ///   国民:                            6   7
    ///   世界:                9  10  11
    ///   界:                 12  13  14  15
    /// types:
    ///   H:      18  19  20  21
    ///           17  18  19  20  21
    ///                       16  17  18  19  20
    ///   K:      25  26  27
    ///           22  23  24  25  26  27
    ///               22  23  24  25  26  27
    ///                   22  23  24  25  26  27
    ///                           22  23  24  25
    ///                               22  23  24
    ///   KH:     30  31  32
    ///                       28  29  30  31  32
    ///   HK:     33  34  35  36  37
    ///                           33  34  35  36
    /// dict:
    ///   国民:                           38  39
    ///   世界:               41  42  43
    ///   世:                 44  46
    fn generate_model_3() -> Model {
        Model {
            char_ngram_model: NgramModel::new(vec![
                NgramData {
                    ngram: "我ら".to_string(),
                    weights: vec![1, 2, 3],
                },
                NgramData {
                    ngram: "全世界".to_string(),
                    weights: vec![4, 5],
                },
                NgramData {
                    ngram: "国民".to_string(),
                    weights: vec![6, 7, 8],
                },
                NgramData {
                    ngram: "世界".to_string(),
                    weights: vec![9, 10, 11],
                },
                NgramData {
                    ngram: "界".to_string(),
                    weights: vec![12, 13, 14, 15],
                },
            ]),
            type_ngram_model: NgramModel::new(vec![
                NgramData {
                    ngram: b"H".to_vec(),
                    weights: vec![16, 17, 18, 19, 20, 21],
                },
                NgramData {
                    ngram: b"K".to_vec(),
                    weights: vec![22, 23, 24, 25, 26, 27],
                },
                NgramData {
                    ngram: b"KH".to_vec(),
                    weights: vec![28, 29, 30, 31, 32],
                },
                NgramData {
                    ngram: b"HK".to_vec(),
                    weights: vec![33, 34, 35, 36, 37],
                },
            ]),
            dict_model: DictModel {
                dict: vec![
                    WordWeightRecord {
                        word: "国民".to_string(),
                        weights: DictWeight {
                            right: 38,
                            inside: 39,
                            left: 40,
                        },
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世界".to_string(),
                        weights: DictWeight {
                            right: 41,
                            inside: 42,
                            left: 43,
                        },
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世".to_string(),
                        weights: DictWeight {
                            right: 44,
                            inside: 45,
                            left: 46,
                        },
                        comment: "".to_string(),
                    },
                ],
            },
            bias: -285,
            char_window_size: 2,
            type_window_size: 3,
        }
    }

    #[test]
    fn test_predict_1() {
        let model = generate_model_1();
        let p = Predictor::new(model).unwrap();
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
    }

    #[test]
    fn test_predict_2() {
        let model = generate_model_2();
        let p = Predictor::new(model).unwrap();
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
    }

    #[test]
    fn test_predict_3() {
        let model = generate_model_3();
        let p = Predictor::new(model).unwrap();
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
    }

    #[test]
    fn test_predict_with_score_1() {
        let model = generate_model_1();
        let p = Predictor::new(model).unwrap();
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_with_score(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
        assert_eq!(
            &[-77, -5, 45, 132, 133, 144, 50, -32],
            s.boundary_scores().unwrap(),
        );
    }

    #[test]
    fn test_predict_with_score_2() {
        let model = generate_model_2();
        let p = Predictor::new(model).unwrap();
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_with_score(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
        assert_eq!(
            &[-138, -109, -39, 57, 104, 34, -79, -114],
            s.boundary_scores().unwrap(),
        );
    }

    #[test]
    fn test_predict_with_score_3() {
        let model = generate_model_3();
        let p = Predictor::new(model).unwrap();
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_with_score(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
        assert_eq!(
            &[-138, -109, -83, 18, 65, -12, -41, -75],
            s.boundary_scores().unwrap(),
        );
    }
}
