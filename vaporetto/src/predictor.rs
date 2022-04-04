use core::mem;

use alloc::vec::Vec;

#[cfg(feature = "tag-prediction")]
use core::cmp::Ordering;

#[cfg(feature = "tag-prediction")]
use alloc::string::String;
#[cfg(feature = "tag-prediction")]
use alloc::sync::Arc;

use bincode::{BorrowDecode, Encode};

use crate::char_scorer::{self, CharScorer};
use crate::errors::Result;
use crate::model::Model;
use crate::sentence::{BoundaryType, Sentence};
use crate::type_scorer::TypeScorer;

#[cfg(feature = "tag-prediction")]
use crate::char_scorer::CharScorerWithTags;

#[derive(BorrowDecode, Encode)]
enum CharScorerWrapper {
    Boundary(CharScorer),

    #[cfg(feature = "tag-prediction")]
    BoundaryAndTags(CharScorerWithTags),
}

/// Predictor.
pub struct Predictor {
    data: PredictorData,
}

/// WARNING: The decode feature is inherently unsafe. Do not publish this feature outside this
/// crate.
#[derive(BorrowDecode, Encode)]
struct PredictorData {
    bias: i32,

    char_scorer: CharScorerWrapper,
    type_scorer: TypeScorer,

    padding: u8,

    #[cfg(feature = "tag-prediction")]
    tag_names: Vec<Arc<String>>,
    #[cfg(feature = "tag-prediction")]
    tag_bias: Vec<i32>,
}

impl Predictor {
    /// Creates a new predictor.
    ///
    /// # Arguments
    ///
    /// * `model` - A model data.
    /// * `predict_tags` - If you want to predict tags, set to true.
    ///
    /// # Returns
    ///
    /// A new predictor.
    pub fn new(
        model: Model,
        #[cfg(feature = "tag-prediction")] predict_tags: bool,
        #[cfg(not(feature = "tag-prediction"))] _predict_tags: bool,
    ) -> Result<Self> {
        #[cfg(feature = "tag-prediction")]
        let mut tag_names = vec![];
        #[cfg(feature = "tag-prediction")]
        let mut tag_bias = vec![];

        #[cfg(feature = "tag-prediction")]
        let char_scorer = if predict_tags {
            for cls in model.data.tag_model.class_info {
                tag_names.push(Arc::new(cls.name));
                tag_bias.push(cls.bias);
            }
            CharScorerWrapper::BoundaryAndTags(CharScorerWithTags::new(
                model.data.char_ngram_model,
                model.data.char_window_size,
                model.data.dict_model,
                tag_names.len(),
                model.data.tag_model.left_char_model,
                model.data.tag_model.right_char_model,
                model.data.tag_model.self_char_model,
            )?)
        } else {
            CharScorerWrapper::Boundary(CharScorer::new(
                model.data.char_ngram_model,
                model.data.char_window_size,
                model.data.dict_model,
            )?)
        };

        #[cfg(not(feature = "tag-prediction"))]
        let char_scorer = CharScorerWrapper::Boundary(CharScorer::new(
            model.data.char_ngram_model,
            model.data.char_window_size,
            model.data.dict_model,
        )?);

        let type_scorer =
            TypeScorer::new(model.data.type_ngram_model, model.data.type_window_size)?;

        Ok(Self {
            data: PredictorData {
                bias: model.data.bias,

                char_scorer,
                type_scorer,

                padding: model.data.char_window_size.max(model.data.type_window_size),

                #[cfg(feature = "tag-prediction")]
                tag_names,
                #[cfg(feature = "tag-prediction")]
                tag_bias,
            },
        })
    }

    /// Serializes the predictor into a Vec.
    pub fn serialize_to_vec(&self) -> Result<Vec<u8>> {
        let config = bincode::config::standard();
        let result = bincode::encode_to_vec(&self.data, config)?;
        Ok(result)
    }

    /// Deserializes a predictor from a given slice.
    ///
    /// # Arguments
    ///
    /// * `data` - A source slice.
    ///
    /// # Returns
    ///
    /// A tuple of a predictor and a slice not used for the deserialization.
    ///
    /// # Safety
    ///
    /// The given data must be a correct predictor exported by [`Predictor::serialize_to_vec()`]
    /// function.
    pub unsafe fn deserialize_from_slice_unchecked(data: &[u8]) -> Result<(Self, &[u8])> {
        let config = bincode::config::standard();
        // Deserialization is unsafe because the automaton will not be verified.
        let (predictor_data, size) = bincode::decode_from_slice(data, config)?;
        Ok((
            Self {
                data: predictor_data,
            },
            &data[size..],
        ))
    }

    fn predict_impl(&self, mut sentence: Sentence) -> Sentence {
        let ys_size =
            sentence.boundaries.len() + usize::from(self.data.padding) + char_scorer::SIMD_SIZE - 1;
        let mut ys = mem::take(&mut sentence.boundary_scores);
        ys.clear();
        ys.resize(ys_size, self.data.bias);
        match &self.data.char_scorer {
            CharScorerWrapper::Boundary(char_scorer) => {
                char_scorer.add_scores(&sentence, self.data.padding, &mut ys);
            }

            #[cfg(feature = "tag-prediction")]
            CharScorerWrapper::BoundaryAndTags(char_scorer) => {
                let mut tag_ys = mem::take(&mut sentence.tag_scores);
                tag_ys.init(sentence.chars.len(), self.data.tag_names.len());
                char_scorer.add_scores(&sentence, self.data.padding, &mut ys, &mut tag_ys);
                sentence.tag_scores = tag_ys;
            }
        }
        self.data
            .type_scorer
            .add_scores(&sentence, self.data.padding, &mut ys);
        for (&y, b) in ys[self.data.padding.into()..]
            .iter()
            .zip(sentence.boundaries.iter_mut())
        {
            *b = if y >= 0 {
                BoundaryType::WordBoundary
            } else {
                BoundaryType::NotWordBoundary
            };
        }
        sentence.boundary_scores = ys;
        sentence
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
    pub fn predict(&self, sentence: Sentence) -> Sentence {
        let mut sentence = self.predict_impl(sentence);
        sentence.boundary_scores.clear();
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
    pub fn predict_with_score(&self, sentence: Sentence) -> Sentence {
        let mut sentence = self.predict_impl(sentence);
        sentence
            .boundary_scores
            .rotate_left(self.data.padding.into());
        sentence.boundary_scores.truncate(sentence.boundaries.len());
        sentence
    }

    #[cfg(feature = "tag-prediction")]
    fn best_tag(&self, scores: &[i32]) -> Arc<String> {
        Arc::clone(
            scores
                .iter()
                .zip(&self.data.tag_names)
                .max_by_key(|(&x, _)| x)
                .unwrap()
                .1,
        )
    }

    /// Fills tags using calculated scores.
    ///
    /// Tags are predicted using token boundaries, so you have to apply boundary post-processors
    /// before filling tags.
    ///
    /// # Arguments
    ///
    /// * `sentence` - A sentence.
    ///
    /// # Returns
    ///
    /// A sentence with tag information. When the predictor is instantiated with
    /// `predict_tag = false`, the sentence is returned without any modification.
    #[cfg(feature = "tag-prediction")]
    #[cfg_attr(docsrs, doc(cfg(feature = "tag-prediction")))]
    pub fn fill_tags(&self, mut sentence: Sentence) -> Sentence {
        if self.data.tag_names.is_empty() {
            return sentence;
        }
        if sentence.tags.is_empty() {
            sentence.tags.resize(sentence.chars().len(), None);
        }
        let n_tags = self.data.tag_names.len();
        let mut tag_score = self.data.tag_bias.clone();
        let mut left_scores_iter = sentence.tag_scores.left_scores.chunks(n_tags);
        for (t, l) in tag_score.iter_mut().zip(left_scores_iter.next().unwrap()) {
            *t += l;
        }
        let mut right_scores_iter = sentence.tag_scores.right_scores.chunks(n_tags);
        let mut last_boundary_idx = 0;
        for (i, ((((b, left_scores), right_scores), self_scores), tag)) in sentence
            .boundaries
            .iter()
            .zip(left_scores_iter)
            .zip(&mut right_scores_iter)
            .zip(&sentence.tag_scores.self_scores)
            .zip(&mut sentence.tags)
            .enumerate()
        {
            if *b == BoundaryType::WordBoundary {
                for (t, r) in tag_score.iter_mut().zip(right_scores) {
                    *t += *r;
                }
                if let Some(self_weights) = self_scores.as_ref() {
                    let diff = -i16::try_from(i + 1 - last_boundary_idx).unwrap_or(0);
                    for self_weight in self_weights.iter() {
                        match self_weight.start_rel_position.cmp(&diff) {
                            Ordering::Greater => continue,
                            Ordering::Equal => {
                                for (t, s) in tag_score.iter_mut().zip(&self_weight.weight) {
                                    *t += *s;
                                }
                            }
                            Ordering::Less => (),
                        }
                        break;
                    }
                }
                tag.replace(self.best_tag(&tag_score));
                for (t, (l, b)) in tag_score
                    .iter_mut()
                    .zip(left_scores.iter().zip(&self.data.tag_bias))
                {
                    *t = *l + *b;
                }
                last_boundary_idx = i + 1;
            }
        }
        for (t, r) in tag_score.iter_mut().zip(right_scores_iter.next().unwrap()) {
            *t += r;
        }
        if let Some(self_weights) = sentence.tag_scores.self_scores.last().unwrap().as_ref() {
            let diff = -i16::try_from(sentence.chars.len() - last_boundary_idx).unwrap_or(0);
            for self_weight in self_weights.iter() {
                match self_weight.start_rel_position.cmp(&diff) {
                    Ordering::Greater => continue,
                    Ordering::Equal => {
                        for (t, s) in tag_score.iter_mut().zip(&self_weight.weight) {
                            *t += *s;
                        }
                    }
                    Ordering::Less => (),
                }
                break;
            }
        }
        sentence
            .tags
            .last_mut()
            .unwrap()
            .replace(self.best_tag(&tag_score));

        sentence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::string::ToString;

    use crate::dict_model::{DictModel, WordWeightRecord};
    use crate::ngram_model::{NgramData, NgramModel};
    use crate::sentence::CharacterType::*;
    use crate::tag_model::TagModel;

    #[cfg(feature = "tag-prediction")]
    use crate::sentence::Token;
    #[cfg(feature = "tag-prediction")]
    use crate::tag_model::TagClassInfo;

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
        Model::new(
            NgramModel {
                data: vec![
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
                ],
            },
            NgramModel {
                data: vec![
                    NgramData {
                        ngram: vec![Hiragana as u8],
                        weights: vec![26, 27, 28, 29],
                    },
                    NgramData {
                        ngram: vec![Kanji as u8],
                        weights: vec![30, 31, 32, 33],
                    },
                    NgramData {
                        ngram: vec![Kanji as u8, Hiragana as u8],
                        weights: vec![34, 35, 36],
                    },
                    NgramData {
                        ngram: vec![Hiragana as u8, Kanji as u8],
                        weights: vec![37, 38, 39],
                    },
                ],
            },
            DictModel {
                dict: vec![
                    WordWeightRecord {
                        word: "全世界".to_string(),
                        weights: vec![43, 44, 44, 45],
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世界".to_string(),
                        weights: vec![43, 44, 45],
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世".to_string(),
                        weights: vec![40, 42],
                        comment: "".to_string(),
                    },
                ],
            },
            -200,
            3,
            2,
            TagModel::default(),
        )
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
        Model::new(
            NgramModel {
                data: vec![
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
                ],
            },
            NgramModel {
                data: vec![
                    NgramData {
                        ngram: vec![Hiragana as u8],
                        weights: vec![16, 17, 18, 19, 20, 21],
                    },
                    NgramData {
                        ngram: vec![Kanji as u8],
                        weights: vec![22, 23, 24, 25, 26, 27],
                    },
                    NgramData {
                        ngram: vec![Kanji as u8, Hiragana as u8],
                        weights: vec![28, 29, 30, 31, 32],
                    },
                    NgramData {
                        ngram: vec![Hiragana as u8, Kanji as u8],
                        weights: vec![33, 34, 35, 36, 37],
                    },
                ],
            },
            DictModel {
                dict: vec![
                    WordWeightRecord {
                        word: "全世界".to_string(),
                        weights: vec![44, 45, 45, 46],
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世界".to_string(),
                        weights: vec![41, 42, 43],
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世".to_string(),
                        weights: vec![38, 40],
                        comment: "".to_string(),
                    },
                ],
            },
            -285,
            2,
            3,
            TagModel::default(),
        )
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
        Model::new(
            NgramModel {
                data: vec![
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
                ],
            },
            NgramModel {
                data: vec![
                    NgramData {
                        ngram: vec![Hiragana as u8],
                        weights: vec![16, 17, 18, 19, 20, 21],
                    },
                    NgramData {
                        ngram: vec![Kanji as u8],
                        weights: vec![22, 23, 24, 25, 26, 27],
                    },
                    NgramData {
                        ngram: vec![Kanji as u8, Hiragana as u8],
                        weights: vec![28, 29, 30, 31, 32],
                    },
                    NgramData {
                        ngram: vec![Hiragana as u8, Kanji as u8],
                        weights: vec![33, 34, 35, 36, 37],
                    },
                ],
            },
            DictModel {
                dict: vec![
                    WordWeightRecord {
                        word: "国民".to_string(),
                        weights: vec![38, 39, 40],
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世界".to_string(),
                        weights: vec![41, 42, 43],
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世".to_string(),
                        weights: vec![44, 46],
                        comment: "".to_string(),
                    },
                ],
            },
            -285,
            2,
            3,
            TagModel::default(),
        )
    }

    /// Input:  我  ら  は  全  世  界  の  国  民
    /// bias:   -200  ..  ..  ..  ..  ..  ..  ..
    /// chars:
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
    ///   世界の国民:         43  44  44  44  44
    ///   は全世界:   43  44  44  44  45
    ///
    ///
    ///   は全世界:   43  44  44  44  45
    ///                   15  16  17  18  19
    ///                   20  21  22  23  24  25
    ///                    6   7   8   9
    fn generate_model_4() -> Model {
        Model::new(
            NgramModel {
                data: vec![
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
                ],
            },
            NgramModel {
                data: vec![
                    NgramData {
                        ngram: vec![Hiragana as u8],
                        weights: vec![26, 27, 28, 29],
                    },
                    NgramData {
                        ngram: vec![Kanji as u8],
                        weights: vec![30, 31, 32, 33],
                    },
                    NgramData {
                        ngram: vec![Kanji as u8, Hiragana as u8],
                        weights: vec![34, 35, 36],
                    },
                    NgramData {
                        ngram: vec![Hiragana as u8, Kanji as u8],
                        weights: vec![37, 38, 39],
                    },
                ],
            },
            DictModel {
                dict: vec![
                    WordWeightRecord {
                        word: "全世界".to_string(),
                        weights: vec![43, 44, 44, 45],
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世界".to_string(),
                        weights: vec![43, 44, 45],
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世".to_string(),
                        weights: vec![40, 42],
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "世界の国民".to_string(),
                        weights: vec![43, 44, 44, 44, 44, 45],
                        comment: "".to_string(),
                    },
                    WordWeightRecord {
                        word: "は全世界".to_string(),
                        weights: vec![43, 44, 44, 44, 45],
                        comment: "".to_string(),
                    },
                ],
            },
            -200,
            3,
            2,
            TagModel::default(),
        )
    }

    /// Input:  人  と  人  を  つ  な  ぐ  人
    /// left:
    ///   \0人: 1   4
    ///         2   5
    ///         3   6
    ///     人:     7  10   7  10
    ///             8  11   8  11
    ///             9  12   9  12
    /// つなぐ:                    13  16  19
    ///                            14  17  20
    ///                            15  18  21
    ///   人\0:                            22
    ///                                    23
    ///                                    24
    ///
    ///    sum: 1  11  10   7  10  13  16  41
    ///         2  13  11   8  11  14  17  43
    ///         3  15  12   9  12  15  18  45
    ///
    /// right:
    /// \0人と:  28
    ///          29
    ///          30
    ///   人を:      31  34  37
    ///              32  35  38
    ///              33  36  39
    ///     を:      40  43
    ///              41  44
    ///              42  45
    ///   人\0:                          46  49
    ///                                  47  50
    ///                                  48  51
    ///
    ///     sum: 28  71  77  37   0   0  46  49
    ///          29  73  79  38   0   0  47  50
    ///          30  75  81  39   0   0  48  51
    #[cfg(feature = "tag-prediction")]
    fn generate_model_5() -> Model {
        Model::new(
            NgramModel {
                data: vec![NgramData {
                    ngram: "xxxx".to_string(),
                    weights: vec![0],
                }],
            },
            NgramModel {
                data: vec![NgramData {
                    ngram: vec![Roman as u8, Roman as u8, Roman as u8, Roman as u8],
                    weights: vec![0],
                }],
            },
            DictModel { dict: vec![] },
            0,
            2,
            2,
            TagModel {
                class_info: vec![
                    TagClassInfo {
                        name: "名詞".to_string(),
                        bias: 5,
                    },
                    TagClassInfo {
                        name: "動詞".to_string(),
                        bias: 3,
                    },
                    TagClassInfo {
                        name: "助詞".to_string(),
                        bias: 1,
                    },
                ],
                left_char_model: NgramModel {
                    data: vec![
                        NgramData {
                            ngram: "\0人".to_string(),
                            weights: vec![1, 2, 3, 4, 5, 6],
                        },
                        NgramData {
                            ngram: "人".to_string(),
                            weights: vec![7, 8, 9, 10, 11, 12],
                        },
                        NgramData {
                            ngram: "つなぐ".to_string(),
                            weights: vec![13, 14, 15, 16, 17, 18, 19, 20, 21],
                        },
                        NgramData {
                            ngram: "ぐ人\0".to_string(),
                            weights: vec![22, 23, 24],
                        },
                    ],
                },
                right_char_model: NgramModel {
                    data: vec![
                        NgramData {
                            ngram: "\0人と".to_string(),
                            weights: vec![25, 26, 27, 28, 29, 30],
                        },
                        NgramData {
                            ngram: "人を".to_string(),
                            weights: vec![31, 32, 33, 34, 35, 36, 37, 38, 39],
                        },
                        NgramData {
                            ngram: "を".to_string(),
                            weights: vec![40, 41, 42, 43, 44, 45],
                        },
                        NgramData {
                            ngram: "人\0".to_string(),
                            weights: vec![46, 47, 48, 49, 50, 51],
                        },
                    ],
                },
                self_char_model: NgramModel {
                    data: vec![
                        NgramData {
                            ngram: "人".to_string(),
                            weights: vec![2, -1, -1],
                        },
                        NgramData {
                            ngram: "と".to_string(),
                            weights: vec![0, 0, 0],
                        },
                        NgramData {
                            ngram: "つなぐ".to_string(),
                            weights: vec![0, 1, 0],
                        },
                        NgramData {
                            ngram: "を".to_string(),
                            weights: vec![0, 0, 0],
                        },
                    ],
                },
            },
        )
    }

    #[test]
    fn test_predict_1() {
        let model = generate_model_1();
        let p = Predictor::new(model, false).unwrap();
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
        let p = Predictor::new(model, false).unwrap();
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
        let p = Predictor::new(model, false).unwrap();
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
    fn test_predict_4() {
        let model = generate_model_4();
        let p = Predictor::new(model, false).unwrap();
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
            ],
            s.boundaries(),
        );
    }

    #[test]
    fn test_predict_with_score_1() {
        let model = generate_model_1();
        let p = Predictor::new(model, false).unwrap();
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_with_score(s);
        assert_eq!(&[-77, -5, 45, 132, 133, 144, 50, -32], s.boundary_scores(),);
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
    fn test_predict_with_score_2() {
        let model = generate_model_2();
        let p = Predictor::new(model, false).unwrap();
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_with_score(s);
        assert_eq!(
            &[-138, -109, -39, 57, 104, 34, -79, -114],
            s.boundary_scores(),
        );
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
    fn test_predict_with_score_3() {
        let model = generate_model_3();
        let p = Predictor::new(model, false).unwrap();
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_with_score(s);
        assert_eq!(
            &[-138, -109, -83, 18, 65, -12, -41, -75],
            s.boundary_scores(),
        );
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
    fn test_predict_with_score_4() {
        let model = generate_model_4();
        let p = Predictor::new(model, false).unwrap();
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_with_score(s);
        assert_eq!(&[-77, 38, 89, 219, 221, 233, 94, 12], s.boundary_scores(),);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
            ],
            s.boundaries(),
        );
    }

    #[cfg(feature = "tag-prediction")]
    #[test]
    fn test_predict_with_score_5() {
        let model = generate_model_5();
        let p = Predictor::new(model, true).unwrap();
        let s = Sentence::from_raw("人と人をつなぐ人").unwrap();
        let mut s = p.predict(s);
        assert_eq!(
            &[
                1, 2, 3, 11, 13, 15, 10, 11, 12, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 41,
                43, 45
            ],
            s.tag_scores.left_scores.as_slice()
        );
        assert_eq!(
            &[
                28, 29, 30, 71, 73, 75, 77, 79, 81, 37, 38, 39, 0, 0, 0, 0, 0, 0, 46, 47, 48, 49,
                50, 51
            ],
            s.tag_scores.right_scores.as_slice()
        );

        s.boundaries_mut().copy_from_slice(&[
            BoundaryType::WordBoundary,
            BoundaryType::WordBoundary,
            BoundaryType::WordBoundary,
            BoundaryType::WordBoundary,
            BoundaryType::NotWordBoundary,
            BoundaryType::NotWordBoundary,
            BoundaryType::WordBoundary,
        ]);
        let s = p.fill_tags(s);

        assert_eq!(
            vec![
                Token {
                    surface: "人",
                    tag: Some("名詞")
                },
                Token {
                    surface: "と",
                    tag: Some("助詞")
                },
                Token {
                    surface: "人",
                    tag: Some("名詞")
                },
                Token {
                    surface: "を",
                    tag: Some("助詞")
                },
                Token {
                    surface: "つなぐ",
                    tag: Some("動詞")
                },
                Token {
                    surface: "人",
                    tag: Some("名詞")
                }
            ],
            s.to_tokenized_vec().unwrap(),
        );
    }
}
