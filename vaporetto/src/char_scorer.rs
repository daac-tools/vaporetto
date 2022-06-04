mod boundary_scorer;

#[cfg(feature = "tag-prediction")]
mod boundary_tag_scorer;

use core::cell::RefCell;
use core::ops::AddAssign;

use alloc::collections::BTreeMap;
use alloc::string::String;
use alloc::vec::Vec;

use bincode::{BorrowDecode, Encode};

use crate::dict_model::DictModel;
use crate::errors::Result;
use crate::ngram_model::NgramModel;
use crate::sentence::Sentence;

#[cfg(feature = "tag-prediction")]
use crate::ngram_model::TagNgramModel;

use boundary_scorer::CharScorerBoundary;

#[cfg(feature = "tag-prediction")]
use boundary_tag_scorer::CharScorerBoundaryTag;

#[derive(Default)]
struct CharWeightMerger<W> {
    map: BTreeMap<String, RefCell<(W, bool)>>,
}

impl<W> CharWeightMerger<W>
where
    for<'a> W: AddAssign<&'a W>,
{
    pub fn add<S>(&mut self, ngram: S, weight: W)
    where
        S: Into<String> + AsRef<str>,
    {
        if let Some(data) = self.map.get_mut(ngram.as_ref()) {
            let (prev_weight, _) = &mut *data.borrow_mut();
            *prev_weight += &weight;
        } else {
            self.map.insert(ngram.into(), RefCell::new((weight, false)));
        }
    }

    #[must_use]
    pub fn merge(self) -> Vec<(String, W)> {
        let mut stack = vec![];
        for (ngram, data) in &self.map {
            if data.borrow().1 {
                continue;
            }
            stack.push(data);
            for (j, _) in ngram.char_indices().skip(1) {
                if let Some(data) = self.map.get(&ngram[j..]) {
                    stack.push(data);
                    if data.borrow().1 {
                        break;
                    }
                }
            }
            let mut data_from = stack.pop().unwrap();
            data_from.borrow_mut().1 = true;
            while let Some(data_to) = stack.pop() {
                let data_to_ref = &mut data_to.borrow_mut();
                data_to_ref.1 = true;
                data_to_ref.0 += &data_from.borrow().0;
                data_from = data_to;
            }
        }
        self.map
            .into_iter()
            .map(|(ngram, weight)| (ngram, weight.into_inner().0))
            .collect()
    }
}

/// WARNING: Decoding is inherently unsafe. Do not publish this struct outside this
/// crate.
#[derive(BorrowDecode, Encode)]
pub enum CharScorer {
    Boundary(CharScorerBoundary),

    #[cfg(feature = "tag-prediction")]
    BoundaryTag(CharScorerBoundaryTag),
}

impl CharScorer {
    pub fn new(
        ngram_model: NgramModel<String>,
        dict_model: DictModel,
        window_size: u8,
        #[cfg(feature = "tag-prediction")] tag_ngram_model: Vec<TagNgramModel<String>>,
    ) -> Result<Option<Self>> {
        if ngram_model.0.is_empty() && dict_model.0.is_empty() || window_size == 0 {
            return Ok(None);
        }

        #[cfg(feature = "tag-prediction")]
        if tag_ngram_model.is_empty() {
            Ok(Some(Self::Boundary(CharScorerBoundary::new(
                ngram_model,
                dict_model,
                window_size,
            )?)))
        } else {
            Ok(Some(Self::BoundaryTag(CharScorerBoundaryTag::new(
                ngram_model,
                dict_model,
                window_size,
                tag_ngram_model,
            )?)))
        }

        #[cfg(not(feature = "tag-prediction"))]
        Ok(Some(Self::Boundary(CharScorerBoundary::new(
            ngram_model,
            dict_model,
            window_size,
        )?)))
    }

    #[inline]
    pub fn add_scores<'a, 'b>(&self, sentence: &mut Sentence<'a, 'b>) {
        match self {
            Self::Boundary(scorer) => scorer.add_scores(sentence),

            #[cfg(feature = "tag-prediction")]
            Self::BoundaryTag(scorer) => scorer.add_scores(sentence),
        }
    }

    /// # Satety
    ///
    /// `token_id` must be smaller than `scorer.tag_weight.len()`.
    /// `pos` must be smaller than `sentence.char_pma_states.len()`.
    #[cfg(feature = "tag-prediction")]
    #[inline]
    pub unsafe fn add_tag_scores(
        &self,
        token_id: u32,
        pos: usize,
        sentence: &Sentence,
        scores: &mut [i32],
    ) {
        match self {
            Self::Boundary(_) => panic!("unsupported"),
            Self::BoundaryTag(scorer) => scorer.add_tag_scores(token_id, pos, sentence, scores),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dict_model::WordWeightRecord;
    use crate::ngram_model::{NgramData, TagWeight};
    use crate::predictor::PositionalWeight;

    use crate::predictor::WEIGHT_FIXED_LEN;

    #[cfg(feature = "tag-prediction")]
    use crate::ngram_model::TagNgramData;

    #[rustfmt::skip]
    #[test]
    fn test_weight_merger() {
        let mut merger = CharWeightMerger::default();
        merger.add("東京都".into(), PositionalWeight::new(-3, vec![1, 2, 3, 4]));
        merger.add("京都".into(), PositionalWeight::new(-3, vec![2, 4, 6, 8, 10]));
        merger.add("京都".into(), PositionalWeight::new(-2, vec![3, 6, 9]));
        merger.add("大阪".into(), PositionalWeight::new(-2, vec![4, 8, 12]));
        assert_eq!(
            vec![
                ("京都".into(), PositionalWeight::new(-3, vec![2, 7, 12, 17, 10])),
                ("大阪".into(), PositionalWeight::new(-2, vec![4, 8, 12])),
                ("東京都".into(), PositionalWeight::new(-3, vec![3, 9, 15, 21, 10])),
            ],
            merger.merge(),
        );
    }

    #[test]
    fn test_add_scores_1() {
        // input:  我  ら  は  全  世  界  の  国  民
        // n-grams:
        //   我ら:    3   4   5
        //   全世界:          6   7   8   9
        //   国民:                       10  11  12
        //   世界:           15  16  17  18  19
        //   界:             20  21  22  23  24  25
        // dict:
        //   全世界:         26  27  28  29
        //   世界:               30  31  32
        //   世:                 33  34
        let scorer = CharScorerBoundary::new(
            NgramModel(vec![
                NgramData {
                    ngram: "我ら".into(),
                    weights: vec![1, 2, 3, 4, 5],
                },
                NgramData {
                    ngram: "全世界".into(),
                    weights: vec![6, 7, 8, 9],
                },
                NgramData {
                    ngram: "国民".into(),
                    weights: vec![10, 11, 12, 13, 14],
                },
                NgramData {
                    ngram: "世界".into(),
                    weights: vec![15, 16, 17, 18, 19],
                },
                NgramData {
                    ngram: "界".into(),
                    weights: vec![20, 21, 22, 23, 24, 25],
                },
            ]),
            DictModel(vec![
                WordWeightRecord {
                    word: "全世界".into(),
                    weights: vec![26, 27, 28, 29],
                    comment: "".into(),
                },
                WordWeightRecord {
                    word: "世界".into(),
                    weights: vec![30, 31, 32],
                    comment: "".into(),
                },
                WordWeightRecord {
                    word: "世".into(),
                    weights: vec![33, 34],
                    comment: "".into(),
                },
            ]),
            3,
        )
        .unwrap();
        let mut sentence = Sentence::from_raw("我らは全世界の国民").unwrap();
        sentence.score_padding = WEIGHT_FIXED_LEN - 1;
        sentence.boundary_scores.clear();
        sentence
            .boundary_scores
            .resize(sentence.score_padding * 2 + sentence.len() - 1, 1);
        scorer.add_scores(&mut sentence);
        assert_eq!(
            &[4, 5, 73, 135, 141, 122, 55, 38],
            sentence.boundary_scores(),
        );
    }

    #[test]
    fn test_add_scores_2() {
        // input:  我  ら  は  全  世  界  の  国  民
        // n-grams:
        //   我ら:    2   3
        //   全世界:              4   5
        //   国民:                            6   7
        //   世界:                9  10  11
        //   界:                 12  13  14  15
        // dict:
        //   全世界:         16  17  18  19
        //   世界:               20  21  22
        //   世:                 23  24
        let scorer = CharScorerBoundary::new(
            NgramModel(vec![
                NgramData {
                    ngram: "我ら".into(),
                    weights: vec![1, 2, 3],
                },
                NgramData {
                    ngram: "全世界".into(),
                    weights: vec![4, 5],
                },
                NgramData {
                    ngram: "国民".into(),
                    weights: vec![6, 7, 8],
                },
                NgramData {
                    ngram: "世界".into(),
                    weights: vec![9, 10, 11],
                },
                NgramData {
                    ngram: "界".into(),
                    weights: vec![12, 13, 14, 15],
                },
            ]),
            DictModel(vec![
                WordWeightRecord {
                    word: "全世界".into(),
                    weights: vec![16, 17, 18, 19],
                    comment: "".into(),
                },
                WordWeightRecord {
                    word: "世界".into(),
                    weights: vec![20, 21, 22],
                    comment: "".into(),
                },
                WordWeightRecord {
                    word: "世".into(),
                    weights: vec![23, 24],
                    comment: "".into(),
                },
            ]),
            2,
        )
        .unwrap();
        let mut sentence = Sentence::from_raw("我らは全世界の国民").unwrap();
        sentence.score_padding = WEIGHT_FIXED_LEN - 1;
        sentence.boundary_scores.clear();
        sentence
            .boundary_scores
            .resize(sentence.score_padding * 2 + sentence.len() - 1, 2);
        scorer.add_scores(&mut sentence);
        assert_eq!(&[4, 5, 18, 87, 93, 68, 23, 9], sentence.boundary_scores(),);
    }

    #[test]
    fn test_add_scores_3() {
        // input:  我  ら  は  全  世  界  の  国  民
        // n-grams:
        //   我ら:    3   4   5
        //   全世界:          6   7   8   9
        //   国民:                       10  11  12
        //   世界:           15  16  17  18  19
        //   界:             20  21  22  23  24  25
        // dict:
        //   全世界:         26  27  28  29
        //   世界:               30  31  32
        //   世:                 33  34
        //   世界の国民:         35  36  37  38  39
        //   は全世界:   41  42  43  44  45
        let scorer = CharScorerBoundary::new(
            NgramModel(vec![
                NgramData {
                    ngram: "我ら".into(),
                    weights: vec![1, 2, 3, 4, 5],
                },
                NgramData {
                    ngram: "全世界".into(),
                    weights: vec![6, 7, 8, 9],
                },
                NgramData {
                    ngram: "国民".into(),
                    weights: vec![10, 11, 12, 13, 14],
                },
                NgramData {
                    ngram: "世界".into(),
                    weights: vec![15, 16, 17, 18, 19],
                },
                NgramData {
                    ngram: "界".into(),
                    weights: vec![20, 21, 22, 23, 24, 25],
                },
            ]),
            DictModel(vec![
                WordWeightRecord {
                    word: "全世界".into(),
                    weights: vec![26, 27, 28, 29],
                    comment: "".into(),
                },
                WordWeightRecord {
                    word: "世界".into(),
                    weights: vec![30, 31, 32],
                    comment: "".into(),
                },
                WordWeightRecord {
                    word: "世".into(),
                    weights: vec![33, 34],
                    comment: "".into(),
                },
                WordWeightRecord {
                    word: "世界の国民".into(),
                    weights: vec![35, 36, 37, 38, 39, 40],
                    comment: "".into(),
                },
                WordWeightRecord {
                    word: "は全世界".into(),
                    weights: vec![41, 42, 43, 44, 45],
                    comment: "".into(),
                },
            ]),
            3,
        )
        .unwrap();
        let mut sentence = Sentence::from_raw("我らは全世界の国民").unwrap();
        sentence.score_padding = WEIGHT_FIXED_LEN - 1;
        sentence.boundary_scores.clear();
        sentence
            .boundary_scores
            .resize(sentence.score_padding * 2 + sentence.len() - 1, 3);
        scorer.add_scores(&mut sentence);
        assert_eq!(
            &[6, 48, 117, 215, 223, 206, 95, 79],
            sentence.boundary_scores(),
        );
    }

    #[cfg(feature = "tag-prediction")]
    #[test]
    fn test_add_scores_with_tags() {
        // input:    こ  の  人  は  火  星  人  だ
        // n-grams:
        //   この人:    2   3   4
        //   人だ:                      5   6   7
        // dict:
        //   人:           10  11          10  11
        //   火星:                 12  13  14
        let scorer = CharScorerBoundaryTag::new(
            NgramModel(vec![
                NgramData {
                    ngram: "この人".into(),
                    weights: vec![1, 2, 3, 4],
                },
                NgramData {
                    ngram: "人だ".into(),
                    weights: vec![5, 6, 7, 8, 9],
                },
            ]),
            DictModel(vec![
                WordWeightRecord {
                    word: "人".into(),
                    weights: vec![10, 11],
                    comment: "".into(),
                },
                WordWeightRecord {
                    word: "火星".into(),
                    weights: vec![12, 13, 14],
                    comment: "".into(),
                },
            ]),
            3,
            vec![
                TagNgramModel(vec![
                    TagNgramData {
                        ngram: "の人".into(),
                        weights: vec![
                            TagWeight {
                                rel_position: 0,
                                weights: vec![15, 16, 17],
                            },
                            TagWeight {
                                rel_position: 1,
                                weights: vec![18, 19, 20],
                            },
                        ],
                    },
                    TagNgramData {
                        ngram: "人は".into(),
                        weights: vec![
                            TagWeight {
                                rel_position: 1,
                                weights: vec![21, 22, 23],
                            },
                            TagWeight {
                                rel_position: 3,
                                weights: vec![24, 25, 26],
                            },
                        ],
                    },
                    TagNgramData {
                        ngram: "火星人".into(),
                        weights: vec![TagWeight {
                            rel_position: 0,
                            weights: vec![27, 28, 29],
                        }],
                    },
                ]),
                TagNgramModel(vec![]),
                TagNgramModel(vec![
                    TagNgramData {
                        ngram: "人は".into(),
                        weights: vec![
                            TagWeight {
                                rel_position: 0,
                                weights: vec![27, 28],
                            },
                            TagWeight {
                                rel_position: 3,
                                weights: vec![29, 30],
                            },
                        ],
                    },
                    TagNgramData {
                        ngram: "は火星人".into(),
                        weights: vec![TagWeight {
                            rel_position: 3,
                            weights: vec![31, 32],
                        }],
                    },
                ]),
            ],
        )
        .unwrap();
        let mut sentence = Sentence::from_raw("この人は火星人だ").unwrap();
        sentence.score_padding = WEIGHT_FIXED_LEN - 1;
        sentence.boundary_scores.clear();
        sentence
            .boundary_scores
            .resize(sentence.score_padding * 2 + sentence.len() - 1, 1);
        scorer.add_scores(&mut sentence);
        assert_eq!(&[3, 14, 16, 13, 19, 31, 19], sentence.boundary_scores());

        let mut tag_scores = [1; 8];
        unsafe {
            scorer.add_tag_scores(0, 2, &sentence, &mut tag_scores);
        }
        assert_eq!(&[37, 39, 41, 1, 1, 1, 1, 1], &tag_scores);

        let mut tag_scores = [1; 8];
        unsafe {
            scorer.add_tag_scores(0, 6, &sentence, &mut tag_scores);
        }
        assert_eq!(&[28, 29, 30, 1, 1, 1, 1, 1], &tag_scores);

        let mut tag_scores = [1; 8];
        unsafe {
            scorer.add_tag_scores(2, 3, &sentence, &mut tag_scores);
        }
        assert_eq!(&[59, 61, 1, 1, 1, 1, 1, 1], &tag_scores);
    }
}
