mod boundary_scorer;

#[cfg(feature = "tag-prediction")]
mod boundary_tag_scorer;

#[cfg(feature = "cache-type-score")]
mod boundary_scorer_cache;

use core::cell::RefCell;
use core::ops::AddAssign;

use alloc::collections::BTreeMap;
use alloc::vec::Vec;

use bincode::{BorrowDecode, Encode};

use crate::errors::Result;
use crate::ngram_model::NgramModel;
use crate::sentence::Sentence;

#[cfg(feature = "tag-prediction")]
use crate::ngram_model::TagNgramModel;

use boundary_scorer::TypeScorerBoundary;

#[cfg(feature = "cache-type-score")]
use boundary_scorer_cache::TypeScorerBoundaryCache;

#[cfg(feature = "tag-prediction")]
use boundary_tag_scorer::TypeScorerBoundaryTag;

// If the cache-type-score feature is enabled and the window size of character type features is
// less than or equal to this value, character type scores are cached.
const CACHE_MAX_WINDOW_SIZE: u8 = 3;

#[derive(Default)]
struct TypeWeightMerger<W> {
    map: BTreeMap<Vec<u8>, RefCell<(W, bool)>>,
}

impl<W> TypeWeightMerger<W>
where
    for<'a> W: AddAssign<&'a W>,
{
    pub fn add(&mut self, ngram: Vec<u8>, weight: W) {
        if let Some(data) = self.map.get_mut(&ngram) {
            let (prev_weight, _) = &mut *data.borrow_mut();
            *prev_weight += &weight;
        } else {
            self.map.insert(ngram, RefCell::new((weight, false)));
        }
    }

    #[must_use]
    pub fn merge(self) -> Vec<(Vec<u8>, W)> {
        let mut stack = vec![];
        for (ngram, data) in &self.map {
            if data.borrow().1 {
                continue;
            }
            stack.push(data);
            for j in 1..ngram.len() {
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
pub enum TypeScorer {
    Nop,

    Boundary(TypeScorerBoundary),

    #[cfg(feature = "cache-type-score")]
    BoundaryCache(TypeScorerBoundaryCache),

    #[cfg(feature = "tag-prediction")]
    BoundaryTag(TypeScorerBoundaryTag),
}

impl TypeScorer {
    pub fn new(
        ngram_model: NgramModel<Vec<u8>>,
        window_size: u8,
        #[cfg(feature = "tag-prediction")] tag_ngram_model: Vec<TagNgramModel<Vec<u8>>>,
    ) -> Result<Self> {
        if window_size == 0 {
            return Ok(Self::Nop);
        }

        #[cfg(feature = "tag-prediction")]
        if tag_ngram_model.is_empty() {
            match window_size {
                #[cfg(feature = "cache-type-score")]
                0..=CACHE_MAX_WINDOW_SIZE => Ok(Self::BoundaryCache(TypeScorerBoundaryCache::new(
                    ngram_model,
                    window_size,
                )?)),

                _ => Ok(Self::Boundary(TypeScorerBoundary::new(
                    ngram_model,
                    window_size,
                )?)),
            }
        } else {
            Ok(Self::BoundaryTag(TypeScorerBoundaryTag::new(
                ngram_model,
                window_size,
                tag_ngram_model,
            )?))
        }

        #[cfg(not(feature = "tag-prediction"))]
        match window_size {
            #[cfg(feature = "cache-type-score")]
            0..=CACHE_MAX_WINDOW_SIZE => Ok(Self::BoundaryCache(TypeScorerBoundaryCache::new(
                ngram_model,
                window_size,
            )?)),

            _ => Ok(Self::Boundary(TypeScorerBoundary::new(
                ngram_model,
                window_size,
            )?)),
        }
    }

    #[inline]
    pub fn add_scores<'a, 'b>(&self, sentence: &mut Sentence<'a, 'b>) {
        match self {
            Self::Nop => (),

            Self::Boundary(scorer) => scorer.add_scores(sentence),

            #[cfg(feature = "cache-type-score")]
            Self::BoundaryCache(scorer) => scorer.add_scores(sentence),

            #[cfg(feature = "tag-prediction")]
            Self::BoundaryTag(scorer) => scorer.add_scores(sentence),
        }
    }

    /// # Satety
    ///
    /// `token_id` must be smaller than `scorer.tag_weight.len()`.
    /// `pos` must be smaller than `sentence.type_pma_states.len()`.
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
            Self::Nop => scores.fill(0),
            Self::BoundaryTag(scorer) => scorer.add_tag_scores(token_id, pos, sentence, scores),
            _ => panic!("unsupported"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ngram_model::NgramData;
    use crate::predictor::PositionalWeight;
    use crate::CharacterType::*;

    use crate::predictor::WEIGHT_FIXED_LEN;

    #[cfg(feature = "tag-prediction")]
    use crate::ngram_model::TagNgramData;

    #[rustfmt::skip]
    #[test]
    fn test_weight_merger() {
        let mut merger = TypeWeightMerger::default();
        merger.add(b"eab".to_vec(), PositionalWeight::new(-3, vec![1, 2, 3, 4]));
        merger.add(b"ab".to_vec(), PositionalWeight::new(-3, vec![2, 4, 6, 8, 10]));
        merger.add(b"ab".to_vec(), PositionalWeight::new(-3, vec![3, 6, 9]));
        merger.add(b"cd".to_vec(), PositionalWeight::new(-2, vec![4, 8, 12]));
        assert_eq!(
            vec![
                (b"ab".to_vec(), PositionalWeight::new(-3, vec![5, 10, 15, 8, 10])),
                (b"cd".to_vec(), PositionalWeight::new(-2, vec![4, 8, 12])),
                (b"eab".to_vec(), PositionalWeight::new(-3, vec![6, 12, 18, 12, 10])),
            ],
            merger.merge(),
        );
    }

    #[test]
    fn test_add_scores() {
        // input:  我  ら  は  全  世  界  の  国  民
        // n-grams:
        //   KH:      4   5   6   7
        //                    1   2   3   4   5   6
        //   KKK:         8   9  10  11  12  13
        //   KK:     14  15  16  17  18  19  20
        //               14  15  16  17  18  19  20
        //                           14  15  16  17
        //   K:      25  26  27  28
        //           22  23  24  25  26  27  28
        //           21  22  23  24  25  26  27  28
        //               21  22  23  24  25  26  27
        //                       21  22  23  24  25
        //                           21  22  23  24
        let scorer = TypeScorerBoundary::new(
            NgramModel(vec![
                NgramData {
                    ngram: vec![Kanji as u8, Hiragana as u8],
                    weights: vec![1, 2, 3, 4, 5, 6, 7],
                },
                NgramData {
                    ngram: vec![Kanji as u8, Kanji as u8, Kanji as u8],
                    weights: vec![8, 9, 10, 11, 12, 13],
                },
                NgramData {
                    ngram: vec![Kanji as u8, Kanji as u8],
                    weights: vec![14, 15, 16, 17, 18, 19, 20],
                },
                NgramData {
                    ngram: vec![Kanji as u8],
                    weights: vec![21, 22, 23, 24, 25, 26, 27, 28],
                },
            ]),
            4,
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
            &[87, 135, 144, 174, 182, 192, 202, 148],
            sentence.boundary_scores(),
        );
    }

    #[cfg(feature = "cache-type-score")]
    #[test]
    fn test_add_scores_cache_1() {
        // input:  我  ら  は  全  世  界  の  国  民
        // n-grams:
        //   KH:      3   4   5
        //                        1   2   3   4   5
        //   KKK:             6   7   8   9
        //   KK:         10  11  12  13  14
        //                   10  11  12  13  14
        //                               10  11  12
        //   K:      18  19  20
        //           15  16  17  18  19  20
        //               15  16  17  18  19  20
        //                   15  16  17  18  19  20
        //                           15  16  17  18
        //                               15  16  17
        let scorer = TypeScorerBoundaryCache::new(
            NgramModel(vec![
                NgramData {
                    ngram: vec![Kanji as u8, Hiragana as u8],
                    weights: vec![1, 2, 3, 4, 5],
                },
                NgramData {
                    ngram: vec![Kanji as u8, Kanji as u8, Kanji as u8],
                    weights: vec![6, 7, 8, 9],
                },
                NgramData {
                    ngram: vec![Kanji as u8, Kanji as u8],
                    weights: vec![10, 11, 12, 13, 14],
                },
                NgramData {
                    ngram: vec![Kanji as u8],
                    weights: vec![15, 16, 17, 18, 19, 20],
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
            .resize(sentence.score_padding * 2 + sentence.len() - 1, 2);
        scorer.add_scores(&mut sentence);
        assert_eq!(
            &[38, 66, 102, 84, 106, 139, 103, 74],
            sentence.boundary_scores(),
        );
    }

    #[cfg(feature = "cache-type-score")]
    #[test]
    fn test_add_scores_cache_2() {
        // input:  我  ら  は  全  世  界  の  国  民
        // n-grams:
        //   KH:      2   3
        //                            1   2   3
        //   KKK:                 4   5
        //   KK:              6   7   8
        //                        6   7   8
        //                                    6   7
        //   K:      11  12
        //                9  10  11  12
        //                    9  10  11  12
        //                        9  10  11  12
        //                                9  10  11
        //                                    9  10
        let scorer = TypeScorerBoundaryCache::new(
            NgramModel(vec![
                NgramData {
                    ngram: vec![Kanji as u8, Hiragana as u8],
                    weights: vec![1, 2, 3],
                },
                NgramData {
                    ngram: vec![Kanji as u8, Kanji as u8, Kanji as u8],
                    weights: vec![4, 5],
                },
                NgramData {
                    ngram: vec![Kanji as u8, Kanji as u8],
                    weights: vec![6, 7, 8],
                },
                NgramData {
                    ngram: vec![Kanji as u8],
                    weights: vec![9, 10, 11, 12],
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
            .resize(sentence.score_padding * 2 + sentence.len() - 1, 3);
        scorer.add_scores(&mut sentence);
        assert_eq!(
            &[16, 27, 28, 50, 57, 45, 43, 31],
            sentence.boundary_scores(),
        );
    }

    #[cfg(feature = "tag-prediction")]
    #[test]
    fn test_add_scores_with_tags() {
        // input:    こ  の  人  は  火  星  人  だ
        // n-grams:
        //   HHK:       2   3   4
        //   KH:        5   6   7   8   9
        //                              5   6   7
        let scorer = TypeScorerBoundaryTag::new(
            NgramModel(vec![
                NgramData {
                    ngram: vec![Hiragana as u8, Hiragana as u8, Kanji as u8],
                    weights: vec![1, 2, 3, 4],
                },
                NgramData {
                    ngram: vec![Kanji as u8, Hiragana as u8],
                    weights: vec![5, 6, 7, 8, 9],
                },
            ]),
            3,
            vec![
                TagNgramModel(vec![
                    TagNgramData {
                        ngram: vec![Hiragana as u8, Kanji as u8],
                        weights: vec![(0, vec![10, 11, 12]), (1, vec![13, 14, 15])],
                    },
                    TagNgramData {
                        ngram: vec![Kanji as u8, Hiragana as u8],
                        weights: vec![(1, vec![16, 17, 18]), (3, vec![19, 20, 21])],
                    },
                    TagNgramData {
                        ngram: vec![Kanji as u8, Kanji as u8, Kanji as u8],
                        weights: vec![(0, vec![22, 23, 24])],
                    },
                ]),
                TagNgramModel(vec![]),
                TagNgramModel(vec![
                    TagNgramData {
                        ngram: vec![Kanji as u8, Hiragana as u8],
                        weights: vec![(0, vec![25, 26]), (3, vec![27, 28])],
                    },
                    TagNgramData {
                        ngram: vec![Hiragana as u8, Kanji as u8, Kanji as u8, Kanji as u8],
                        weights: vec![(3, vec![29, 30])],
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
        assert_eq!(&[8, 10, 12, 9, 15, 7, 8], sentence.boundary_scores());

        let mut tag_scores = [1; 8];
        unsafe {
            scorer.add_tag_scores(0, 2, &sentence, &mut tag_scores);
        }
        assert_eq!(&[27, 29, 31, 1, 1, 1, 1, 1], &tag_scores);

        let mut tag_scores = [1; 8];
        unsafe {
            scorer.add_tag_scores(0, 6, &sentence, &mut tag_scores);
        }
        assert_eq!(&[39, 41, 43, 1, 1, 1, 1, 1], &tag_scores);

        let mut tag_scores = [1; 8];
        unsafe {
            scorer.add_tag_scores(2, 3, &sentence, &mut tag_scores);
        }
        assert_eq!(&[55, 57, 1, 1, 1, 1, 1, 1], &tag_scores);
    }
}
