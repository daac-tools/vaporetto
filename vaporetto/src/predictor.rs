use core::ops::AddAssign;

#[cfg(all(feature = "fix-weight-length", feature = "portable-simd"))]
use core::simd::Simd;

use alloc::vec::Vec;

#[cfg(feature = "tag-prediction")]
use alloc::borrow::Cow;
#[cfg(feature = "tag-prediction")]
use alloc::string::String;

use bincode::{
    de::{BorrowDecoder, Decoder},
    enc::Encoder,
    error::{DecodeError, EncodeError},
    BorrowDecode, Decode, Encode,
};

#[cfg(feature = "tag-prediction")]
use hashbrown::HashMap;

use crate::char_scorer::CharScorer;
use crate::errors::Result;
use crate::model::Model;
use crate::sentence::{CharacterBoundary, Sentence};
use crate::type_scorer::TypeScorer;

#[cfg(feature = "tag-prediction")]
use crate::utils::SerializableHashMap;

pub const WEIGHT_FIXED_LEN: usize = 8;

#[cfg(all(feature = "fix-weight-length", not(feature = "portable-simd")))]
pub type I32Simd = [i32; WEIGHT_FIXED_LEN];
#[cfg(all(feature = "fix-weight-length", feature = "portable-simd"))]
pub type I32Simd = Simd<i32, WEIGHT_FIXED_LEN>;

#[derive(Clone, Debug)]
pub enum WeightVector {
    Variable(Vec<i32>),

    #[cfg(feature = "fix-weight-length")]
    Fixed(I32Simd),
}

impl Default for WeightVector {
    fn default() -> Self {
        Self::Variable(vec![])
    }
}

impl Decode for WeightVector {
    fn decode<D: Decoder>(decoder: &mut D) -> Result<Self, DecodeError> {
        let weight: Vec<i32> = Decode::decode(decoder)?;
        Ok(Self::from(weight))
    }
}
bincode::impl_borrow_decode!(WeightVector);

impl Encode for WeightVector {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        match self {
            Self::Variable(w) => {
                Encode::encode(&w, encoder)?;
            }

            #[cfg(feature = "fix-weight-length")]
            Self::Fixed(w) => {
                #[cfg(feature = "portable-simd")]
                let w = w.as_array();

                Encode::encode(&crate::utils::trim_end_zeros(w).to_vec(), encoder)?;
            }
        }
        Ok(())
    }
}

#[cfg(feature = "tag-prediction")]
impl WeightVector {
    pub fn add_scores(&self, ys: &mut [i32]) {
        match self {
            Self::Variable(w) => {
                for (y, x) in ys.iter_mut().zip(w) {
                    *y += *x;
                }
            }

            #[cfg(feature = "fix-weight-length")]
            Self::Fixed(w) => {
                #[cfg(not(feature = "portable-simd"))]
                for (y, x) in ys[..WEIGHT_FIXED_LEN].iter_mut().zip(w) {
                    *y += *x
                }

                #[cfg(feature = "portable-simd")]
                {
                    let ys = &mut ys[..WEIGHT_FIXED_LEN];
                    let mut y = I32Simd::from_slice(ys);
                    y += w;
                    ys.copy_from_slice(y.as_array());
                }
            }
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::Variable(w) => w.len(),

            #[cfg(feature = "fix-weight-length")]
            Self::Fixed(_) => WEIGHT_FIXED_LEN,
        }
    }
}

impl From<Vec<i32>> for WeightVector {
    fn from(src: Vec<i32>) -> Self {
        match src.len() {
            #[cfg(feature = "fix-weight-length")]
            0..=WEIGHT_FIXED_LEN => {
                let mut weight = [0; WEIGHT_FIXED_LEN];
                weight[..src.len()].copy_from_slice(&src);

                #[cfg(feature = "portable-simd")]
                let weight = I32Simd::from(weight);

                Self::Fixed(weight)
            }

            _ => Self::Variable(src),
        }
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Decode, Encode)]
pub struct PositionalWeight<W> {
    offset: i16,
    weight: W,
}

impl PositionalWeight<Vec<i32>> {
    pub fn new(offset: i16, weight: Vec<i32>) -> Self {
        Self { offset, weight }
    }
}

impl AddAssign<&Self> for PositionalWeight<Vec<i32>> {
    fn add_assign(&mut self, other: &Self) {
        let new_offset = self.offset.min(other.offset);
        let shift = usize::try_from(self.offset - new_offset).unwrap();
        let new_size = (shift + self.weight.len())
            .max(usize::try_from(other.offset - new_offset).unwrap() + other.weight.len());
        self.weight.resize(new_size, 0);
        self.weight.rotate_right(shift);
        for (y, x) in self.weight[usize::try_from(other.offset - new_offset).unwrap()..]
            .iter_mut()
            .zip(&other.weight)
        {
            *y += *x;
        }
        self.offset = new_offset;
    }
}

impl From<PositionalWeight<Vec<i32>>> for PositionalWeight<WeightVector> {
    fn from(src: PositionalWeight<Vec<i32>>) -> Self {
        Self {
            offset: src.offset,
            weight: src.weight.into(),
        }
    }
}

impl PositionalWeight<WeightVector> {
    #[inline(always)]
    pub fn add_score(&self, end: isize, ys: &mut [i32]) {
        let pos = end + isize::from(self.offset);
        match &self.weight {
            WeightVector::Variable(w) => {
                if pos >= 0 {
                    for (y, x) in ys[pos as usize..].iter_mut().zip(w) {
                        *y += *x;
                    }
                } else if let Some(xs) = w.get((-pos) as usize..) {
                    for (y, x) in ys.iter_mut().zip(xs) {
                        *y += *x;
                    }
                }
            }

            #[cfg(feature = "fix-weight-length")]
            WeightVector::Fixed(w) => {
                #[cfg(not(feature = "portable-simd"))]
                for (y, x) in ys[pos as usize..pos as usize + WEIGHT_FIXED_LEN]
                    .iter_mut()
                    .zip(w)
                {
                    *y += *x
                }

                #[cfg(feature = "portable-simd")]
                {
                    let ys = &mut ys[pos as usize..pos as usize + WEIGHT_FIXED_LEN];
                    let mut y = I32Simd::from_slice(ys);
                    y += w;
                    ys.copy_from_slice(y.as_array());
                }
            }
        }
    }
}

#[cfg(feature = "tag-prediction")]
#[derive(Debug, Default, Eq, PartialEq)]
pub struct PositionalWeightWithTag {
    pub weight: Option<PositionalWeight<Vec<i32>>>,
    pub tag_info: HashMap<(usize, u8), Vec<i32>>,
}

#[cfg(feature = "tag-prediction")]
impl PositionalWeightWithTag {
    pub fn with_boundary(offset: i16, weight: Vec<i32>) -> Self {
        Self {
            weight: Some(PositionalWeight::new(offset, weight)),
            tag_info: HashMap::new(),
        }
    }

    pub fn with_tag(token_id: usize, rel_position: u8, tag_weight: Vec<i32>) -> Self {
        let mut tag_info = HashMap::new();
        tag_info.insert((token_id, rel_position), tag_weight);
        Self {
            weight: None,
            tag_info,
        }
    }
}

#[cfg(feature = "tag-prediction")]
impl AddAssign<&Self> for PositionalWeightWithTag {
    fn add_assign(&mut self, other: &Self) {
        if let Some(y) = self.weight.as_mut() {
            if let Some(x) = other.weight.as_ref() {
                *y += x;
            }
        } else {
            self.weight = other.weight.clone();
        }
        for (k, v) in &other.tag_info {
            self.tag_info
                .entry(*k)
                .and_modify(|w| {
                    for (y, x) in w.iter_mut().zip(v) {
                        *y += *x;
                    }
                })
                .or_insert_with(|| v.clone());
        }
    }
}

#[cfg(feature = "tag-prediction")]
#[derive(Decode, Encode)]
struct TagPredictor {
    tags: Vec<Vec<String>>,
    bias: WeightVector,
}

#[cfg(feature = "tag-prediction")]
impl TagPredictor {
    pub fn new(tags: Vec<Vec<String>>, bias: Vec<i32>) -> Self {
        Self {
            tags,
            bias: bias.into(),
        }
    }

    #[inline]
    pub const fn bias(&self) -> &WeightVector {
        &self.bias
    }

    #[inline]
    pub fn predict<'a>(&'a self, scores: &[i32], tags: &mut [Option<Cow<'a, str>>]) {
        let mut offset = 0;
        for (tag_cands, tag) in self.tags.iter().zip(tags) {
            if tag_cands.len() >= 2 {
                let mut idx = 0;
                let mut max_score = i32::MIN;
                for (i, &s) in scores[offset..offset + tag_cands.len()].iter().enumerate() {
                    if s > max_score {
                        idx = i;
                        max_score = s;
                    }
                }
                tag.replace(Cow::Borrowed(&tag_cands[idx]));
                offset += tag_cands.len();
            } else {
                *tag = tag_cands.first().map(|t| Cow::Borrowed(t.as_str()));
            }
        }
    }
}

pub struct PredictorData {
    char_scorer: Option<CharScorer>,
    type_scorer: Option<TypeScorer>,
    bias: i32,

    #[cfg(feature = "tag-prediction")]
    tag_predictor: Option<SerializableHashMap<String, (u32, TagPredictor)>>,
    #[cfg(feature = "tag-prediction")]
    n_tags: usize,
}

impl<'de> BorrowDecode<'de> for PredictorData {
    /// WARNING: This function is inherently unsafe. Do not publish this function outside this
    /// crate.
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let config = bincode::config::standard();
        let char_scorer_data: Option<&[u8]> = BorrowDecode::borrow_decode(decoder)?;
        let char_scorer = if let Some(data) = char_scorer_data {
            Some(bincode::borrow_decode_from_slice(data, config)?.0)
        } else {
            None
        };
        let type_scorer_data: Option<&[u8]> = BorrowDecode::borrow_decode(decoder)?;
        let type_scorer = if let Some(data) = type_scorer_data {
            Some(bincode::borrow_decode_from_slice(data, config)?.0)
        } else {
            None
        };
        let bias = Decode::decode(decoder)?;
        #[cfg(feature = "tag-prediction")]
        let tag_predictor = Decode::decode(decoder)?;
        #[cfg(feature = "tag-prediction")]
        let n_tags = Decode::decode(decoder)?;
        Ok(Self {
            char_scorer,
            type_scorer,
            bias,
            #[cfg(feature = "tag-prediction")]
            tag_predictor,
            #[cfg(feature = "tag-prediction")]
            n_tags,
        })
    }
}

impl Encode for PredictorData {
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let config = bincode::config::standard();
        let char_scorer_data = if let Some(char_scorer) = self.char_scorer.as_ref() {
            Some(bincode::encode_to_vec(char_scorer, config)?)
        } else {
            None
        };
        Encode::encode(&char_scorer_data, encoder)?;
        let type_scorer_data = if let Some(type_scorer) = self.type_scorer.as_ref() {
            Some(bincode::encode_to_vec(type_scorer, config)?)
        } else {
            None
        };
        Encode::encode(&type_scorer_data, encoder)?;
        Encode::encode(&self.bias, encoder)?;
        #[cfg(feature = "tag-prediction")]
        Encode::encode(&self.tag_predictor, encoder)?;
        #[cfg(feature = "tag-prediction")]
        Encode::encode(&self.n_tags, encoder)?;
        Ok(())
    }
}

/// Predictor created from the model.
///
#[cfg_attr(
    feature = "std",
    doc = "
# Example 1: without tag prediction

```
use std::fs::File;

use vaporetto::{Model, Predictor, Sentence};

let f = File::open(\"../resources/model.bin\").unwrap();
let model = Model::read(f).unwrap();
let predictor = Predictor::new(model, false).unwrap();

let mut s = Sentence::from_raw(\"まぁ社長は火星猫だ\").unwrap();
predictor.predict(&mut s);
// s.fill_tags(); will panic!

let mut buf = String::new();
s.write_tokenized_text(&mut buf);
assert_eq!(
    \"まぁ 社長 は 火星 猫 だ\",
    buf,
);
```
"
)]
#[cfg_attr(
    all(feature = "std", feature = "tag-prediction"),
    doc = "
# Example 2: with tag prediction

Tag prediction requires **crate feature** `tag-prediction`.
```
use std::fs::File;

use vaporetto::{Model, Predictor, Sentence};

let mut f = File::open(\"../resources/model.bin\").unwrap();
let model = Model::read(f).unwrap();
let predictor = Predictor::new(model, true).unwrap();

let mut s = Sentence::from_raw(\"まぁ社長は火星猫だ\").unwrap();
predictor.predict(&mut s);
s.fill_tags();

let mut buf = String::new();
s.write_tokenized_text(&mut buf);
assert_eq!(
    \"まぁ/名詞/マー 社長/名詞/シャチョー は/助詞/ワ 火星/名詞/カセー 猫/名詞/ネコ だ/助動詞/ダ\",
    buf,
);
```
"
)]
pub struct Predictor(PredictorData);

impl Predictor {
    /// Creates a new predictor from the model.
    ///
    /// # Arguments
    ///
    /// * `model` - A model data.
    /// * `predict_tags` - If you want to predict tags, set to true.
    ///
    /// # Errors
    ///
    /// Returns an error variant when the model is invalid.
    pub fn new(model: Model, predict_tags: bool) -> Result<Self> {
        #[cfg(feature = "tag-prediction")]
        let mut tag_char_ngram_model = vec![];
        #[cfg(feature = "tag-prediction")]
        let mut tag_type_ngram_model = vec![];
        #[cfg(feature = "tag-prediction")]
        let mut n_tags = 0;

        #[cfg(not(feature = "tag-prediction"))]
        if predict_tags {
            panic!("tag prediction is unsupported");
        }
        #[cfg(feature = "tag-prediction")]
        let tag_predictor = predict_tags.then(|| {
            let mut tag_predictor = HashMap::new();
            for (i, tag_model) in model.0.tag_models.into_iter().enumerate() {
                n_tags = n_tags.max(tag_model.tags.len());
                // token does not duplicate in the model.
                tag_predictor.insert(
                    tag_model.token,
                    (
                        u32::try_from(i).unwrap(),
                        TagPredictor::new(tag_model.tags, tag_model.bias),
                    ),
                );
                tag_char_ngram_model.push(tag_model.char_ngram_model);
                tag_type_ngram_model.push(tag_model.type_ngram_model);
            }
            SerializableHashMap(tag_predictor)
        });

        let char_scorer = CharScorer::new(
            model.0.char_ngram_model,
            model.0.dict_model,
            model.0.char_window_size,
            #[cfg(feature = "tag-prediction")]
            tag_char_ngram_model,
        )?;
        let type_scorer = TypeScorer::new(
            model.0.type_ngram_model,
            model.0.type_window_size,
            #[cfg(feature = "tag-prediction")]
            tag_type_ngram_model,
        )?;
        Ok(Self(PredictorData {
            char_scorer,
            type_scorer,
            bias: model.0.bias,

            #[cfg(feature = "tag-prediction")]
            tag_predictor,
            #[cfg(feature = "tag-prediction")]
            n_tags,
        }))
    }

    /// Predicts word boundaries of the given sentence.
    /// If necessary, this function also prepares for predicting tags.
    pub fn predict<'a>(&'a self, sentence: &mut Sentence<'_, 'a>) {
        sentence.score_padding = WEIGHT_FIXED_LEN - 1;
        sentence.boundary_scores.clear();
        sentence
            .boundary_scores
            .resize(sentence.score_padding * 2 + sentence.len() - 1, self.0.bias);
        if let Some(scorer) = self.0.char_scorer.as_ref() {
            scorer.add_scores(sentence);
        }
        if let Some(scorer) = self.0.type_scorer.as_ref() {
            scorer.add_scores(sentence);
        }
        for (b, s) in sentence
            .boundaries
            .iter_mut()
            .zip(&sentence.boundary_scores[sentence.score_padding..])
        {
            if *s > 0 {
                *b = CharacterBoundary::WordBoundary;
            } else {
                *b = CharacterBoundary::NotWordBoundary;
            }
        }
        sentence.set_predictor(self);
    }

    #[cfg(feature = "tag-prediction")]
    pub(crate) fn predict_tags<'a>(&'a self, sentence: &mut Sentence<'_, 'a>) {
        let tag_predictor = self
            .0
            .tag_predictor
            .as_ref()
            .expect("this predictor is created with predict_tags = false");

        if self.0.n_tags == 0 {
            return;
        }
        let mut scores = vec![];
        let mut range_start = Some(0);
        sentence.n_tags = self.0.n_tags;
        sentence.tags.clear();
        sentence.tags.resize(sentence.len() * self.0.n_tags, None);
        for (i, &b) in sentence.boundaries.iter().enumerate() {
            if b == CharacterBoundary::Unknown {
                range_start.take();
            } else if b == CharacterBoundary::WordBoundary {
                if let Some(&range_start) = range_start.as_ref() {
                    let token = sentence.text_substring(range_start, i + 1);
                    if let Some((token_id, tag_predictor)) = tag_predictor.get(token) {
                        scores.clear();
                        scores.resize(tag_predictor.bias().len(), 0);
                        tag_predictor.bias().add_scores(&mut scores);
                        if let Some(scorer) = self.0.char_scorer.as_ref() {
                            debug_assert!(i < sentence.char_pma_states.len());
                            // token_id is always smaller than tag_weight.len() because
                            // tag_predictor is created to contain such values in the new()
                            // function.
                            unsafe {
                                scorer.add_tag_scores(*token_id, i, sentence, &mut scores);
                            }
                        }
                        if let Some(scorer) = self.0.type_scorer.as_ref() {
                            debug_assert!(i < sentence.type_pma_states.len());
                            // token_id is always smaller than tag_weight.len() because
                            // tag_predictor is created to contain such values in the new()
                            // function.
                            unsafe {
                                scorer.add_tag_scores(*token_id, i, sentence, &mut scores);
                            }
                        }
                        tag_predictor.predict(
                            &scores,
                            &mut sentence.tags[i * self.0.n_tags..(i + 1) * self.0.n_tags],
                        );
                    }
                }
                range_start.replace(i + 1);
            }
        }
        if let Some(&range_start) = range_start.as_ref() {
            let token = sentence.text_substring(range_start, sentence.len());
            if let Some((token_id, tag_predictor)) = tag_predictor.get(token) {
                scores.clear();
                scores.resize(tag_predictor.bias().len(), 0);
                tag_predictor.bias().add_scores(&mut scores);
                if let Some(scorer) = self.0.char_scorer.as_ref() {
                    debug_assert!(sentence.len() <= sentence.char_pma_states.len());
                    // token_id is always smaller than tag_weight.len() because tag_predictor is
                    // created to contain such values in the new() function.
                    unsafe {
                        scorer.add_tag_scores(*token_id, sentence.len() - 1, sentence, &mut scores);
                    }
                }
                if let Some(scorer) = self.0.type_scorer.as_ref() {
                    debug_assert!(sentence.len() <= sentence.type_pma_states.len());
                    // token_id is always smaller than tag_weight.len() because tag_predictor is
                    // created to contain such values in the new() function.
                    unsafe {
                        scorer.add_tag_scores(*token_id, sentence.len() - 1, sentence, &mut scores);
                    }
                }
                let i = sentence.len() - 1;
                tag_predictor.predict(&scores, &mut sentence.tags[i * self.0.n_tags..]);
            }
        }
    }

    /// Serializes the predictor into a Vec.
    pub fn serialize_to_vec(&self) -> Result<Vec<u8>> {
        let config = bincode::config::standard();
        let result = bincode::encode_to_vec(&self.0, config)?;
        Ok(result)
    }

    /// Deserializes a predictor from a given slice and returns a tuple of the predictor and the remaining slice.
    ///
    /// # Safety
    ///
    /// The given data must be a correct predictor exported by [`Predictor::serialize_to_vec()`]
    /// function.
    pub unsafe fn deserialize_from_slice_unchecked(data: &[u8]) -> Result<(Self, &[u8])> {
        let config = bincode::config::standard();
        // Deserialization is unsafe because the automaton will not be verified.
        let (predictor_data, size) = bincode::borrow_decode_from_slice(data, config)?;
        Ok((Self(predictor_data), &data[size..]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::dict_model::{DictModel, WordWeightRecord};
    use crate::model::TagModel;
    use crate::ngram_model::{NgramData, NgramModel, TagNgramData, TagNgramModel, TagWeight};
    use crate::CharacterBoundary::*;
    use crate::CharacterType::*;

    #[test]
    fn test_positional_weight_add_assign_1() {
        let mut y = PositionalWeight::new(-2, vec![1, 2, 3, 4]);
        let x = PositionalWeight::new(4, vec![2, 4, 8]);
        y += &x;
        assert_eq!(-2, y.offset);
        assert_eq!(vec![1, 2, 3, 4, 0, 0, 2, 4, 8], y.weight);
    }

    #[test]
    fn test_positional_weight_add_assign_2() {
        let mut y = PositionalWeight::new(-2, vec![1, 2, 3, 4]);
        let x = PositionalWeight::new(2, vec![2, 4, 8]);
        y += &x;
        assert_eq!(-2, y.offset);
        assert_eq!(vec![1, 2, 3, 4, 2, 4, 8], y.weight);
    }

    #[test]
    fn test_positional_weight_add_assign_3() {
        let mut y = PositionalWeight::new(-2, vec![1, 2, 3, 4]);
        let x = PositionalWeight::new(0, vec![2, 4, 8]);
        y += &x;
        assert_eq!(-2, y.offset);
        assert_eq!(vec![1, 2, 5, 8, 8], y.weight);
    }

    #[test]
    fn test_positional_weight_add_assign_4() {
        let mut y = PositionalWeight::new(-2, vec![1, 2, 3, 4]);
        let x = PositionalWeight::new(-1, vec![2, 4, 8]);
        y += &x;
        assert_eq!(-2, y.offset);
        assert_eq!(vec![1, 4, 7, 12], y.weight);
    }

    #[test]
    fn test_positional_weight_add_assign_5() {
        let mut y = PositionalWeight::new(-2, vec![1, 2, 3, 4]);
        let x = PositionalWeight::new(-2, vec![2, 4, 8]);
        y += &x;
        assert_eq!(-2, y.offset);
        assert_eq!(vec![3, 6, 11, 4], y.weight);
    }

    #[test]
    fn test_positional_weight_add_assign_6() {
        let mut y = PositionalWeight::new(-2, vec![1, 2, 3, 4]);
        let x = PositionalWeight::new(-4, vec![2, 4, 8]);
        y += &x;
        assert_eq!(-4, y.offset);
        assert_eq!(vec![2, 4, 9, 2, 3, 4], y.weight);
    }

    #[test]
    fn test_positional_weight_add_assign_7() {
        let mut y = PositionalWeight::new(-2, vec![1, 2, 3, 4]);
        let x = PositionalWeight::new(-5, vec![2, 4, 8]);
        y += &x;
        assert_eq!(-5, y.offset);
        assert_eq!(vec![2, 4, 8, 1, 2, 3, 4], y.weight);
    }

    #[test]
    fn test_positional_weight_add_assign_8() {
        let mut y = PositionalWeight::new(-2, vec![1, 2, 3, 4]);
        let x = PositionalWeight::new(-7, vec![2, 4, 8]);
        y += &x;
        assert_eq!(-7, y.offset);
        assert_eq!(vec![2, 4, 8, 0, 0, 1, 2, 3, 4], y.weight);
    }

    fn create_test_model() -> Model {
        // input:    こ  の  人  は  地  球  人  だ
        // n-grams:
        //   この人:   -2   3   4
        //   人だ:                     -5   6   7
        // n-grams:
        //   HHK:     -11  12  13
        //   KH:      -14  15  16  17 -18
        //                            -14  15  16
        // dict:
        //   人:           19  20          19  20
        //   地球:                 21 -22  23
        Model::new(
            NgramModel(vec![
                NgramData {
                    ngram: "この人".into(),
                    weights: vec![1, -2, 3, 4],
                },
                NgramData {
                    ngram: "人だ".into(),
                    weights: vec![-5, 6, 7, 8, 9],
                },
            ]),
            NgramModel(vec![
                NgramData {
                    ngram: vec![Hiragana as u8, Hiragana as u8, Kanji as u8],
                    weights: vec![10, -11, 12, 13],
                },
                NgramData {
                    ngram: vec![Kanji as u8, Hiragana as u8],
                    weights: vec![-14, 15, 16, 17, -18],
                },
            ]),
            DictModel(vec![
                WordWeightRecord {
                    word: "人".into(),
                    weights: vec![19, 20],
                    comment: "".into(),
                },
                WordWeightRecord {
                    word: "地球".into(),
                    weights: vec![21, -22, 23],
                    comment: "".into(),
                },
            ]),
            5,
            3,
            3,
            vec![
                TagModel {
                    token: "人".into(),
                    tags: vec![
                        vec!["名詞".into(), "接尾辞".into()],
                        vec!["ジン".into(), "ヒト".into()],
                    ],
                    char_ngram_model: TagNgramModel(vec![TagNgramData {
                        ngram: "は地球人".into(),
                        weights: vec![TagWeight {
                            rel_position: 0,
                            weights: vec![-32, 33, 34, -35],
                        }],
                    }]),
                    type_ngram_model: TagNgramModel(vec![TagNgramData {
                        ngram: vec![Hiragana as u8, Kanji as u8, Hiragana as u8],
                        weights: vec![TagWeight {
                            rel_position: 1,
                            weights: vec![36, -37, -38, 39],
                        }],
                    }]),
                    bias: vec![40, 41, 42, 43],
                },
                TagModel {
                    token: "地球".into(),
                    tags: vec![
                        vec!["名詞".into()],
                        vec!["マンホーム".into(), "チキュー".into()],
                    ],
                    char_ngram_model: TagNgramModel(vec![TagNgramData {
                        ngram: "は地球人".into(),
                        weights: vec![TagWeight {
                            rel_position: 1,
                            weights: vec![-44, 45],
                        }],
                    }]),
                    type_ngram_model: TagNgramModel(vec![]),
                    bias: vec![46, 47],
                },
            ],
        )
    }

    #[test]
    fn test_predict_boundaries() {
        let model = create_test_model();
        let predictor = Predictor::new(model, false).unwrap();
        let mut sentence = Sentence::from_raw("この人は地球人だ").unwrap();
        predictor.predict(&mut sentence);
        assert_eq!(&[-22, 54, 58, 43, -54, 68, 48], sentence.boundary_scores(),);
        assert_eq!(
            &[
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
                WordBoundary
            ],
            sentence.boundaries(),
        );
    }

    #[cfg(feature = "tag-prediction")]
    #[test]
    fn test_predict_tags() {
        let model = create_test_model();
        let predictor = Predictor::new(model, true).unwrap();
        let mut sentence = Sentence::from_raw("この人は地球人だ").unwrap();
        predictor.predict(&mut sentence);
        sentence.fill_tags();
        assert_eq!(&[-22, 54, 58, 43, -54, 68, 48], sentence.boundary_scores(),);
        assert_eq!(
            &[
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
            ],
            sentence.boundaries(),
        );
        assert_eq!(
            &[
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("名詞")),
                Some(Cow::Borrowed("ヒト")),
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("名詞")),
                Some(Cow::Borrowed("チキュー")),
                Some(Cow::Borrowed("接尾辞")),
                Some(Cow::Borrowed("ジン")),
                None,
                None,
            ],
            sentence.tags()
        );
    }

    #[test]
    fn test_serialization() {
        let model = create_test_model();
        let predictor = Predictor::new(model, false).unwrap();
        let data = predictor.serialize_to_vec().unwrap();
        let (predictor, _) = unsafe { Predictor::deserialize_from_slice_unchecked(&data).unwrap() };
        let mut sentence = Sentence::from_raw("この人は地球人だ").unwrap();
        predictor.predict(&mut sentence);
        assert_eq!(&[-22, 54, 58, 43, -54, 68, 48], sentence.boundary_scores(),);
        assert_eq!(
            &[
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
                WordBoundary
            ],
            sentence.boundaries(),
        );
    }

    #[cfg(feature = "tag-prediction")]
    #[test]
    fn test_serialization_tags() {
        let model = create_test_model();
        let predictor = Predictor::new(model, true).unwrap();
        let data = predictor.serialize_to_vec().unwrap();
        let (predictor, _) = unsafe { Predictor::deserialize_from_slice_unchecked(&data).unwrap() };
        let mut sentence = Sentence::from_raw("この人は地球人だ").unwrap();
        predictor.predict(&mut sentence);
        sentence.fill_tags();
        assert_eq!(&[-22, 54, 58, 43, -54, 68, 48], sentence.boundary_scores(),);
        assert_eq!(
            &[
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
                WordBoundary
            ],
            sentence.boundaries(),
        );
        assert_eq!(
            &[
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("名詞")),
                Some(Cow::Borrowed("ヒト")),
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("名詞")),
                Some(Cow::Borrowed("チキュー")),
                Some(Cow::Borrowed("接尾辞")),
                Some(Cow::Borrowed("ジン")),
                None,
                None,
            ],
            sentence.tags()
        );
    }

    #[cfg(feature = "tag-prediction")]
    #[test]
    #[should_panic]
    fn test_fill_tags_unsupported() {
        let model = create_test_model();
        let predictor = Predictor::new(model, false).unwrap();
        let mut sentence = Sentence::from_raw("この人は地球人だ").unwrap();
        predictor.predict(&mut sentence);
        sentence.fill_tags();
    }

    #[cfg(feature = "tag-prediction")]
    #[test]
    #[should_panic]
    fn test_fill_tags_unsupported_overwrite_prediction() {
        let mut sentence = Sentence::from_raw("この人は地球人だ").unwrap();

        let model = create_test_model();
        let predictor = Predictor::new(model, true).unwrap();
        predictor.predict(&mut sentence);
        sentence.fill_tags();

        let model = create_test_model();
        let predictor = Predictor::new(model, false).unwrap();
        predictor.predict(&mut sentence);
        sentence.fill_tags();
    }
}
