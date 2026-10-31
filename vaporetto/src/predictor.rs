use core::ops::AddAssign;

#[cfg(all(feature = "fix-weight-length", feature = "portable-simd"))]
use core::simd::Simd;

use alloc::vec::Vec;

#[cfg(feature = "tag-prediction")]
use alloc::borrow::Cow;
#[cfg(feature = "tag-prediction")]
use alloc::string::String;

use daachorse::Serializable;
#[cfg(feature = "tag-prediction")]
use hashbrown::HashMap;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::char_scorer::CharScorer;
use crate::errors::{Result, VaporettoError};
use crate::model::Model;
use crate::sentence::{CharacterBoundary, Sentence};
use crate::type_scorer::TypeScorer;

#[cfg(feature = "tag-prediction")]
use crate::utils::SplitMix64Builder;

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

impl<'de> Deserialize<'de> for WeightVector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let weight: Vec<i32> = Deserialize::deserialize(deserializer)?;
        Ok(Self::from(weight))
    }
}

impl Serialize for WeightVector {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        match self {
            Self::Variable(w) => w.serialize(serializer),

            #[cfg(feature = "fix-weight-length")]
            Self::Fixed(w) => {
                #[cfg(feature = "portable-simd")]
                let w = w.as_array();

                crate::utils::trim_end_zeros(w).serialize(serializer)
            }
        }
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

#[derive(Clone, Debug, Default, Eq, PartialEq, Deserialize, Serialize)]
pub struct PositionalWeight<W> {
    offset: i16,
    weight: W,
}

impl PositionalWeight<Vec<i32>> {
    pub const fn new(offset: i16, weight: Vec<i32>) -> Self {
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

/// Flag of a pattern value indicating that the weight is stored in the variable-length storage of
/// [`BoundaryWeights`].
const VARIABLE_WEIGHT_FLAG: u32 = 1 << 31;

/// Flag of a pattern value indicating that the pattern has no boundary weight.
#[cfg(any(feature = "fix-weight-length", feature = "tag-prediction"))]
const NO_WEIGHT_FLAG: u32 = 1 << 30;

/// Flag of a pattern value indicating that the pattern has tag weights.
#[cfg(feature = "tag-prediction")]
pub const TAG_WEIGHT_FLAG: u32 = 1 << 29;

/// Mask to extract an index from a pattern value.
const VALUE_INDEX_MASK: u32 = (1 << 29) - 1;

/// Value associated with each pattern of an automaton.
///
/// If the boundary weight of the pattern fits in [`WEIGHT_FIXED_LEN`] elements starting from a
/// common offset, the weight is stored in this value itself. Since the automaton returns this
/// value on a match, the weight can be added without accessing another array. Otherwise, `id`
/// indicates the weight stored in [`BoundaryWeights`].
#[derive(Clone, Copy)]
pub struct PatternValue {
    #[cfg(feature = "fix-weight-length")]
    weight: [i32; WEIGHT_FIXED_LEN],

    /// Unique identifier of the pattern containing flags.
    pub id: u32,
}

impl Serializable for PatternValue {
    fn serialize_to_vec(&self, dst: &mut Vec<u8>) {
        #[cfg(feature = "fix-weight-length")]
        for w in &self.weight {
            w.serialize_to_vec(dst);
        }
        self.id.serialize_to_vec(dst);
    }

    fn deserialize_from_slice(src: &[u8]) -> daachorse::Result<(Self, &[u8])> {
        #[cfg(feature = "fix-weight-length")]
        let (weight, src) = {
            let mut weight = [0; WEIGHT_FIXED_LEN];
            let mut src = src;
            for w in &mut weight {
                let (x, rest) = i32::deserialize_from_slice(src)?;
                *w = x;
                src = rest;
            }
            (weight, src)
        };
        let (id, src) = u32::deserialize_from_slice(src)?;
        Ok((
            Self {
                #[cfg(feature = "fix-weight-length")]
                weight,
                id,
            },
            src,
        ))
    }

    fn serialized_bytes() -> usize {
        #[cfg(feature = "fix-weight-length")]
        let n = WEIGHT_FIXED_LEN + 1;
        #[cfg(not(feature = "fix-weight-length"))]
        let n = 1;
        core::mem::size_of::<u32>() * n
    }
}

/// Boundary weights associated with the patterns of an automaton.
///
/// [`BoundaryWeights::push()`] returns a [`PatternValue`] that should be associated with the
/// pattern, and [`BoundaryWeights::add_score()`] adds the weight specified by the value. Most
/// weights are stored in the values, and the other weights are stored in this storage.
#[derive(Deserialize, Serialize)]
pub struct BoundaryWeights {
    /// Weights stored in values cover positions
    /// [`end + fixed_offset`, `end + fixed_offset + WEIGHT_FIXED_LEN`).
    #[cfg(feature = "fix-weight-length")]
    fixed_offset: i16,

    /// Number of weights stored in values, which is used to assign identifiers.
    #[cfg(feature = "fix-weight-length")]
    #[serde(skip)]
    n_fixed: usize,

    variable: Vec<PositionalWeight<WeightVector>>,
}

impl BoundaryWeights {
    /// Creates a new storage for weights of patterns with the given window size.
    #[cfg_attr(not(feature = "fix-weight-length"), allow(unused_variables))]
    pub fn new(window_size: u8) -> Self {
        Self {
            #[cfg(feature = "fix-weight-length")]
            fixed_offset: i16::from(window_size) - i16::try_from(WEIGHT_FIXED_LEN).unwrap(),
            #[cfg(feature = "fix-weight-length")]
            n_fixed: 0,
            variable: vec![],
        }
    }

    /// Adds a weight and returns the value that should be associated with the pattern.
    pub fn push(&mut self, weight: PositionalWeight<Vec<i32>>) -> Result<PatternValue> {
        let PositionalWeight { offset, weight } = weight;

        #[cfg(feature = "fix-weight-length")]
        {
            let start = i32::from(offset) - i32::from(self.fixed_offset);
            // The range of weights stored in values must be within the padding of the score
            // array, which has `WEIGHT_FIXED_LEN - 1` elements on each side. This holds when
            // `fixed_offset < 0`.
            if self.fixed_offset < 0
                && start >= 0
                && usize::try_from(start)? + weight.len() <= WEIGHT_FIXED_LEN
            {
                let start = usize::try_from(start)?;
                let mut w = [0; WEIGHT_FIXED_LEN];
                w[start..start + weight.len()].copy_from_slice(&weight);
                let id = Self::check_index(self.n_fixed)?;
                self.n_fixed += 1;
                return Ok(PatternValue { weight: w, id });
            }
        }

        let id = Self::check_index(self.variable.len())? | VARIABLE_WEIGHT_FLAG;
        // The fixed-length vector can only be used if the range of the weight is within the
        // padding of the score array, which has `WEIGHT_FIXED_LEN - 1` elements on each side.
        let weight = if (1 - i16::try_from(WEIGHT_FIXED_LEN)?..0).contains(&offset) {
            WeightVector::from(weight)
        } else {
            WeightVector::Variable(weight)
        };
        self.variable.push(PositionalWeight { offset, weight });
        Ok(PatternValue {
            #[cfg(feature = "fix-weight-length")]
            weight: [0; WEIGHT_FIXED_LEN],
            id,
        })
    }

    /// Returns a value that should be associated with the `index`-th pattern without a boundary
    /// weight.
    #[cfg(feature = "tag-prediction")]
    pub fn no_weight_value(index: usize) -> Result<PatternValue> {
        Ok(PatternValue {
            #[cfg(feature = "fix-weight-length")]
            weight: [0; WEIGHT_FIXED_LEN],
            id: Self::check_index(index)? | NO_WEIGHT_FLAG,
        })
    }

    fn check_index(index: usize) -> Result<u32> {
        let index = u32::try_from(index)?;
        if index > VALUE_INDEX_MASK {
            return Err(VaporettoError::invalid_model("too many patterns"));
        }
        Ok(index)
    }

    /// Adds the weight specified by `value` to `ys`. `end` is the position in `ys` corresponding
    /// to the end of the pattern.
    ///
    /// # Safety
    ///
    /// `value` must be a value returned by [`BoundaryWeights::push()`] or
    /// [`BoundaryWeights::no_weight_value()`] of this storage.
    #[inline(always)]
    pub unsafe fn add_score(&self, value: &PatternValue, end: usize, ys: &mut [i32]) {
        if value.id & VARIABLE_WEIGHT_FLAG != 0 {
            let index = usize::try_from(value.id & VALUE_INDEX_MASK).unwrap();
            debug_assert!(index < self.variable.len());
            self.variable
                .get_unchecked(index)
                .add_score(isize::try_from(end).unwrap(), ys);
            return;
        }

        #[cfg(feature = "fix-weight-length")]
        if value.id & NO_WEIGHT_FLAG == 0 {
            let pos = end.wrapping_add_signed(isize::from(self.fixed_offset));
            let ys = &mut ys[pos..pos + WEIGHT_FIXED_LEN];

            #[cfg(not(feature = "portable-simd"))]
            for (y, x) in ys.iter_mut().zip(&value.weight) {
                *y += *x;
            }

            #[cfg(feature = "portable-simd")]
            {
                let mut y = I32Simd::from_slice(ys);
                y += I32Simd::from_array(value.weight);
                ys.copy_from_slice(y.as_array());
            }
        }
    }
}

/// Tag weights of all tokens and relative positions.
///
/// All weights are stored in a single hash map instead of a map for each token and position
/// because it causes fewer cache misses.
///
/// Values of patterns having tag weights must contain [`TAG_WEIGHT_FLAG`]. Patterns without the
/// flag are skipped without accessing the map.
#[cfg(feature = "tag-prediction")]
#[derive(Deserialize, Serialize)]
pub struct TagWeights {
    n_positions: u32,
    map: HashMap<u64, WeightVector, SplitMix64Builder>,
}

#[cfg(feature = "tag-prediction")]
impl TagWeights {
    /// Creates a new storage for tag weights of `n_tokens` tokens. Relative positions must be
    /// smaller than or equal to `window_size`.
    pub fn new(n_tokens: usize, window_size: u8) -> Result<Self> {
        let n_positions = u32::from(window_size) + 1;
        // Keys of the map contain `token_id * n_positions + rel_position` in the upper 32 bits.
        if u32::try_from(n_tokens)?.checked_mul(n_positions).is_none() {
            return Err(VaporettoError::invalid_model("too many tag models"));
        }
        Ok(Self {
            n_positions,
            map: HashMap::default(),
        })
    }

    #[inline(always)]
    fn key(&self, token_id: u32, rel_position: u32, value: u32) -> u64 {
        ((u64::from(token_id) * u64::from(self.n_positions) + u64::from(rel_position)) << 32)
            | u64::from(value)
    }

    /// Inserts a weight of the given token and relative position associated with the pattern of
    /// the given value.
    ///
    /// `token_id` must be smaller than `n_tokens` given to [`TagWeights::new()`], and `value` must
    /// contain [`TAG_WEIGHT_FLAG`].
    pub fn insert(
        &mut self,
        token_id: usize,
        rel_position: u8,
        value: u32,
        weight: Vec<i32>,
    ) -> Result<()> {
        debug_assert!(value & TAG_WEIGHT_FLAG != 0);
        if u32::from(rel_position) >= self.n_positions {
            return Err(VaporettoError::invalid_model(
                "relative positions of tag n-grams must be smaller than or equal to the window size",
            ));
        }
        let key = self.key(u32::try_from(token_id)?, u32::from(rel_position), value);
        self.map.insert(key, weight.into());
        Ok(())
    }

    /// Adds tag weights of the given token to `scores`. `values` is a sequence of pattern
    /// values whose `i`-th item corresponds to the relative position `i`.
    #[inline(always)]
    pub fn add_scores(&self, token_id: u32, values: &[u32], scores: &mut [i32]) {
        for (rel_position, &value) in (0..self.n_positions).zip(values) {
            // Most patterns have no tag weights, so they are skipped without accessing the map.
            if value & TAG_WEIGHT_FLAG == 0 {
                continue;
            }
            if let Some(weight) = self.map.get(&self.key(token_id, rel_position, value)) {
                weight.add_scores(scores);
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
            self.weight.clone_from(&other.weight);
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
#[derive(Deserialize, Serialize)]
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

/// WARNING: Deserialization is inherently unsafe. Do not publish this struct outside this crate.
#[derive(Deserialize, Serialize)]
pub struct PredictorData {
    char_scorer: Option<CharScorer>,
    type_scorer: Option<TypeScorer>,
    bias: i32,

    #[cfg(feature = "tag-prediction")]
    tag_predictor: Option<HashMap<String, (u32, TagPredictor)>>,
    #[cfg(feature = "tag-prediction")]
    n_tags: usize,
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
pub struct Predictor {
    data: PredictorData,
    #[cfg(feature = "tag-prediction")]
    tag_scores: bool,
}

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
            tag_predictor
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
        Ok(Self {
            data: PredictorData {
                char_scorer,
                type_scorer,
                bias: model.0.bias,

                #[cfg(feature = "tag-prediction")]
                tag_predictor,
                #[cfg(feature = "tag-prediction")]
                n_tags,
            },
            #[cfg(feature = "tag-prediction")]
            tag_scores: false,
        })
    }

    /// Stores tag scores if the given `flag` is `true`.
    #[cfg(feature = "tag-prediction")]
    pub const fn store_tag_scores(&mut self, flag: bool) {
        self.tag_scores = flag;
    }

    /// Predicts word boundaries of the given sentence.
    /// If necessary, this function also prepares for predicting tags.
    ///
    /// Tags of the given sentence are removed. Use [`Sentence::fill_tags()`] to predict tags.
    pub fn predict<'a>(&'a self, sentence: &mut Sentence<'_, 'a>) {
        sentence.score_padding = WEIGHT_FIXED_LEN - 1;
        sentence.boundary_scores.clear();
        sentence.boundary_scores.resize(
            sentence.score_padding * 2 + sentence.len() - 1,
            self.data.bias,
        );
        if let Some(scorer) = self.data.char_scorer.as_ref() {
            scorer.add_scores(sentence);
        }
        if let Some(scorer) = self.data.type_scorer.as_ref() {
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
        sentence.tags.clear();
        #[cfg(feature = "tag-prediction")]
        sentence.tag_scores.clear();
        sentence.n_tags = 0;
        sentence.set_predictor(self);
    }

    #[cfg(feature = "tag-prediction")]
    pub(crate) fn predict_tags<'a>(&'a self, sentence: &mut Sentence<'_, 'a>) {
        let tag_predictor = self
            .data
            .tag_predictor
            .as_ref()
            .expect("this predictor is created with predict_tags = false");

        if self.data.n_tags == 0 {
            return;
        }
        let mut scores = vec![];
        let mut range_start = Some(0);
        sentence.n_tags = self.data.n_tags;
        sentence.tags.clear();
        sentence
            .tags
            .resize(sentence.len() * self.data.n_tags, None);
        sentence.tag_scores.clear();
        if self.tag_scores {
            sentence.tag_scores.resize(sentence.len(), None);
        }
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
                        if let Some(scorer) = self.data.char_scorer.as_ref() {
                            debug_assert!(i < sentence.char_pma_states.len());
                            // The states contain `sentence.len()` items because `predict()` initializes them.
                            unsafe {
                                scorer.add_tag_scores(*token_id, i, sentence, &mut scores);
                            }
                        }
                        if let Some(scorer) = self.data.type_scorer.as_ref() {
                            debug_assert!(i < sentence.type_pma_states.len());
                            // The states contain `sentence.len()` items because `predict()` initializes them.
                            unsafe {
                                scorer.add_tag_scores(*token_id, i, sentence, &mut scores);
                            }
                        }
                        tag_predictor.predict(
                            &scores,
                            &mut sentence.tags[i * self.data.n_tags..(i + 1) * self.data.n_tags],
                        );
                        if !sentence.tag_scores.is_empty() {
                            sentence.tag_scores[i].replace((&tag_predictor.tags, scores));
                            scores = vec![];
                        }
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
                if let Some(scorer) = self.data.char_scorer.as_ref() {
                    debug_assert!(sentence.len() <= sentence.char_pma_states.len());
                    // The states contain `sentence.len()` items because `predict()` initializes them.
                    unsafe {
                        scorer.add_tag_scores(*token_id, sentence.len() - 1, sentence, &mut scores);
                    }
                }
                if let Some(scorer) = self.data.type_scorer.as_ref() {
                    debug_assert!(sentence.len() <= sentence.type_pma_states.len());
                    // The states contain `sentence.len()` items because `predict()` initializes them.
                    unsafe {
                        scorer.add_tag_scores(*token_id, sentence.len() - 1, sentence, &mut scores);
                    }
                }
                let i = sentence.len() - 1;
                tag_predictor.predict(&scores, &mut sentence.tags[i * self.data.n_tags..]);
                if !sentence.tag_scores.is_empty() {
                    sentence.tag_scores[i].replace((&tag_predictor.tags, scores));
                }
            }
        }
    }

    /// Serializes the predictor into a Vec.
    ///
    /// # Errors
    ///
    /// When postcard generates an error, it will be returned as is.
    pub fn serialize_to_vec(&self) -> Result<Vec<u8>> {
        Ok(postcard::to_allocvec(&self.data)?)
    }

    /// Deserializes a predictor from a given slice and returns a tuple of the predictor and the remaining slice.
    ///
    /// # Safety
    ///
    /// The given data must be a correct predictor exported by [`Predictor::serialize_to_vec()`]
    /// function.
    pub unsafe fn deserialize_from_slice_unchecked(data: &[u8]) -> Result<(Self, &[u8])> {
        // Deserialization is unsafe because the automaton will not be verified.
        let (predictor_data, rest) = postcard::take_from_bytes(data)?;
        Ok((
            Self {
                data: predictor_data,
                #[cfg(feature = "tag-prediction")]
                tag_scores: false,
            },
            rest,
        ))
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

    #[test]
    fn test_predict_removes_tags() {
        let model = create_test_model();
        let predictor = Predictor::new(model, false).unwrap();
        let mut sentence = Sentence::from_tokenized("この/名詞 人は地球人だ").unwrap();
        predictor.predict(&mut sentence);

        assert_eq!(0, sentence.n_tags());
        assert!(sentence.tags().is_empty());
        let mut buf = alloc::string::String::new();
        sentence.write_tokenized_text(&mut buf);
        assert_eq!("この 人 は 地球 人 だ", buf);
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

    #[test]
    fn test_predict_short_weight_with_long_window() {
        // Weights shorter than the standard length are placed far from the end of the n-gram.
        let model = Model::new(
            NgramModel(vec![
                NgramData {
                    ngram: "a".into(),
                    weights: vec![5],
                },
                NgramData {
                    ngram: "b".into(),
                    weights: vec![7, 11],
                },
            ]),
            NgramModel(vec![]),
            DictModel(vec![]),
            0,
            10,
            3,
            vec![],
        );
        let predictor = Predictor::new(model, false).unwrap();
        let mut sentence = Sentence::from_raw("abc").unwrap();
        predictor.predict(&mut sentence);
        assert_eq!(&[0, 0], sentence.boundary_scores());

        let mut sentence = Sentence::from_raw("cccccccccccccb").unwrap();
        predictor.predict(&mut sentence);
        assert_eq!(
            &[0, 0, 0, 7, 11, 0, 0, 0, 0, 0, 0, 0, 0],
            sentence.boundary_scores()
        );
    }

    #[test]
    fn test_predict_dictionary_with_window_size_0() {
        let model = Model::new(
            NgramModel(vec![]),
            NgramModel(vec![]),
            DictModel(vec![WordWeightRecord {
                word: "火星".into(),
                weights: vec![100, -100, 100],
                comment: "".into(),
            }]),
            -1,
            0,
            3,
            vec![],
        );
        let predictor = Predictor::new(model, false).unwrap();
        let mut sentence = Sentence::from_raw("の火星だ").unwrap();
        predictor.predict(&mut sentence);
        assert_eq!(&[99, -101, 99], sentence.boundary_scores());
    }

    #[cfg(feature = "tag-prediction")]
    fn create_tag_only_model(char_ngram: bool, type_ngram: bool) -> Model {
        let char_ngram_model = if char_ngram {
            TagNgramModel(vec![TagNgramData {
                ngram: "人".into(),
                weights: vec![TagWeight {
                    rel_position: 0,
                    weights: vec![0, 20],
                }],
            }])
        } else {
            TagNgramModel(vec![])
        };
        let type_ngram_model = if type_ngram {
            TagNgramModel(vec![TagNgramData {
                ngram: vec![Kanji as u8],
                weights: vec![TagWeight {
                    rel_position: 0,
                    weights: vec![0, 20],
                }],
            }])
        } else {
            TagNgramModel(vec![])
        };
        Model::new(
            NgramModel(vec![]),
            NgramModel(vec![]),
            DictModel(vec![]),
            1,
            3,
            3,
            vec![TagModel {
                token: "人".into(),
                tags: vec![vec!["名詞".into(), "接尾辞".into()]],
                char_ngram_model,
                type_ngram_model,
                bias: vec![10, 0],
            }],
        )
    }

    #[cfg(feature = "tag-prediction")]
    #[test]
    fn test_predict_tags_without_boundary_ngrams() {
        // The bias selects 名詞, but the tag n-gram weights select 接尾辞.
        for (char_ngram, type_ngram) in [(true, false), (false, true)] {
            let model = create_tag_only_model(char_ngram, type_ngram);
            let predictor = Predictor::new(model, true).unwrap();
            let mut sentence = Sentence::from_raw("人").unwrap();
            predictor.predict(&mut sentence);
            sentence.fill_tags();
            let mut buf = String::new();
            sentence.write_tokenized_text(&mut buf);
            assert_eq!(
                "人/接尾辞", buf,
                "char_ngram={char_ngram}, type_ngram={type_ngram}"
            );
        }
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
