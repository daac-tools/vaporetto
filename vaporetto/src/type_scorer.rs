use std::cell::RefCell;
use std::collections::BTreeMap;

use bincode::{de::BorrowDecoder, error::DecodeError, BorrowDecode, Decode, Encode};
use daachorse::DoubleArrayAhoCorasick;

use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::sentence::Sentence;
use crate::utils::AddWeight;

/// WARNING: The decode feature is inherently unsafe. Do not publish this feature outside this
/// crate.
#[derive(BorrowDecode, Encode)]
pub enum TypeScorer {
    Pma(TypeScorerPma),

    #[cfg(feature = "cache-type-score")]
    Cache(TypeScorerCache),
}

impl TypeScorer {
    pub fn new(model: NgramModel<Vec<u8>>, window_size: u8) -> Result<Self> {
        #[cfg(feature = "cache-type-score")]
        let scorer = if window_size <= 3 {
            Self::Cache(TypeScorerCache::new(model, window_size)?)
        } else {
            Self::Pma(TypeScorerPma::new(model, window_size)?)
        };

        #[cfg(not(feature = "cache-type-score"))]
        let scorer = Self::Pma(TypeScorerPma::new(model, window_size)?);

        Ok(scorer)
    }

    pub fn add_scores(&self, sentence: &Sentence, padding: u8, ys: &mut [i32]) {
        match self {
            TypeScorer::Pma(pma) => pma.add_scores(sentence, padding, ys),

            #[cfg(feature = "cache-type-score")]
            TypeScorer::Cache(cache) => cache.add_scores(sentence, &mut ys[padding.into()..]),
        }
    }
}

pub struct TypeScorerPma {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<Vec<i32>>,
    window_size: u8,
}

impl TypeScorerPma {
    pub fn new(model: NgramModel<Vec<u8>>, window_size: u8) -> Result<Self> {
        // key: ngram, value: (weight, check)
        let mut weights_map: BTreeMap<Vec<u8>, RefCell<(Vec<i32>, bool)>> = BTreeMap::new();

        for d in model.data {
            weights_map.insert(d.ngram, RefCell::new((d.weights, false)));
        }

        let mut stack = vec![];
        for (ngram, data) in &weights_map {
            if data.borrow().1 {
                continue;
            }
            stack.push(data);
            for j in 1..ngram.len() {
                if let Some(data) = weights_map.get(&ngram[j..]) {
                    stack.push(data);
                    if data.borrow().1 {
                        break;
                    }
                }
            }
            let mut data_from = stack.pop().unwrap();
            data_from.borrow_mut().1 = true;
            while let Some(data_to) = stack.pop() {
                let mut new_weight = data_from.borrow().0.clone();
                for (w1, w2) in new_weight.iter_mut().zip(&data_to.borrow().0) {
                    *w1 += w2;
                }
                let new_data = (new_weight, true);
                *data_to.borrow_mut() = new_data;
                data_from = data_to;
            }
        }
        let mut ngrams = vec![];
        let mut weights = vec![];
        for (ngram, data) in weights_map {
            ngrams.push(ngram);
            weights.push(data.into_inner().0);
        }
        let pma = DoubleArrayAhoCorasick::new(ngrams)
            .map_err(|_| VaporettoError::invalid_model("invalid character type n-grams"))?;
        Ok(Self {
            pma,
            weights,
            window_size,
        })
    }

    pub fn add_scores(&self, sentence: &Sentence, padding: u8, ys: &mut [i32]) {
        for m in self
            .pma
            .find_overlapping_no_suffix_iter(&sentence.char_type)
        {
            let offset = usize::from(padding) + m.end() - usize::from(self.window_size) - 1;
            // Both the weights and the PMA always have the same number of items.
            // Therefore, the following code is safe.
            let weights = unsafe { self.weights.get_unchecked(m.value()) };
            weights.add_weight(ys, offset);
        }
    }
}

impl<'de> BorrowDecode<'de> for TypeScorerPma {
    /// WARNING: This function is inherently unsafe. Do not publish this function outside this
    /// crate.
    fn borrow_decode<D: BorrowDecoder<'de>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let pma_data: &[u8] = BorrowDecode::borrow_decode(decoder)?;
        let (pma, _) =
            unsafe { DoubleArrayAhoCorasick::deserialize_from_slice_unchecked(pma_data) };
        Ok(Self {
            pma,
            weights: Decode::decode(decoder)?,
            window_size: Decode::decode(decoder)?,
        })
    }
}

impl Encode for TypeScorerPma {
    fn encode<E: bincode::enc::Encoder>(
        &self,
        encoder: &mut E,
    ) -> Result<(), bincode::error::EncodeError> {
        let pma_data = self.pma.serialize_to_vec();
        Encode::encode(&pma_data, encoder)?;
        Encode::encode(&self.weights, encoder)?;
        Encode::encode(&self.window_size, encoder)?;
        Ok(())
    }
}

#[cfg(feature = "cache-type-score")]
#[derive(Decode, Encode)]
pub struct TypeScorerCache {
    scores: Vec<i32>,
    window_size: u8,
    sequence_mask: usize,
}

#[cfg(feature = "cache-type-score")]
impl TypeScorerCache {
    pub fn new(model: NgramModel<Vec<u8>>, window_size: u8) -> Result<Self> {
        let pma = DoubleArrayAhoCorasick::new(model.data.iter().map(|d| &d.ngram))
            .map_err(|_| VaporettoError::invalid_model("invalid character type n-grams"))?;
        let mut weights = vec![];
        for d in model.data {
            if d.weights.len() <= 2 * usize::from(window_size) - d.ngram.len() {
                return Err(VaporettoError::invalid_model(
                    "invalid size of weight vector",
                ));
            }
            weights.push(d.weights);
        }

        let sequence_size = u16::from(window_size) * 2;
        let all_sequences = ALPHABET_SIZE.pow(sequence_size.into());

        let mut sequence = vec![0u8; sequence_size.into()];
        let mut scores = vec![0; all_sequences];

        for (i, score) in scores.iter_mut().enumerate() {
            if !Self::seqid_to_seq(i, &mut sequence) {
                continue;
            }
            let mut y = 0;
            for m in pma.find_overlapping_iter(&sequence) {
                y += weights[m.value()][usize::from(sequence_size) - m.end()];
            }
            *score = y;
        }

        Ok(Self {
            scores,
            window_size,
            sequence_mask: (1 << (ALPHABET_SHIFT * usize::from(sequence_size))) - 1,
        })
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [i32]) {
        let mut seqid = 0;
        for i in 0..self.window_size {
            if let Some(ct) = sentence.char_type.get(usize::from(i)) {
                seqid = self.increment_seqid(seqid, *ct);
            } else {
                seqid = self.increment_seqid_without_char(seqid);
            };
        }
        for (i, y) in ys.iter_mut().enumerate() {
            if let Some(ct) = sentence.char_type.get(i + usize::from(self.window_size)) {
                seqid = self.increment_seqid(seqid, *ct);
            } else {
                seqid = self.increment_seqid_without_char(seqid);
            };
            *y += self.get_score(seqid);
        }
    }

    #[allow(clippy::cast_possible_truncation)]
    fn seqid_to_seq(mut seqid: usize, sequence: &mut [u8]) -> bool {
        for type_id in sequence.iter_mut().rev() {
            *type_id = (seqid & ALPHABET_MASK) as u8;
            if usize::from(*type_id) == ALPHABET_MASK {
                return false; // invalid
            }
            seqid >>= ALPHABET_SHIFT;
        }
        assert_eq!(seqid, 0);
        true
    }

    #[inline(always)]
    fn get_score(&self, seqid: usize) -> i32 {
        self.scores[seqid]
    }

    #[inline(always)]
    fn increment_seqid(&self, seqid: usize, char_type: u8) -> usize {
        let char_id = usize::from(char_type);
        debug_assert!((1..=6).contains(&char_id));
        ((seqid << ALPHABET_SHIFT) | char_id) & self.sequence_mask
    }

    #[inline(always)]
    const fn increment_seqid_without_char(&self, seqid: usize) -> usize {
        (seqid << ALPHABET_SHIFT) & self.sequence_mask
    }
}

#[cfg(feature = "cache-type-score")]
const ALPHABET_SIZE: usize = 8;
#[cfg(feature = "cache-type-score")]
const ALPHABET_MASK: usize = ALPHABET_SIZE - 1;
#[cfg(feature = "cache-type-score")]
const ALPHABET_SHIFT: usize = 3;
