use std::cell::RefCell;
use std::collections::BTreeMap;

use daachorse::DoubleArrayAhoCorasick;

use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::sentence::Sentence;
use crate::utils::AddWeight;

pub enum TypeScorer {
    Pma(TypeScorerPma),
    Cache(TypeScorerCache),
}

impl TypeScorer {
    pub fn new(model: NgramModel<Vec<u8>>, window_size: usize) -> Result<Self> {
        Ok(if window_size <= 3 {
            Self::Cache(TypeScorerCache::new(model, window_size)?)
        } else {
            Self::Pma(TypeScorerPma::new(model, window_size)?)
        })
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [i32]) {
        match self {
            TypeScorer::Pma(pma) => pma.add_scores(sentence, ys),
            TypeScorer::Cache(cache) => cache.add_scores(sentence, ys),
        }
    }
}

pub struct TypeScorerPma {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<Vec<i32>>,
    window_size: usize,
}

impl TypeScorerPma {
    pub fn new(model: NgramModel<Vec<u8>>, window_size: usize) -> Result<Self> {
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

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [i32]) {
        for m in self
            .pma
            .find_overlapping_no_suffix_iter(&sentence.char_type)
        {
            let offset = m.end() as isize - self.window_size as isize - 1;
            // Both the weights and the PMA always have the same number of items.
            // Therefore, the following code is safe.
            let weights = unsafe { self.weights.get_unchecked(m.value()) };
            weights.add_weight(ys, offset);
        }
    }
}

pub struct TypeScorerCache {
    scores: Vec<i32>,
    window_size: usize,
    sequence_mask: usize,
}

impl TypeScorerCache {
    pub fn new(model: NgramModel<Vec<u8>>, window_size: usize) -> Result<Self> {
        let pma = DoubleArrayAhoCorasick::new(model.data.iter().map(|d| &d.ngram))
            .map_err(|_| VaporettoError::invalid_model("invalid character type n-grams"))?;
        let mut weights = vec![];
        for d in model.data {
            if d.weights.len() <= 2 * window_size - d.ngram.len() {
                return Err(VaporettoError::invalid_model(
                    "invalid size of weight vector",
                ));
            }
            weights.push(d.weights);
        }

        let sequence_size = window_size * 2;
        let all_sequences = ALPHABET_SIZE.pow(sequence_size as u32);

        let mut sequence = vec![0u8; sequence_size];
        let mut scores = vec![0; all_sequences];

        for (i, score) in scores.iter_mut().enumerate() {
            if !Self::seqid_to_seq(i, &mut sequence) {
                continue;
            }
            let mut y = 0;
            for m in pma.find_overlapping_iter(&sequence) {
                y += weights[m.value()][sequence_size - m.end()];
            }
            *score = y;
        }

        Ok(Self {
            scores,
            window_size,
            sequence_mask: (1 << (ALPHABET_SHIFT * sequence_size)) - 1,
        })
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [i32]) {
        let mut seqid = 0;
        for i in 0..self.window_size {
            if let Some(ct) = sentence.char_type.get(i) {
                seqid = self.increment_seqid(seqid, *ct);
            } else {
                seqid = self.increment_seqid_without_char(seqid);
            };
        }
        for (i, y) in ys.iter_mut().enumerate() {
            if let Some(ct) = sentence.char_type.get(i + self.window_size) {
                seqid = self.increment_seqid(seqid, *ct);
            } else {
                seqid = self.increment_seqid_without_char(seqid);
            };
            *y += self.get_score(seqid);
        }
    }

    fn seqid_to_seq(mut seqid: usize, sequence: &mut [u8]) -> bool {
        for i in (0..sequence.len()).rev() {
            let x = seqid & ALPHABET_MASK;
            if x == ALPHABET_MASK {
                return false; // invalid
            }
            sequence[i] = ID_TO_TYPE[x];
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
        let char_id = TYPE_TO_ID[char_type as usize] as usize;
        debug_assert!((1..=6).contains(&char_id));
        ((seqid << ALPHABET_SHIFT) | char_id) & self.sequence_mask
    }

    #[inline(always)]
    const fn increment_seqid_without_char(&self, seqid: usize) -> usize {
        (seqid << ALPHABET_SHIFT) & self.sequence_mask
    }
}

const ALPHABET_SIZE: usize = 8;
const ALPHABET_MASK: usize = ALPHABET_SIZE - 1;
const ALPHABET_SHIFT: usize = 3;
const TYPE_TO_ID: [u32; 256] = make_type_to_id();
const ID_TO_TYPE: [u8; 256] = make_id_to_type();

const fn make_type_to_id() -> [u32; 256] {
    use crate::sentence::CharacterType::*;

    let mut type_to_id = [0u32; 256];
    type_to_id[Digit as usize] = 1;
    type_to_id[Roman as usize] = 2;
    type_to_id[Hiragana as usize] = 3;
    type_to_id[Katakana as usize] = 4;
    type_to_id[Kanji as usize] = 5;
    type_to_id[Other as usize] = 6;
    type_to_id
}

const fn make_id_to_type() -> [u8; 256] {
    use crate::sentence::CharacterType::*;

    let mut id_to_type = [0u8; 256];
    id_to_type[1] = Digit as u8;
    id_to_type[2] = Roman as u8;
    id_to_type[3] = Hiragana as u8;
    id_to_type[4] = Katakana as u8;
    id_to_type[5] = Kanji as u8;
    id_to_type[6] = Other as u8;
    id_to_type
}
