use alloc::vec::Vec;

use bincode::{Decode, Encode};
use daachorse::DoubleArrayAhoCorasick;

use crate::errors::{Result, VaporettoError};
use crate::ngram_model::NgramModel;
use crate::sentence::Sentence;

const ALPHABET_SIZE: usize = 8;
const ALPHABET_MASK: usize = ALPHABET_SIZE - 1;
const ALPHABET_SHIFT: usize = 3;

#[derive(Decode, Encode)]
pub struct TypeScorerBoundaryCache {
    scores: Vec<i32>,
    window_size: u8,
    sequence_mask: usize,
}

impl TypeScorerBoundaryCache {
    pub fn new(model: NgramModel<Vec<u8>>, window_size: u8) -> Result<Self> {
        let pma = DoubleArrayAhoCorasick::new(model.0.iter().map(|d| &d.ngram))
            .map_err(|_| VaporettoError::invalid_model("invalid character type n-grams"))?;
        let mut weights = vec![];
        for d in model.0 {
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

    #[inline(always)]
    pub fn add_scores<'a, 'b>(&self, sentence: &mut Sentence<'a, 'b>) {
        sentence.type_pma_states.clear();
        let mut seqid = 0;
        for i in 0..self.window_size {
            if let Some(ct) = sentence.char_types.get(usize::from(i)) {
                seqid = self.increment_seqid(seqid, *ct);
            } else {
                seqid = self.increment_seqid_without_char(seqid);
            };
        }
        for (i, y) in sentence.boundary_scores
            [sentence.score_padding..sentence.score_padding + sentence.boundaries.len()]
            .iter_mut()
            .enumerate()
        {
            if let Some(ct) = sentence.char_types.get(i + usize::from(self.window_size)) {
                seqid = self.increment_seqid(seqid, *ct);
            } else {
                seqid = self.increment_seqid_without_char(seqid);
            };
            *y += self.get_score(seqid);
        }
    }

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
