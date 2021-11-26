use crate::model::ScoreValue;
use crate::sentence::Sentence;
use daachorse::DoubleArrayAhoCorasick;

pub enum TypeScorer {
    Pma(TypeScorerPma),
    Cache(TypeScorerCache),
}

impl TypeScorer {
    pub fn new(
        pma: DoubleArrayAhoCorasick,
        weights: Vec<Vec<ScoreValue>>,
        window_size: usize,
    ) -> Self {
        if window_size <= 3 {
            Self::Cache(TypeScorerCache::new(pma, weights, window_size))
        } else {
            Self::Pma(TypeScorerPma::new(pma, weights, window_size))
        }
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [ScoreValue]) {
        match self {
            TypeScorer::Pma(pma) => pma.add_scores(sentence, ys),
            TypeScorer::Cache(cache) => cache.add_scores(sentence, ys),
        }
    }
}

pub struct TypeScorerPma {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<Vec<ScoreValue>>,
    window_size: usize,
}

impl TypeScorerPma {
    pub fn new(
        pma: DoubleArrayAhoCorasick,
        weights: Vec<Vec<ScoreValue>>,
        window_size: usize,
    ) -> Self {
        Self {
            pma,
            weights,
            window_size,
        }
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [ScoreValue]) {
        for m in self
            .pma
            .find_overlapping_no_suffix_iter(&sentence.char_type)
        {
            let offset = m.end() as isize - self.window_size as isize - 1;
            let weights = &self.weights[m.pattern()];
            if offset >= 0 {
                for (w, y) in weights.iter().zip(&mut ys[offset as usize..]) {
                    *y += w;
                }
            } else {
                for (w, y) in weights[-offset as usize..].iter().zip(ys.iter_mut()) {
                    *y += w;
                }
            }
        }
    }
}

pub struct TypeScorerCache {
    scores: Vec<ScoreValue>,
    window_size: usize,
    sequence_mask: usize,
}

impl TypeScorerCache {
    pub fn new(
        pma: DoubleArrayAhoCorasick,
        weights: Vec<Vec<ScoreValue>>,
        window_size: usize,
    ) -> Self {
        let sequence_size = window_size * 2;
        let all_sequences = ALPHABET_SIZE.pow(sequence_size as u32);

        let mut sequence = vec![0u8; sequence_size];
        let mut scores = vec![0 as ScoreValue; all_sequences];

        for (i, score) in scores.iter_mut().enumerate() {
            if !Self::seqid_to_seq(i, &mut sequence) {
                continue;
            }
            let mut y = ScoreValue::default();
            for m in pma.find_overlapping_no_suffix_iter(&sequence) {
                y += weights[m.pattern()][sequence_size - m.end()];
            }
            *score = y;
        }

        Self {
            scores,
            window_size,
            sequence_mask: (1 << (ALPHABET_SHIFT * sequence_size)) - 1,
        }
    }

    pub fn add_scores(&self, sentence: &Sentence, ys: &mut [ScoreValue]) {
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
    fn get_score(&self, seqid: usize) -> ScoreValue {
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
