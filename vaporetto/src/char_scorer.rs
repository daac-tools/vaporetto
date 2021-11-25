use crate::model::ScoreValue;
use crate::sentence::Sentence;
use daachorse::DoubleArrayAhoCorasick;

pub enum CharScorer {
    Pma(CharScorerPma),
}

impl CharScorer {
    pub fn new(
        pma: DoubleArrayAhoCorasick,
        weights: Vec<Vec<ScoreValue>>,
        window_size: usize,
    ) -> Self {
        Self::Pma(CharScorerPma::new(pma, weights, window_size))
    }

    pub fn add_scores(&self, sentence: &Sentence, start: usize, ys: &mut [ScoreValue]) {
        match self {
            CharScorer::Pma(pma) => pma.add_scores(sentence, start, ys),
        }
    }
}

pub struct CharScorerPma {
    pma: DoubleArrayAhoCorasick,
    weights: Vec<Vec<ScoreValue>>,
    window_size: usize,
}

impl CharScorerPma {
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

    pub fn add_scores(&self, sentence: &Sentence, start: usize, ys: &mut [ScoreValue]) {
        let char_start = if start >= self.window_size {
            start + 1 - self.window_size
        } else {
            0
        };
        let text_start = sentence.char_to_str_pos[char_start];
        let char_end = std::cmp::min(
            start + ys.len() + self.window_size,
            sentence.char_to_str_pos.len() - 1,
        );
        let text_end = sentence.char_to_str_pos[char_end];
        let text = &sentence.text[text_start..text_end];
        let padding = start - char_start + 1;
        for m in self.pma.find_overlapping_no_suffix_iter(&text) {
            let m_end = sentence.str_to_char_pos[m.end() + text_start] - char_start;
            let offset = m_end as isize - self.window_size as isize - padding as isize;
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
