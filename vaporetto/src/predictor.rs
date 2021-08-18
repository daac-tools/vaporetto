use std::ops::Range;

#[cfg(feature = "multithreading")]
use std::cell::RefCell;
#[cfg(feature = "multithreading")]
use std::sync::Arc;
#[cfg(feature = "multithreading")]
use std::thread;

use aho_corasick::{AhoCorasick, AhoCorasickBuilder};

#[cfg(feature = "multithreading")]
use crossbeam_channel::{Receiver, Sender};

use crate::model::{Model, ScoreValue};
use crate::sentence::{BoundaryType, Sentence};

pub struct Predictor {
    word_pma: AhoCorasick,
    type_pma: AhoCorasick,
    dict_pma: AhoCorasick,
    word_weights: Vec<Vec<ScoreValue>>,
    type_weights: Vec<Vec<ScoreValue>>,
    dict_weights: Vec<[ScoreValue; 3]>,
    dict_word_wise: bool,
    bias: ScoreValue,
    char_window_size: usize,
    type_window_size: usize,
    dict_overwrap_size: usize,

    #[cfg(feature = "model-quantize")]
    quantize_multiplier: f64,
}

impl Predictor {
    pub fn new(model: Model, use_dfa: bool) -> Self {
        let mut words = Vec::with_capacity(model.word_fst.len());
        for i in 0..model.word_fst.len() as u64 {
            words.push(model.word_fst.get_key(i).unwrap())
        }
        let mut types = Vec::with_capacity(model.type_fst.len());
        for i in 0..model.type_fst.len() as u64 {
            types.push(model.type_fst.get_key(i).unwrap())
        }
        let mut dict = Vec::with_capacity(model.dict_fst.len());
        for i in 0..model.dict_fst.len() as u64 {
            dict.push(model.dict_fst.get_key(i).unwrap())
        }

        let bias = model.bias;

        let mut word_weights = vec![];
        for word in &words {
            let mut weights: Option<Vec<_>> = None;
            for st in (0..word.len()).rev() {
                if let Some(idx) = model.word_fst.get(&word[st..]) {
                    let idx = idx.value() as usize;
                    if let Some(weights) = weights.as_mut() {
                        for (i, &w) in model.word_weights[idx].iter().enumerate() {
                            weights[i] += w as ScoreValue;
                        }
                    } else {
                        weights.replace(
                            model.word_weights[idx]
                                .iter()
                                .map(|&w| w as ScoreValue)
                                .collect(),
                        );
                    }
                }
            }
            word_weights.push(weights.unwrap());
        }

        let mut type_weights = vec![];
        for type_seq in &types {
            let mut weights: Option<Vec<_>> = None;
            for st in (0..type_seq.len()).rev() {
                if let Some(idx) = model.type_fst.get(&type_seq[st..]) {
                    let idx = idx.value() as usize;
                    if let Some(weights) = weights.as_mut() {
                        for (i, &w) in model.type_weights[idx].iter().enumerate() {
                            weights[i] += w as ScoreValue;
                        }
                    } else {
                        weights.replace(
                            model.type_weights[idx]
                                .iter()
                                .map(|&w| w as ScoreValue)
                                .collect(),
                        );
                    }
                }
            }
            type_weights.push(weights.unwrap());
        }

        let dict_weights = model.dict_weights;

        #[cfg(feature = "model-quantize")]
        let bias = bias as i32;
        #[cfg(feature = "model-quantize")]
        let dict_weights = dict_weights
            .iter()
            .map(|ws| [ws[0] as i32, ws[1] as i32, ws[2] as i32])
            .collect();

        let word_pma = AhoCorasickBuilder::new().dfa(use_dfa).build(words);
        let type_pma = AhoCorasickBuilder::new().dfa(use_dfa).build(types);
        let dict_pma = AhoCorasickBuilder::new().dfa(use_dfa).build(dict);
        Self {
            word_pma,
            type_pma,
            dict_pma,
            word_weights,
            type_weights,
            dict_weights,
            dict_word_wise: model.dict_word_wise,
            bias,
            char_window_size: model.char_window_size,
            type_window_size: model.type_window_size,
            dict_overwrap_size: 1,

            #[cfg(feature = "model-quantize")]
            quantize_multiplier: model.quantize_multiplier,
        }
    }

    unsafe fn add_word_ngram_scores(
        &self,
        sentence: &Sentence,
        start: usize,
        ys: &mut [ScoreValue],
    ) {
        let char_start = if start >= self.char_window_size {
            start + 1 - self.char_window_size
        } else {
            0
        };
        let text_start = *sentence.char_to_str_pos.get_unchecked(char_start);
        let char_end = std::cmp::min(
            start + ys.len() + self.char_window_size,
            sentence.char_to_str_pos.len() - 1,
        );
        let text_end = *sentence.char_to_str_pos.get_unchecked(char_end);
        let text = &sentence.text.get_unchecked(text_start..text_end);

        let padding = start - char_start + 1;

        let mut prev_end = 0;
        for m in self.word_pma.find_overlapping_iter(&text) {
            if m.end() == prev_end {
                continue;
            }
            prev_end = m.end();

            let m_end = *sentence.str_to_char_pos.get_unchecked(m.end() + text_start) - char_start;
            let offset = m_end as isize - self.char_window_size as isize - padding as isize;
            if offset >= 0 {
                let weights = self.word_weights.get_unchecked(m.pattern());
                let ys = ys.get_unchecked_mut(offset as usize..);
                for (w, y) in weights.iter().zip(ys.iter_mut()) {
                    *y += w;
                }
            } else {
                let weights = self
                    .word_weights
                    .get_unchecked(m.pattern())
                    .get_unchecked(-offset as usize..);
                for (w, y) in weights.iter().zip(ys.iter_mut()) {
                    *y += w;
                }
            }
        }
    }

    unsafe fn add_type_ngram_scores(
        &self,
        sentence: &Sentence,
        start: usize,
        ys: &mut [ScoreValue],
    ) {
        let type_start = if start >= self.type_window_size {
            start + 1 - self.type_window_size
        } else {
            0
        };
        let type_end = std::cmp::min(
            start + ys.len() + self.type_window_size,
            sentence.char_type.len(),
        );
        let char_type = sentence.char_type.get_unchecked(type_start..type_end);

        let padding = start - type_start + 1;

        let mut prev_end = 0;
        for m in self.type_pma.find_overlapping_iter(&char_type) {
            if m.end() == prev_end {
                continue;
            }
            prev_end = m.end();

            let offset = m.end() as isize - self.type_window_size as isize - padding as isize;
            if offset >= 0 {
                let weights = self.type_weights.get_unchecked(m.pattern());
                let ys = ys.get_unchecked_mut(offset as usize..);
                for (w, y) in weights.iter().zip(ys.iter_mut()) {
                    *y += w;
                }
            } else {
                let weights = self
                    .type_weights
                    .get_unchecked(m.pattern())
                    .get_unchecked(-offset as usize..);
                for (w, y) in weights.iter().zip(ys.iter_mut()) {
                    *y += w;
                }
            }
        }
    }

    unsafe fn add_dict_scores(&self, sentence: &Sentence, start: usize, ys: &mut [ScoreValue]) {
        let char_start = if start >= self.dict_overwrap_size {
            start + 1 - self.dict_overwrap_size
        } else {
            0
        };
        let text_start = *sentence.char_to_str_pos.get_unchecked(char_start);
        let char_end = std::cmp::min(
            start + ys.len() + self.dict_overwrap_size,
            sentence.char_to_str_pos.len() - 1,
        );
        let text_end = *sentence.char_to_str_pos.get_unchecked(char_end);
        let text = &sentence.text.get_unchecked(text_start..text_end);

        let padding = start - char_start + 1;

        for m in self.dict_pma.find_overlapping_iter(&text) {
            let m_start = *sentence
                .str_to_char_pos
                .get_unchecked(m.start() + text_start)
                - char_start;
            let m_end = *sentence.str_to_char_pos.get_unchecked(m.end() + text_start) - char_start;

            let idx = if self.dict_word_wise {
                m.pattern()
            } else {
                std::cmp::min(m_end - m_start - 1, self.dict_weights.len())
            };
            let weights = self.dict_weights.get_unchecked(idx);
            if m_start >= padding && m_start < padding + ys.len() {
                *ys.get_unchecked_mut(m_start - padding) += weights.get_unchecked(0);
            }
            let range_start = std::cmp::max(0, m_start as isize - padding as isize + 1);
            let range_end = std::cmp::min(m_end as isize - padding as isize, ys.len() as isize);
            if range_start < range_end {
                for y in ys.get_unchecked_mut(range_start as usize..range_end as usize) {
                    *y += weights.get_unchecked(1);
                }
            }
            if m_end >= padding && m_end < ys.len() + padding {
                *ys.get_unchecked_mut(m_end - padding) += weights.get_unchecked(2);
            }
        }
    }

    fn predict_partial_impl(
        &self,
        sentence: &Sentence,
        range: Range<usize>,
        ys: &mut [ScoreValue],
    ) {
        if range.start >= range.end || range.start >= sentence.boundaries.len() {
            panic!("invalid range: {:?}", range);
        }
        ys.fill(self.bias);
        unsafe {
            self.add_word_ngram_scores(sentence, range.start, ys);
            self.add_type_ngram_scores(sentence, range.start, ys);
            self.add_dict_scores(sentence, range.start, ys);
        }
    }

    pub fn predict_partial(&self, sentence: Sentence, range: Range<usize>) -> Sentence {
        let mut ys = vec![ScoreValue::default(); range.end - range.start];
        self.predict_partial_impl(&sentence, range.clone(), &mut ys);
        let mut sentence = sentence;
        for (y, b) in ys.into_iter().zip(sentence.boundaries[range].iter_mut()) {
            if y >= ScoreValue::default() {
                *b = BoundaryType::WordBoundary;
            } else {
                *b = BoundaryType::NotWordBoundary;
            }
        }
        sentence
    }

    pub fn predict_partial_with_score(&self, sentence: Sentence, range: Range<usize>) -> Sentence {
        let mut ys = vec![ScoreValue::default(); range.end - range.start];
        self.predict_partial_impl(&sentence, range.clone(), &mut ys);
        let mut sentence = sentence;
        let mut scores = sentence.boundary_scores.take().unwrap_or_else(|| vec![0.; sentence.boundaries.len()]);
        for (y, (b, s)) in ys.into_iter().zip(sentence.boundaries[range.clone()].iter_mut().zip(scores[range].iter_mut())) {
            if y >= ScoreValue::default() {
                *b = BoundaryType::WordBoundary;
            } else {
                *b = BoundaryType::NotWordBoundary;
            }
            #[cfg(not(feature = "model-quantize"))]
            {
                *s = y;
            }
            #[cfg(feature = "model-quantize")]
            {
                *s = y as f64 * self.quantize_multiplier;
            }
        }
        sentence.boundary_scores.replace(scores);
        sentence
    }

    pub fn predict(&self, sentence: Sentence) -> Sentence {
        let boundaries_size = sentence.boundaries.len();
        if boundaries_size == 0 {
            sentence
        } else {
            self.predict_partial(sentence, 0..boundaries_size)
        }
    }

    pub fn predict_with_score(&self, sentence: Sentence) -> Sentence {
        let boundaries_size = sentence.boundaries.len();
        if boundaries_size == 0 {
            sentence
        } else {
            self.predict_partial_with_score(sentence, 0..boundaries_size)
        }
    }

    pub fn dict_overwrap_size(mut self, size: usize) -> Self {
        if size >= 1 {
            self.dict_overwrap_size = size;
        } else {
            self.dict_overwrap_size = 1;
        }
        self
    }

    #[cfg(feature = "multithreading")]
    pub fn multithreading(self, n_threads: usize, chunk_size: usize) -> MultithreadPredictor {
        MultithreadPredictor::new(self, n_threads, chunk_size)
    }
}

#[cfg(feature = "multithreading")]
pub struct MultithreadPredictor {
    task_tx: Sender<(Arc<Sentence>, Range<usize>, Vec<ScoreValue>)>,
    result_rx: Receiver<(Vec<ScoreValue>, Range<usize>)>,
    chunk_size: usize,
    ys_pool: RefCell<Vec<Vec<ScoreValue>>>,
}

#[cfg(feature = "multithreading")]
impl MultithreadPredictor {
    pub fn new(predictor: Predictor, n_threads: usize, chunk_size: usize) -> Self {
        let predictor = Arc::new(predictor);

        let (result_tx, result_rx) = crossbeam_channel::unbounded();
        let (task_tx, task_rx) =
            crossbeam_channel::unbounded::<(Arc<Sentence>, Range<usize>, Vec<ScoreValue>)>();
        for _ in 0..n_threads {
            let predictor = Arc::clone(&predictor);
            let result_tx = result_tx.clone();
            let task_rx = task_rx.clone();
            thread::spawn(move || {
                for (sentence, range, mut ys) in task_rx {
                    predictor.predict_partial_impl(
                        &sentence,
                        range.clone(),
                        &mut ys[..range.end - range.start],
                    );
                    std::mem::drop(sentence);
                    result_tx.send((ys, range)).unwrap();
                }
            });
        }

        Self {
            task_tx,
            result_rx,
            chunk_size,
            ys_pool: RefCell::new(vec![]),
        }
    }

    pub fn predict(&self, sentence: Sentence) -> Sentence {
        let sentence = Arc::new(sentence);

        let mut n_chunks = 0;
        let mut ys_pool = self.ys_pool.borrow_mut();
        for start in (0..sentence.boundaries.len()).step_by(self.chunk_size) {
            let ys = if let Some(ys) = ys_pool.pop() {
                ys
            } else {
                vec![ScoreValue::default(); self.chunk_size]
            };
            let sentence = Arc::clone(&sentence);
            let end = std::cmp::min(start + self.chunk_size, sentence.boundaries.len());
            self.task_tx.send((sentence, start..end, ys)).unwrap();
            n_chunks += 1;
        }
        let mut boundaries = vec![BoundaryType::Unknown; sentence.boundaries.len()];
        for _ in 0..n_chunks {
            let (ys, range) = self.result_rx.recv().unwrap();
            for (y, b) in ys.iter().zip(&mut boundaries[range]) {
                if *y >= ScoreValue::default() {
                    *b = BoundaryType::WordBoundary;
                } else {
                    *b = BoundaryType::NotWordBoundary;
                }
            }
            ys_pool.push(ys);
        }

        let mut sentence = Arc::try_unwrap(sentence).unwrap();
        sentence.boundaries = boundaries;
        sentence
    }
}
