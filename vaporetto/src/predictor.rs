use std::collections::HashMap;
use std::ops::Range;

#[cfg(feature = "multithreading")]
use std::cell::RefCell;
#[cfg(feature = "multithreading")]
use std::sync::Arc;
#[cfg(feature = "multithreading")]
use std::thread;

#[cfg(feature = "multithreading")]
use crossbeam_channel::{Receiver, Sender};

use crate::model::{Model, ScoreValue, WeightValue};
use crate::sentence::{BoundaryType, Sentence};
use daachorse::DoubleArrayAhoCorasick;

/// Predictor.
pub struct Predictor {
    word_pma: DoubleArrayAhoCorasick,
    type_pma: DoubleArrayAhoCorasick,
    dict_pma: DoubleArrayAhoCorasick,
    word_weights: Vec<Vec<ScoreValue>>,
    type_weights: Vec<Vec<ScoreValue>>,
    dict_weights: Vec<[ScoreValue; 3]>,
    dict_word_wise: bool,
    bias: ScoreValue,
    char_window_size: usize,
    type_window_size: usize,
    dict_window_size: usize,

    #[cfg(feature = "model-quantize")]
    quantize_multiplier: f64,
}

impl Predictor {
    /// Creates a new predictor.
    ///
    /// # Arguments
    ///
    /// * `model` - A model data.
    ///
    /// # Returns
    ///
    /// A new predictor.
    pub fn new(model: Model) -> Self {
        let bias = model.bias;
        let word_weights = Self::merge_weights(&model.words, &model.word_weights);
        let type_weights = Self::merge_weights(&model.types, &model.type_weights);
        let dict_weights = model.dict_weights;

        #[cfg(feature = "model-quantize")]
        let bias = bias as i32;
        #[cfg(feature = "model-quantize")]
        let dict_weights = dict_weights
            .iter()
            .map(|ws| [ws[0] as i32, ws[1] as i32, ws[2] as i32])
            .collect();

        let word_pma = DoubleArrayAhoCorasick::new(model.words).unwrap();
        let type_pma = DoubleArrayAhoCorasick::new(model.types).unwrap();
        let dict_pma = DoubleArrayAhoCorasick::new(model.dict).unwrap();
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
            dict_window_size: 1,

            #[cfg(feature = "model-quantize")]
            quantize_multiplier: model.quantize_multiplier,
        }
    }

    fn merge_weights(words: &[Vec<u8>], weights: &[Vec<WeightValue>]) -> Vec<Vec<ScoreValue>> {
        let mut result = vec![];
        let word_ids = words
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, w)| (w, i))
            .collect::<HashMap<Vec<u8>, usize>>();
        for seq in words {
            let mut new_weights: Option<Vec<_>> = None;
            for st in (0..seq.len()).rev() {
                if let Some(&idx) = word_ids.get(&seq[st..]) {
                    if let Some(new_weights) = new_weights.as_mut() {
                        for (w_new, w) in new_weights.iter_mut().zip(&weights[idx]) {
                            *w_new += *w as ScoreValue;
                        }
                    } else {
                        new_weights
                            .replace(weights[idx].iter().map(|&w| w as ScoreValue).collect());
                    }
                }
            }
            result.push(new_weights.unwrap());
        }
        result
    }

    fn add_word_ngram_scores(&self, sentence: &Sentence, start: usize, ys: &mut [ScoreValue]) {
        let char_start = if start >= self.char_window_size {
            start + 1 - self.char_window_size
        } else {
            0
        };
        let text_start = sentence.char_to_str_pos[char_start];
        let char_end = std::cmp::min(
            start + ys.len() + self.char_window_size,
            sentence.char_to_str_pos.len() - 1,
        );
        let text_end = sentence.char_to_str_pos[char_end];
        let text = &sentence.text[text_start..text_end];
        let padding = start - char_start + 1;
        for m in self.word_pma.find_overlapping_no_suffix_iter(&text) {
            let m_end = sentence.str_to_char_pos[m.end() + text_start] - char_start;
            let offset = m_end as isize - self.char_window_size as isize - padding as isize;
            let weights = &self.word_weights[m.pattern()];
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

    fn add_type_ngram_scores(&self, sentence: &Sentence, start: usize, ys: &mut [ScoreValue]) {
        let type_start = if start >= self.type_window_size {
            start + 1 - self.type_window_size
        } else {
            0
        };
        let type_end = std::cmp::min(
            start + ys.len() + self.type_window_size,
            sentence.char_type.len(),
        );
        let char_type = &sentence.char_type[type_start..type_end];
        let padding = start - type_start + 1;
        for m in self.type_pma.find_overlapping_no_suffix_iter(&char_type) {
            let offset = m.end() as isize - self.type_window_size as isize - padding as isize;
            let weights = &self.type_weights[m.pattern()];
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

    fn add_dict_scores(&self, sentence: &Sentence, start: usize, ys: &mut [ScoreValue]) {
        let char_start = if start >= self.dict_window_size {
            start + 1 - self.dict_window_size
        } else {
            0
        };
        let text_start = sentence.char_to_str_pos[char_start];
        let char_end = std::cmp::min(
            start + ys.len() + self.dict_window_size,
            sentence.char_to_str_pos.len() - 1,
        );
        let text_end = sentence.char_to_str_pos[char_end];
        let text = &sentence.text[text_start..text_end];
        let padding = start - char_start + 1;
        for m in self.dict_pma.find_overlapping_iter(&text) {
            let m_start = sentence.str_to_char_pos[m.start() + text_start] - char_start;
            let m_end = sentence.str_to_char_pos[m.end() + text_start] - char_start;
            let idx = if self.dict_word_wise {
                m.pattern()
            } else {
                std::cmp::min(m_end - m_start, self.dict_weights.len()) - 1
            };
            let [w_right, w_center, w_left] = self.dict_weights[idx];
            if m_start >= padding && m_start < padding + ys.len() {
                ys[m_start - padding] += w_right;
            }
            let range_start = std::cmp::max(0, m_start as isize - padding as isize + 1);
            let range_end = std::cmp::min(m_end as isize - padding as isize, ys.len() as isize);
            if range_start < range_end {
                for y in &mut ys[range_start as usize..range_end as usize] {
                    *y += w_center;
                }
            }
            if m_end >= padding && m_end < ys.len() + padding {
                ys[m_end - padding] += w_left;
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
        self.add_word_ngram_scores(sentence, range.start, ys);
        self.add_type_ngram_scores(sentence, range.start, ys);
        self.add_dict_scores(sentence, range.start, ys);
    }

    /// Predicts word boundaries of the specified range of a sentence.
    ///
    /// # Arguments
    ///
    /// * `sentence` - A sentence.
    /// * `range` - The range of the sentence.
    ///
    /// # Returns
    ///
    /// A sentence with predicted boundary information.
    pub fn predict_partial(&self, mut sentence: Sentence, range: Range<usize>) -> Sentence {
        let mut ys = vec![ScoreValue::default(); range.len()];
        self.predict_partial_impl(&sentence, range.clone(), &mut ys);
        for (y, b) in ys.into_iter().zip(sentence.boundaries[range].iter_mut()) {
            *b = if y >= ScoreValue::default() {
                BoundaryType::WordBoundary
            } else {
                BoundaryType::NotWordBoundary
            };
        }
        sentence
    }

    /// Predicts word boundaries of the specified range of a sentence. This function inserts
    /// scores.
    ///
    /// # Arguments
    ///
    /// * `sentence` - A sentence.
    /// * `range` - The range of the sentence.
    ///
    /// # Returns
    ///
    /// A sentence with predicted boundary information.
    pub fn predict_partial_with_score(
        &self,
        mut sentence: Sentence,
        range: Range<usize>,
    ) -> Sentence {
        let mut ys = vec![ScoreValue::default(); range.len()];
        self.predict_partial_impl(&sentence, range.clone(), &mut ys);
        let mut scores = sentence
            .boundary_scores
            .take()
            .unwrap_or_else(|| vec![0.; sentence.boundaries.len()]);
        for (y, (b, s)) in ys.into_iter().zip(
            sentence.boundaries[range.clone()]
                .iter_mut()
                .zip(scores[range].iter_mut()),
        ) {
            *b = if y >= ScoreValue::default() {
                BoundaryType::WordBoundary
            } else {
                BoundaryType::NotWordBoundary
            };

            #[cfg(feature = "model-quantize")]
            let y = y as f64 * self.quantize_multiplier;

            *s = y;
        }
        sentence.boundary_scores.replace(scores);
        sentence
    }

    /// Predicts word boundaries.
    ///
    /// # Arguments
    ///
    /// * `sentence` - A sentence.
    ///
    /// # Returns
    ///
    /// A sentence with predicted boundary information.
    pub fn predict(&self, sentence: Sentence) -> Sentence {
        let boundaries_size = sentence.boundaries.len();
        if boundaries_size == 0 {
            sentence
        } else {
            self.predict_partial(sentence, 0..boundaries_size)
        }
    }

    /// Predicts word boundaries. This function inserts scores.
    ///
    /// # Arguments
    ///
    /// * `sentence` - A sentence.
    ///
    /// # Returns
    ///
    /// A sentence with predicted boundary information.
    pub fn predict_with_score(&self, sentence: Sentence) -> Sentence {
        let boundaries_size = sentence.boundaries.len();
        if boundaries_size == 0 {
            sentence
        } else {
            self.predict_partial_with_score(sentence, 0..boundaries_size)
        }
    }

    /// Sets the window size of words in the dictionary.
    ///
    /// # Arguments
    ///
    /// * `size` - The window size.
    ///
    /// # Returns
    ///
    /// A predictor with the specified window size.
    pub fn dict_window_size(mut self, size: usize) -> Self {
        self.dict_window_size = std::cmp::max(size, 1);
        self
    }

    /// Creates a multithreading predictor. This function is the alias of
    /// [`MultithreadPredictor::new()`].
    ///
    /// # Arguments
    ///
    /// * `n_threads` - The number of threads.
    /// * `chunk_size` - The chunk size of each thread.
    ///
    /// # Returns
    ///
    /// A multithread predictor.
    #[cfg(feature = "multithreading")]
    #[cfg_attr(docsrs, doc(cfg(feature = "multithreading")))]
    pub fn multithreading(self, n_threads: usize, chunk_size: usize) -> MultithreadPredictor {
        MultithreadPredictor::new(self, n_threads, chunk_size)
    }
}

/// Predictor for multithreading.
#[cfg(feature = "multithreading")]
#[cfg_attr(docsrs, doc(cfg(feature = "multithreading")))]
pub struct MultithreadPredictor {
    task_tx: Sender<(Arc<Sentence>, Range<usize>, Vec<ScoreValue>)>,
    result_rx: Receiver<(Vec<ScoreValue>, Range<usize>)>,
    chunk_size: usize,
    ys_pool: RefCell<Vec<Vec<ScoreValue>>>,

    #[cfg(feature = "model-quantize")]
    quantize_multiplier: f64,
}

#[cfg(feature = "multithreading")]
impl MultithreadPredictor {
    /// Creates a multithreading predictor.
    ///
    /// # Arguments
    ///
    /// * `predictor` - A normal predictor.
    /// * `n_threads` - The number of threads.
    /// * `chunk_size` - The chunk size of each thread.
    ///
    /// # Returns
    ///
    /// A multithread predictor.
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
                        &mut ys[..range.len()],
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

            #[cfg(feature = "model-quantize")]
            quantize_multiplier: predictor.quantize_multiplier,
        }
    }

    /// Predicts word boundaries.
    ///
    /// # Arguments
    ///
    /// * `sentence` - A sentence.
    ///
    /// # Returns
    ///
    /// A sentence with predicted boundary information.
    pub fn predict(&self, sentence: Sentence) -> Sentence {
        let sentence = Arc::new(sentence);

        let mut n_chunks = 0;
        let mut ys_pool = self.ys_pool.borrow_mut();
        for start in (0..sentence.boundaries.len()).step_by(self.chunk_size) {
            let ys = ys_pool
                .pop()
                .unwrap_or_else(|| vec![ScoreValue::default(); self.chunk_size]);
            let sentence = Arc::clone(&sentence);
            let end = std::cmp::min(start + self.chunk_size, sentence.boundaries.len());
            self.task_tx.send((sentence, start..end, ys)).unwrap();
            n_chunks += 1;
        }
        let mut boundaries = vec![BoundaryType::Unknown; sentence.boundaries.len()];
        for _ in 0..n_chunks {
            let (ys, range) = self.result_rx.recv().unwrap();
            for (&y, b) in ys.iter().zip(&mut boundaries[range]) {
                *b = if y >= ScoreValue::default() {
                    BoundaryType::WordBoundary
                } else {
                    BoundaryType::NotWordBoundary
                };
            }
            ys_pool.push(ys);
        }

        let mut sentence = Arc::try_unwrap(sentence).unwrap();
        sentence.boundaries = boundaries;
        sentence
    }

    /// Predicts word boundaries. This function inserts scores.
    ///
    /// # Arguments
    ///
    /// * `sentence` - A sentence.
    ///
    /// # Returns
    ///
    /// A sentence with predicted boundary information.
    pub fn predict_with_score(&self, mut sentence: Sentence) -> Sentence {
        let mut scores = sentence
            .boundary_scores
            .take()
            .unwrap_or_else(|| vec![0.; sentence.boundaries.len()]);
        let sentence = Arc::new(sentence);
        let mut n_chunks = 0;
        let mut ys_pool = self.ys_pool.borrow_mut();
        for start in (0..sentence.boundaries.len()).step_by(self.chunk_size) {
            let ys = ys_pool
                .pop()
                .unwrap_or_else(|| vec![ScoreValue::default(); self.chunk_size]);
            let sentence = Arc::clone(&sentence);
            let end = std::cmp::min(start + self.chunk_size, sentence.boundaries.len());
            self.task_tx.send((sentence, start..end, ys)).unwrap();
            n_chunks += 1;
        }
        let mut boundaries = vec![BoundaryType::Unknown; sentence.boundaries.len()];
        for _ in 0..n_chunks {
            let (ys, range) = self.result_rx.recv().unwrap();
            for (&y, (b, s)) in ys
                .iter()
                .zip(boundaries[range.clone()].iter_mut().zip(&mut scores[range]))
            {
                *b = if y >= ScoreValue::default() {
                    BoundaryType::WordBoundary
                } else {
                    BoundaryType::NotWordBoundary
                };

                #[cfg(feature = "model-quantize")]
                let y = y as f64 * self.quantize_multiplier;

                *s = y;
            }
            ys_pool.push(ys);
        }

        let mut sentence = Arc::try_unwrap(sentence).unwrap();
        sentence.boundaries = boundaries;
        sentence.boundary_scores.replace(scores);
        sentence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Input:  我  ら  は  全  世  界  の  国  民
    /// bias:   -200  ..  ..  ..  ..  ..  ..  ..
    /// words:
    ///   我ら:    3   4   5
    ///   全世界:          6   7   8   9
    ///   国民:                       10  11  12
    ///   世界:           15  16  17  18  19
    ///   界:             20  21  22  23  24  25
    /// types:
    ///   H:      27  28  29
    ///           26  27  28  29
    ///                           26  27  28  29
    ///   K:      32  33
    ///               30  31  32  33
    ///                   30  31  32  33
    ///                       30  31  32  33
    ///                               30  31  32
    ///                                   30  31
    ///   KH:     35  36
    ///                           34  35  36
    ///   HK:         37  38  39
    ///                               37  38  39
    /// dict:
    ///   全世界:         43  44  44  45
    ///   世界:               43  44  45
    ///   世:                 40  42
    fn generate_model_1() -> Model {
        Model {
            words: vec![
                "我ら".as_bytes().to_vec(),
                "全世界".as_bytes().to_vec(),
                "国民".as_bytes().to_vec(),
                "世界".as_bytes().to_vec(),
                "界".as_bytes().to_vec(),
            ],
            types: vec![b"H".to_vec(), b"K".to_vec(), b"KH".to_vec(), b"HK".to_vec()],
            dict: vec![
                "全世界".as_bytes().to_vec(),
                "世界".as_bytes().to_vec(),
                "世".as_bytes().to_vec(),
            ],
            #[cfg(not(feature = "model-quantize"))]
            word_weights: vec![
                vec![0.5, 1.0, 1.5, 2.0, 2.5],
                vec![3.0, 3.5, 4.0, 4.5],
                vec![5.0, 5.5, 6.0, 6.5, 7.0],
                vec![7.5, 8.0, 8.5, 9.0, 9.5],
                vec![10.0, 10.5, 11.0, 11.5, 12.0, 12.5],
            ],
            #[cfg(feature = "model-quantize")]
            word_weights: vec![
                vec![1, 2, 3, 4, 5],
                vec![6, 7, 8, 9],
                vec![10, 11, 12, 13, 14],
                vec![15, 16, 17, 18, 19],
                vec![20, 21, 22, 23, 24, 25],
            ],
            #[cfg(not(feature = "model-quantize"))]
            type_weights: vec![
                vec![13.0, 13.5, 14.0, 14.5],
                vec![15.0, 15.5, 16.0, 16.5],
                vec![17.0, 17.5, 18.0],
                vec![18.5, 19.0, 19.5],
            ],
            #[cfg(feature = "model-quantize")]
            type_weights: vec![
                vec![26, 27, 28, 29],
                vec![30, 31, 32, 33],
                vec![34, 35, 36],
                vec![37, 38, 39],
            ],
            #[cfg(not(feature = "model-quantize"))]
            dict_weights: vec![[20.0, 20.5, 21.0], [21.5, 22.0, 22.5]],
            #[cfg(feature = "model-quantize")]
            dict_weights: vec![[40, 41, 42], [43, 44, 45]],
            #[cfg(feature = "model-quantize")]
            quantize_multiplier: 0.5,
            dict_word_wise: false,
            #[cfg(not(feature = "model-quantize"))]
            bias: -100.0,
            #[cfg(feature = "model-quantize")]
            bias: -200,
            char_window_size: 3,
            type_window_size: 2,
        }
    }

    /// Input:  我  ら  は  全  世  界  の  国  民
    /// bias:   -285  ..  ..  ..  ..  ..  ..  ..
    /// words:
    ///   我ら:    2   3
    ///   全世界:              4   5
    ///   国民:                            6   7
    ///   世界:                9  10  11
    ///   界:                 12  13  14  15
    /// types:
    ///   H:      18  19  20  21
    ///           17  18  19  20  21
    ///                       16  17  18  19  20
    ///   K:      25  26  27
    ///           22  23  24  25  26  27
    ///               22  23  24  25  26  27
    ///                   22  23  24  25  26  27
    ///                           22  23  24  25
    ///                               22  23  24
    ///   KH:     30  31  32
    ///                       28  29  30  31  32
    ///   HK:     33  34  35  36  37
    ///                           33  34  35  36
    /// dict:
    ///   全世界:         44  45  45  46
    ///   世界:               41  42  43
    ///   世:                 38  40
    fn generate_model_2() -> Model {
        Model {
            words: vec![
                "我ら".as_bytes().to_vec(),
                "全世界".as_bytes().to_vec(),
                "国民".as_bytes().to_vec(),
                "世界".as_bytes().to_vec(),
                "界".as_bytes().to_vec(),
            ],
            types: vec![b"H".to_vec(), b"K".to_vec(), b"KH".to_vec(), b"HK".to_vec()],
            dict: vec![
                "全世界".as_bytes().to_vec(),
                "世界".as_bytes().to_vec(),
                "世".as_bytes().to_vec(),
            ],
            #[cfg(not(feature = "model-quantize"))]
            word_weights: vec![
                vec![0.25, 0.5, 0.75],
                vec![1.0, 1.25],
                vec![1.5, 1.75, 2.0],
                vec![2.25, 2.5, 2.75],
                vec![3.0, 3.25, 3.5, 3.75],
            ],
            #[cfg(feature = "model-quantize")]
            word_weights: vec![
                vec![1, 2, 3],
                vec![4, 5],
                vec![6, 7, 8],
                vec![9, 10, 11],
                vec![12, 13, 14, 15],
            ],
            #[cfg(not(feature = "model-quantize"))]
            type_weights: vec![
                vec![4.0, 4.25, 4.5, 4.75, 5.0, 5.25],
                vec![5.5, 5.75, 6.0, 6.25, 6.5, 6.75],
                vec![7.0, 7.25, 7.5, 7.75, 8.0],
                vec![8.25, 8.5, 8.75, 9.0, 9.25],
            ],
            #[cfg(feature = "model-quantize")]
            type_weights: vec![
                vec![16, 17, 18, 19, 20, 21],
                vec![22, 23, 24, 25, 26, 27],
                vec![28, 29, 30, 31, 32],
                vec![33, 34, 35, 36, 37],
            ],
            #[cfg(not(feature = "model-quantize"))]
            dict_weights: vec![[9.5, 9.75, 10.0], [10.25, 10.5, 10.75], [11.0, 11.25, 11.5]],
            #[cfg(feature = "model-quantize")]
            dict_weights: vec![[38, 39, 40], [41, 42, 43], [44, 45, 46]],
            #[cfg(feature = "model-quantize")]
            quantize_multiplier: 0.25,
            dict_word_wise: false,
            #[cfg(not(feature = "model-quantize"))]
            bias: -71.25,
            #[cfg(feature = "model-quantize")]
            bias: -285,
            char_window_size: 2,
            type_window_size: 3,
        }
    }

    /// Input:  我  ら  は  全  世  界  の  国  民
    /// bias:   -285  ..  ..  ..  ..  ..  ..  ..
    /// words:
    ///   我ら:    2   3
    ///   全世界:              4   5
    ///   国民:                            6   7
    ///   世界:                9  10  11
    ///   界:                 12  13  14  15
    /// types:
    ///   H:      18  19  20  21
    ///           17  18  19  20  21
    ///                       16  17  18  19  20
    ///   K:      25  26  27
    ///           22  23  24  25  26  27
    ///               22  23  24  25  26  27
    ///                   22  23  24  25  26  27
    ///                           22  23  24  25
    ///                               22  23  24
    ///   KH:     30  31  32
    ///                       28  29  30  31  32
    ///   HK:     33  34  35  36  37
    ///                           33  34  35  36
    /// dict:
    ///   国民:                           38  39
    ///   世界:               41  42  43
    ///   世:                 44  46
    ///
    ///                       85  88  43  38  39
    ///                 -44 -124 -127 -89
    ///
    ///                 -44  -39 -39 -46  38  39
    fn generate_model_3() -> Model {
        Model {
            words: vec![
                "我ら".as_bytes().to_vec(),
                "全世界".as_bytes().to_vec(),
                "国民".as_bytes().to_vec(),
                "世界".as_bytes().to_vec(),
                "界".as_bytes().to_vec(),
            ],
            types: vec![b"H".to_vec(), b"K".to_vec(), b"KH".to_vec(), b"HK".to_vec()],
            dict: vec![
                "国民".as_bytes().to_vec(),
                "世界".as_bytes().to_vec(),
                "世".as_bytes().to_vec(),
            ],
            #[cfg(not(feature = "model-quantize"))]
            word_weights: vec![
                vec![0.25, 0.5, 0.75],
                vec![1.0, 1.25],
                vec![1.5, 1.75, 2.0],
                vec![2.25, 2.5, 2.75],
                vec![3.0, 3.25, 3.5, 3.75],
            ],
            #[cfg(feature = "model-quantize")]
            word_weights: vec![
                vec![1, 2, 3],
                vec![4, 5],
                vec![6, 7, 8],
                vec![9, 10, 11],
                vec![12, 13, 14, 15],
            ],
            #[cfg(not(feature = "model-quantize"))]
            type_weights: vec![
                vec![4.0, 4.25, 4.5, 4.75, 5.0, 5.25],
                vec![5.5, 5.75, 6.0, 6.25, 6.5, 6.75],
                vec![7.0, 7.25, 7.5, 7.75, 8.0],
                vec![8.25, 8.5, 8.75, 9.0, 9.25],
            ],
            #[cfg(feature = "model-quantize")]
            type_weights: vec![
                vec![16, 17, 18, 19, 20, 21],
                vec![22, 23, 24, 25, 26, 27],
                vec![28, 29, 30, 31, 32],
                vec![33, 34, 35, 36, 37],
            ],
            #[cfg(not(feature = "model-quantize"))]
            dict_weights: vec![[9.5, 9.75, 11.0], [10.25, 10.5, 10.75], [11.0, 11.25, 11.5]],
            #[cfg(feature = "model-quantize")]
            dict_weights: vec![[38, 39, 40], [41, 42, 43], [44, 45, 46]],
            #[cfg(feature = "model-quantize")]
            quantize_multiplier: 0.25,
            dict_word_wise: true,
            #[cfg(not(feature = "model-quantize"))]
            bias: -71.25,
            #[cfg(feature = "model-quantize")]
            bias: -285,
            char_window_size: 2,
            type_window_size: 3,
        }
    }

    #[test]
    fn test_predict_1() {
        let model = generate_model_1();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
    }

    #[test]
    fn test_predict_2() {
        let model = generate_model_2();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
    }

    #[test]
    fn test_predict_3() {
        let model = generate_model_3();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
    }

    #[test]
    fn test_predict_with_score_1() {
        let model = generate_model_1();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_with_score(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
        assert_eq!(
            &[-38.5, -2.5, 22.5, 66.0, 66.5, 72.0, 25.0, -16.0],
            s.boundary_scores().unwrap(),
        );
    }

    #[test]
    fn test_predict_with_score_2() {
        let model = generate_model_2();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_with_score(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
        assert_eq!(
            &[-34.5, -27.25, -9.75, 14.25, 26.0, 8.5, -19.75, -28.5],
            s.boundary_scores().unwrap(),
        );
    }

    #[test]
    fn test_predict_with_score_3() {
        let model = generate_model_3();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_with_score(s);
        assert_eq!(
            &[
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::NotWordBoundary,
            ],
            s.boundaries(),
        );
        assert_eq!(
            &[-34.5, -27.25, -20.75, 4.5, 16.25, -3.0, -10.25, -18.75],
            s.boundary_scores().unwrap(),
        );
    }

    #[test]
    fn test_predict_partial_1() {
        let model = generate_model_1();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_partial(s, 1..5);
        assert_eq!(
            &[
                BoundaryType::Unknown,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::Unknown,
                BoundaryType::Unknown,
                BoundaryType::Unknown,
            ],
            s.boundaries(),
        );
    }

    #[test]
    fn test_predict_partial_2() {
        let model = generate_model_2();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_partial(s, 2..7);
        assert_eq!(
            &[
                BoundaryType::Unknown,
                BoundaryType::Unknown,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::Unknown,
            ],
            s.boundaries(),
        );
    }

    #[test]
    fn test_predict_partial_3() {
        let model = generate_model_3();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_partial(s, 2..6);
        assert_eq!(
            &[
                BoundaryType::Unknown,
                BoundaryType::Unknown,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::Unknown,
                BoundaryType::Unknown,
            ],
            s.boundaries(),
        );
    }

    #[test]
    fn test_predict_partial_with_score_1() {
        let model = generate_model_1();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_partial_with_score(s, 1..5);
        assert_eq!(
            &[
                BoundaryType::Unknown,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::Unknown,
                BoundaryType::Unknown,
                BoundaryType::Unknown,
            ],
            s.boundaries(),
        );
        assert_eq!(
            &[0.0, -2.5, 22.5, 66.0, 66.5, 0.0, 0.0, 0.0],
            s.boundary_scores().unwrap(),
        );
    }

    #[test]
    fn test_predict_partial_with_score_2() {
        let model = generate_model_2();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_partial_with_score(s, 2..7);
        assert_eq!(
            &[
                BoundaryType::Unknown,
                BoundaryType::Unknown,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::Unknown,
            ],
            s.boundaries(),
        );
        assert_eq!(
            &[0.0, 0.0, -9.75, 14.25, 26.0, 8.5, -19.75, 0.0],
            s.boundary_scores().unwrap(),
        );
    }

    #[test]
    fn test_predict_partial_with_score_3() {
        let model = generate_model_3();
        let p = Predictor::new(model);
        let s = Sentence::from_raw("我らは全世界の国民").unwrap();
        let s = p.predict_partial_with_score(s, 2..6);
        assert_eq!(
            &[
                BoundaryType::Unknown,
                BoundaryType::Unknown,
                BoundaryType::NotWordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::WordBoundary,
                BoundaryType::NotWordBoundary,
                BoundaryType::Unknown,
                BoundaryType::Unknown,
            ],
            s.boundaries(),
        );
        assert_eq!(
            &[0.0, 0.0, -20.75, 4.5, 16.25, -3.0, 0.0, 0.0],
            s.boundary_scores().unwrap(),
        );
    }
}
