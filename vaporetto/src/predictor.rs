use std::collections::HashMap;
use std::ops::Range;

use crate::model::{DictWeight, Model, ScoreValue};
use crate::sentence::{BoundaryType, Sentence};
use crate::type_scorer::TypeScorer;

use daachorse::DoubleArrayAhoCorasick;

/// Predictor.
pub struct Predictor {
    word_pma: DoubleArrayAhoCorasick,
    dict_pma: DoubleArrayAhoCorasick,
    word_weights: Vec<Vec<ScoreValue>>,
    dict_weights: Vec<DictWeight>,
    dict_word_wise: bool,
    bias: ScoreValue,
    char_window_size: usize,
    dict_window_size: usize,

    type_scorer: TypeScorer,

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

        let words = model.words;
        let dict = model.dict;
        let dict_weights = model.dict_weights;

        let mut word_weights: Vec<_> = model
            .word_weights
            .into_iter()
            .map(|ws| ws.into_iter().map(|w| w as ScoreValue).collect())
            .collect();
        let type_weights: Vec<_> = model
            .type_weights
            .into_iter()
            .map(|ws| ws.into_iter().map(|w| w as ScoreValue).collect())
            .collect();

        let (dict, dict_weights) = Self::merge_dict_weights(
            dict,
            dict_weights,
            &words,
            &mut word_weights,
            model.char_window_size,
            model.dict_word_wise,
        );

        let word_weights = Self::merge_weights(&words, &word_weights);
        let type_weights = Self::merge_weights(&model.types, &type_weights);

        #[cfg(feature = "model-quantize")]
        let bias = bias as i32;

        let word_pma = DoubleArrayAhoCorasick::new(words).unwrap();
        let type_pma = DoubleArrayAhoCorasick::new(model.types).unwrap();
        let dict_pma = DoubleArrayAhoCorasick::new(dict).unwrap();

        let type_scorer = TypeScorer::new(type_pma, type_weights, model.type_window_size);

        Self {
            word_pma,
            dict_pma,
            word_weights,
            dict_weights,
            dict_word_wise: model.dict_word_wise,
            bias,
            char_window_size: model.char_window_size,
            dict_window_size: 1,

            type_scorer,

            #[cfg(feature = "model-quantize")]
            quantize_multiplier: model.quantize_multiplier,
        }
    }

    fn merge_dict_weights(
        dict: Vec<Vec<u8>>,
        dict_weights: Vec<DictWeight>,
        words: &[Vec<u8>],
        word_weights: &mut Vec<Vec<ScoreValue>>,
        char_window_size: usize,
        dict_word_wise: bool,
    ) -> (Vec<Vec<u8>>, Vec<DictWeight>) {
        let mut word_map = HashMap::new();
        for (i, word) in words.iter().cloned().enumerate() {
            word_map.insert(word, i);
        }
        let mut new_dict = vec![];
        if dict_word_wise {
            let mut new_dict_weights = vec![];
            for (word, weight) in dict.into_iter().zip(dict_weights) {
                let word_size = std::str::from_utf8(&word).unwrap().chars().count();
                match word_map.get(&word) {
                    Some(&idx) if char_window_size >= word_size => {
                        let start = char_window_size - word_size;
                        let end = start + word_size;
                        word_weights[idx][start] += weight.right;
                        for i in start + 1..end {
                            word_weights[idx][i] += weight.inner;
                        }
                        word_weights[idx][end] += weight.left;
                    }
                    _ => {
                        new_dict.push(word);
                        new_dict_weights.push(weight);
                    }
                }
            }
            (new_dict, new_dict_weights)
        } else {
            for word in dict {
                let word_size = std::str::from_utf8(&word).unwrap().chars().count();
                match word_map.get(&word) {
                    Some(&idx) if char_window_size >= word_size => {
                        let start = char_window_size - word_size;
                        let end = start + word_size;
                        let word_size_idx = std::cmp::min(word_size, dict_weights.len()) - 1;
                        let weight = &dict_weights[word_size_idx];
                        word_weights[idx][start] += weight.right;
                        for i in start + 1..end {
                            word_weights[idx][i] += weight.inner;
                        }
                        word_weights[idx][end] += weight.left;
                    }
                    _ => new_dict.push(word),
                }
            }
            (new_dict, dict_weights)
        }
    }

    fn merge_weights(words: &[Vec<u8>], weights: &[Vec<ScoreValue>]) -> Vec<Vec<ScoreValue>> {
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
                            *w_new += *w;
                        }
                    } else {
                        new_weights.replace(weights[idx].clone());
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
            let dict_weight = self.dict_weights[idx];
            if m_start >= padding && m_start < padding + ys.len() {
                ys[m_start - padding] += dict_weight.right;
            }
            let range_start = std::cmp::max(0, m_start as isize - padding as isize + 1);
            let range_end = std::cmp::min(m_end as isize - padding as isize, ys.len() as isize);
            if range_start < range_end {
                for y in &mut ys[range_start as usize..range_end as usize] {
                    *y += dict_weight.inner;
                }
            }
            if m_end >= padding && m_end < ys.len() + padding {
                ys[m_end - padding] += dict_weight.left;
            }
        }
    }

    fn predict_partial_impl(
        &self,
        sentence: &Sentence,
        range: Range<usize>,
        ys: &mut [ScoreValue],
    ) {
        ys.fill(self.bias);
        self.add_word_ngram_scores(sentence, range.start, ys);
        self.type_scorer.add_scores(sentence, range.start, ys);
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
            dict_weights: vec![
                DictWeight {
                    right: 20.0,
                    inner: 20.5,
                    left: 21.0,
                },
                DictWeight {
                    right: 21.5,
                    inner: 22.0,
                    left: 22.5,
                },
            ],
            #[cfg(feature = "model-quantize")]
            dict_weights: vec![
                DictWeight {
                    right: 40,
                    inner: 41,
                    left: 42,
                },
                DictWeight {
                    right: 43,
                    inner: 44,
                    left: 45,
                },
            ],
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
            dict_weights: vec![
                DictWeight {
                    right: 9.5,
                    inner: 9.75,
                    left: 10.0,
                },
                DictWeight {
                    right: 10.25,
                    inner: 10.5,
                    left: 10.75,
                },
                DictWeight {
                    right: 11.0,
                    inner: 11.25,
                    left: 11.5,
                },
            ],
            #[cfg(feature = "model-quantize")]
            dict_weights: vec![
                DictWeight {
                    right: 38,
                    inner: 39,
                    left: 40,
                },
                DictWeight {
                    right: 41,
                    inner: 42,
                    left: 43,
                },
                DictWeight {
                    right: 44,
                    inner: 45,
                    left: 46,
                },
            ],
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
            dict_weights: vec![
                DictWeight {
                    right: 9.5,
                    inner: 9.75,
                    left: 11.0,
                },
                DictWeight {
                    right: 10.25,
                    inner: 10.5,
                    left: 10.75,
                },
                DictWeight {
                    right: 11.0,
                    inner: 11.25,
                    left: 11.5,
                },
            ],
            #[cfg(feature = "model-quantize")]
            dict_weights: vec![
                DictWeight {
                    right: 38,
                    inner: 39,
                    left: 40,
                },
                DictWeight {
                    right: 41,
                    inner: 42,
                    left: 43,
                },
                DictWeight {
                    right: 44,
                    inner: 45,
                    left: 46,
                },
            ],
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
