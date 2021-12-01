use crate::errors::{Result, VaporettoError};
use crate::sentence::{BoundaryType, Sentence};

use daachorse::DoubleArrayAhoCorasick;

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum FeatureContent<'a> {
    CharacterNgram(&'a str),
    CharacterTypeNgram(&'a [u8]),
    DictionaryWord(usize),
}

#[derive(Debug, PartialEq)]
pub struct FeatureSpan<'a> {
    start: usize,
    end: usize,
    feature: FeatureContent<'a>,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct Feature<'a> {
    pub(crate) rel_position: usize,
    pub(crate) feature: FeatureContent<'a>,
}

#[derive(Debug, PartialEq)]
pub struct Example<'a> {
    pub features: Vec<Feature<'a>>,
    pub label: BoundaryType,
}

pub struct FeatureExtractor {
    char_ngram_size: usize,
    type_ngram_size: usize,
    dict_ac: DoubleArrayAhoCorasick,
    dict_word_size: Vec<usize>,
}

impl FeatureExtractor {
    pub fn new<D, P>(
        char_ngram_size: usize,
        type_ngram_size: usize,
        dictionary: D,
        dict_word_max_size: usize,
    ) -> Result<Self>
    where
        D: AsRef<[P]>,
        P: AsRef<[u8]> + AsRef<str>,
    {
        let dictionary = dictionary.as_ref();
        let mut dict_word_size = Vec::with_capacity(dictionary.len());
        for word in dictionary {
            let size = std::cmp::min(
                AsRef::<str>::as_ref(word).chars().count(),
                dict_word_max_size,
            );
            if size == 0 {
                return Err(VaporettoError::invalid_argument(
                    "dictionary",
                    "contains an empty string",
                ));
            }
            dict_word_size.push(size);
        }
        Ok(Self {
            char_ngram_size,
            type_ngram_size,
            dict_ac: DoubleArrayAhoCorasick::new(dictionary).unwrap(),
            dict_word_size,
        })
    }

    pub fn extract<'a>(&self, sentence: &'a Sentence) -> Vec<FeatureSpan<'a>> {
        let mut features = vec![];
        for n in 0..self.char_ngram_size as isize {
            for i in 0..sentence.char_type.len() as isize - n {
                let start = i as usize;
                let end = (i + n + 1) as usize;
                let feature = FeatureContent::CharacterNgram(sentence.char_substring(start, end));
                features.push(FeatureSpan {
                    start,
                    end,
                    feature,
                })
            }
        }
        for n in 0..self.type_ngram_size as isize {
            for i in 0..sentence.char_type.len() as isize - n {
                let start = i as usize;
                let end = (i + n + 1) as usize;
                let feature =
                    FeatureContent::CharacterTypeNgram(sentence.type_substring(start, end));
                features.push(FeatureSpan {
                    start,
                    end,
                    feature,
                });
            }
        }
        for m in self.dict_ac.find_overlapping_iter(&sentence.text) {
            let start = sentence.str_to_char_pos[m.start()];
            let end = sentence.str_to_char_pos[m.end()];
            let feature = FeatureContent::DictionaryWord(self.dict_word_size[m.pattern()]);
            features.push(FeatureSpan {
                start,
                end,
                feature,
            });
        }
        features
    }
}

pub struct ExampleGenerator {
    char_window_size: usize,
    type_window_size: usize,
}

impl ExampleGenerator {
    pub const fn new(char_window_size: usize, type_window_size: usize) -> Self {
        Self {
            char_window_size,
            type_window_size,
        }
    }

    pub fn generate<'a>(
        &self,
        sentence: &'a Sentence,
        feature_spans: impl Into<Vec<FeatureSpan<'a>>>,
        include_unknown: bool,
    ) -> Vec<Example<'a>> {
        let mut examples: Vec<Example> = sentence
            .boundaries
            .iter()
            .map(|&label| Example {
                features: vec![],
                label,
            })
            .collect();
        for span in feature_spans.into() {
            match span.feature {
                FeatureContent::CharacterNgram(_) => {
                    let start =
                        std::cmp::max(span.end - 1, self.char_window_size) - self.char_window_size;
                    let end = std::cmp::min(
                        span.start + self.char_window_size,
                        sentence.boundaries.len(),
                    );
                    for (i, example) in examples.iter_mut().enumerate().take(end).skip(start) {
                        if include_unknown || example.label != BoundaryType::Unknown {
                            example.features.push(Feature {
                                rel_position: self.char_window_size + i + 1 - span.end,
                                feature: span.feature,
                            });
                        }
                    }
                }
                FeatureContent::CharacterTypeNgram(_) => {
                    let start =
                        std::cmp::max(span.end - 1, self.type_window_size) - self.type_window_size;
                    let end = std::cmp::min(
                        span.start + self.type_window_size,
                        sentence.boundaries.len(),
                    );
                    for (i, example) in examples.iter_mut().enumerate().take(end).skip(start) {
                        if include_unknown || example.label != BoundaryType::Unknown {
                            example.features.push(Feature {
                                rel_position: self.type_window_size + i + 1 - span.end,
                                feature: span.feature,
                            });
                        }
                    }
                }
                FeatureContent::DictionaryWord(_) => {
                    if span.start >= 1 {
                        let example = &mut examples[span.start - 1];
                        if include_unknown || example.label != BoundaryType::Unknown {
                            example.features.push(Feature {
                                rel_position: 0,
                                feature: span.feature,
                            });
                        }
                    }
                    for example in &mut examples[span.start..span.end - 1] {
                        if include_unknown || example.label != BoundaryType::Unknown {
                            example.features.push(Feature {
                                rel_position: 1,
                                feature: span.feature,
                            });
                        }
                    }
                    if span.end <= examples.len() {
                        let example = &mut examples[span.end - 1];
                        if include_unknown || example.label != BoundaryType::Unknown {
                            example.features.push(Feature {
                                rel_position: 2,
                                feature: span.feature,
                            });
                        }
                    }
                }
            }
        }
        if include_unknown {
            examples
        } else {
            examples
                .into_iter()
                .filter(|example| example.label != BoundaryType::Unknown)
                .collect()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sentence::CharacterType::*;
    use BoundaryType::*;
    use FeatureContent::*;

    #[test]
    fn test_feature_extractor_new_empty_dict_string() {
        let dict = ["東京特許許可局", "", "猫"];
        let fe = FeatureExtractor::new(3, 2, dict, 4);

        assert!(fe.is_err());
        assert_eq!(
            "InvalidArgumentError: dictionary: contains an empty string",
            &fe.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_feature_extractor_new_empty_dict() {
        let dict: &[String] = &[];
        let fe = FeatureExtractor::new(3, 2, dict, 4).unwrap();

        assert_eq!(3, fe.char_ngram_size);
        assert_eq!(2, fe.type_ngram_size);
        assert_eq!(Vec::<usize>::new(), fe.dict_word_size);
    }

    #[test]
    fn test_feature_extractor_new() {
        let dict = ["東京特許許可局", "火星猫", "猫"];
        let fe = FeatureExtractor::new(3, 2, dict, 4).unwrap();

        assert_eq!(3, fe.char_ngram_size);
        assert_eq!(2, fe.type_ngram_size);
        assert_eq!(vec![4, 3, 1], fe.dict_word_size);
    }

    #[test]
    fn test_feature_extractor_extract_one() {
        let dict = ["東京特許許可局", "火星猫", "猫"];
        let fe = FeatureExtractor::new(3, 2, dict, 4).unwrap();
        let s = Sentence::from_raw("A").unwrap();
        let feature_spans = fe.extract(&s);

        let expected = vec![
            FeatureSpan {
                start: 0,
                end: 1,
                feature: CharacterNgram("A"),
            },
            FeatureSpan {
                start: 0,
                end: 1,
                feature: CharacterTypeNgram(&ct2u8![Roman]),
            },
        ];
        assert_eq!(expected, feature_spans);
    }

    #[test]
    fn test_feature_extractor_extract() {
        let dict = ["東京特許許可局", "火星猫", "猫"];
        let fe = FeatureExtractor::new(3, 2, dict, 2).unwrap();
        let s = Sentence::from_raw("Ariaは火星猫だ").unwrap();
        let feature_spans = fe.extract(&s);

        #[rustfmt::skip]
        let expected = vec![
            FeatureSpan { start: 0, end: 1, feature: CharacterNgram("A") },
            FeatureSpan { start: 1, end: 2, feature: CharacterNgram("r") },
            FeatureSpan { start: 2, end: 3, feature: CharacterNgram("i") },
            FeatureSpan { start: 3, end: 4, feature: CharacterNgram("a") },
            FeatureSpan { start: 4, end: 5, feature: CharacterNgram("は") },
            FeatureSpan { start: 5, end: 6, feature: CharacterNgram("火") },
            FeatureSpan { start: 6, end: 7, feature: CharacterNgram("星") },
            FeatureSpan { start: 7, end: 8, feature: CharacterNgram("猫") },
            FeatureSpan { start: 8, end: 9, feature: CharacterNgram("だ") },
            FeatureSpan { start: 0, end: 2, feature: CharacterNgram("Ar") },
            FeatureSpan { start: 1, end: 3, feature: CharacterNgram("ri") },
            FeatureSpan { start: 2, end: 4, feature: CharacterNgram("ia") },
            FeatureSpan { start: 3, end: 5, feature: CharacterNgram("aは") },
            FeatureSpan { start: 4, end: 6, feature: CharacterNgram("は火") },
            FeatureSpan { start: 5, end: 7, feature: CharacterNgram("火星") },
            FeatureSpan { start: 6, end: 8, feature: CharacterNgram("星猫") },
            FeatureSpan { start: 7, end: 9, feature: CharacterNgram("猫だ") },
            FeatureSpan { start: 0, end: 3, feature: CharacterNgram("Ari") },
            FeatureSpan { start: 1, end: 4, feature: CharacterNgram("ria") },
            FeatureSpan { start: 2, end: 5, feature: CharacterNgram("iaは") },
            FeatureSpan { start: 3, end: 6, feature: CharacterNgram("aは火") },
            FeatureSpan { start: 4, end: 7, feature: CharacterNgram("は火星") },
            FeatureSpan { start: 5, end: 8, feature: CharacterNgram("火星猫") },
            FeatureSpan { start: 6, end: 9, feature: CharacterNgram("星猫だ") },
            FeatureSpan { start: 0, end: 1, feature: CharacterTypeNgram(&ct2u8![Roman]) },
            FeatureSpan { start: 1, end: 2, feature: CharacterTypeNgram(&ct2u8![Roman]) },
            FeatureSpan { start: 2, end: 3, feature: CharacterTypeNgram(&ct2u8![Roman]) },
            FeatureSpan { start: 3, end: 4, feature: CharacterTypeNgram(&ct2u8![Roman]) },
            FeatureSpan { start: 4, end: 5, feature: CharacterTypeNgram(&ct2u8![Hiragana]) },
            FeatureSpan { start: 5, end: 6, feature: CharacterTypeNgram(&ct2u8![Kanji]) },
            FeatureSpan { start: 6, end: 7, feature: CharacterTypeNgram(&ct2u8![Kanji]) },
            FeatureSpan { start: 7, end: 8, feature: CharacterTypeNgram(&ct2u8![Kanji]) },
            FeatureSpan { start: 8, end: 9, feature: CharacterTypeNgram(&ct2u8![Hiragana]) },
            FeatureSpan { start: 0, end: 2, feature: CharacterTypeNgram(&ct2u8![Roman, Roman]) },
            FeatureSpan { start: 1, end: 3, feature: CharacterTypeNgram(&ct2u8![Roman, Roman]) },
            FeatureSpan { start: 2, end: 4, feature: CharacterTypeNgram(&ct2u8![Roman, Roman]) },
            FeatureSpan { start: 3, end: 5, feature: CharacterTypeNgram(&ct2u8![Roman, Hiragana]) },
            FeatureSpan { start: 4, end: 6, feature: CharacterTypeNgram(&ct2u8![Hiragana, Kanji]) },
            FeatureSpan { start: 5, end: 7, feature: CharacterTypeNgram(&ct2u8![Kanji, Kanji]) },
            FeatureSpan { start: 6, end: 8, feature: CharacterTypeNgram(&ct2u8![Kanji, Kanji]) },
            FeatureSpan { start: 7, end: 9, feature: CharacterTypeNgram(&ct2u8![Kanji, Hiragana]) },
            FeatureSpan { start: 5, end: 8, feature: DictionaryWord(2) },
            FeatureSpan { start: 7, end: 8, feature: DictionaryWord(1) },
        ];
        assert_eq!(expected, feature_spans);
    }

    #[test]
    fn test_example_generator_new() {
        let gen = ExampleGenerator::new(3, 2);

        assert_eq!(3, gen.char_window_size);
        assert_eq!(2, gen.type_window_size);
    }

    #[test]
    fn test_example_generator_generate_one() {
        let dict = ["東京特許許可局", "火星猫", "猫"];
        let fe = FeatureExtractor::new(3, 2, dict, 2).unwrap();
        let gen = ExampleGenerator::new(3, 2);

        let s = Sentence::from_raw("猫").unwrap();
        let feature_spans = fe.extract(&s);
        let examples = gen.generate(&s, feature_spans, true);

        assert_eq!(Vec::<Example>::new(), examples);
    }

    #[test]
    fn test_example_generator_generate_all() {
        let dict = ["東京特許許可局", "火星猫", "猫"];
        let fe = FeatureExtractor::new(3, 2, dict, 2).unwrap();
        let gen = ExampleGenerator::new(3, 2);

        let s = Sentence::from_partial_annotation("A-r-i-a|は|火-星 猫|だ").unwrap();
        let feature_spans = fe.extract(&s);
        let examples = gen.generate(&s, feature_spans, true);

        assert_eq!(8, examples.len());

        // pos 3 "A-r"
        #[rustfmt::skip]
        let expected = Example {
            features: vec![
                Feature { rel_position: 3, feature: CharacterNgram("A") },
                Feature { rel_position: 2, feature: CharacterNgram("r") },
                Feature { rel_position: 1, feature: CharacterNgram("i") },
                Feature { rel_position: 0, feature: CharacterNgram("a") },
                Feature { rel_position: 2, feature: CharacterNgram("Ar") },
                Feature { rel_position: 1, feature: CharacterNgram("ri") },
                Feature { rel_position: 0, feature: CharacterNgram("ia") },
                Feature { rel_position: 1, feature: CharacterNgram("Ari") },
                Feature { rel_position: 0, feature: CharacterNgram("ria") },
                Feature { rel_position: 2, feature: CharacterTypeNgram(&ct2u8![Roman]) },
                Feature { rel_position: 1, feature: CharacterTypeNgram(&ct2u8![Roman]) },
                Feature { rel_position: 0, feature: CharacterTypeNgram(&ct2u8![Roman]) },
                Feature { rel_position: 1, feature: CharacterTypeNgram(&ct2u8![Roman, Roman]) },
                Feature { rel_position: 0, feature: CharacterTypeNgram(&ct2u8![Roman, Roman]) },
            ],
            label: NotWordBoundary,
        };
        assert_eq!(expected, examples[0]);

        // pos 3 "a|は"
        #[rustfmt::skip]
        let expected = Example {
            features: vec![
                Feature { rel_position: 5, feature: CharacterNgram("r") },
                Feature { rel_position: 4, feature: CharacterNgram("i") },
                Feature { rel_position: 3, feature: CharacterNgram("a") },
                Feature { rel_position: 2, feature: CharacterNgram("は") },
                Feature { rel_position: 1, feature: CharacterNgram("火") },
                Feature { rel_position: 0, feature: CharacterNgram("星") },
                Feature { rel_position: 4, feature: CharacterNgram("ri") },
                Feature { rel_position: 3, feature: CharacterNgram("ia") },
                Feature { rel_position: 2, feature: CharacterNgram("aは") },
                Feature { rel_position: 1, feature: CharacterNgram("は火") },
                Feature { rel_position: 0, feature: CharacterNgram("火星") },
                Feature { rel_position: 3, feature: CharacterNgram("ria") },
                Feature { rel_position: 2, feature: CharacterNgram("iaは") },
                Feature { rel_position: 1, feature: CharacterNgram("aは火") },
                Feature { rel_position: 0, feature: CharacterNgram("は火星") },
                Feature { rel_position: 3, feature: CharacterTypeNgram(&ct2u8![Roman]) },
                Feature { rel_position: 2, feature: CharacterTypeNgram(&ct2u8![Roman]) },
                Feature { rel_position: 1, feature: CharacterTypeNgram(&ct2u8![Hiragana]) },
                Feature { rel_position: 0, feature: CharacterTypeNgram(&ct2u8![Kanji]) },
                Feature { rel_position: 2, feature: CharacterTypeNgram(&ct2u8![Roman, Roman]) },
                Feature { rel_position: 1, feature: CharacterTypeNgram(&ct2u8![Roman, Hiragana]) },
                Feature { rel_position: 0, feature: CharacterTypeNgram(&ct2u8![Hiragana, Kanji]) },
            ],
            label: WordBoundary,
        };
        assert_eq!(expected, examples[3]);

        // pos 6 "星 猫"
        #[rustfmt::skip]
        let expected = Example {
            features: vec![
                Feature { rel_position: 5, feature: CharacterNgram("は") },
                Feature { rel_position: 4, feature: CharacterNgram("火") },
                Feature { rel_position: 3, feature: CharacterNgram("星") },
                Feature { rel_position: 2, feature: CharacterNgram("猫") },
                Feature { rel_position: 1, feature: CharacterNgram("だ") },
                Feature { rel_position: 4, feature: CharacterNgram("は火") },
                Feature { rel_position: 3, feature: CharacterNgram("火星") },
                Feature { rel_position: 2, feature: CharacterNgram("星猫") },
                Feature { rel_position: 1, feature: CharacterNgram("猫だ") },
                Feature { rel_position: 3, feature: CharacterNgram("は火星") },
                Feature { rel_position: 2, feature: CharacterNgram("火星猫") },
                Feature { rel_position: 1, feature: CharacterNgram("星猫だ") },
                Feature { rel_position: 3, feature: CharacterTypeNgram(&ct2u8![Kanji]) },
                Feature { rel_position: 2, feature: CharacterTypeNgram(&ct2u8![Kanji]) },
                Feature { rel_position: 1, feature: CharacterTypeNgram(&ct2u8![Kanji]) },
                Feature { rel_position: 0, feature: CharacterTypeNgram(&ct2u8![Hiragana]) },
                Feature { rel_position: 2, feature: CharacterTypeNgram(&ct2u8![Kanji, Kanji]) },
                Feature { rel_position: 1, feature: CharacterTypeNgram(&ct2u8![Kanji, Kanji]) },
                Feature { rel_position: 0, feature: CharacterTypeNgram(&ct2u8![Kanji, Hiragana]) },
                Feature { rel_position: 1, feature: DictionaryWord(2) },
                Feature { rel_position: 0, feature: DictionaryWord(1) },
            ],
            label: Unknown,
        };
        assert_eq!(expected, examples[6]);

        // pos 7 "猫|だ"
        #[rustfmt::skip]
        let expected = Example {
            features: vec![
                Feature { rel_position: 5, feature: CharacterNgram("火") },
                Feature { rel_position: 4, feature: CharacterNgram("星") },
                Feature { rel_position: 3, feature: CharacterNgram("猫") },
                Feature { rel_position: 2, feature: CharacterNgram("だ") },
                Feature { rel_position: 4, feature: CharacterNgram("火星") },
                Feature { rel_position: 3, feature: CharacterNgram("星猫") },
                Feature { rel_position: 2, feature: CharacterNgram("猫だ") },
                Feature { rel_position: 3, feature: CharacterNgram("火星猫") },
                Feature { rel_position: 2, feature: CharacterNgram("星猫だ") },
                Feature { rel_position: 3, feature: CharacterTypeNgram(&ct2u8![Kanji]) },
                Feature { rel_position: 2, feature: CharacterTypeNgram(&ct2u8![Kanji]) },
                Feature { rel_position: 1, feature: CharacterTypeNgram(&ct2u8![Hiragana]) },
                Feature { rel_position: 2, feature: CharacterTypeNgram(&ct2u8![Kanji, Kanji]) },
                Feature { rel_position: 1, feature: CharacterTypeNgram(&ct2u8![Kanji, Hiragana]) },
                Feature { rel_position: 2, feature: DictionaryWord(2) },
                Feature { rel_position: 2, feature: DictionaryWord(1) },
            ],
            label: WordBoundary,
        };
        assert_eq!(expected, examples[7]);
    }

    #[test]
    fn test_example_generator_generate_without_unknown() {
        let dict = ["東京特許許可局", "火星猫", "猫"];
        let fe = FeatureExtractor::new(3, 2, dict, 2).unwrap();
        let gen = ExampleGenerator::new(3, 2);

        let s = Sentence::from_partial_annotation("A-r-i-a|は|火-星 猫|だ").unwrap();
        let feature_spans = fe.extract(&s);
        let examples = gen.generate(&s, feature_spans, false);

        assert_eq!(7, examples.len());
    }
}
