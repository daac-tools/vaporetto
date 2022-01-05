use std::hash::Hash;

use daachorse::DoubleArrayAhoCorasick;

use crate::errors::{Result, VaporettoError};
use crate::sentence::BoundaryType;
use crate::sentence::Sentence;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct StringNgramFeature<'a> {
    pub(crate) rel_position: isize,
    pub(crate) ngram: &'a str,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct BytesNgramFeature<'a> {
    pub(crate) rel_position: isize,
    pub(crate) ngram: &'a [u8],
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DictionaryWordPosition {
    Right,
    Left,
    Inside,
}

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub struct DictionaryWordFeature {
    pub(crate) position: DictionaryWordPosition,
    pub(crate) length: usize,
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryFeature<'a> {
    CharacterNgram(StringNgramFeature<'a>),
    CharacterTypeNgram(BytesNgramFeature<'a>),
    DictionaryWord(DictionaryWordFeature),
}

impl<'a> BoundaryFeature<'a> {
    pub const fn char_ngram(rel_position: isize, ngram: &'a str) -> Self {
        Self::CharacterNgram(StringNgramFeature {
            rel_position,
            ngram,
        })
    }

    pub const fn type_ngram(rel_position: isize, ngram: &'a [u8]) -> Self {
        Self::CharacterTypeNgram(BytesNgramFeature {
            rel_position,
            ngram,
        })
    }

    pub const fn dict_word(position: DictionaryWordPosition, length: usize) -> Self {
        Self::DictionaryWord(DictionaryWordFeature { position, length })
    }
}

#[derive(Debug, PartialEq)]
pub struct BoundaryExample<'a> {
    pub features: Vec<BoundaryFeature<'a>>,
    pub label: BoundaryType,
}

pub struct BoundaryExampleGenerator {
    char_ngram_size: usize,
    type_ngram_size: usize,
    char_window_size: usize,
    type_window_size: usize,
    dict_ac: DoubleArrayAhoCorasick,
    dict_max_word_size: usize,
}

impl BoundaryExampleGenerator {
    pub fn new<I, P>(
        char_ngram_size: usize,
        type_ngram_size: usize,
        char_window_size: usize,
        type_window_size: usize,
        dict: I,
        dict_max_word_size: usize,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        Ok(Self {
            char_ngram_size,
            type_ngram_size,
            char_window_size,
            type_window_size,
            dict_ac: DoubleArrayAhoCorasick::new(dict)
                .map_err(|e| VaporettoError::invalid_argument("dict", format!("{:?}", e)))?,
            dict_max_word_size,
        })
    }

    pub fn generate<'a>(&self, s: &'a Sentence) -> Vec<BoundaryExample<'a>> {
        let mut result = vec![];
        for (i, &label) in s.boundaries().iter().enumerate() {
            let mut features = vec![];
            for n in 1..self.char_ngram_size + 1 {
                let begin = (i + 1).saturating_sub(self.char_window_size);
                let end = (i + 1 + self.char_window_size)
                    .min(s.chars.len())
                    .saturating_sub(n - 1);
                for pos in begin..end {
                    let rel_position = pos as isize - i as isize - 1;
                    let ngram = s.char_substring(pos, pos + n);
                    features.push(BoundaryFeature::char_ngram(rel_position, ngram));
                }
            }
            for n in 1..self.type_ngram_size + 1 {
                let begin = (i + 1).saturating_sub(self.type_window_size);
                let end = (i + 1 + self.type_window_size)
                    .min(s.chars.len())
                    .saturating_sub(n - 1);
                for pos in begin..end {
                    let rel_position = pos as isize - i as isize - 1;
                    let ngram = &s.char_types()[pos..pos + n];
                    features.push(BoundaryFeature::type_ngram(rel_position, ngram));
                }
            }
            result.push(BoundaryExample { features, label })
        }
        for m in self.dict_ac.find_overlapping_iter(&s.text) {
            let m_start = s.str_to_char_pos[m.start()];
            let m_end = s.str_to_char_pos[m.end()];
            let length = (m_end - m_start).min(self.dict_max_word_size);
            if m_start != 0 {
                result[m_start - 1]
                    .features
                    .push(BoundaryFeature::dict_word(
                        DictionaryWordPosition::Right,
                        length,
                    ));
            }
            for example in &mut result[m_start..m_end - 1] {
                example.features.push(BoundaryFeature::dict_word(
                    DictionaryWordPosition::Inside,
                    length,
                ));
            }
            if m_end != s.chars().len() {
                result[m_end - 1].features.push(BoundaryFeature::dict_word(
                    DictionaryWordPosition::Left,
                    length,
                ));
            }
        }
        result
            .into_iter()
            .filter(|example| example.label != BoundaryType::Unknown)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sentence::CharacterType::*;
    use BoundaryFeature::*;
    use BoundaryType::*;

    #[test]
    fn test_example_generator_generate_one() {
        let dict = ["東京特許許可局", "火星猫", "猫"];
        let gen = BoundaryExampleGenerator::new(3, 2, 3, 2, dict, 2).unwrap();

        let s = Sentence::from_raw("猫").unwrap();
        let examples = gen.generate(&s);

        assert!(examples.is_empty());
    }

    #[test]
    fn test_example_generator_generate_all() {
        let dict = ["東京特許許可局", "火星猫", "猫"];
        let gen = BoundaryExampleGenerator::new(3, 2, 3, 2, dict, 2).unwrap();

        let s = Sentence::from_partial_annotation("A-r-i-a|は|火-星 猫|だ").unwrap();
        let examples = gen.generate(&s);

        assert_eq!(7, examples.len());

        // pos 3 "A-r"
        #[rustfmt::skip]
        let expected = BoundaryExample {
            features: vec![
                CharacterNgram(StringNgramFeature { rel_position: -1, ngram: "A" }),
                CharacterNgram(StringNgramFeature { rel_position: 0, ngram: "r" }),
                CharacterNgram(StringNgramFeature { rel_position: 1, ngram: "i" }),
                CharacterNgram(StringNgramFeature { rel_position: 2, ngram: "a" }),
                CharacterNgram(StringNgramFeature { rel_position: -1, ngram: "Ar" }),
                CharacterNgram(StringNgramFeature { rel_position: 0, ngram: "ri" }),
                CharacterNgram(StringNgramFeature { rel_position: 1, ngram: "ia" }),
                CharacterNgram(StringNgramFeature { rel_position: -1, ngram: "Ari" }),
                CharacterNgram(StringNgramFeature { rel_position: 0, ngram: "ria" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: b"R" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 0, ngram: b"R" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 1, ngram: b"R" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: b"RR" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 0, ngram: b"RR" }),
            ],
            label: NotWordBoundary,
        };
        assert_eq!(expected, examples[0]);

        // pos 3 "a|は"
        #[rustfmt::skip]
        let expected = BoundaryExample {
            features: vec![
                CharacterNgram(StringNgramFeature { rel_position: -3, ngram: "r" }),
                CharacterNgram(StringNgramFeature { rel_position: -2, ngram: "i" }),
                CharacterNgram(StringNgramFeature { rel_position: -1, ngram: "a" }),
                CharacterNgram(StringNgramFeature { rel_position: 0, ngram: "は" }),
                CharacterNgram(StringNgramFeature { rel_position: 1, ngram: "火" }),
                CharacterNgram(StringNgramFeature { rel_position: 2, ngram: "星" }),
                CharacterNgram(StringNgramFeature { rel_position: -3, ngram: "ri" }),
                CharacterNgram(StringNgramFeature { rel_position: -2, ngram: "ia" }),
                CharacterNgram(StringNgramFeature { rel_position: -1, ngram: "aは" }),
                CharacterNgram(StringNgramFeature { rel_position: 0, ngram: "は火" }),
                CharacterNgram(StringNgramFeature { rel_position: 1, ngram: "火星" }),
                CharacterNgram(StringNgramFeature { rel_position: -3, ngram: "ria" }),
                CharacterNgram(StringNgramFeature { rel_position: -2, ngram: "iaは" }),
                CharacterNgram(StringNgramFeature { rel_position: -1, ngram: "aは火" }),
                CharacterNgram(StringNgramFeature { rel_position: 0, ngram: "は火星" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -2, ngram: b"R" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: b"R" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 0, ngram: b"H" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 1, ngram: b"K" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -2, ngram: b"RR" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: b"RH" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 0, ngram: b"HK" }),
            ],
            label: WordBoundary,
        };
        assert_eq!(expected, examples[3]);

        // pos 6 "星 猫" (skipped)

        // pos 7 "猫|だ"
        #[rustfmt::skip]
        let expected = BoundaryExample {
            features: vec![
                CharacterNgram(StringNgramFeature { rel_position: -3, ngram: "火" }),
                CharacterNgram(StringNgramFeature { rel_position: -2, ngram: "星" }),
                CharacterNgram(StringNgramFeature { rel_position: -1, ngram: "猫" }),
                CharacterNgram(StringNgramFeature { rel_position: 0, ngram: "だ" }),
                CharacterNgram(StringNgramFeature { rel_position: -3, ngram: "火星" }),
                CharacterNgram(StringNgramFeature { rel_position: -2, ngram: "星猫" }),
                CharacterNgram(StringNgramFeature { rel_position: -1, ngram: "猫だ" }),
                CharacterNgram(StringNgramFeature { rel_position: -3, ngram: "火星猫" }),
                CharacterNgram(StringNgramFeature { rel_position: -2, ngram: "星猫だ" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -2, ngram: b"K" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: b"K" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 0, ngram: b"H" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -2, ngram: b"KK" }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: b"KH" }),
                DictionaryWord(DictionaryWordFeature { position: DictionaryWordPosition::Left, length: 2 }),
                DictionaryWord(DictionaryWordFeature { position: DictionaryWordPosition::Left, length: 1 }),
            ],
            label: WordBoundary,
        };
        assert_eq!(expected, examples[6]);
    }

    #[test]
    fn test_example_generator_generate_without_unknown() {
        let dict = ["東京特許許可局", "火星猫", "猫"];
        let gen = BoundaryExampleGenerator::new(3, 2, 3, 2, dict, 2).unwrap();

        let s = Sentence::from_partial_annotation("A-r-i-a|は|火-星 猫|だ").unwrap();
        let examples = gen.generate(&s);

        assert_eq!(7, examples.len());
    }
}
