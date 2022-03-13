use std::hash::Hash;
use std::sync::Arc;

use daachorse::DoubleArrayAhoCorasick;

use crate::errors::{Result, VaporettoError};
use crate::sentence::BoundaryType;
use crate::sentence::Sentence;

#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq, PartialOrd, Ord)]
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
    char_ngram_size: u8,
    type_ngram_size: u8,
    char_window_size: u8,
    type_window_size: u8,
    dict_ac: Option<DoubleArrayAhoCorasick>,
    dict_max_word_size: u8,
}

impl BoundaryExampleGenerator {
    pub fn new<I, P>(
        char_ngram_size: u8,
        type_ngram_size: u8,
        char_window_size: u8,
        type_window_size: u8,
        dict: Option<I>,
        dict_max_word_size: u8,
    ) -> Result<Self>
    where
        I: IntoIterator<Item = P>,
        P: AsRef<[u8]>,
    {
        let dict_ac = if let Some(dict) = dict {
            Some(
                DoubleArrayAhoCorasick::new(dict)
                    .map_err(|e| VaporettoError::invalid_argument("dict", format!("{:?}", e)))?,
            )
        } else {
            None
        };
        Ok(Self {
            char_ngram_size,
            type_ngram_size,
            char_window_size,
            type_window_size,
            dict_ac,
            dict_max_word_size,
        })
    }

    pub fn generate<'a>(&self, s: &'a Sentence) -> Result<Vec<BoundaryExample<'a>>> {
        let mut result = vec![];
        for (i, &label) in s.boundaries().iter().enumerate() {
            let mut features = vec![];
            for n in 0..usize::from(self.char_ngram_size) {
                let begin = (i + 1).saturating_sub(usize::from(self.char_window_size));
                let end = (i + 1 + usize::from(self.char_window_size))
                    .min(s.chars.len())
                    .saturating_sub(n);
                for pos in begin..end {
                    let rel_position = isize::try_from(pos)? - isize::try_from(i)? - 1;
                    let ngram = s.char_substring(pos, pos + n + 1);
                    features.push(BoundaryFeature::char_ngram(rel_position, ngram));
                }
            }
            for n in 0..usize::from(self.type_ngram_size) {
                let begin = (i + 1).saturating_sub(usize::from(self.type_window_size));
                let end = (i + 1 + usize::from(self.type_window_size))
                    .min(s.chars.len())
                    .saturating_sub(n);
                for pos in begin..end {
                    let rel_position = isize::try_from(pos)? - isize::try_from(i)? - 1;
                    let ngram = &s.char_types()[pos..pos + n + 1];
                    features.push(BoundaryFeature::type_ngram(rel_position, ngram));
                }
            }
            result.push(BoundaryExample { features, label })
        }
        if let Some(dict_ac) = self.dict_ac.as_ref() {
            for m in dict_ac.find_overlapping_iter(&s.text) {
                let m_start = s.str_to_char_pos[m.start()];
                let m_end = s.str_to_char_pos[m.end()];
                let length = (m_end - m_start).min(usize::from(self.dict_max_word_size));
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
        }
        Ok(result
            .into_iter()
            .filter(|example| example.label != BoundaryType::Unknown)
            .collect())
    }
}

#[derive(Debug, Hash, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TagFeature<'a> {
    LeftCharacterNgram(StringNgramFeature<'a>),
    LeftCharacterNgramBos(StringNgramFeature<'a>),
    RightCharacterNgram(StringNgramFeature<'a>),
    RightCharacterNgramEos(StringNgramFeature<'a>),
    Character(&'a str),
}

impl<'a> TagFeature<'a> {
    pub const fn left_char_ngram(rel_position: isize, ngram: &'a str) -> Self {
        Self::LeftCharacterNgram(StringNgramFeature {
            rel_position,
            ngram,
        })
    }

    pub const fn left_char_ngram_bos(rel_position: isize, ngram: &'a str) -> Self {
        Self::LeftCharacterNgramBos(StringNgramFeature {
            rel_position,
            ngram,
        })
    }

    pub const fn right_char_ngram(rel_position: isize, ngram: &'a str) -> Self {
        Self::RightCharacterNgram(StringNgramFeature {
            rel_position,
            ngram,
        })
    }

    pub const fn right_char_ngram_eos(rel_position: isize, ngram: &'a str) -> Self {
        Self::RightCharacterNgramEos(StringNgramFeature {
            rel_position,
            ngram,
        })
    }

    pub const fn chars(chars: &'a str) -> Self {
        Self::Character(chars)
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct TagExample<'a> {
    pub features: Vec<TagFeature<'a>>,
    pub tag: Arc<String>,
}

pub struct TagExampleGenerator {
    char_ngram_size: u8,
    char_window_size: u8,
}

impl TagExampleGenerator {
    pub const fn new(char_ngram_size: u8, char_window_size: u8) -> Self {
        Self {
            char_ngram_size,
            char_window_size,
        }
    }

    pub fn generate<'a>(&self, sentence: &'a Sentence) -> Result<Vec<TagExample<'a>>> {
        let mut result = vec![];
        let mut features = vec![];
        for start in (sentence.chars.len() + 1).saturating_sub(usize::from(self.char_ngram_size))
            ..sentence.chars.len() + 1
        {
            features.push(TagFeature::right_char_ngram_eos(
                1,
                sentence.char_substring(start, sentence.chars.len()),
            ));
        }
        let mut current_tag: Option<Arc<String>> = sentence
            .tags
            .last()
            .and_then(|x| x.as_ref())
            .map(Arc::clone);
        let mut tag_right_pos = sentence.chars.len();
        for (i, (t, b)) in sentence
            .tags
            .iter()
            .zip(sentence.boundaries())
            .enumerate()
            .rev()
        {
            match b {
                BoundaryType::WordBoundary => {
                    if let Some(tag) = current_tag.take() {
                        if i + 2 <= usize::from(self.char_window_size) {
                            let rel_position = -isize::try_from(i)? - 2;
                            for end in
                                0..sentence.chars.len().min(usize::from(self.char_ngram_size))
                            {
                                features.push(TagFeature::left_char_ngram_bos(
                                    rel_position,
                                    sentence.char_substring(0, end),
                                ));
                            }
                        }
                        for j in (i + 1).saturating_sub(usize::from(self.char_window_size))..i + 1 {
                            let rel_position = isize::try_from(j)? - isize::try_from(i)? - 1;
                            for end in j + 1
                                ..sentence
                                    .chars
                                    .len()
                                    .min(j + usize::from(self.char_ngram_size))
                                    + 1
                            {
                                features.push(TagFeature::left_char_ngram(
                                    rel_position,
                                    sentence.char_substring(j, end),
                                ));
                            }
                        }
                        features.push(TagFeature::chars(
                            sentence.char_substring(i + 1, tag_right_pos),
                        ));
                        result.push(TagExample { features, tag });
                        features = vec![];
                    }
                    if let Some(tag) = t.as_ref() {
                        current_tag.replace(Arc::clone(tag));
                        tag_right_pos = i + 1;
                        for j in (i + 2)
                            ..(i + 2 + usize::from(self.char_window_size))
                                .min(sentence.chars.len() + 1)
                        {
                            let rel_position = isize::try_from(j - i)? - 1;
                            for start in j.saturating_sub(usize::from(self.char_ngram_size))..j {
                                features.push(TagFeature::right_char_ngram(
                                    rel_position,
                                    sentence.char_substring(start, j),
                                ));
                            }
                        }
                        if i + usize::from(self.char_window_size) >= sentence.chars.len() {
                            let rel_position = isize::try_from(sentence.chars.len() - i)?;
                            for start in (sentence.chars.len() + 1)
                                .saturating_sub(usize::from(self.char_ngram_size))
                                ..sentence.chars.len() + 1
                            {
                                features.push(TagFeature::right_char_ngram_eos(
                                    rel_position,
                                    sentence.char_substring(start, sentence.chars.len()),
                                ));
                            }
                        }
                    }
                }
                BoundaryType::NotWordBoundary => (),
                BoundaryType::Unknown => {
                    if current_tag.is_some() {
                        return Err(VaporettoError::invalid_argument("sentence", ""));
                    }
                }
            }
        }
        if let Some(tag) = current_tag.take() {
            for end in 0..sentence.chars.len().min(usize::from(self.char_ngram_size)) {
                features.push(TagFeature::left_char_ngram_bos(
                    -1,
                    sentence.char_substring(0, end),
                ));
            }
            features.push(TagFeature::chars(sentence.char_substring(0, tag_right_pos)));
            result.push(TagExample { features, tag });
        }
        Ok(result)
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
        let dict = Some(["東京特許許可局", "火星猫", "猫"]);
        let gen = BoundaryExampleGenerator::new(3, 2, 3, 2, dict, 2).unwrap();

        let s = Sentence::from_raw("猫").unwrap();
        let examples = gen.generate(&s);

        assert!(examples.is_empty());
    }

    #[test]
    fn test_example_generator_generate_all() {
        let dict = Some(["東京特許許可局", "火星猫", "猫"]);
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
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: &[Roman as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 0, ngram: &[Roman as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 1, ngram: &[Roman as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: &[Roman as u8, Roman as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 0, ngram: &[Roman as u8, Roman as u8] }),
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
                CharacterTypeNgram(BytesNgramFeature { rel_position: -2, ngram: &[Roman as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: &[Roman as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 0, ngram: &[Hiragana as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 1, ngram: &[Kanji as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -2, ngram: &[Roman as u8, Roman as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: &[Roman as u8, Hiragana as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 0, ngram: &[Hiragana as u8, Kanji as u8] }),
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
                CharacterTypeNgram(BytesNgramFeature { rel_position: -2, ngram: &[Kanji as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: &[Kanji as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: 0, ngram: &[Hiragana as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -2, ngram: &[Kanji as u8, Kanji as u8] }),
                CharacterTypeNgram(BytesNgramFeature { rel_position: -1, ngram: &[Kanji as u8, Hiragana as u8] }),
                DictionaryWord(DictionaryWordFeature { position: DictionaryWordPosition::Left, length: 2 }),
                DictionaryWord(DictionaryWordFeature { position: DictionaryWordPosition::Left, length: 1 }),
            ],
            label: WordBoundary,
        };
        assert_eq!(expected, examples[6]);
    }

    #[test]
    fn test_example_generator_generate_without_unknown() {
        let dict = Some(["東京特許許可局", "火星猫", "猫"]);
        let gen = BoundaryExampleGenerator::new(3, 2, 3, 2, dict, 2).unwrap();

        let s = Sentence::from_partial_annotation("A-r-i-a|は|火-星 猫|だ").unwrap();
        let examples = gen.generate(&s);

        assert_eq!(7, examples.len());
    }

    #[test]
    fn test_tag_example_generate_33() {
        let gen = TagExampleGenerator::new(3, 3);

        let s =
            Sentence::from_partial_annotation("A-r-i-a/名詞|は/助詞|火-星 猫|だ/助動詞").unwrap();
        let mut examples = gen.generate(&s).unwrap();

        // The order of examples is unimportant.
        examples
            .iter_mut()
            .for_each(|example| example.features.sort_unstable());
        examples.sort_unstable();

        let mut expected = vec![
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram(1, "iaは"),
                    TagFeature::right_char_ngram(1, "aは"),
                    TagFeature::right_char_ngram(1, "は"),
                    TagFeature::right_char_ngram(2, "aは火"),
                    TagFeature::right_char_ngram(2, "は火"),
                    TagFeature::right_char_ngram(2, "火"),
                    TagFeature::right_char_ngram(3, "は火星"),
                    TagFeature::right_char_ngram(3, "火星"),
                    TagFeature::right_char_ngram(3, "星"),
                    TagFeature::left_char_ngram_bos(-1, ""),
                    TagFeature::left_char_ngram_bos(-1, "A"),
                    TagFeature::left_char_ngram_bos(-1, "Ar"),
                    TagFeature::chars("Aria"),
                ],
                tag: Arc::new("名詞".to_string()),
            },
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram(1, "aは火"),
                    TagFeature::right_char_ngram(1, "は火"),
                    TagFeature::right_char_ngram(1, "火"),
                    TagFeature::right_char_ngram(2, "は火星"),
                    TagFeature::right_char_ngram(2, "火星"),
                    TagFeature::right_char_ngram(2, "星"),
                    TagFeature::right_char_ngram(3, "火星猫"),
                    TagFeature::right_char_ngram(3, "星猫"),
                    TagFeature::right_char_ngram(3, "猫"),
                    TagFeature::left_char_ngram(-3, "r"),
                    TagFeature::left_char_ngram(-3, "ri"),
                    TagFeature::left_char_ngram(-3, "ria"),
                    TagFeature::left_char_ngram(-2, "i"),
                    TagFeature::left_char_ngram(-2, "ia"),
                    TagFeature::left_char_ngram(-2, "iaは"),
                    TagFeature::left_char_ngram(-1, "a"),
                    TagFeature::left_char_ngram(-1, "aは"),
                    TagFeature::left_char_ngram(-1, "aは火"),
                    TagFeature::chars("は"),
                ],
                tag: Arc::new("助詞".to_string()),
            },
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram_eos(1, "猫だ"),
                    TagFeature::right_char_ngram_eos(1, "だ"),
                    TagFeature::right_char_ngram_eos(1, ""),
                    TagFeature::left_char_ngram(-3, "火"),
                    TagFeature::left_char_ngram(-3, "火星"),
                    TagFeature::left_char_ngram(-3, "火星猫"),
                    TagFeature::left_char_ngram(-2, "星"),
                    TagFeature::left_char_ngram(-2, "星猫"),
                    TagFeature::left_char_ngram(-2, "星猫だ"),
                    TagFeature::left_char_ngram(-1, "猫"),
                    TagFeature::left_char_ngram(-1, "猫だ"),
                    TagFeature::chars("だ"),
                ],
                tag: Arc::new("助動詞".to_string()),
            },
        ];

        expected
            .iter_mut()
            .for_each(|example| example.features.sort_unstable());
        expected.sort_unstable();

        assert_eq!(expected, examples);
    }

    #[test]
    fn test_tag_example_generate_32() {
        let gen = TagExampleGenerator::new(3, 2);

        let s =
            Sentence::from_partial_annotation("A-r-i-a/名詞|は/助詞|火-星 猫|だ/助動詞").unwrap();
        let mut examples = gen.generate(&s).unwrap();

        // The order of examples is unimportant.
        examples
            .iter_mut()
            .for_each(|example| example.features.sort_unstable());
        examples.sort_unstable();

        let mut expected = vec![
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram(1, "iaは"),
                    TagFeature::right_char_ngram(1, "aは"),
                    TagFeature::right_char_ngram(1, "は"),
                    TagFeature::right_char_ngram(2, "aは火"),
                    TagFeature::right_char_ngram(2, "は火"),
                    TagFeature::right_char_ngram(2, "火"),
                    TagFeature::left_char_ngram_bos(-1, ""),
                    TagFeature::left_char_ngram_bos(-1, "A"),
                    TagFeature::left_char_ngram_bos(-1, "Ar"),
                    TagFeature::chars("Aria"),
                ],
                tag: Arc::new("名詞".to_string()),
            },
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram(1, "aは火"),
                    TagFeature::right_char_ngram(1, "は火"),
                    TagFeature::right_char_ngram(1, "火"),
                    TagFeature::right_char_ngram(2, "は火星"),
                    TagFeature::right_char_ngram(2, "火星"),
                    TagFeature::right_char_ngram(2, "星"),
                    TagFeature::left_char_ngram(-2, "i"),
                    TagFeature::left_char_ngram(-2, "ia"),
                    TagFeature::left_char_ngram(-2, "iaは"),
                    TagFeature::left_char_ngram(-1, "a"),
                    TagFeature::left_char_ngram(-1, "aは"),
                    TagFeature::left_char_ngram(-1, "aは火"),
                    TagFeature::chars("は"),
                ],
                tag: Arc::new("助詞".to_string()),
            },
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram_eos(1, "猫だ"),
                    TagFeature::right_char_ngram_eos(1, "だ"),
                    TagFeature::right_char_ngram_eos(1, ""),
                    TagFeature::left_char_ngram(-2, "星"),
                    TagFeature::left_char_ngram(-2, "星猫"),
                    TagFeature::left_char_ngram(-2, "星猫だ"),
                    TagFeature::left_char_ngram(-1, "猫"),
                    TagFeature::left_char_ngram(-1, "猫だ"),
                    TagFeature::chars("だ"),
                ],
                tag: Arc::new("助動詞".to_string()),
            },
        ];

        expected
            .iter_mut()
            .for_each(|example| example.features.sort_unstable());
        expected.sort_unstable();

        assert_eq!(expected, examples);
    }

    #[test]
    fn test_tag_example_generate_23() {
        let gen = TagExampleGenerator::new(2, 3);

        let s =
            Sentence::from_partial_annotation("A-r-i-a/名詞|は/助詞|火-星 猫|だ/助動詞").unwrap();
        let mut examples = gen.generate(&s).unwrap();

        // The order of examples is unimportant.
        examples
            .iter_mut()
            .for_each(|example| example.features.sort_unstable());
        examples.sort_unstable();

        let mut expected = vec![
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram(1, "aは"),
                    TagFeature::right_char_ngram(1, "は"),
                    TagFeature::right_char_ngram(2, "は火"),
                    TagFeature::right_char_ngram(2, "火"),
                    TagFeature::right_char_ngram(3, "火星"),
                    TagFeature::right_char_ngram(3, "星"),
                    TagFeature::left_char_ngram_bos(-1, ""),
                    TagFeature::left_char_ngram_bos(-1, "A"),
                    TagFeature::chars("Aria"),
                ],
                tag: Arc::new("名詞".to_string()),
            },
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram(1, "は火"),
                    TagFeature::right_char_ngram(1, "火"),
                    TagFeature::right_char_ngram(2, "火星"),
                    TagFeature::right_char_ngram(2, "星"),
                    TagFeature::right_char_ngram(3, "星猫"),
                    TagFeature::right_char_ngram(3, "猫"),
                    TagFeature::left_char_ngram(-3, "r"),
                    TagFeature::left_char_ngram(-3, "ri"),
                    TagFeature::left_char_ngram(-2, "i"),
                    TagFeature::left_char_ngram(-2, "ia"),
                    TagFeature::left_char_ngram(-1, "a"),
                    TagFeature::left_char_ngram(-1, "aは"),
                    TagFeature::chars("は"),
                ],
                tag: Arc::new("助詞".to_string()),
            },
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram_eos(1, "だ"),
                    TagFeature::right_char_ngram_eos(1, ""),
                    TagFeature::left_char_ngram(-3, "火"),
                    TagFeature::left_char_ngram(-3, "火星"),
                    TagFeature::left_char_ngram(-2, "星"),
                    TagFeature::left_char_ngram(-2, "星猫"),
                    TagFeature::left_char_ngram(-1, "猫"),
                    TagFeature::left_char_ngram(-1, "猫だ"),
                    TagFeature::chars("だ"),
                ],
                tag: Arc::new("助動詞".to_string()),
            },
        ];

        expected
            .iter_mut()
            .for_each(|example| example.features.sort_unstable());
        expected.sort_unstable();

        assert_eq!(expected, examples);
    }

    #[test]
    fn test_tag_example_generate_check_sentence_boundary() {
        let gen = TagExampleGenerator::new(3, 3);

        let s = Sentence::from_tokenized("僕/代名詞 は/助詞 人間/名詞").unwrap();
        let mut examples = gen.generate(&s).unwrap();

        // The order of examples is unimportant.
        examples
            .iter_mut()
            .for_each(|example| example.features.sort_unstable());
        examples.sort_unstable();

        let mut expected = vec![
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram(1, "僕は"),
                    TagFeature::right_char_ngram(1, "は"),
                    TagFeature::right_char_ngram(2, "僕は人"),
                    TagFeature::right_char_ngram(2, "は人"),
                    TagFeature::right_char_ngram(2, "人"),
                    TagFeature::right_char_ngram(3, "は人間"),
                    TagFeature::right_char_ngram(3, "人間"),
                    TagFeature::right_char_ngram(3, "間"),
                    TagFeature::left_char_ngram_bos(-1, ""),
                    TagFeature::left_char_ngram_bos(-1, "僕"),
                    TagFeature::left_char_ngram_bos(-1, "僕は"),
                    TagFeature::chars("僕"),
                ],
                tag: Arc::new("代名詞".to_string()),
            },
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram(1, "僕は人"),
                    TagFeature::right_char_ngram(1, "は人"),
                    TagFeature::right_char_ngram(1, "人"),
                    TagFeature::right_char_ngram(2, "は人間"),
                    TagFeature::right_char_ngram(2, "人間"),
                    TagFeature::right_char_ngram(2, "間"),
                    TagFeature::right_char_ngram_eos(3, "人間"),
                    TagFeature::right_char_ngram_eos(3, "間"),
                    TagFeature::right_char_ngram_eos(3, ""),
                    TagFeature::left_char_ngram_bos(-2, "僕は"),
                    TagFeature::left_char_ngram_bos(-2, "僕"),
                    TagFeature::left_char_ngram_bos(-2, ""),
                    TagFeature::left_char_ngram(-1, "僕は人"),
                    TagFeature::left_char_ngram(-1, "僕は"),
                    TagFeature::left_char_ngram(-1, "僕"),
                    TagFeature::chars("は"),
                ],
                tag: Arc::new("助詞".to_string()),
            },
            TagExample {
                features: vec![
                    TagFeature::right_char_ngram_eos(1, "人間"),
                    TagFeature::right_char_ngram_eos(1, "間"),
                    TagFeature::right_char_ngram_eos(1, ""),
                    TagFeature::left_char_ngram_bos(-3, "僕は"),
                    TagFeature::left_char_ngram_bos(-3, "僕"),
                    TagFeature::left_char_ngram_bos(-3, ""),
                    TagFeature::left_char_ngram(-2, "僕は人"),
                    TagFeature::left_char_ngram(-2, "僕は"),
                    TagFeature::left_char_ngram(-2, "僕"),
                    TagFeature::left_char_ngram(-1, "は人間"),
                    TagFeature::left_char_ngram(-1, "は人"),
                    TagFeature::left_char_ngram(-1, "は"),
                    TagFeature::chars("人間"),
                ],
                tag: Arc::new("名詞".to_string()),
            },
        ];

        expected
            .iter_mut()
            .for_each(|example| example.features.sort_unstable());
        expected.sort_unstable();

        assert_eq!(expected, examples);
    }
}
