use vaporetto::{BoundaryType, CharacterType, Sentence};

use crate::SentenceFilter;

/// Character type concatenator. This filter works like KyTea's wsconst option.
#[derive(Clone)]
pub struct KyteaWsConstFilter {
    char_type: CharacterType,
}

impl KyteaWsConstFilter {
    /// Creates a new KyteaWsConstFilter.
    ///
    /// # Arguments
    ///
    /// * `char_type` - Character type.
    ///
    /// # Returns
    ///
    /// A new KyteaWsConstFilter.
    pub const fn new(char_type: CharacterType) -> Self {
        Self { char_type }
    }
}

impl SentenceFilter for KyteaWsConstFilter {
    fn filter(&self, mut sentence: Sentence) -> Sentence {
        let t_flag = self.char_type as u8;
        let (_, char_types, boundaries) = sentence.chars_and_boundaries_mut();
        for ((t1, t2), b) in char_types.iter().zip(&char_types[1..]).zip(boundaries) {
            if *t1 == t_flag && *t2 == t_flag {
                *b = BoundaryType::NotWordBoundary;
            }
        }
        sentence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_cons_char_types_no_boundary() {
        let s = Sentence::from_tokenized("5").unwrap();
        let filter = KyteaWsConstFilter::new(CharacterType::Digit);
        let s = filter.filter(s);
        assert_eq!("5", s.to_tokenized_string().unwrap());
    }

    #[test]
    fn test_concat_cons_char_types() {
        let s = Sentence::from_tokenized("5 00 0").unwrap();
        let filter = KyteaWsConstFilter::new(CharacterType::Digit);
        let s = filter.filter(s);
        assert_eq!("5000", s.to_tokenized_string().unwrap());
    }

    #[test]
    fn test_concat_cons_char_types_combined() {
        let s = Sentence::from_tokenized("20 21 年 8 月 2 4 日").unwrap();
        let filter = KyteaWsConstFilter::new(CharacterType::Digit);
        let s = filter.filter(s);
        assert_eq!("2021 年 8 月 24 日", s.to_tokenized_string().unwrap());
    }
}
