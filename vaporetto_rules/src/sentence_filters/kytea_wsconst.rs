use vaporetto::{CharacterBoundary, CharacterType, Sentence};

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
    fn filter(&self, sentence: &mut Sentence) {
        let t_flag = self.char_type as u8;
        let len = sentence.char_types().len() - 1;
        for i in 0..len {
            unsafe {
                if *sentence.char_types().get_unchecked(i) == t_flag
                    && *sentence.char_types().get_unchecked(i + 1) == t_flag
                {
                    *sentence.boundaries_mut().get_unchecked_mut(i) =
                        CharacterBoundary::NotWordBoundary;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::string::String;

    #[test]
    fn test_concat_cons_char_types_no_boundary() {
        let mut s = Sentence::from_tokenized("5").unwrap();
        let filter = KyteaWsConstFilter::new(CharacterType::Digit);
        filter.filter(&mut s);
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);
        assert_eq!("5", buf);
    }

    #[test]
    fn test_concat_cons_char_types() {
        let mut s = Sentence::from_tokenized("5 00 0").unwrap();
        let filter = KyteaWsConstFilter::new(CharacterType::Digit);
        filter.filter(&mut s);
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);
        assert_eq!("5000", buf);
    }

    #[test]
    fn test_concat_cons_char_types_combined() {
        let mut s = Sentence::from_tokenized("20 21 年 8 月 2 4 日").unwrap();
        let filter = KyteaWsConstFilter::new(CharacterType::Digit);
        filter.filter(&mut s);
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);
        assert_eq!("2021 年 8 月 24 日", buf);
    }
}
