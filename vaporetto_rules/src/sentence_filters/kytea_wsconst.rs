use vaporetto::{BoundaryType, CharacterType, Sentence};

use crate::SentenceFilter;

/// Character type concatenator. This filter works like KyTea's wsconst option.
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
    pub fn new(char_type: CharacterType) -> Self {
        Self { char_type }
    }
}

impl SentenceFilter for KyteaWsConstFilter {
    /// Concatenates consecutive character types.
    ///
    /// # Arguments:
    ///
    /// * `sentence` - Input sentence.
    ///
    /// # Returns
    ///
    /// A processed sentence.
    fn filter(&self, mut sentence: Sentence) -> Sentence {
        let t_flag = self.char_type as u8;
        let mut tmp = sentence.boundaries().to_vec();
        for (i, (b, &t)) in tmp.iter_mut().zip(sentence.char_types()).enumerate() {
            if t == t_flag && t == sentence.char_types()[i + 1] {
                *b = BoundaryType::NotWordBoundary;
            }
        }
        for (b, t) in sentence.boundaries_mut().iter_mut().zip(&tmp) {
            *b = *t;
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
