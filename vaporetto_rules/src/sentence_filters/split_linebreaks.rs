use vaporetto::{BoundaryType, Sentence};

use crate::SentenceFilter;

/// Line breaks splitter.
#[derive(Clone, Default)]
pub struct SplitLinebreaksFilter;

impl SentenceFilter for SplitLinebreaksFilter {
    fn filter(&self, mut sentence: Sentence) -> Sentence {
        let (chars, _, boundaries) = sentence.chars_and_boundaries_mut();
        for ((c1, c2), b) in chars.iter().zip(&chars[1..]).zip(boundaries) {
            match (*c1, *c2) {
                ('\r' | '\n', _) | (_, '\r' | '\n') => {
                    *b = BoundaryType::WordBoundary;
                }
                _ => {}
            }
        }
        sentence
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_lf() {
        let s = Sentence::from_tokenized("前の行\n次の行").unwrap();
        let filter = SplitLinebreaksFilter;
        let s = filter.filter(s);
        assert_eq!("前の行 \n 次の行", s.to_tokenized_string().unwrap());
    }

    #[test]
    fn test_split_cr() {
        let s = Sentence::from_tokenized("前の行\r次の行").unwrap();
        let filter = SplitLinebreaksFilter;
        let s = filter.filter(s);
        assert_eq!("前の行 \r 次の行", s.to_tokenized_string().unwrap());
    }

    #[test]
    fn test_split_crlf() {
        let s = Sentence::from_tokenized("前の行\r\n次の行").unwrap();
        let filter = SplitLinebreaksFilter;
        let s = filter.filter(s);
        assert_eq!("前の行 \r \n 次の行", s.to_tokenized_string().unwrap());
    }
}
