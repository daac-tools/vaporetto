use vaporetto::{CharacterBoundary, Sentence};

use crate::SentenceFilter;

/// Line breaks splitter.
#[derive(Clone, Default)]
pub struct SplitLinebreaksFilter;

impl SentenceFilter for SplitLinebreaksFilter {
    fn filter(&self, sentence: &mut Sentence) {
        unsafe {
            let mut prev_c = sentence.as_raw_text().chars().next().unwrap_unchecked();
            let mut offset = prev_c.len_utf8();
            let mut i = 0;
            while let Some(c) = sentence
                .as_raw_text()
                .get_unchecked(offset..)
                .chars()
                .next()
            {
                offset += c.len_utf8();
                match (prev_c, c) {
                    ('\r' | '\n', _) | (_, '\r' | '\n') => {
                        *sentence.boundaries_mut().get_unchecked_mut(i) =
                            CharacterBoundary::WordBoundary;
                    }
                    _ => {}
                }
                prev_c = c;
                i += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::string::String;

    #[test]
    fn test_split_lf() {
        let mut s = Sentence::from_tokenized("前の行\n次の行").unwrap();
        let filter = SplitLinebreaksFilter;
        filter.filter(&mut s);
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);
        assert_eq!("前の行 \n 次の行", buf);
    }

    #[test]
    fn test_split_cr() {
        let mut s = Sentence::from_tokenized("前の行\r次の行").unwrap();
        let filter = SplitLinebreaksFilter;
        filter.filter(&mut s);
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);
        assert_eq!("前の行 \r 次の行", buf);
    }

    #[test]
    fn test_split_crlf() {
        let mut s = Sentence::from_tokenized("前の行\r\n次の行").unwrap();
        let filter = SplitLinebreaksFilter;
        filter.filter(&mut s);
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);
        assert_eq!("前の行 \r \n 次の行", buf);
    }
}
