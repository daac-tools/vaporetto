use unicode_segmentation::UnicodeSegmentation;
use vaporetto::{CharacterBoundary, Sentence};

use crate::SentenceFilter;

/// Grapheme cluster concatenator.
#[derive(Clone, Default)]
pub struct ConcatGraphemeClustersFilter;

impl SentenceFilter for ConcatGraphemeClustersFilter {
    fn filter(&self, sentence: &mut Sentence) {
        let mut start = 0;
        let mut offset = 0;
        unsafe {
            debug_assert!(sentence.as_raw_text().is_char_boundary(offset));
            while let Some((len, n_chars)) = sentence
                .as_raw_text()
                .get_unchecked(offset..)
                .graphemes(true)
                .next()
                .map(|x| (x.len(), x.chars().count()))
            {
                offset += len;
                let end = start + n_chars;
                debug_assert!(start <= sentence.boundaries().len());
                debug_assert!(end <= sentence.boundaries().len() + 1);
                sentence
                    .boundaries_mut()
                    .get_unchecked_mut(start..end - 1)
                    .fill(CharacterBoundary::NotWordBoundary);
                start = end;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::string::String;

    #[test]
    fn test_concat_grapheme_clusters_no_boundary() {
        let mut s = Sentence::from_tokenized("\u{200d}").unwrap();
        let filter = ConcatGraphemeClustersFilter;
        filter.filter(&mut s);
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);
        assert_eq!("\u{200d}", buf);
    }

    #[test]
    fn test_concat_grapheme_clusters_zwj() {
        let mut s =
            Sentence::from_tokenized("\u{1f468} \u{200d} \u{1f469} \u{200d} \u{1f466}").unwrap();
        let filter = ConcatGraphemeClustersFilter;
        filter.filter(&mut s);
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);
        assert_eq!("\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f466}", buf);
    }

    #[test]
    fn test_concat_grapheme_clusters_color() {
        let mut s = Sentence::from_tokenized("\u{1f44f} \u{1f3fd}").unwrap();
        let filter = ConcatGraphemeClustersFilter;
        filter.filter(&mut s);
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);
        assert_eq!("\u{1f44f}\u{1f3fd}", buf);
    }

    #[test]
    fn test_concat_grapheme_clusters_combined() {
        let mut s = Sentence::from_tokenized("これ は 手 \u{1f44f} \u{1f3fd} で す").unwrap();
        let filter = ConcatGraphemeClustersFilter;
        filter.filter(&mut s);
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);
        assert_eq!("これ は 手 \u{1f44f}\u{1f3fd} で す", buf);
    }
}
