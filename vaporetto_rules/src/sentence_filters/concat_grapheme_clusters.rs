use unicode_segmentation::UnicodeSegmentation;
use vaporetto::{BoundaryType, Sentence};

use crate::SentenceFilter;

/// Grapheme cluster concatenator.
pub struct ConcatGraphemeClustersFilter;

impl ConcatGraphemeClustersFilter {
    /// Creates a new ConcatGraphemeClustersFilter.
    ///
    /// # Returns
    ///
    /// A new ConcatGraphemeClustersFilter.
    pub fn new() -> Self {
        Self {}
    }
}

impl Default for ConcatGraphemeClustersFilter {
    fn default() -> Self {
        Self::new()
    }
}

impl SentenceFilter for ConcatGraphemeClustersFilter {
    /// Concatenates grapheme clusters.
    ///
    /// # Arguments:
    ///
    /// * `sentence` - Input sentence.
    ///
    /// # Returns
    ///
    /// A processed sentence.
    fn filter(&self, mut sentence: Sentence) -> Sentence {
        let mut tmp = sentence.boundaries().to_vec();
        for (i, c) in UnicodeSegmentation::grapheme_indices(sentence.to_raw_string(), true) {
            let start = sentence.get_char_pos(i).unwrap();
            let end = sentence.get_char_pos(i + c.len()).unwrap() - 1;
            for b in &mut tmp[start..end] {
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
    fn test_concat_grapheme_clusters_no_boundary() {
        let s = Sentence::from_tokenized("\u{200d}").unwrap();
        let filter = ConcatGraphemeClustersFilter::new();
        let s = filter.filter(s);
        assert_eq!("\u{200d}", s.to_tokenized_string().unwrap());
    }

    #[test]
    fn test_concat_grapheme_clusters_zwj() {
        let s =
            Sentence::from_tokenized("\u{1f468} \u{200d} \u{1f469} \u{200d} \u{1f466}").unwrap();
        let filter = ConcatGraphemeClustersFilter::new();
        let s = filter.filter(s);
        assert_eq!(
            "\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f466}",
            s.to_tokenized_string().unwrap()
        );
    }

    #[test]
    fn test_concat_grapheme_clusters_color() {
        let s = Sentence::from_tokenized("\u{1f44f} \u{1f3fd}").unwrap();
        let filter = ConcatGraphemeClustersFilter::new();
        let s = filter.filter(s);
        assert_eq!("\u{1f44f}\u{1f3fd}", s.to_tokenized_string().unwrap());
    }

    #[test]
    fn test_concat_grapheme_clusters_combined() {
        let s = Sentence::from_tokenized("これ は 手 \u{1f44f} \u{1f3fd} で す").unwrap();
        let filter = ConcatGraphemeClustersFilter::new();
        let s = filter.filter(s);
        assert_eq!(
            "これ は 手 \u{1f44f}\u{1f3fd} で す",
            s.to_tokenized_string().unwrap()
        );
    }
}
