//! Rule base filters for Vaporetto.

use unicode_segmentation::UnicodeSegmentation;
use vaporetto::{BoundaryType, CharacterType, Sentence};

/// Concatenates grapheme clusters.
///
/// # Arguments:
///
/// * `sentence` - Input sentence.
///
/// # Returns
///
/// A processed sentence.
pub fn concat_grapheme_clusters(mut sentence: Sentence) -> Sentence {
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

/// Concatenates consecutive character types.
///
/// # Arguments:
///
/// * `sentence` - Input sentence.
/// * `t_flag` - Character type.
///
/// # Returns
///
/// A processed sentence.
pub fn concat_cons_char_types(mut sentence: Sentence, t_flag: CharacterType) -> Sentence {
    let t_flag = t_flag as u8;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concat_grapheme_clusters_no_boundary() {
        let s = Sentence::from_tokenized("\u{200d}").unwrap();
        let s = concat_grapheme_clusters(s);
        assert_eq!("\u{200d}", s.to_tokenized_string().unwrap());
    }

    #[test]
    fn test_concat_grapheme_clusters_zwj() {
        let s =
            Sentence::from_tokenized("\u{1f468} \u{200d} \u{1f469} \u{200d} \u{1f466}").unwrap();
        let s = concat_grapheme_clusters(s);
        assert_eq!(
            "\u{1f468}\u{200d}\u{1f469}\u{200d}\u{1f466}",
            s.to_tokenized_string().unwrap()
        );
    }

    #[test]
    fn test_concat_grapheme_clusters_color() {
        let s = Sentence::from_tokenized("\u{1f44f} \u{1f3fd}").unwrap();
        let s = concat_grapheme_clusters(s);
        assert_eq!("\u{1f44f}\u{1f3fd}", s.to_tokenized_string().unwrap());
    }

    #[test]
    fn test_concat_grapheme_clusters_combined() {
        let s = Sentence::from_tokenized("これ は 手 \u{1f44f} \u{1f3fd} で す").unwrap();
        let s = concat_grapheme_clusters(s);
        assert_eq!(
            "これ は 手 \u{1f44f}\u{1f3fd} で す",
            s.to_tokenized_string().unwrap()
        );
    }

    #[test]
    fn test_concat_cons_char_types_no_boundary() {
        let s = Sentence::from_tokenized("5").unwrap();
        let s = concat_cons_char_types(s, CharacterType::Digit);
        assert_eq!("5", s.to_tokenized_string().unwrap());
    }

    #[test]
    fn test_concat_cons_char_types() {
        let s = Sentence::from_tokenized("5 00 0").unwrap();
        let s = concat_cons_char_types(s, CharacterType::Digit);
        assert_eq!("5000", s.to_tokenized_string().unwrap());
    }

    #[test]
    fn test_concat_cons_char_types_combined() {
        let s = Sentence::from_tokenized("20 21 年 8 月 2 4 日").unwrap();
        let s = concat_cons_char_types(s, CharacterType::Digit);
        assert_eq!("2021 年 8 月 24 日", s.to_tokenized_string().unwrap());
    }
}
