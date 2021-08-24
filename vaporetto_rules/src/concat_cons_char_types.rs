use vaporetto::{BoundaryType, CharacterType, Sentence};

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
