use anyhow::{anyhow, Result};

/// Character type.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(u8)]
pub enum CharacterType {
    /// Digit character. (e.g. 0, 1, 2, ...)
    Digit = b'D',

    /// Roman character. (e.g. A, B, C, ...)
    Roman = b'R',

    /// Japanese Hiragana character. (e.g. あ, い, う, ...)
    Hiragana = b'H',

    /// Japanese Katakana character. (e.g. ア, イ, ウ, ...)
    Katakana = b'T',

    /// Kanji (a.k.a. Hanzi or Hanja) character. (e.g. 漢, 字, ...)
    Kanji = b'K',

    /// Other character.
    Other = b'O',
}

impl CharacterType {
    /// Gets a character type of a given character.
    ///
    /// # Arguments
    ///
    /// * `c` - A character.
    ///
    /// # Returns
    ///
    /// A character type.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::CharacterType;
    ///
    /// let t = CharacterType::get_type('A');
    /// assert_eq!(CharacterType::Roman, t);
    /// ```
    pub const fn get_type(c: char) -> Self {
        match c as u32 {
            0x30..=0x39 | 0xFF10..=0xFF19 => Self::Digit,
            0x41..=0x5A | 0x61..=0x7A | 0xFF21..=0xFF3A | 0xFF41..=0xFF5A => Self::Roman,
            0x3040..=0x3096 => Self::Hiragana,
            0x30A0..=0x30FA | 0x30FC..=0x30FF | 0xFF66..=0xFF9F => Self::Katakana,
            0x3400..=0x4DBF          // CJK Unified Ideographs Extension A
                | 0x4E00..=0x9FFF    // CJK Unified Ideographs
                | 0xF900..=0xFAFF    // CJK Compatibility Ideographs
                | 0x20000..=0x2A6DF  // CJK Unified Ideographs Extension B
                | 0x2A700..=0x2B73F  // CJK Unified Ideographs Extension C
                | 0x2B740..=0x2B81F  // CJK Unified Ideographs Extension D
                | 0x2B820..=0x2CEAF  // CJK Unified Ideographs Extension E
                | 0x2F800..=0x2FA1F  // CJK Compatibility Ideographs Supplement
                => Self::Kanji,
            _ => Self::Other,
        }
    }
}

/// Boundary type.
#[derive(Debug, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum BoundaryType {
    /// Inner of a word.
    NotWordBoundary = 0,

    /// Word boundary.
    WordBoundary = 1,

    /// Unknown. (Not annotated.)
    Unknown = 2,
}

/// Sentence with boundary annotations.
#[derive(Debug, PartialEq, Clone)]
pub struct Sentence {
    pub(crate) text: String,
    pub(crate) str_to_char_pos: Vec<usize>,
    pub(crate) char_to_str_pos: Vec<usize>,
    pub(crate) char_type: Vec<u8>,
    pub(crate) boundaries: Vec<BoundaryType>,
    pub(crate) boundary_scores: Option<Vec<f64>>,
}

impl Sentence {
    fn common_info(chars: &[char]) -> (Vec<usize>, Vec<usize>, Vec<u8>) {
        let mut char_to_str_pos = Vec::with_capacity(chars.len() + 1);
        let mut char_type = Vec::with_capacity(chars.len());
        let mut pos = 0;
        char_to_str_pos.push(0);
        for &c in chars {
            pos += c.len_utf8();
            char_to_str_pos.push(pos);
            char_type.push(CharacterType::get_type(c) as u8)
        }
        let mut str_to_char_pos = vec![0; char_to_str_pos.last().unwrap_or(&0) + 1];
        for (i, &j) in char_to_str_pos.iter().enumerate() {
            // j < str_to_char_pos.len()
            unsafe {
                *str_to_char_pos.get_unchecked_mut(j) = i;
            }
        }
        (char_to_str_pos, str_to_char_pos, char_type)
    }

    /// Creates a new [`Sentence`] from a given string.
    ///
    /// # Arguments
    ///
    /// * `text` - A raw string without any annotation.
    ///
    /// # Returns
    ///
    /// A new [`Sentence`].
    ///
    /// # Errors
    ///
    /// If the given `text` is empty, an error variant will be returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_raw("How are you?");
    /// assert!(s.is_ok());
    ///
    /// let s = Sentence::from_raw("");
    /// assert!(s.is_err());
    /// ```
    pub fn from_raw<S>(text: S) -> Result<Self>
    where
        S: Into<String>,
    {
        let text = text.into();

        if text.is_empty() {
            return Err(anyhow!("`text` is empty"));
        }

        let chars: Vec<char> = text.chars().collect();
        let boundaries = vec![BoundaryType::Unknown; chars.len() - 1];

        let (char_to_str_pos, str_to_char_pos, char_type) = Self::common_info(&chars);

        Ok(Self {
            text,
            str_to_char_pos,
            char_to_str_pos,
            char_type,
            boundaries,
            boundary_scores: None,
        })
    }

    /// Gets a string without any annotation.
    ///
    /// # Returns
    ///
    /// A reference to the string.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_raw("How are you?").unwrap();
    /// assert_eq!("How are you?", s.to_raw_string());
    /// ```
    pub fn to_raw_string(&self) -> &str {
        &self.text
    }

    /// Creates a new [`Sentence`] from a tokenized string.
    ///
    /// # Arguments
    ///
    /// * `tokenized_text` - A tokenized string containing whitespaces for word boundaries.
    ///
    /// # Returns
    ///
    /// A new [`Sentence`].
    ///
    /// # Errors
    ///
    /// This function will return an error variant when:
    ///
    /// * `tokenized_text` is empty.
    /// * `tokenized_text` starts/ends with a whitespace.
    /// * `tokenized_text` contains consecutive whitespaces.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_tokenized("How are you?");
    /// assert!(s.is_ok());
    ///
    /// let s = Sentence::from_tokenized("How  are you?");
    /// assert!(s.is_err());
    /// ```
    pub fn from_tokenized<S>(tokenized_text: S) -> Result<Self>
    where
        S: AsRef<str>,
    {
        let tokenized_text = tokenized_text.as_ref();

        if tokenized_text.is_empty() {
            return Err(anyhow!("`tokenized_text` is empty"));
        }

        let tokenized_chars: Vec<char> = tokenized_text.chars().collect();
        let mut chars = Vec::with_capacity(tokenized_chars.len());
        let mut boundaries = Vec::with_capacity(tokenized_chars.len() - 1);

        let mut prev_boundary = false;
        let mut escape = false;
        for c in tokenized_chars {
            match (escape, c) {
                (false, '\\') => {
                    escape = true;
                }
                (false, ' ') => {
                    if chars.is_empty() {
                        return Err(anyhow!("`tokenized_text` starts with a whitespace"));
                    } else if prev_boundary {
                        return Err(anyhow!("`tokenized_text` contains consecutive whitespaces"));
                    }
                    prev_boundary = true;
                }
                (_, _) => {
                    if !chars.is_empty() {
                        boundaries.push(if prev_boundary {
                            BoundaryType::WordBoundary
                        } else {
                            BoundaryType::NotWordBoundary
                        });
                    }
                    prev_boundary = false;
                    escape = false;
                    chars.push(c);
                }
            };
        }
        if prev_boundary {
            return Err(anyhow!("`tokenized_text` ends with a whitespace"));
        }

        let (char_to_str_pos, str_to_char_pos, char_type) = Self::common_info(&chars);
        Ok(Self {
            text: chars.iter().collect(),
            char_to_str_pos,
            str_to_char_pos,
            char_type,
            boundaries,
            boundary_scores: None,
        })
    }

    /// Generates a string with whitespaces for word boundaries.
    ///
    /// # Returns
    ///
    /// A newly allocated string containing whitespaces for word boundaries.
    ///
    /// # Errors
    ///
    /// If the sentence contains unknown boundary, an error variant will be returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_tokenized("How are you?").unwrap();
    /// assert_eq!("How are you?", s.to_tokenized_string().unwrap());
    /// ```
    pub fn to_tokenized_string(&self) -> Result<String> {
        let chars: Vec<char> = self.text.chars().collect();
        let mut result = String::with_capacity(self.text.len() + chars.len() - 1);
        match chars[0] {
            '\\' | '/' | '&' | ' ' => result.push('\\'),
            _ => (),
        }
        result.push(chars[0]);
        for (&c, b) in chars[1..].iter().zip(&self.boundaries) {
            match b {
                BoundaryType::WordBoundary => {
                    result.push(' ');
                }
                BoundaryType::NotWordBoundary => (),
                BoundaryType::Unknown => {
                    return Err(anyhow!("sentence contains an unknown boundary"));
                }
            }
            match c {
                '\\' | '/' | '&' | ' ' => result.push('\\'),
                _ => (),
            }
            result.push(c);
        }
        Ok(result)
    }

    /// Generates a vector of words.
    ///
    /// # Returns
    ///
    /// A newly allocated vector of words.
    ///
    /// # Errors
    ///
    /// If the sentence contains unknown boundaries, an error variant will be returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_tokenized("How are you ?").unwrap();
    /// assert_eq!(vec![
    ///     "How",
    ///     "are",
    ///     "you",
    ///     "?",
    /// ], s.to_tokenized_vec().unwrap());
    /// ```
    pub fn to_tokenized_vec(&self) -> Result<Vec<&str>> {
        let mut result = vec![];
        let mut start = 0;
        for (i, b) in self.boundaries.iter().enumerate() {
            match b {
                BoundaryType::WordBoundary => {
                    let end = unsafe { *self.char_to_str_pos.get_unchecked(i + 1) };
                    let word = unsafe { self.text.get_unchecked(start..end) };
                    result.push(word);
                    start = end;
                }
                BoundaryType::NotWordBoundary => (),
                BoundaryType::Unknown => {
                    return Err(anyhow!("sentence contains an unknown boundary"));
                }
            }
        }
        let word = unsafe { self.text.get_unchecked(start..) };
        result.push(word);
        Ok(result)
    }

    /// Creates a new [`Sentence`] from a string with partial annotations.
    ///
    /// # Arguments
    ///
    /// * `labeled_text` - A string with partial annotations.
    ///
    /// # Returns
    ///
    /// A new [`Sentence`].
    ///
    /// # Errors
    ///
    /// This function will return an error variant when:
    ///
    /// * `labeled_text` is empty.
    /// * The length of `lsbeled_text` is even numbers.
    /// * `labeled_text` contains invalid boundary characters.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_partial_annotation("g-o-o-d|i-d e-a");
    /// assert!(s.is_ok());
    ///
    /// let s = Sentence::from_partial_annotation("b-a-d/i-d-e-a");
    /// assert!(s.is_err());
    /// ```
    pub fn from_partial_annotation<S>(labeled_text: S) -> Result<Self>
    where
        S: AsRef<str>,
    {
        let labeled_text = labeled_text.as_ref();

        if labeled_text.is_empty() {
            return Err(anyhow!("`labeled_text` is empty"));
        }

        let labeled_chars: Vec<char> = labeled_text.chars().collect();
        if labeled_chars.len() & 0x01 == 0 {
            return Err(anyhow!(
                "invalid length for `labeled_text`: {}",
                labeled_chars.len()
            ));
        }
        let mut chars = Vec::with_capacity(labeled_chars.len() / 2 + 1);
        let mut boundaries = Vec::with_capacity(labeled_chars.len() / 2);

        for c in labeled_chars.iter().skip(1).step_by(2) {
            boundaries.push(match c {
                ' ' => BoundaryType::Unknown,
                '|' => BoundaryType::WordBoundary,
                '-' => BoundaryType::NotWordBoundary,
                _ => return Err(anyhow!("invalid boundary character: '{}'", c)),
            });
        }
        for c in labeled_chars.into_iter().step_by(2) {
            chars.push(c);
        }

        let (char_to_str_pos, str_to_char_pos, char_type) = Self::common_info(&chars);
        Ok(Self {
            text: chars.iter().collect(),
            char_to_str_pos,
            str_to_char_pos,
            char_type,
            boundaries,
            boundary_scores: None,
        })
    }

    /// Generates a string with partial annotations.
    ///
    /// # Returns
    ///
    /// A newly allocated string with partial annotations.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_tokenized("How are you ?").unwrap();
    /// assert_eq!("H-o-w|a-r-e|y-o-u|?", &s.to_partial_annotation_string());
    /// ```
    pub fn to_partial_annotation_string(&self) -> String {
        let chars: Vec<char> = self.text.chars().collect();
        let mut result = String::with_capacity(self.text.len() + chars.len() - 1);
        result.push(chars[0]);
        for (&c, b) in chars[1..].iter().zip(&self.boundaries) {
            result.push(match b {
                BoundaryType::WordBoundary => '|',
                BoundaryType::NotWordBoundary => '-',
                BoundaryType::Unknown => ' ',
            });
            result.push(c);
        }
        result
    }

    /// Gets a reference to the boundary information.
    ///
    /// # Returns
    ///
    /// A reference to the boundary information.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::{BoundaryType, Sentence};
    ///
    /// let s = Sentence::from_partial_annotation("a|b-c d").unwrap();
    /// assert_eq!(&[
    ///     BoundaryType::WordBoundary,
    ///     BoundaryType::NotWordBoundary,
    ///     BoundaryType::Unknown,
    /// ], s.boundaries());
    /// ```
    pub fn boundaries(&self) -> &[BoundaryType] {
        &self.boundaries
    }

    /// Gets a mutable reference to the boundary information.
    ///
    /// # Returns
    ///
    /// A mutable reference to the boundary information.
    pub fn boundaries_mut(&mut self) -> &mut [BoundaryType] {
        &mut self.boundaries
    }

    /// Gets a reference to the character type information.
    ///
    /// # Returns
    ///
    /// A reference to the character type information.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_raw("A1あエ漢?").unwrap();
    /// assert_eq!(&[b'R', b'D', b'H', b'T', b'K', b'O',], s.char_types());
    /// ```
    pub fn char_types(&self) -> &[u8] {
        &self.char_type
    }

    /// Gets a reference to the boundary score information.
    ///
    /// # Returns
    ///
    /// If the predictor inserted, the boundary score information is returned. Otherwise, None.
    pub fn boundary_scores(&self) -> Option<&[f64]> {
        self.boundary_scores.as_deref()
    }

    /// Gets a character position in the code point unit.
    ///
    /// # Returns
    ///
    /// A position in the code point unit.
    ///
    /// # Errors
    ///
    /// `index` must be a valid position.
    pub fn get_char_pos(&self, index: usize) -> Result<usize> {
        if index == 0 {
            Ok(0)
        } else {
            match self.str_to_char_pos.get(index) {
                Some(index) if *index != 0 => Ok(*index),
                _ => Err(anyhow!("invalid index")),
            }
        }
    }

    #[cfg(feature = "train")]
    pub(crate) fn char_substring(&self, start: usize, end: usize) -> &str {
        let begin = self.char_to_str_pos[start];
        let end = self.char_to_str_pos[end];
        &self.text.as_str()[begin..end]
    }

    #[cfg(feature = "train")]
    pub(crate) fn type_substring(&self, start: usize, end: usize) -> &[u8] {
        &self.char_type[start..end]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use BoundaryType::*;
    use CharacterType::*;

    #[test]
    fn test_sentence_from_raw_empty() {
        let s = Sentence::from_raw("");

        assert!(s.is_err());
        assert_eq!("`text` is empty", &s.err().unwrap().to_string());
    }

    #[test]
    fn test_sentence_from_raw_one() {
        let s = Sentence::from_raw("あ");

        let expected = Sentence {
            text: "あ".to_string(),
            str_to_char_pos: vec![0, 0, 0, 1],
            char_to_str_pos: vec![0, 3],
            char_type: ct2u8vec![Hiragana],
            boundaries: vec![],
            boundary_scores: None,
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_from_raw() {
        let s = Sentence::from_raw("Rustで良いプログラミング体験を！");

        let expected = Sentence {
            text: "Rustで良いプログラミング体験を！".to_string(),
            str_to_char_pos: vec![
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            char_to_str_pos: vec![
                0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
            ],
            char_type: ct2u8vec![
                Roman, Roman, Roman, Roman, Hiragana, Kanji, Hiragana, Katakana, Katakana,
                Katakana, Katakana, Katakana, Katakana, Katakana, Kanji, Kanji, Hiragana, Other,
            ],
            boundaries: vec![Unknown; 17],
            boundary_scores: None,
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_to_raw() {
        let s = Sentence::from_raw("Rustで良いプログラミング体験を！");

        assert_eq!(
            "Rustで良いプログラミング体験を！",
            s.unwrap().to_raw_string()
        );
    }

    #[test]
    fn test_sentence_from_tokenized_empty() {
        let s = Sentence::from_tokenized("");

        assert!(s.is_err());
        assert_eq!("`tokenized_text` is empty", &s.err().unwrap().to_string());
    }

    #[test]
    fn test_sentence_from_tokenized_start_with_space() {
        let s = Sentence::from_tokenized(" Rust で 良い プログラミング 体験 を ！");

        assert!(s.is_err());
        assert_eq!(
            "`tokenized_text` starts with a whitespace",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_from_tokenized_end_with_space() {
        let s = Sentence::from_tokenized("Rust で 良い プログラミング 体験 を ！ ");

        assert!(s.is_err());
        assert_eq!(
            "`tokenized_text` ends with a whitespace",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_from_tokenized_two_spaces() {
        let s = Sentence::from_tokenized("Rust で 良い  プログラミング 体験 を ！");

        assert!(s.is_err());
        assert_eq!(
            "`tokenized_text` contains consecutive whitespaces",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_from_tokenized_one() {
        let s = Sentence::from_tokenized("あ");

        let expected = Sentence {
            text: "あ".to_string(),
            str_to_char_pos: vec![0, 0, 0, 1],
            char_to_str_pos: vec![0, 3],
            char_type: ct2u8vec![Hiragana],
            boundaries: vec![],
            boundary_scores: None,
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_from_tokenized() {
        let s = Sentence::from_tokenized("Rust で 良い プログラミング 体験 を ！");

        let expected = Sentence {
            text: "Rustで良いプログラミング体験を！".to_string(),
            str_to_char_pos: vec![
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            char_to_str_pos: vec![
                0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
            ],
            char_type: ct2u8vec![
                Roman, Roman, Roman, Roman, Hiragana, Kanji, Hiragana, Katakana, Katakana,
                Katakana, Katakana, Katakana, Katakana, Katakana, Kanji, Kanji, Hiragana, Other,
            ],
            boundaries: vec![
                NotWordBoundary,
                NotWordBoundary,
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
                NotWordBoundary,
                NotWordBoundary,
                NotWordBoundary,
                NotWordBoundary,
                NotWordBoundary,
                NotWordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
            ],
            boundary_scores: None,
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_from_tokenized_with_escape_whitespace() {
        let s = Sentence::from_tokenized("火星 猫 の 生態 ( M \\  et\\ al. )");

        let expected = Sentence {
            text: "火星猫の生態(M et al.)".to_string(),
            str_to_char_pos: vec![
                0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13,
                14, 15, 16,
            ],
            char_to_str_pos: vec![
                0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            ],
            char_type: ct2u8vec![
                Kanji, Kanji, Kanji, Hiragana, Kanji, Kanji, Other, Roman, Other, Roman, Roman,
                Other, Roman, Roman, Other, Other,
            ],
            boundaries: vec![
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                NotWordBoundary,
                NotWordBoundary,
                NotWordBoundary,
                NotWordBoundary,
                WordBoundary,
            ],
            boundary_scores: None,
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_from_tokenized_with_escape_backslash() {
        let s = Sentence::from_tokenized("改行 に \\\\n を 用い る");

        let expected = Sentence {
            text: "改行に\\nを用いる".to_string(),
            str_to_char_pos: vec![
                0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 4, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9,
            ],
            char_to_str_pos: vec![0, 3, 6, 9, 10, 11, 14, 17, 20, 23],
            char_type: ct2u8vec![
                Kanji, Kanji, Hiragana, Other, Roman, Hiragana, Kanji, Hiragana, Hiragana,
            ],
            boundaries: vec![
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
            ],
            boundary_scores: None,
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_to_tokenized_string_unknown() {
        let s = Sentence::from_partial_annotation("火-星 猫|の|生-態");
        let result = s.unwrap().to_tokenized_string();

        assert!(result.is_err());
        assert_eq!(
            "sentence contains an unknown boundary",
            result.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_to_tokenized_string() {
        let s = Sentence::from_tokenized("Rust で 良い プログラミング 体験 を ！");

        assert_eq!(
            "Rust で 良い プログラミング 体験 を ！",
            s.unwrap().to_tokenized_string().unwrap()
        );
    }

    #[test]
    fn test_sentence_to_tokenized_string_escape() {
        let s = Sentence::from_partial_annotation("火-星-猫|の| |生-態|\\-n");

        assert_eq!(
            "火星猫 の \\  生態 \\\\n",
            s.unwrap().to_tokenized_string().unwrap()
        );
    }

    #[test]
    fn test_sentence_to_tokenized_vec_unknown() {
        let s = Sentence::from_partial_annotation("火-星 猫|の|生-態").unwrap();
        let result = s.to_tokenized_vec();

        assert!(result.is_err());
        assert_eq!(
            "sentence contains an unknown boundary",
            result.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_to_tokenized_vec() {
        let s = Sentence::from_tokenized("Rust で 良い プログラミング 体験 を ！").unwrap();

        assert_eq!(
            vec!["Rust", "で", "良い", "プログラミング", "体験", "を", "！"],
            s.to_tokenized_vec().unwrap()
        );
    }

    #[test]
    fn test_sentence_from_partial_annotation_empty() {
        let s = Sentence::from_partial_annotation("");

        assert!(s.is_err());
        assert_eq!("`labeled_text` is empty", &s.err().unwrap().to_string());
    }

    #[test]
    fn test_sentence_from_partial_annotation_invalid_length() {
        let s = Sentence::from_partial_annotation("火-星 猫|の|生-態 ");

        assert!(s.is_err());
        assert_eq!(
            "invalid length for `labeled_text`: 12",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_from_partial_annotation_invalid_boundary_character() {
        let s = Sentence::from_partial_annotation("火-星?猫|の|生-態");

        assert!(s.is_err());
        assert_eq!(
            "invalid boundary character: '?'",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_from_partial_annotation_one() {
        let s = Sentence::from_partial_annotation("火-星 猫|の|生-態");

        let expected = Sentence {
            text: "火星猫の生態".to_string(),
            str_to_char_pos: vec![0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6],
            char_to_str_pos: vec![0, 3, 6, 9, 12, 15, 18],
            char_type: ct2u8vec![Kanji, Kanji, Kanji, Hiragana, Kanji, Kanji],
            boundaries: vec![
                NotWordBoundary,
                Unknown,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
            ],
            boundary_scores: None,
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_to_partial_annotation_string() {
        let s = Sentence::from_partial_annotation("火-星 猫|の|生-態");

        assert_eq!(
            "火-星 猫|の|生-態",
            s.unwrap().to_partial_annotation_string()
        );
    }
}
