use std::sync::Arc;

use crate::errors::{Result, VaporettoError};

/// Character type.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
#[repr(u8)]
pub enum CharacterType {
    /// Digit character. (e.g. 0, 1, 2, ...)
    Digit = 1,

    /// Roman character. (e.g. A, B, C, ...)
    Roman = 2,

    /// Japanese Hiragana character. (e.g. あ, い, う, ...)
    Hiragana = 3,

    /// Japanese Katakana character. (e.g. ア, イ, ウ, ...)
    Katakana = 4,

    /// Kanji (a.k.a. Hanzi or Hanja) character. (e.g. 漢, 字, ...)
    Kanji = 5,

    /// Other character.
    Other = 6,
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
    pub fn get_type(c: char) -> Self {
        match u32::from(c) {
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

/// Token information.
#[derive(Debug, PartialEq, Clone)]
pub struct Token<'a> {
    /// A surface of this token.
    pub surface: &'a str,

    /// A part-of-speech tag of this token.
    pub tag: Option<&'a str>,
}

/// Weight array with the corresponding range.
///
/// This data is placed on the end of each range.
#[derive(Debug, PartialEq, Clone)]
pub struct TagRangeScore {
    /// Weight array.
    pub weight: Vec<i32>,

    /// The relative position of the start position from the end position.
    pub start_rel_position: i16,
}

impl TagRangeScore {
    #[allow(dead_code)]
    pub fn new(start_rel_position: i16, weight: Vec<i32>) -> Self {
        Self {
            start_rel_position,
            weight,
        }
    }
}

pub type TagRangeScores = Arc<Vec<TagRangeScore>>;

#[derive(Debug, PartialEq, Clone, Default)]
pub struct TagScores {
    pub left_scores: Vec<i32>,
    pub right_scores: Vec<i32>,
    pub self_scores: Vec<Option<TagRangeScores>>,
}

impl TagScores {
    /// Clears scores.
    pub fn clear(&mut self) {
        self.left_scores.clear();
        self.right_scores.clear();
        self.self_scores.clear();
    }

    /// Initializes score arrays.
    ///
    /// # Arguments
    ///
    /// * `n_chars` - Length of characters in code points.
    /// * `n_tags` - The number of tags.
    #[allow(dead_code)]
    pub fn init(&mut self, n_chars: usize, n_tags: usize) {
        self.clear();
        self.left_scores.resize(n_chars * n_tags, 0);
        self.right_scores.resize(n_chars * n_tags, 0);
        self.self_scores.resize(n_chars, None);
    }
}

/// Sentence with boundary annotations.
#[derive(Debug, PartialEq, Clone)]
pub struct Sentence {
    pub(crate) text: String,
    pub(crate) chars: Vec<char>,
    pub(crate) str_to_char_pos: Vec<usize>,
    pub(crate) char_to_str_pos: Vec<usize>,
    pub(crate) char_type: Vec<u8>,
    pub(crate) boundaries: Vec<BoundaryType>,
    pub(crate) boundary_scores: Vec<i32>,
    pub(crate) tag_scores: TagScores,
    pub(crate) tags: Vec<Option<Arc<String>>>,
}

impl Sentence {
    fn internal_new(
        text: String,
        chars: Vec<char>,
        boundaries: Vec<BoundaryType>,
        tags: Vec<Option<Arc<String>>>,
    ) -> Self {
        let mut s = Self {
            text,
            chars,
            str_to_char_pos: vec![],
            char_to_str_pos: vec![],
            char_type: vec![],
            boundaries,
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags,
        };
        s.update_common_info();
        s
    }

    fn clear(&mut self) {
        self.text.clear();
        self.text.push(' ');
        self.chars.clear();
        self.chars.push(' ');
        self.str_to_char_pos.clear();
        self.str_to_char_pos.push(0);
        self.str_to_char_pos.push(1);
        self.char_to_str_pos.clear();
        self.char_to_str_pos.push(0);
        self.char_to_str_pos.push(1);
        self.char_type.clear();
        self.char_type.push(CharacterType::Other as u8);
        self.boundaries.clear();
        self.boundary_scores.clear();
        self.tag_scores.clear();
        self.tags.clear();
        self.tags.push(None);
    }

    fn parse_raw_text(
        raw_text: &str,
        chars: &mut Vec<char>,
        boundaries: &mut Vec<BoundaryType>,
        tags: &mut Vec<Option<Arc<String>>>,
    ) -> Result<()> {
        if raw_text.is_empty() {
            return Err(VaporettoError::invalid_argument(
                "raw_text",
                "must contain at least one character",
            ));
        }

        chars.clear();

        for c in raw_text.chars() {
            if c == '\0' {
                return Err(VaporettoError::invalid_argument(
                    "raw_text",
                    "must not contain NULL",
                ));
            }
            chars.push(c);
        }
        boundaries.clear();
        boundaries.resize(chars.len() - 1, BoundaryType::Unknown);
        tags.clear();
        tags.resize(chars.len(), None);

        Ok(())
    }

    fn parse_tokenized_text(
        tokenized_text: &str,
        text: &mut String,
        chars: &mut Vec<char>,
        boundaries: &mut Vec<BoundaryType>,
        tags: &mut Vec<Option<Arc<String>>>,
    ) -> Result<()> {
        if tokenized_text.is_empty() {
            return Err(VaporettoError::invalid_argument(
                "tokenized_text",
                "must contain at least one character",
            ));
        }

        text.clear();
        text.reserve(tokenized_text.len());
        chars.clear();
        boundaries.clear();
        tags.clear();

        let mut tag_str_tmp = None;
        let mut tag_str = None;
        let mut prev_boundary = false;
        let mut escape = false;
        for c in tokenized_text.chars() {
            match (escape, c) {
                // escape a following character
                (false, '\\') => {
                    escape = true;
                }
                // token boundary
                (false, ' ') => {
                    if chars.is_empty() {
                        return Err(VaporettoError::invalid_argument(
                            "tokenized_text",
                            "must not start with a whitespace",
                        ));
                    }
                    if prev_boundary {
                        return Err(VaporettoError::invalid_argument(
                            "tokenized_text",
                            "must not contain consecutive whitespaces",
                        ));
                    }
                    prev_boundary = true;
                    tag_str = tag_str_tmp.take();
                }
                // POS tag
                (false, '/') => {
                    if chars.is_empty() {
                        return Err(VaporettoError::invalid_argument(
                            "tokenized_text",
                            "must not start with a slash",
                        ));
                    }
                    if prev_boundary {
                        return Err(VaporettoError::invalid_argument(
                            "tokenized_text",
                            "a slash must follow a character",
                        ));
                    }
                    tag_str_tmp.replace("".to_string());
                }
                // escaped character or other character
                (_, _) => {
                    if let Some(tag) = tag_str_tmp.as_mut() {
                        tag.push(c);
                        continue;
                    }
                    if !chars.is_empty() {
                        boundaries.push(if prev_boundary {
                            BoundaryType::WordBoundary
                        } else {
                            BoundaryType::NotWordBoundary
                        });
                        tags.push(tag_str.take().map(Arc::new));
                    }
                    if c == '\0' {
                        return Err(VaporettoError::invalid_argument(
                            "tokenized_text",
                            "must not contain NULL",
                        ));
                    }
                    prev_boundary = false;
                    escape = false;
                    text.push(c);
                    chars.push(c);
                }
            };
        }

        if prev_boundary {
            return Err(VaporettoError::invalid_argument(
                "tokenized_text",
                "must not end with a whitespace",
            ));
        }
        tags.push(tag_str_tmp.take().map(Arc::new));

        Ok(())
    }

    fn parse_partial_annotation(
        labeled_text: &str,
        text: &mut String,
        chars: &mut Vec<char>,
        boundaries: &mut Vec<BoundaryType>,
        tags: &mut Vec<Option<Arc<String>>>,
    ) -> Result<()> {
        if labeled_text.is_empty() {
            return Err(VaporettoError::invalid_argument(
                "labeled_text",
                "must contain at least one character",
            ));
        }

        text.clear();
        chars.clear();
        boundaries.clear();
        tags.clear();

        let mut tag_str = None;
        let mut is_char = true;
        let mut fixed_token = true;
        for c in labeled_text.chars() {
            if is_char {
                if c == '\0' {
                    return Err(VaporettoError::invalid_argument(
                        "labeled_text",
                        "must not contain NULL",
                    ));
                }
                text.push(c);
                chars.push(c);
                is_char = false;
                continue;
            }
            match c {
                // unannotated boundary
                ' ' => {
                    if tag_str.is_some() {
                        return Err(VaporettoError::invalid_argument(
                            "labeled_text",
                            "POS tag must be annotated to a token".to_string(),
                        ));
                    }
                    tags.push(None);
                    boundaries.push(BoundaryType::Unknown);
                    is_char = true;
                    fixed_token = false;
                }
                // token boundary
                '|' => {
                    if !fixed_token && tag_str.is_some() {
                        return Err(VaporettoError::invalid_argument(
                            "labeled_text",
                            "POS tag must be annotated to a token".to_string(),
                        ));
                    }
                    tags.push(tag_str.take().map(Arc::new));
                    boundaries.push(BoundaryType::WordBoundary);
                    is_char = true;
                    fixed_token = true;
                }
                // not token boundary
                '-' => {
                    if tag_str.is_some() {
                        return Err(VaporettoError::invalid_argument(
                            "labeled_text",
                            "POS tag must be annotated to a token".to_string(),
                        ));
                    }
                    tags.push(None);
                    boundaries.push(BoundaryType::NotWordBoundary);
                    is_char = true;
                }
                // POS tag
                '/' => {
                    tag_str.replace("".to_string());
                }
                _ => {
                    if let Some(tag) = tag_str.as_mut() {
                        tag.push(c);
                    } else {
                        return Err(VaporettoError::invalid_argument(
                            "labeled_text",
                            format!("contains an invalid boundary character: '{}'", c),
                        ));
                    }
                }
            }
        }
        tags.push(tag_str.take().map(Arc::new));
        if chars.len() != boundaries.len() + 1 {
            return Err(VaporettoError::invalid_argument(
                "labeled_text",
                "invalid annotation".to_string(),
            ));
        }

        Ok(())
    }

    /// Updates char_to_str_pos, str_to_char_pos, and char_type.
    ///
    /// This function allocates:
    ///
    /// * char_to_str_pos: chars.len() + 1
    /// * str_to_char_pos: text.len() + 1
    /// * char_type: chars.len()
    ///
    /// If these variables already have sufficient spaces, this function reuses them.
    fn update_common_info(&mut self) {
        self.char_to_str_pos.clear();
        self.str_to_char_pos.clear();
        self.char_type.clear();
        self.boundary_scores.clear();
        self.tag_scores.clear();

        let mut pos = 0;
        self.char_to_str_pos.push(0);
        for &c in &self.chars {
            pos += c.len_utf8();
            self.char_to_str_pos.push(pos);
            self.char_type.push(CharacterType::get_type(c) as u8)
        }

        debug_assert!(pos == self.text.len());

        self.str_to_char_pos.resize(self.text.len() + 1, 0);
        for (i, &j) in self.char_to_str_pos.iter().enumerate() {
            // j is always lower than pos + 1, so the following is safe.
            unsafe {
                *self.str_to_char_pos.get_unchecked_mut(j) = i;
            }
        }
    }

    /// Creates a new [`Sentence`] from a given string.
    ///
    /// # Arguments
    ///
    /// * `raw_text` - A raw string without any annotation.
    ///
    /// # Returns
    ///
    /// A new [`Sentence`].
    ///
    /// # Errors
    ///
    /// If the given `raw_text` is empty, an error variant will be returned.
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
    pub fn from_raw<S>(raw_text: S) -> Result<Self>
    where
        S: Into<String>,
    {
        let raw_text = raw_text.into();

        let mut chars = Vec::with_capacity(0);
        let mut boundaries = Vec::with_capacity(0);
        let mut tags = Vec::with_capacity(0);
        Self::parse_raw_text(&raw_text, &mut chars, &mut boundaries, &mut tags)?;

        Ok(Self::internal_new(raw_text, chars, boundaries, tags))
    }

    /// Updates the [`Sentence`] using a given string.
    ///
    /// # Arguments
    ///
    /// * `raw_text` - A raw string without any annotation.
    ///
    /// # Errors
    ///
    /// If the given `raw_text` is empty, an error variant will be returned.
    /// When an error is occurred, the sentence will be replaced with a white space.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut s = Sentence::from_raw("How are you?").unwrap();
    /// s.update_raw("I am file.").unwrap();
    /// assert_eq!("I am file.", s.to_raw_string());
    /// ```
    pub fn update_raw<S>(&mut self, raw_text: S) -> Result<()>
    where
        S: Into<String>,
    {
        let raw_text = raw_text.into();

        match Self::parse_raw_text(
            &raw_text,
            &mut self.chars,
            &mut self.boundaries,
            &mut self.tags,
        ) {
            Ok(_) => {
                self.text = raw_text;
                self.update_common_info();
                Ok(())
            }
            Err(e) => {
                self.clear();
                Err(e)
            }
        }
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
    /// * `tokenized_text` - A tokenized text that is annotated by the following rules:
    ///   - A whitespace (`' '`) is inserted to each token boundary.
    ///   - If necessary, a POS tag following a slash (`'/'`) can be added to each token.
    ///   - Each character following a back slash (`'\\'`) is escaped.
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
    /// let s = Sentence::from_tokenized("How/WRB are/VBP you?");
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

        let mut text = String::with_capacity(0);
        let mut chars = Vec::with_capacity(0);
        let mut boundaries = Vec::with_capacity(0);
        let mut tags = Vec::with_capacity(0);

        Self::parse_tokenized_text(
            tokenized_text,
            &mut text,
            &mut chars,
            &mut boundaries,
            &mut tags,
        )?;

        Ok(Self::internal_new(text, chars, boundaries, tags))
    }

    /// Updates the [`Sentence`] using tokenized string.
    ///
    /// # Arguments
    ///
    /// * `tokenized_text` - A tokenized text that is annotated by the following rules:
    ///   - A whitespace (`' '`) is inserted to each token boundary.
    ///   - If necessary, a POS tag following a slash (`'/'`) can be added to each token.
    ///   - Each character following a back slash (`'\\'`) is escaped.
    ///
    /// # Errors
    ///
    /// This function will return an error variant when:
    ///
    /// * `tokenized_text` is empty.
    /// * `tokenized_text` starts/ends with a whitespace.
    /// * `tokenized_text` contains consecutive whitespaces.
    ///
    /// When an error is occurred, the sentence will be replaced with a white space.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut s = Sentence::from_tokenized("How are you?").unwrap();
    ///
    /// s.update_tokenized("I am fine").unwrap();
    /// assert_eq!("Iamfine", s.to_raw_string());
    ///
    /// s.update_tokenized("How/WRB are/VBP you ?/.").unwrap();
    /// assert_eq!("Howareyou?", s.to_raw_string());
    /// ```
    pub fn update_tokenized<S>(&mut self, tokenized_text: S) -> Result<()>
    where
        S: AsRef<str>,
    {
        let tokenized_text = tokenized_text.as_ref();

        match Self::parse_tokenized_text(
            tokenized_text,
            &mut self.text,
            &mut self.chars,
            &mut self.boundaries,
            &mut self.tags,
        ) {
            Ok(_) => {
                self.update_common_info();
                Ok(())
            }
            Err(e) => {
                self.clear();
                Err(e)
            }
        }
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
    ///
    /// let s = Sentence::from_tokenized("How/WRB are/VBP you?").unwrap();
    /// assert_eq!("How/WRB are/VBP you?", s.to_tokenized_string().unwrap());
    /// ```
    pub fn to_tokenized_string(&self) -> Result<String> {
        let mut result = String::with_capacity(self.text.len() * 2 - 1);
        self.write_tokenized_string(&mut result)?;
        Ok(result)
    }

    /// Writes a string with whitespaces for word boundaries.
    ///
    /// # Arguments
    ///
    /// * `buffer` - A string buffer.
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
    /// let mut buf = String::new();
    ///
    /// let s = Sentence::from_tokenized("How are you?").unwrap();
    /// s.write_tokenized_string(&mut buf).unwrap();
    /// assert_eq!("How are you?", buf);
    ///
    /// let s = Sentence::from_tokenized("How/WRB are/VBP you?").unwrap();
    /// s.write_tokenized_string(&mut buf).unwrap();
    /// assert_eq!("How/WRB are/VBP you?", buf);
    /// ```
    pub fn write_tokenized_string(&self, buffer: &mut String) -> Result<()> {
        let mut chars_iter = self.text.chars();
        buffer.clear();
        let c = chars_iter.next().unwrap();
        match c {
            '\\' | '/' | '&' | ' ' => buffer.push('\\'),
            _ => (),
        }
        buffer.push(c);
        for (i, (c, b)) in chars_iter.zip(&self.boundaries).enumerate() {
            match b {
                BoundaryType::WordBoundary => {
                    if !self.tags.is_empty() {
                        if let Some(tag) = self.tags.get(i).and_then(|x| x.as_ref()) {
                            buffer.push('/');
                            buffer.push_str(tag);
                        }
                    }
                    buffer.push(' ');
                }
                BoundaryType::NotWordBoundary => (),
                BoundaryType::Unknown => {
                    return Err(VaporettoError::invalid_sentence(
                        "contains an unknown boundary",
                    ));
                }
            }
            match c {
                '\\' | '/' | '&' | ' ' => buffer.push('\\'),
                _ => (),
            }
            buffer.push(c);
        }
        if let Some(tag) = self.tags.last().and_then(|x| x.as_ref()) {
            buffer.push('/');
            buffer.push_str(tag);
        }
        Ok(())
    }

    /// Generates a vector of tokens.
    ///
    /// # Returns
    ///
    /// A newly allocated vector of tokens.
    ///
    /// # Errors
    ///
    /// If the sentence contains unknown boundaries, an error variant will be returned.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::{Sentence, Token};
    ///
    /// let s = Sentence::from_tokenized("How are you ?").unwrap();
    /// assert_eq!(vec![
    ///     Token { surface: "How", tag: None },
    ///     Token { surface: "are", tag: None },
    ///     Token { surface: "you", tag: None },
    ///     Token { surface: "?", tag: None },
    /// ], s.to_tokenized_vec().unwrap());
    ///
    /// let s = Sentence::from_tokenized("How/WRB are/VBP you/PRP ?/.").unwrap();
    /// assert_eq!(vec![
    ///     Token { surface: "How", tag: Some("WRB") },
    ///     Token { surface: "are", tag: Some("VBP") },
    ///     Token { surface: "you", tag: Some("PRP") },
    ///     Token { surface: "?", tag: Some(".") },
    /// ], s.to_tokenized_vec().unwrap());
    /// ```
    pub fn to_tokenized_vec(&self) -> Result<Vec<Token>> {
        let mut result = vec![];
        let mut start = 0;
        if self.tags.is_empty() {
            for (i, b) in self.boundaries.iter().enumerate() {
                match b {
                    BoundaryType::WordBoundary => {
                        let end = unsafe { *self.char_to_str_pos.get_unchecked(i + 1) };
                        let surface = unsafe { self.text.get_unchecked(start..end) };
                        result.push(Token { surface, tag: None });
                        start = end;
                    }
                    BoundaryType::NotWordBoundary => (),
                    BoundaryType::Unknown => {
                        return Err(VaporettoError::invalid_sentence(
                            "contains an unknown boundary",
                        ));
                    }
                }
            }
            let surface = unsafe { self.text.get_unchecked(start..) };
            result.push(Token { surface, tag: None });
        } else {
            for (i, (b, tag)) in self.boundaries.iter().zip(&self.tags).enumerate() {
                match b {
                    BoundaryType::WordBoundary => {
                        let end = unsafe { *self.char_to_str_pos.get_unchecked(i + 1) };
                        let surface = unsafe { self.text.get_unchecked(start..end) };
                        let tag = tag.as_ref().map(|x| x.as_str());
                        result.push(Token { surface, tag });
                        start = end;
                    }
                    BoundaryType::NotWordBoundary => (),
                    BoundaryType::Unknown => {
                        return Err(VaporettoError::invalid_sentence(
                            "contains an unknown boundary",
                        ));
                    }
                }
            }
            let surface = unsafe { self.text.get_unchecked(start..) };
            let tag = self
                .tags
                .last()
                .and_then(|x| x.as_ref())
                .map(|x| x.as_str());
            result.push(Token { surface, tag });
        }
        Ok(result)
    }

    /// Creates a new [`Sentence`] from a string with partial annotations.
    ///
    /// # Arguments
    ///
    /// * `labeled_text` - A partially annotated text. Each character boundary is annotated by the following rules:
    ///   - If the boundary is a token boundary, a pipe symbol (`'|'`) is inserted.
    ///   - If the boundary is not a token boundary, a dash symobl (`'-'`) is inserted.
    ///   - If the boundary is not annotated, a whitespace (`' '`) is inserted.
    ///
    ///   In addition, a POS tag following a slash (`'/'`) can be inserted to each token.
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
    /// let s = Sentence::from_partial_annotation("I-t/PRP|'-s/VBZ|o-k-a-y/JJ|./.");
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

        let mut text = String::with_capacity(0);
        let mut chars = Vec::with_capacity(0);
        let mut boundaries = Vec::with_capacity(0);
        let mut tags = Vec::with_capacity(0);
        Self::parse_partial_annotation(
            labeled_text,
            &mut text,
            &mut chars,
            &mut boundaries,
            &mut tags,
        )?;

        Ok(Self::internal_new(text, chars, boundaries, tags))
    }

    /// Updates the [`Sentence`] using a string with partial annotations.
    ///
    /// # Arguments
    ///
    /// * `labeled_text` - A partially annotated text. Each character boundary is annotated by the following rules:
    ///   - If the boundary is a token boundary, a pipe symbol (`'|'`) is inserted.
    ///   - If the boundary is not a token boundary, a dash symobl (`'-'`) is inserted.
    ///   - If the boundary is not annotated, a whitespace (`' '`) is inserted.
    ///
    ///   In addition, a POS tag following a slash (`'/'`) can be inserted to each token.
    ///
    /// # Errors
    ///
    /// This function will return an error variant when:
    ///
    /// * `labeled_text` is empty.
    /// * The length of `lsbeled_text` is even numbers.
    /// * `labeled_text` contains invalid boundary characters.
    ///
    /// When an error is occurred, the sentence will be replaced with a white space.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut s = Sentence::from_raw("g-o-o-d|i-d e-a").unwrap();
    /// s.update_partial_annotation("h-e-l-l-o").unwrap();
    /// assert_eq!("hello", s.to_raw_string());
    ///
    /// s.update_partial_annotation("I-t/PRP|'-s/VBZ|o-k-a-y/JJ|./.").unwrap();
    /// assert_eq!("It'sokay.", s.to_raw_string());
    /// ```
    pub fn update_partial_annotation<S>(&mut self, labeled_text: S) -> Result<()>
    where
        S: AsRef<str>,
    {
        let labeled_text = labeled_text.as_ref();

        match Self::parse_partial_annotation(
            labeled_text,
            &mut self.text,
            &mut self.chars,
            &mut self.boundaries,
            &mut self.tags,
        ) {
            Ok(_) => {
                self.update_common_info();
                Ok(())
            }
            Err(e) => {
                self.clear();
                Err(e)
            }
        }
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
    ///
    /// let s = Sentence::from_tokenized("How/WRB are you/PRP ?").unwrap();
    /// assert_eq!("H-o-w/WRB|a-r-e|y-o-u/PRP|?", &s.to_partial_annotation_string());
    /// ```
    pub fn to_partial_annotation_string(&self) -> String {
        let mut result = String::with_capacity(self.text.len() * 2 - 1);
        self.write_partial_annotation_string(&mut result);
        result
    }

    /// Write a string with partial annotations.
    ///
    /// # Arguments
    ///
    /// * `buffer` - A string buffer.
    ///
    /// A newly allocated string with partial annotations.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut buf = String::new();
    ///
    /// let s = Sentence::from_tokenized("How are you ?").unwrap();
    /// s.write_partial_annotation_string(&mut buf);
    /// assert_eq!("H-o-w|a-r-e|y-o-u|?", buf);
    ///
    /// let s = Sentence::from_tokenized("How/WRB are you/PRP ?").unwrap();
    /// s.write_partial_annotation_string(&mut buf);
    /// assert_eq!("H-o-w/WRB|a-r-e|y-o-u/PRP|?", buf);
    /// ```
    pub fn write_partial_annotation_string(&self, buffer: &mut String) {
        let mut chars_iter = self.text.chars();
        buffer.clear();
        buffer.push(chars_iter.next().unwrap());
        for (i, (c, b)) in chars_iter.zip(&self.boundaries).enumerate() {
            match b {
                BoundaryType::WordBoundary => {
                    if let Some(tag) = self.tags.get(i).and_then(|x| x.as_ref()) {
                        buffer.push('/');
                        buffer.push_str(tag);
                    }
                    buffer.push('|');
                }
                BoundaryType::NotWordBoundary => {
                    buffer.push('-');
                }
                BoundaryType::Unknown => {
                    buffer.push(' ');
                }
            }
            buffer.push(c);
        }
        if let Some(tag) = self.tags.last().and_then(|x| x.as_ref()) {
            buffer.push('/');
            buffer.push_str(tag);
        }
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

    /// Gets a reference to the part-of-speech information.
    ///
    /// Each tag is placed at the last of the corresponding token. For example, when the first token
    /// containing three characters has a tag, that tag will be placed at the third element of the
    /// returned slice.
    ///
    /// # Returns
    ///
    /// A reference to the POS information.
    ///
    /// # Examples
    ///
    /// ```
    /// use std::sync::Arc;
    ///
    /// use vaporetto::{BoundaryType, Sentence};
    ///
    /// let s = Sentence::from_tokenized("I/PRP am a/DT cat/NN ./.").unwrap();
    /// assert_eq!(&[
    ///     Some(Arc::new("PRP".to_string())), // 'I'
    ///     None,                             // 'a'
    ///     None,                             // 'm'
    ///     Some(Arc::new("DT".to_string())),  // 'a'
    ///     None,                             // 'c'
    ///     None,                             // 'a'
    ///     Some(Arc::new("NN".to_string())),  // 't'
    ///     Some(Arc::new(".".to_string())),   // '.'
    /// ], s.tags());
    /// ```
    pub fn tags(&self) -> &[Option<Arc<String>>] {
        &self.tags
    }

    /// Gets a mutable reference to the part-of-speech information.
    ///
    /// # Returns
    ///
    /// A mutable reference to the part-of-speech information.
    pub fn tags_mut(&mut self) -> &mut [Option<Arc<String>>] {
        &mut self.tags
    }

    /// Gets a reference to the characters.
    ///
    /// # Returns
    ///
    /// A reference to the characters.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_raw("A1あエ漢?").unwrap();
    /// assert_eq!(&['A', '1', 'あ', 'エ', '漢', '?'], s.chars());
    /// ```
    pub fn chars(&self) -> &[char] {
        &self.chars
    }

    /// Gets immutable references to the characters and character types, and a mutable reference to
    /// boundaries.
    ///
    /// # Returns
    ///
    /// A tuple of references.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::{BoundaryType, CharacterType, Sentence};
    ///
    /// let mut s = Sentence::from_partial_annotation("A-1|あ エ-漢|?").unwrap();
    /// let (chars, char_types, boundaries) = s.chars_and_boundaries_mut();
    /// assert_eq!(&['A', '1', 'あ', 'エ', '漢', '?'], chars);
    /// assert_eq!(&[
    ///     CharacterType::Roman as u8,
    ///     CharacterType::Digit as u8,
    ///     CharacterType::Hiragana as u8,
    ///     CharacterType::Katakana as u8,
    ///     CharacterType::Kanji as u8,
    ///     CharacterType::Other as u8,
    /// ], char_types);
    /// assert_eq!(&[
    ///     BoundaryType::NotWordBoundary,
    ///     BoundaryType::WordBoundary,
    ///     BoundaryType::Unknown,
    ///     BoundaryType::NotWordBoundary,
    ///     BoundaryType::WordBoundary,
    /// ], boundaries);
    /// ```
    pub fn chars_and_boundaries_mut(&mut self) -> (&[char], &[u8], &mut [BoundaryType]) {
        (&self.chars, &self.char_type, &mut self.boundaries)
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
    /// use vaporetto::{CharacterType, Sentence};
    ///
    /// let s = Sentence::from_raw("A1あエ漢?").unwrap();
    /// assert_eq!(&[
    ///     CharacterType::Roman as u8,
    ///     CharacterType::Digit as u8,
    ///     CharacterType::Hiragana as u8,
    ///     CharacterType::Katakana as u8,
    ///     CharacterType::Kanji as u8,
    ///     CharacterType::Other as u8,
    /// ], s.char_types());
    /// ```
    pub fn char_types(&self) -> &[u8] {
        &self.char_type
    }

    /// Gets a reference to the boundary score information.
    ///
    /// # Returns
    ///
    /// If the predictor inserted, the boundary score information is returned. Otherwise, None.
    pub fn boundary_scores(&self) -> &[i32] {
        &self.boundary_scores
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
                _ => Err(VaporettoError::invalid_argument("index", "invalid index")),
            }
        }
    }

    #[cfg(feature = "train")]
    pub(crate) fn char_substring(&self, start: usize, end: usize) -> &str {
        let begin = self.char_to_str_pos[start];
        let end = self.char_to_str_pos[end];
        &self.text.as_str()[begin..end]
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

        assert_eq!(
            "InvalidArgumentError: raw_text: must contain at least one character",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_raw_empty() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_raw("");

        assert_eq!(
            "InvalidArgumentError: raw_text: must contain at least one character",
            &result.err().unwrap().to_string()
        );

        let expected = Sentence {
            text: " ".to_string(),
            chars: vec![' '],
            str_to_char_pos: vec![0, 1],
            char_to_str_pos: vec![0, 1],
            char_type: vec![Other as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_raw_null() {
        let s = Sentence::from_raw("A1あ\0ア亜");

        assert_eq!(
            "InvalidArgumentError: raw_text: must not contain NULL",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_raw_null() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_raw("A1あ\0ア亜");

        assert_eq!(
            "InvalidArgumentError: raw_text: must not contain NULL",
            &result.err().unwrap().to_string()
        );

        let expected = Sentence {
            text: " ".to_string(),
            chars: vec![' '],
            str_to_char_pos: vec![0, 1],
            char_to_str_pos: vec![0, 1],
            char_type: vec![Other as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_raw_one() {
        let s = Sentence::from_raw("あ");

        let expected = Sentence {
            text: "あ".to_string(),
            chars: vec!['あ'],
            str_to_char_pos: vec![0, 0, 0, 1],
            char_to_str_pos: vec![0, 3],
            char_type: vec![Hiragana as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_update_raw_one() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_raw("あ").unwrap();

        let expected = Sentence {
            text: "あ".to_string(),
            chars: vec!['あ'],
            str_to_char_pos: vec![0, 0, 0, 1],
            char_to_str_pos: vec![0, 3],
            char_type: vec![Hiragana as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_raw() {
        let s = Sentence::from_raw("Rustで良いプログラミング体験を！");

        let expected = Sentence {
            text: "Rustで良いプログラミング体験を！".to_string(),
            chars: vec![
                'R', 'u', 's', 't', 'で', '良', 'い', 'プ', 'ロ', 'グ', 'ラ', 'ミ', 'ン', 'グ',
                '体', '験', 'を', '！',
            ],
            str_to_char_pos: vec![
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            char_to_str_pos: vec![
                0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
            ],
            char_type: vec![
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
            ],
            boundaries: vec![Unknown; 17],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 18],
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_update_raw() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_raw("Rustで良いプログラミング体験を！").unwrap();

        let expected = Sentence {
            text: "Rustで良いプログラミング体験を！".to_string(),
            chars: vec![
                'R', 'u', 's', 't', 'で', '良', 'い', 'プ', 'ロ', 'グ', 'ラ', 'ミ', 'ン', 'グ',
                '体', '験', 'を', '！',
            ],
            str_to_char_pos: vec![
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            char_to_str_pos: vec![
                0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
            ],
            char_type: vec![
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
            ],
            boundaries: vec![Unknown; 17],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 18],
        };
        assert_eq!(expected, s);
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

        assert_eq!(
            "InvalidArgumentError: tokenized_text: must contain at least one character",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_tokenized_empty() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_tokenized("");

        assert_eq!(
            "InvalidArgumentError: tokenized_text: must contain at least one character",
            &result.err().unwrap().to_string()
        );

        let expected = Sentence {
            text: " ".to_string(),
            chars: vec![' '],
            str_to_char_pos: vec![0, 1],
            char_to_str_pos: vec![0, 1],
            char_type: vec![Other as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_tokenized_null() {
        let s = Sentence::from_tokenized("A1あ\0ア亜");

        assert_eq!(
            "InvalidArgumentError: tokenized_text: must not contain NULL",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_tokenized_null() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_tokenized("A1あ\0ア亜");

        assert_eq!(
            "InvalidArgumentError: tokenized_text: must not contain NULL",
            &result.err().unwrap().to_string()
        );

        let expected = Sentence {
            text: " ".to_string(),
            chars: vec![' '],
            str_to_char_pos: vec![0, 1],
            char_to_str_pos: vec![0, 1],
            char_type: vec![Other as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_tokenized_start_with_space() {
        let s = Sentence::from_tokenized(" Rust で 良い プログラミング 体験 を ！");

        assert_eq!(
            "InvalidArgumentError: tokenized_text: must not start with a whitespace",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_tokenized_start_with_space() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_tokenized(" Rust で 良い プログラミング 体験 を ！");

        assert_eq!(
            "InvalidArgumentError: tokenized_text: must not start with a whitespace",
            &result.err().unwrap().to_string()
        );

        let expected = Sentence {
            text: " ".to_string(),
            chars: vec![' '],
            str_to_char_pos: vec![0, 1],
            char_to_str_pos: vec![0, 1],
            char_type: vec![Other as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_tokenized_end_with_space() {
        let s = Sentence::from_tokenized("Rust で 良い プログラミング 体験 を ！ ");

        assert_eq!(
            "InvalidArgumentError: tokenized_text: must not end with a whitespace",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_tokenized_end_with_space() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_tokenized("Rust で 良い プログラミング 体験 を ！ ");

        assert_eq!(
            "InvalidArgumentError: tokenized_text: must not end with a whitespace",
            &result.err().unwrap().to_string()
        );

        let expected = Sentence {
            text: " ".to_string(),
            chars: vec![' '],
            str_to_char_pos: vec![0, 1],
            char_to_str_pos: vec![0, 1],
            char_type: vec![Other as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_tokenized_two_spaces() {
        let s = Sentence::from_tokenized("Rust で 良い  プログラミング 体験 を ！");

        assert_eq!(
            "InvalidArgumentError: tokenized_text: must not contain consecutive whitespaces",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_tokenized_two_spaces() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_tokenized("Rust で 良い  プログラミング 体験 を ！");

        assert_eq!(
            "InvalidArgumentError: tokenized_text: must not contain consecutive whitespaces",
            &result.err().unwrap().to_string()
        );

        let expected = Sentence {
            text: " ".to_string(),
            chars: vec![' '],
            str_to_char_pos: vec![0, 1],
            char_to_str_pos: vec![0, 1],
            char_type: vec![Other as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_tokenized_one() {
        let s = Sentence::from_tokenized("あ");

        let expected = Sentence {
            text: "あ".to_string(),
            chars: vec!['あ'],
            str_to_char_pos: vec![0, 0, 0, 1],
            char_to_str_pos: vec![0, 3],
            char_type: vec![Hiragana as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_update_tokenized_one() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("あ").unwrap();

        let expected = Sentence {
            text: "あ".to_string(),
            chars: vec!['あ'],
            str_to_char_pos: vec![0, 0, 0, 1],
            char_to_str_pos: vec![0, 3],
            char_type: vec![Hiragana as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_tokenized() {
        let s = Sentence::from_tokenized("Rust で 良い プログラミング 体験 を ！");

        let expected = Sentence {
            text: "Rustで良いプログラミング体験を！".to_string(),
            chars: vec![
                'R', 'u', 's', 't', 'で', '良', 'い', 'プ', 'ロ', 'グ', 'ラ', 'ミ', 'ン', 'グ',
                '体', '験', 'を', '！',
            ],
            str_to_char_pos: vec![
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            char_to_str_pos: vec![
                0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
            ],
            char_type: vec![
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
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
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 18],
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_from_tokenized_with_tags() {
        let s =
            Sentence::from_tokenized("Rust/名詞 で 良い/形容詞 プログラミング 体験 を ！/補助記号");

        let expected = Sentence {
            text: "Rustで良いプログラミング体験を！".to_string(),
            chars: vec![
                'R', 'u', 's', 't', 'で', '良', 'い', 'プ', 'ロ', 'グ', 'ラ', 'ミ', 'ン', 'グ',
                '体', '験', 'を', '！',
            ],
            str_to_char_pos: vec![
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            char_to_str_pos: vec![
                0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
            ],
            char_type: vec![
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
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
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![
                None,
                None,
                None,
                Some(Arc::new("名詞".to_string())),
                None,
                None,
                Some(Arc::new("形容詞".to_string())),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(Arc::new("補助記号".to_string())),
            ],
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_update_tokenized() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("Rust で 良い プログラミング 体験 を ！")
            .unwrap();

        let expected = Sentence {
            text: "Rustで良いプログラミング体験を！".to_string(),
            chars: vec![
                'R', 'u', 's', 't', 'で', '良', 'い', 'プ', 'ロ', 'グ', 'ラ', 'ミ', 'ン', 'グ',
                '体', '験', 'を', '！',
            ],
            str_to_char_pos: vec![
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            char_to_str_pos: vec![
                0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
            ],
            char_type: vec![
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
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
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 18],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_update_tokenized_with_tags() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("Rust/名詞 で 良い/形容詞 プログラミング 体験 を ！/補助記号")
            .unwrap();

        let expected = Sentence {
            text: "Rustで良いプログラミング体験を！".to_string(),
            chars: vec![
                'R', 'u', 's', 't', 'で', '良', 'い', 'プ', 'ロ', 'グ', 'ラ', 'ミ', 'ン', 'グ',
                '体', '験', 'を', '！',
            ],
            str_to_char_pos: vec![
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            char_to_str_pos: vec![
                0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,
            ],
            char_type: vec![
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Roman as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Katakana as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
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
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![
                None,
                None,
                None,
                Some(Arc::new("名詞".to_string())),
                None,
                None,
                Some(Arc::new("形容詞".to_string())),
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(Arc::new("補助記号".to_string())),
            ],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_tokenized_with_escape_whitespace() {
        let s = Sentence::from_tokenized("火星 猫 の 生態 ( M \\  et\\ al. )").unwrap();

        let expected = Sentence {
            text: "火星猫の生態(M et al.)".to_string(),
            chars: vec![
                '火', '星', '猫', 'の', '生', '態', '(', 'M', ' ', 'e', 't', ' ', 'a', 'l', '.',
                ')',
            ],
            str_to_char_pos: vec![
                0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13,
                14, 15, 16,
            ],
            char_to_str_pos: vec![
                0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            ],
            char_type: vec![
                Kanji as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Kanji as u8,
                Kanji as u8,
                Other as u8,
                Roman as u8,
                Other as u8,
                Roman as u8,
                Roman as u8,
                Other as u8,
                Roman as u8,
                Roman as u8,
                Other as u8,
                Other as u8,
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
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 16],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_update_tokenized_escape_whitespace() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("火星 猫 の 生態 ( M \\  et\\ al. )")
            .unwrap();

        let expected = Sentence {
            text: "火星猫の生態(M et al.)".to_string(),
            chars: vec![
                '火', '星', '猫', 'の', '生', '態', '(', 'M', ' ', 'e', 't', ' ', 'a', 'l', '.',
                ')',
            ],
            str_to_char_pos: vec![
                0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13,
                14, 15, 16,
            ],
            char_to_str_pos: vec![
                0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
            ],
            char_type: vec![
                Kanji as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Kanji as u8,
                Kanji as u8,
                Other as u8,
                Roman as u8,
                Other as u8,
                Roman as u8,
                Roman as u8,
                Other as u8,
                Roman as u8,
                Roman as u8,
                Other as u8,
                Other as u8,
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
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 16],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_tokenized_with_escape_backslash() {
        let s = Sentence::from_tokenized("改行 に \\\\n を 用い る");

        let expected = Sentence {
            text: "改行に\\nを用いる".to_string(),
            chars: vec!['改', '行', 'に', '\\', 'n', 'を', '用', 'い', 'る'],
            str_to_char_pos: vec![
                0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 4, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9,
            ],
            char_to_str_pos: vec![0, 3, 6, 9, 10, 11, 14, 17, 20, 23],
            char_type: vec![
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
                Roman as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Hiragana as u8,
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
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 9],
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_update_tokenized_with_escape_backslash() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("改行 に \\\\n を 用い る").unwrap();

        let expected = Sentence {
            text: "改行に\\nを用いる".to_string(),
            chars: vec!['改', '行', 'に', '\\', 'n', 'を', '用', 'い', 'る'],
            str_to_char_pos: vec![
                0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 4, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9,
            ],
            char_to_str_pos: vec![0, 3, 6, 9, 10, 11, 14, 17, 20, 23],
            char_type: vec![
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
                Roman as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Hiragana as u8,
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
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 9],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_tokenized_escape_slash() {
        let s = Sentence::from_tokenized("品詞 に \\/ を 用い る");

        let expected = Sentence {
            text: "品詞に/を用いる".to_string(),
            chars: vec!['品', '詞', 'に', '/', 'を', '用', 'い', 'る'],
            str_to_char_pos: vec![
                0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8,
            ],
            char_to_str_pos: vec![0, 3, 6, 9, 10, 13, 16, 19, 22],
            char_type: vec![
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Hiragana as u8,
            ],
            boundaries: vec![
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
            ],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 8],
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_update_tokenized_escape_slash() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("品詞 に \\/ を 用い る").unwrap();

        let expected = Sentence {
            text: "品詞に/を用いる".to_string(),
            chars: vec!['品', '詞', 'に', '/', 'を', '用', 'い', 'る'],
            str_to_char_pos: vec![
                0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8,
            ],
            char_to_str_pos: vec![0, 3, 6, 9, 10, 13, 16, 19, 22],
            char_type: vec![
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Hiragana as u8,
            ],
            boundaries: vec![
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
            ],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 8],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_to_tokenized_string_unknown() {
        let s = Sentence::from_partial_annotation("火-星 猫|の|生-態");
        let result = s.unwrap().to_tokenized_string();

        assert_eq!(
            "InvalidSentenceError: contains an unknown boundary",
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
    fn test_sentence_to_tokenized_string_with_tags() {
        let s =
            Sentence::from_tokenized("Rust/名詞 で 良い/形容詞 プログラミング 体験 を ！/補助記号");

        assert_eq!(
            "Rust/名詞 で 良い/形容詞 プログラミング 体験 を ！/補助記号",
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

        assert_eq!(
            "InvalidSentenceError: contains an unknown boundary",
            result.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_to_tokenized_vec() {
        let s = Sentence::from_tokenized("Rust で 良い プログラミング 体験 を ！").unwrap();

        assert_eq!(
            vec![
                Token {
                    surface: "Rust",
                    tag: None
                },
                Token {
                    surface: "で",
                    tag: None
                },
                Token {
                    surface: "良い",
                    tag: None
                },
                Token {
                    surface: "プログラミング",
                    tag: None
                },
                Token {
                    surface: "体験",
                    tag: None
                },
                Token {
                    surface: "を",
                    tag: None
                },
                Token {
                    surface: "！",
                    tag: None
                },
            ],
            s.to_tokenized_vec().unwrap()
        );
    }

    #[test]
    fn test_sentence_to_tokenized_vec_with_tags() {
        let s =
            Sentence::from_tokenized("Rust/名詞 で 良い/形容詞 プログラミング 体験 を ！/補助記号")
                .unwrap();

        assert_eq!(
            vec![
                Token {
                    surface: "Rust",
                    tag: Some("名詞"),
                },
                Token {
                    surface: "で",
                    tag: None,
                },
                Token {
                    surface: "良い",
                    tag: Some("形容詞"),
                },
                Token {
                    surface: "プログラミング",
                    tag: None,
                },
                Token {
                    surface: "体験",
                    tag: None,
                },
                Token {
                    surface: "を",
                    tag: None,
                },
                Token {
                    surface: "！",
                    tag: Some("補助記号"),
                },
            ],
            s.to_tokenized_vec().unwrap()
        );
    }

    #[test]
    fn test_sentence_from_partial_annotation_empty() {
        let s = Sentence::from_partial_annotation("");

        assert_eq!(
            "InvalidArgumentError: labeled_text: must contain at least one character",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_partial_annotation_empty() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_partial_annotation("");

        assert_eq!(
            "InvalidArgumentError: labeled_text: must contain at least one character",
            &result.err().unwrap().to_string()
        );

        let expected = Sentence {
            text: " ".to_string(),
            chars: vec![' '],
            str_to_char_pos: vec![0, 1],
            char_to_str_pos: vec![0, 1],
            char_type: vec![Other as u8],
            boundaries: vec![],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_from_partial_annotation_null() {
        let s = Sentence::from_partial_annotation("A-1-あ-\0-ア-亜");

        assert_eq!(
            "InvalidArgumentError: labeled_text: must not contain NULL",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_partial_annotation_null() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_partial_annotation("A-1-あ-\0-ア-亜");

        assert_eq!(
            "InvalidArgumentError: labeled_text: must not contain NULL",
            &result.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_from_partial_annotation_invalid_length() {
        let result = Sentence::from_partial_annotation("火-星 猫|の|生-態 ");

        assert_eq!(
            "InvalidArgumentError: labeled_text: invalid annotation",
            &result.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_partial_annotation_invalid_length() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_partial_annotation("火-星 猫|の|生-態 ");

        assert_eq!(
            "InvalidArgumentError: labeled_text: invalid annotation",
            &result.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_from_partial_annotation_invalid_boundary_character() {
        let s = Sentence::from_partial_annotation("火-星?猫|の|生-態");

        assert_eq!(
            "InvalidArgumentError: labeled_text: contains an invalid boundary character: '?'",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_partial_annotation_invalid_boundary_character() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_partial_annotation("火-星?猫|の|生-態");

        assert_eq!(
            "InvalidArgumentError: labeled_text: contains an invalid boundary character: '?'",
            &result.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_from_partial_annotation_one() {
        let s = Sentence::from_partial_annotation("火-星 猫|の|生-態");

        let expected = Sentence {
            text: "火星猫の生態".to_string(),
            chars: vec!['火', '星', '猫', 'の', '生', '態'],
            str_to_char_pos: vec![0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6],
            char_to_str_pos: vec![0, 3, 6, 9, 12, 15, 18],
            char_type: vec![
                Kanji as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Kanji as u8,
                Kanji as u8,
            ],
            boundaries: vec![
                NotWordBoundary,
                Unknown,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
            ],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 6],
        };
        assert_eq!(expected, s.unwrap());
    }

    #[test]
    fn test_sentence_update_partial_annotation_one() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_partial_annotation("火-星 猫|の|生-態").unwrap();

        let expected = Sentence {
            text: "火星猫の生態".to_string(),
            chars: vec!['火', '星', '猫', 'の', '生', '態'],
            str_to_char_pos: vec![0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6],
            char_to_str_pos: vec![0, 3, 6, 9, 12, 15, 18],
            char_type: vec![
                Kanji as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Kanji as u8,
                Kanji as u8,
            ],
            boundaries: vec![
                NotWordBoundary,
                Unknown,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
            ],
            boundary_scores: vec![],
            tag_scores: TagScores::default(),
            tags: vec![None; 6],
        };
        assert_eq!(expected, s);
    }

    #[test]
    fn test_sentence_to_partial_annotation_string() {
        let s = Sentence::from_partial_annotation("火-星 猫|の|生-態");

        assert_eq!(
            "火-星 猫|の|生-態",
            s.unwrap().to_partial_annotation_string()
        );
    }

    #[test]
    fn test_sentence_to_partial_annotation_string_with_tags() {
        let s = Sentence::from_partial_annotation("火-星 猫|の/助詞|生-態/名詞");

        assert_eq!(
            "火-星 猫|の/助詞|生-態/名詞",
            s.unwrap().to_partial_annotation_string()
        );
    }
}
