use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

use crate::errors::{Result, VaporettoError};
use crate::predictor::Predictor;

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
#[derive(Debug, Eq, PartialEq, Clone, Copy)]
#[repr(u8)]
pub enum CharacterBoundary {
    /// Inner of a word.
    NotWordBoundary = 0,

    /// Word boundary.
    WordBoundary = 1,

    /// Unknown. (Not annotated.)
    Unknown = 2,
}

/// Sentence data containing boundary and tag annotations.
pub struct Sentence<'a, 'b> {
    pub(crate) text: Cow<'a, str>,
    pub(crate) char_types: Vec<u8>,
    pub(crate) boundaries: Vec<CharacterBoundary>,
    pub(crate) boundary_scores: Vec<i32>,
    pub(crate) score_padding: usize,
    pub(crate) char_pma_states: Vec<u32>,
    pub(crate) type_pma_states: Vec<u32>,
    pub(crate) tags: Vec<Option<Cow<'b, str>>>,
    #[allow(clippy::type_complexity)]
    pub(crate) tag_scores: Vec<Option<(&'b [Vec<String>], Vec<i32>)>>,
    pub(crate) n_tags: usize,
    predictor: Option<&'b Predictor>,
    str_to_char_pos: Vec<usize>,
    char_to_str_pos: Vec<usize>,
}

impl<'a, 'b> Default for Sentence<'a, 'b> {
    /// Creates a new [`Sentence`] consisting of a space.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::default();
    ///
    /// assert_eq!(" ", s.as_raw_text());
    /// assert_eq!(0, s.n_tags());
    /// ```
    fn default() -> Self {
        let mut s = Self {
            text: Cow::Borrowed(""),
            char_types: vec![],
            boundaries: vec![],
            boundary_scores: vec![],
            score_padding: 0,
            char_pma_states: vec![],
            type_pma_states: vec![],
            tags: vec![],
            tag_scores: vec![],
            n_tags: 0,
            predictor: None,
            str_to_char_pos: vec![],
            char_to_str_pos: vec![],
        };
        s.set_default();
        s
    }
}

impl<'a, 'b> Sentence<'a, 'b> {
    #[inline(always)]
    fn set_default(&mut self) {
        self.text = Cow::Borrowed(" ");
        self.char_types.clear();
        self.char_types.push(CharacterType::Other as u8);
        self.boundaries.clear();
        self.boundary_scores.clear();
        self.score_padding = 0;
        self.char_pma_states.clear();
        self.type_pma_states.clear();
        self.tags.clear();
        self.n_tags = 0;
        self.predictor.take();
        self.str_to_char_pos.clear();
        self.str_to_char_pos.push(0);
        self.str_to_char_pos.push(1);
        self.char_to_str_pos.clear();
        self.char_to_str_pos.push(0);
        self.char_to_str_pos.push(1);
    }

    fn parse_raw(
        text: &str,
        char_types: &mut Vec<u8>,
        boundaries: &mut Vec<CharacterBoundary>,
        str_to_char_pos: &mut Vec<usize>,
        char_to_str_pos: &mut Vec<usize>,
    ) -> Result<()> {
        char_types.clear();
        boundaries.clear();
        str_to_char_pos.clear();
        char_to_str_pos.clear();
        char_to_str_pos.push(0);
        let mut pos = 0;
        for c in text.chars() {
            if c == '\0' {
                return Err(VaporettoError::invalid_argument(
                    "text",
                    "must not contain NULL",
                ));
            }
            char_types.push(CharacterType::get_type(c) as u8);
            pos += c.len_utf8();
            char_to_str_pos.push(pos);
        }
        if char_types.is_empty() {
            return Err(VaporettoError::invalid_argument(
                "text",
                "must contain at least one character",
            ));
        }
        str_to_char_pos.resize(pos + 1, 0);
        for (i, &pos) in char_to_str_pos.iter().enumerate() {
            str_to_char_pos[pos] = i;
        }
        boundaries.resize(char_types.len() - 1, CharacterBoundary::Unknown);
        Ok(())
    }

    /// Creates a new [`Sentence`] from a given text without any annotation.
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
    /// let s = Sentence::from_raw("まぁ良いだろう").unwrap();
    /// let mut buf = String::new();
    /// s.write_partial_annotation_text(&mut buf);
    /// assert_eq!("ま ぁ 良 い だ ろ う", buf);
    ///
    /// let s = Sentence::from_raw("");
    /// assert!(s.is_err());
    /// ```
    pub fn from_raw(text: impl Into<Cow<'a, str>>) -> Result<Self> {
        let text = text.into();
        let mut char_types = vec![];
        let mut boundaries = vec![];
        let mut str_to_char_pos = vec![];
        let mut char_to_str_pos = vec![];
        Self::parse_raw(
            &text,
            &mut char_types,
            &mut boundaries,
            &mut str_to_char_pos,
            &mut char_to_str_pos,
        )?;
        Ok(Self {
            text,
            char_types,
            boundaries,
            boundary_scores: vec![],
            score_padding: 0,
            char_pma_states: vec![],
            type_pma_states: vec![],
            predictor: None,
            tags: vec![],
            tag_scores: vec![],
            n_tags: 0,
            str_to_char_pos,
            char_to_str_pos,
        })
    }

    /// Updates the [`Sentence`] using a given text without any annotation.
    ///
    /// # Errors
    ///
    /// If the given `text` is empty, an error variant will be returned.
    /// When an error is occurred, the sentence will be replaced with a white space.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut s = Sentence::from_raw("まぁ良いだろう").unwrap();
    /// s.update_raw("まぁ社長は火星猫だ").unwrap();
    /// assert_eq!("まぁ社長は火星猫だ", s.as_raw_text());
    /// ```
    pub fn update_raw(&mut self, text: impl Into<Cow<'a, str>>) -> Result<()> {
        self.text = text.into();
        if let Err(e) = Self::parse_raw(
            &self.text,
            &mut self.char_types,
            &mut self.boundaries,
            &mut self.str_to_char_pos,
            &mut self.char_to_str_pos,
        ) {
            self.set_default();
            return Err(e);
        }
        self.boundary_scores.clear();
        self.score_padding = 0;
        self.char_pma_states.clear();
        self.type_pma_states.clear();
        self.predictor.take();
        self.tags.clear();
        Ok(())
    }

    fn parse_tokenized(
        tokenized_text: &str,
        text: &mut String,
        char_types: &mut Vec<u8>,
        boundaries: &mut Vec<CharacterBoundary>,
        str_to_char_pos: &mut Vec<usize>,
        char_to_str_pos: &mut Vec<usize>,
        tags: &mut Vec<Option<Cow<'b, str>>>,
    ) -> Result<()> {
        if tokenized_text.is_empty() {
            return Err(VaporettoError::invalid_argument(
                "tokenized_text",
                "must contain at least one character",
            ));
        }
        text.clear();
        char_types.clear();
        boundaries.clear();
        str_to_char_pos.clear();
        char_to_str_pos.clear();
        char_to_str_pos.push(0);
        let mut tag_str = None;
        let mut prev_boundary = false;
        let mut escape = false;
        let mut tags_tmp: Vec<Vec<_>> = vec![];
        let mut pos = 0;
        for c in tokenized_text.chars() {
            match (escape, c) {
                // escape a following character
                (false, '\\') => {
                    escape = true;
                }
                // token boundary
                (false, ' ') => {
                    if text.is_empty() {
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
                    if let Some(tag) = tag_str.take() {
                        tags_tmp.last_mut().unwrap().push(tag);
                    }
                    prev_boundary = true;
                }
                // tag
                (false, '/') => {
                    if text.is_empty() || prev_boundary {
                        return Err(VaporettoError::invalid_argument(
                            "tokenized_text",
                            "a slash must follow a character",
                        ));
                    }
                    if let Some(tag) = tag_str.replace(String::new()) {
                        tags_tmp.last_mut().unwrap().push(tag);
                    }
                }
                // escaped character or other character
                (_, _) => {
                    escape = false;
                    if c == '\0' {
                        return Err(VaporettoError::invalid_argument(
                            "tokenized_text",
                            "must not contain NULL",
                        ));
                    }
                    if let Some(tag) = tag_str.as_mut() {
                        tag.push(c);
                        continue;
                    }
                    if !text.is_empty() {
                        boundaries.push(if prev_boundary {
                            CharacterBoundary::WordBoundary
                        } else {
                            CharacterBoundary::NotWordBoundary
                        });
                    }
                    prev_boundary = false;
                    text.push(c);
                    char_types.push(CharacterType::get_type(c) as u8);
                    pos += c.len_utf8();
                    char_to_str_pos.push(pos);
                    tags_tmp.push(vec![]);
                }
            };
        }
        if prev_boundary {
            return Err(VaporettoError::invalid_argument(
                "tokenized_text",
                "must not end with a whitespace",
            ));
        }
        str_to_char_pos.resize(pos + 1, 0);
        for (i, &pos) in char_to_str_pos.iter().enumerate() {
            str_to_char_pos[pos] = i;
        }
        if let Some(tag) = tag_str.take() {
            tags_tmp.last_mut().unwrap().push(tag);
        }
        let n_tags = tags_tmp.iter().fold(0, |acc, x| acc.max(x.len()));
        tags.clear();
        for ts in tags_tmp {
            let n_fill_none = n_tags - ts.len();
            for t in ts {
                if t.is_empty() {
                    tags.push(None);
                } else {
                    tags.push(Some(Cow::Owned(t)));
                }
            }
            for _ in 0..n_fill_none {
                tags.push(None);
            }
        }
        Ok(())
    }

    /// Creates a new [`Sentence`] from a tokenized text.
    ///
    /// A tokenized text must be annotated by the following rules:
    ///   - A whitespace (`' '`) is inserted to each token boundary.
    ///   - If necessary, multiple tags following each slash (`'/'`) can be added to each token.
    ///   - Each character following a back slash (`'\\'`) is escaped.
    ///
    /// # Errors
    ///
    /// This function will return an error variant when the given text is empty, starts/ends with a
    /// whitespace, or contains consecutive whitespaces.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_tokenized("まぁ 社長 は 火星 猫 だ");
    /// assert_eq!("まぁ社長は火星猫だ", s.unwrap().as_raw_text());
    ///
    /// let s = Sentence::from_tokenized("まぁ/名詞 社長/名詞 は/助詞 火星/名詞 猫/名詞 だ/助動詞");
    /// assert_eq!("まぁ社長は火星猫だ", s.unwrap().as_raw_text());
    ///
    /// let s = Sentence::from_tokenized("まぁ 社長  は 火星 猫 だ");
    /// assert!(s.is_err());
    /// ```
    pub fn from_tokenized(tokenized_text: &str) -> Result<Self> {
        let mut text = String::new();
        let mut char_types = vec![];
        let mut boundaries = vec![];
        let mut str_to_char_pos = vec![];
        let mut char_to_str_pos = vec![];
        let mut tags = vec![];
        Self::parse_tokenized(
            tokenized_text,
            &mut text,
            &mut char_types,
            &mut boundaries,
            &mut str_to_char_pos,
            &mut char_to_str_pos,
            &mut tags,
        )?;
        let n_tags = tags.len() / char_types.len();
        Ok(Self {
            text: Cow::Owned(text),
            char_types,
            boundaries,
            boundary_scores: vec![],
            score_padding: 0,
            char_pma_states: vec![],
            type_pma_states: vec![],
            predictor: None,
            tags,
            tag_scores: vec![],
            n_tags,
            str_to_char_pos,
            char_to_str_pos,
        })
    }

    /// Updates the [`Sentence`] using a tokenized text.
    ///
    /// A tokenized text must be annotated by the following rules:
    ///   - A whitespace (`' '`) is inserted to each token boundary.
    ///   - If necessary, multiple tags following each slash (`'/'`) can be added to each token.
    ///   - Each character following a back slash (`'\\'`) is escaped.
    ///
    /// # Errors
    ///
    /// This function will return an error variant when the given text is empty, starts/ends with a
    /// whitespace, or contains consecutive whitespaces.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut s = Sentence::default();
    ///
    /// s.update_tokenized("まぁ 良い だろう").unwrap();
    /// assert_eq!("まぁ良いだろう", s.as_raw_text());
    ///
    /// s.update_tokenized("まぁ/副詞/マー 良い/形容詞/ヨイ だろう/助動詞/ダロー").unwrap();
    /// assert_eq!("まぁ良いだろう", s.as_raw_text());
    /// ```
    pub fn update_tokenized(&mut self, tokenized_text: &str) -> Result<()> {
        if let Err(e) = Self::parse_tokenized(
            tokenized_text,
            self.text.to_mut(),
            &mut self.char_types,
            &mut self.boundaries,
            &mut self.str_to_char_pos,
            &mut self.char_to_str_pos,
            &mut self.tags,
        ) {
            self.set_default();
            return Err(e);
        }
        self.boundary_scores.clear();
        self.score_padding = 0;
        self.char_pma_states.clear();
        self.type_pma_states.clear();
        self.predictor.take();
        self.n_tags = self.tags.len() / self.char_types.len();
        Ok(())
    }

    fn parse_partial_annotation(
        partial_annotation_text: &str,
        text: &mut String,
        char_types: &mut Vec<u8>,
        boundaries: &mut Vec<CharacterBoundary>,
        str_to_char_pos: &mut Vec<usize>,
        char_to_str_pos: &mut Vec<usize>,
        tags: &mut Vec<Option<Cow<'b, str>>>,
    ) -> Result<()> {
        if partial_annotation_text.is_empty() {
            return Err(VaporettoError::invalid_argument(
                "partial_annotation_text",
                "must contain at least one character",
            ));
        }
        text.clear();
        char_types.clear();
        boundaries.clear();
        str_to_char_pos.clear();
        char_to_str_pos.clear();
        char_to_str_pos.push(0);
        let mut tag_str = None;
        let mut escape = false;
        let mut tags_tmp: Vec<Vec<_>> = vec![];
        let mut pos = 0;
        let mut is_char = true;
        for c in partial_annotation_text.chars() {
            if is_char {
                if c == '\0' {
                    return Err(VaporettoError::invalid_argument(
                        "partial_annotation_text",
                        "must not contain NULL",
                    ));
                }
                text.push(c);
                char_types.push(CharacterType::get_type(c) as u8);
                pos += c.len_utf8();
                char_to_str_pos.push(pos);
                tags_tmp.push(vec![]);
                is_char = false;
                continue;
            }
            match (escape, c) {
                (false, '\\') => {
                    escape = true;
                }
                (false, ' ') => {
                    if let Some(tag) = tag_str.take() {
                        tags_tmp.last_mut().unwrap().push(tag);
                    }
                    boundaries.push(CharacterBoundary::Unknown);
                    is_char = true;
                }
                (false, '-') => {
                    if let Some(tag) = tag_str.take() {
                        tags_tmp.last_mut().unwrap().push(tag);
                    }
                    boundaries.push(CharacterBoundary::NotWordBoundary);
                    is_char = true;
                }
                (false, '|') => {
                    if let Some(tag) = tag_str.take() {
                        tags_tmp.last_mut().unwrap().push(tag);
                    }
                    boundaries.push(CharacterBoundary::WordBoundary);
                    is_char = true;
                }
                (false, '/') => {
                    let tag = tag_str.replace(String::new());
                    if let Some(tag) = tag {
                        tags_tmp.last_mut().unwrap().push(tag);
                    }
                }
                _ => {
                    escape = false;
                    if let Some(tag) = tag_str.as_mut() {
                        tag.push(c);
                    } else {
                        return Err(VaporettoError::invalid_argument(
                            "partial_annotation_text",
                            format!("contains an invalid boundary character: '{c}'"),
                        ));
                    }
                }
            }
        }
        if is_char {
            return Err(VaporettoError::invalid_argument(
                "partial_annotation_text",
                "invalid annotation",
            ));
        }
        str_to_char_pos.resize(pos + 1, 0);
        for (i, &pos) in char_to_str_pos.iter().enumerate() {
            str_to_char_pos[pos] = i;
        }
        if let Some(tag) = tag_str.take() {
            tags_tmp.last_mut().unwrap().push(tag);
        }
        let n_tags = tags_tmp.iter().fold(0, |acc, x| acc.max(x.len()));
        tags.clear();
        for ts in tags_tmp {
            let n_fill_none = n_tags - ts.len();
            for t in ts {
                if t.is_empty() {
                    tags.push(None);
                } else {
                    tags.push(Some(Cow::Owned(t)));
                }
            }
            for _ in 0..n_fill_none {
                tags.push(None);
            }
        }
        Ok(())
    }

    /// Creates a new [`Sentence`] from a text with partial annotations.
    ///
    /// Each character boundary must be annotated by the following rules:
    ///   - If the boundary is a token boundary, a pipe symbol (`'|'`) is inserted.
    ///   - If the boundary is not a token boundary, a dash symobl (`'-'`) is inserted.
    ///   - If the boundary is not annotated, a whitespace (`' '`) is inserted.
    ///
    /// In addition, multiple tags following each slash (`'/'`) can be inserted to each token.
    /// Tags can also be inserted at non-word boundaries, but such tags are ignored.
    ///
    /// # Errors
    ///
    /// This function will return an error variant when the text is empty, the length of the text
    /// is even numbers, or the text contains invalid boundary characters.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut buf = String::new();
    ///
    /// let s = Sentence::from_partial_annotation(
    ///     "ま-ぁ|良-い|だ-ろ-う"
    /// ).unwrap();
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("まぁ 良い だろう", buf);
    ///
    /// let s = Sentence::from_partial_annotation(
    ///     "ま-ぁ/名詞/マー|社-長/名詞/シャチョー|は/助詞/ワ|火-星 猫|だ/助動詞/ダ"
    /// ).unwrap();
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("まぁ/名詞/マー 社長/名詞/シャチョー は/助詞/ワ だ/助動詞/ダ", buf);
    ///
    /// let s = Sentence::from_partial_annotation(
    ///     "ま-ぁ/名詞/マー|社-長/名詞/シャチョー|は/助詞/ワ|火/名詞/ヒ-星|猫|だ/助動詞/ダ"
    /// ).unwrap();
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("まぁ/名詞/マー 社長/名詞/シャチョー は/助詞/ワ 火星 猫 だ/助動詞/ダ", buf);
    /// ```
    pub fn from_partial_annotation(partial_annotation_text: &str) -> Result<Self> {
        let mut text = String::new();
        let mut char_types = vec![];
        let mut boundaries = vec![];
        let mut str_to_char_pos = vec![];
        let mut char_to_str_pos = vec![];
        let mut tags = vec![];
        Self::parse_partial_annotation(
            partial_annotation_text,
            &mut text,
            &mut char_types,
            &mut boundaries,
            &mut str_to_char_pos,
            &mut char_to_str_pos,
            &mut tags,
        )?;
        let n_tags = tags.len() / char_types.len();
        Ok(Self {
            text: Cow::Owned(text),
            char_types,
            boundaries,
            boundary_scores: vec![],
            score_padding: 0,
            char_pma_states: vec![],
            type_pma_states: vec![],
            predictor: None,
            tags,
            tag_scores: vec![],
            n_tags,
            str_to_char_pos,
            char_to_str_pos,
        })
    }

    /// Updates the [`Sentence`] using a text with partial annotations.
    ///
    /// Each character boundary must be annotated by the following rules:
    ///   - If the boundary is a token boundary, a pipe symbol (`'|'`) is inserted.
    ///   - If the boundary is not a token boundary, a dash symobl (`'-'`) is inserted.
    ///   - If the boundary is not annotated, a whitespace (`' '`) is inserted.
    ///
    /// In addition, multiple tags following each slash (`'/'`) can be inserted to each token.
    /// Tags can also be inserted at non-word boundaries, but such tags are ignored.
    ///
    /// # Errors
    ///
    /// This function will return an error variant when the text is empty, the length of the text
    /// is even numbers, or the text contains invalid boundary characters.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut buf = String::new();
    /// let mut s = Sentence::default();
    ///
    /// s.update_partial_annotation(
    ///     "ま-ぁ|良-い|だ-ろ-う"
    /// ).unwrap();
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("まぁ 良い だろう", buf);
    ///
    /// s.update_partial_annotation(
    ///     "ま-ぁ/名詞/マー|社-長/名詞/シャチョー|は/助詞/ワ|火-星 猫|だ/助動詞/ダ"
    /// ).unwrap();
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("まぁ/名詞/マー 社長/名詞/シャチョー は/助詞/ワ だ/助動詞/ダ", buf);
    ///
    /// s.update_partial_annotation(
    ///     "ま-ぁ/名詞/マー|社-長/名詞/シャチョー|は/助詞/ワ|火/名詞/ヒ-星|猫|だ/助動詞/ダ"
    /// ).unwrap();
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("まぁ/名詞/マー 社長/名詞/シャチョー は/助詞/ワ 火星 猫 だ/助動詞/ダ", buf);
    /// ```
    pub fn update_partial_annotation(&mut self, partial_annotation_text: &str) -> Result<()> {
        if let Err(e) = Self::parse_partial_annotation(
            partial_annotation_text,
            self.text.to_mut(),
            &mut self.char_types,
            &mut self.boundaries,
            &mut self.str_to_char_pos,
            &mut self.char_to_str_pos,
            &mut self.tags,
        ) {
            self.set_default();
            return Err(e);
        }
        self.boundary_scores.clear();
        self.score_padding = 0;
        self.char_pma_states.clear();
        self.type_pma_states.clear();
        self.predictor.take();
        self.n_tags = self.tags.len() / self.char_types.len();
        Ok(())
    }

    /// Gets a text without any annotation.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_tokenized("まぁ/副詞 良い/形容詞 だろう/助動詞").unwrap();
    /// assert_eq!("まぁ良いだろう", s.as_raw_text());
    /// ```
    #[inline]
    pub fn as_raw_text(&self) -> &str {
        &self.text
    }

    /// Returns an iterator of tokens. Tokens adjacent to [`CharacterBoundary::Unknown`] will be
    /// skipped.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_partial_annotation("ま-ぁ|社-長|は|火-星 猫|だ").unwrap();
    /// let mut it = s.iter_tokens();
    ///
    /// let token = it.next().unwrap();
    /// assert_eq!("まぁ", token.surface());
    /// assert_eq!(0, token.start());
    /// assert_eq!(2, token.end());
    ///
    /// let token = it.next().unwrap();
    /// assert_eq!("社長", token.surface());
    /// assert_eq!(2, token.start());
    /// assert_eq!(4, token.end());
    ///
    /// let token = it.next().unwrap();
    /// assert_eq!("は", token.surface());
    /// assert_eq!(4, token.start());
    /// assert_eq!(5, token.end());
    ///
    /// let token = it.next().unwrap();
    /// assert_eq!("だ", token.surface());
    /// assert_eq!(8, token.start());
    /// assert_eq!(9, token.end());
    ///
    /// assert!(it.next().is_none());
    /// ```
    pub const fn iter_tokens(&'a self) -> TokenIterator<'a, 'b> {
        TokenIterator {
            token: Token {
                sentence: self,
                start: 0,
                end: 0,
            },
        }
    }

    /// Writes a tokenized text. Tokens adjacent to [`CharacterBoundary::Unknown`] will be skipped.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut buf = String::new();
    ///
    /// let s = Sentence::from_partial_annotation(
    ///     "ま-ぁ/名詞|社-長/名詞|は/助詞|火-星/名詞|猫/名詞|だ/助動詞"
    /// ).unwrap();
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("まぁ/名詞 社長/名詞 は/助詞 火星/名詞 猫/名詞 だ/助動詞", buf);
    ///
    /// let s = Sentence::from_partial_annotation(
    ///     "ま-ぁ/名詞|社-長/名詞|は/助詞|火-星 猫|だ/助動詞"
    /// ).unwrap();
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("まぁ/名詞 社長/名詞 は/助詞 だ/助動詞", buf);
    /// ```
    pub fn write_tokenized_text(&self, buf: &mut String) {
        buf.clear();
        // `buf` always consists of a valid UTF-8 sequence because
        // `Token::surface` and `Token::tags` return values in `str`.
        unsafe {
            let buf = buf.as_mut_vec();
            for token in self.iter_tokens() {
                if !buf.is_empty() {
                    buf.push(b' ');
                }
                for &b in token.surface().as_bytes() {
                    match b {
                        b' ' | b'\\' | b'/' => {
                            buf.push(b'\\');
                        }
                        _ => (),
                    }
                    buf.push(b);
                }
                let ts = token.tags();
                for tag in &ts[..ts.iter().rposition(|x| x.is_some()).map_or(0, |x| x + 1)] {
                    buf.push(b'/');
                    if let Some(tag) = tag {
                        for &b in tag.as_bytes() {
                            match b {
                                b' ' | b'\\' | b'/' => {
                                    buf.push(b'\\');
                                }
                                _ => (),
                            }
                            buf.push(b);
                        }
                    }
                }
            }
        }
    }

    /// Writes a text with partial annotations.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut buf = String::new();
    ///
    /// let s = Sentence::from_tokenized("まぁ 良い だろう").unwrap();
    /// s.write_partial_annotation_text(&mut buf);
    /// assert_eq!("ま-ぁ|良-い|だ-ろ-う", buf);
    ///
    /// let s = Sentence::from_tokenized(
    ///     "まぁ/副詞/マー 良い/形容詞/ヨイ だろう/助動詞/ダロー"
    /// ).unwrap();
    /// s.write_partial_annotation_text(&mut buf);
    /// assert_eq!("ま-ぁ/副詞/マー|良-い/形容詞/ヨイ|だ-ろ-う/助動詞/ダロー", buf);
    /// ```
    pub fn write_partial_annotation_text(&self, buf: &mut String) {
        buf.clear();
        let mut char_iter = self.text.chars();
        buf.push(char_iter.next().unwrap());
        if self.n_tags != 0 {
            let mut tag_iter = self.tags.chunks_exact(self.n_tags);
            let ts = tag_iter.next().unwrap();
            for tag in &ts[..ts.iter().rposition(|x| x.is_some()).map_or(0, |x| x + 1)] {
                buf.push('/');
                if let Some(tag) = tag {
                    buf.push_str(tag);
                }
            }
            for ((c, ts), &b) in char_iter.zip(tag_iter).zip(&self.boundaries) {
                buf.push(match b {
                    CharacterBoundary::NotWordBoundary => '-',
                    CharacterBoundary::WordBoundary => '|',
                    CharacterBoundary::Unknown => ' ',
                });
                buf.push(c);
                for tag in &ts[..ts.iter().rposition(|x| x.is_some()).map_or(0, |x| x + 1)] {
                    buf.push('/');
                    if let Some(tag) = tag {
                        buf.push_str(tag);
                    }
                }
            }
        } else {
            for (c, &b) in char_iter.zip(&self.boundaries) {
                buf.push(match b {
                    CharacterBoundary::NotWordBoundary => '-',
                    CharacterBoundary::WordBoundary => '|',
                    CharacterBoundary::Unknown => ' ',
                });
                buf.push(c);
            }
        }
    }

    /// Removes tag information and updates the number of tags.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut s = Sentence::from_tokenized("火星/名詞/カセー に 行き/動詞 まし/助動詞/マシ た").unwrap();
    /// let mut buf = String::new();
    /// assert_eq!(2, s.n_tags());
    /// assert_eq!(16, s.tags().len());
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("火星/名詞/カセー に 行き/動詞 まし/助動詞/マシ た", buf);
    ///
    /// s.reset_tags(1);
    /// assert_eq!(1, s.n_tags());
    /// assert_eq!(8, s.tags().len());
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("火星 に 行き まし た", buf);
    /// ```
    #[inline]
    pub fn reset_tags(&mut self, n_tags: usize) {
        self.tags.clear();
        self.tags.resize(n_tags * self.len(), None);
        self.n_tags = n_tags;
    }

    /// Returns a slice of character types.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::{CharacterType, Sentence};
    ///
    /// let s = Sentence::from_tokenized("火星 に 行き まし た").unwrap();
    /// assert_eq!(&[
    ///     CharacterType::Kanji as u8,
    ///     CharacterType::Kanji as u8,
    ///     CharacterType::Hiragana as u8,
    ///     CharacterType::Kanji as u8,
    ///     CharacterType::Hiragana as u8,
    ///     CharacterType::Hiragana as u8,
    ///     CharacterType::Hiragana as u8,
    ///     CharacterType::Hiragana as u8,
    /// ], s.char_types());
    /// ```
    #[inline]
    pub fn char_types(&self) -> &[u8] {
        &self.char_types
    }

    /// Returns a slice of boundary types.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::{CharacterBoundary, Sentence};
    ///
    /// let s = Sentence::from_partial_annotation("火-星|に|行-き|ま-し た").unwrap();
    /// assert_eq!(&[
    ///     CharacterBoundary::NotWordBoundary,
    ///     CharacterBoundary::WordBoundary,
    ///     CharacterBoundary::WordBoundary,
    ///     CharacterBoundary::NotWordBoundary,
    ///     CharacterBoundary::WordBoundary,
    ///     CharacterBoundary::NotWordBoundary,
    ///     CharacterBoundary::Unknown,
    /// ], s.boundaries());
    /// ```
    #[inline]
    pub fn boundaries(&self) -> &[CharacterBoundary] {
        &self.boundaries
    }

    /// Returns a mutable slice of boundary types.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::{CharacterBoundary, Sentence};
    ///
    /// let mut s = Sentence::from_partial_annotation("火-星|に|行-き|ま-し た").unwrap();
    /// s.boundaries_mut()[6] = CharacterBoundary::WordBoundary;
    /// let mut buf = String::new();
    /// s.write_partial_annotation_text(&mut buf);
    /// assert_eq!("火-星|に|行-き|ま-し|た", buf);
    /// ```
    #[inline]
    pub fn boundaries_mut(&mut self) -> &mut [CharacterBoundary] {
        &mut self.boundaries
    }

    /// Returns a slice of boundary scores.
    #[inline]
    pub fn boundary_scores(&self) -> &[i32] {
        if self.boundary_scores.is_empty() {
            &[]
        } else {
            &self.boundary_scores[self.score_padding..self.score_padding + self.boundaries.len()]
        }
    }

    /// Returns a reference to the internal representation of tags.
    ///
    /// In the representation, tags are stored in an array, and
    /// the `j`-th tag of the `i`-th character is stored in the `i*k+j`-th element,
    /// where `k` is the maximum number of tags (i.e., [`Sentence::n_tags()`]).
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut s = Sentence::from_tokenized("火星/名詞/カセー に 行き/動詞 まし/助動詞/マシ た").unwrap();
    /// assert_eq!(16, s.tags().len());
    /// assert_eq!("名詞", s.tags()[2].as_ref().unwrap().as_ref());
    /// assert_eq!("カセー", s.tags()[3].as_ref().unwrap().as_ref());
    /// assert_eq!("動詞", s.tags()[8].as_ref().unwrap().as_ref());
    /// assert_eq!("助動詞", s.tags()[12].as_ref().unwrap().as_ref());
    /// assert_eq!("マシ", s.tags()[13].as_ref().unwrap().as_ref());
    /// ```
    #[inline]
    pub fn tags(&self) -> &[Option<Cow<'b, str>>] {
        &self.tags
    }

    /// Returns a mutable reference to the internal representation of tags.
    ///
    /// In the representation, tags are stored in an array, and
    /// the `j`-th tag of the `i`-th character is stored in the `i*k+j`-th element,
    /// where `k` is the maximum number of tags (i.e., [`Sentence::n_tags()`]).
    ///
    /// Tags can also be inserted at other positions, but such tags are ignored.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let mut buf = String::new();
    ///
    /// let mut s = Sentence::from_tokenized("火星/名詞/カセー に 行き/動詞 まし/助動詞/マシ た").unwrap();
    /// s.tags_mut()[4].replace("助詞".into());
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("火星/名詞/カセー に/助詞 行き/動詞 まし/助動詞/マシ た", buf);
    ///
    /// // Sets a pronunciation of the first character (`火`), but this character is not the last
    /// // of a word.
    /// s.tags_mut()[1].replace("ヒ".into());
    /// s.write_tokenized_text(&mut buf);
    /// assert_eq!("火星/名詞/カセー に/助詞 行き/動詞 まし/助動詞/マシ た", buf);
    /// ```
    #[inline]
    pub fn tags_mut(&mut self) -> &mut [Option<Cow<'b, str>>] {
        &mut self.tags
    }

    /// Update the tag information.
    /// If you want to predict tags, call this function after calling [`Predictor::predict()`] and
    /// word boundaries are fixed.
    ///
    /// # Panics
    ///
    /// The predictor must be created with `predict_tags = true`.
    ///
    #[cfg_attr(
        feature = "std",
        doc = "
# Examples

```
use std::fs::File;

use vaporetto::{Model, Predictor, Sentence};

let f = File::open(\"../resources/model.bin\").unwrap();
let model = Model::read(f).unwrap();
let predictor = Predictor::new(model, true).unwrap();

let mut s = Sentence::from_raw(\"まぁ良いだろう\").unwrap();
predictor.predict(&mut s);
let mut buf = String::new();
s.write_tokenized_text(&mut buf);
assert_eq!(\"まぁ 良い だろう\", buf);

s.fill_tags();

s.write_tokenized_text(&mut buf);
assert_eq!(
    \"まぁ/副詞/マー 良い/形容詞/ヨイ だろう/助動詞/ダロー\",
    buf,
);
```
"
    )]
    #[cfg(feature = "tag-prediction")]
    #[cfg_attr(docsrs, doc(cfg(feature = "tag-prediction")))]
    #[inline]
    pub fn fill_tags(&mut self) {
        if let Some(p) = self.predictor.as_ref() {
            p.predict_tags(self);
        }
    }

    /// Returns the maximum number of tags.
    ///
    /// # Examples
    ///
    /// ```
    /// use vaporetto::Sentence;
    ///
    /// let s = Sentence::from_tokenized("火星/名詞/カセー に 行き/動詞 まし/助動詞/マシ た").unwrap();
    /// assert_eq!(2, s.n_tags());
    /// ```
    #[inline]
    pub const fn n_tags(&self) -> usize {
        self.n_tags
    }

    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.char_types.len()
    }

    #[inline]
    pub(crate) fn set_predictor(&mut self, predictor: &'b Predictor) {
        self.predictor.replace(predictor);
    }

    /// # Safety
    ///
    /// `pos` must be a position corresponding to a boundary in the UTF-8 format.
    #[inline(always)]
    pub(crate) unsafe fn str_to_char_pos(&self, pos: usize) -> usize {
        *self.str_to_char_pos.get_unchecked(pos)
    }

    #[inline]
    pub(crate) fn text_substring(&self, start: usize, end: usize) -> &str {
        &self.text[self.char_to_str_pos[start]..self.char_to_str_pos[end]]
    }

    #[cfg(test)]
    pub(crate) fn char_to_str_pos(&self) -> &[usize] {
        &self.char_to_str_pos
    }
}

/// A Token information.
#[derive(Clone, Copy)]
pub struct Token<'a, 'b> {
    sentence: &'a Sentence<'a, 'b>,
    start: usize,
    end: usize,
}

impl<'a, 'b> Token<'a, 'b> {
    /// Returns the surface of this token.
    #[inline]
    pub fn surface(&self) -> &'a str {
        self.sentence.text_substring(self.start, self.end)
    }

    /// Returns tags of this token.
    #[inline]
    pub fn tags(&self) -> &'a [Option<Cow<'b, str>>] {
        let start = (self.end - 1) * self.sentence.n_tags();
        let end = self.end * self.sentence.n_tags();
        &self.sentence.tags[start..end]
    }

    /// Returns tag candidates with scores.
    ///
    /// The return value is a two-dimensional array. The outer array index corresponding to the
    /// return value of [`Token::tags()`]. The inner array is a candidate set, where each element
    /// is a tuple of the tag name and its score.
    pub fn tag_candidates(&self) -> Vec<Vec<(&'b str, i32)>> {
        let mut results = vec![];
        if let Some((tags, scores)) = self.sentence.tag_scores[self.end - 1].as_ref() {
            let mut i = 0;
            for cands in *tags {
                let mut inner = vec![];
                if cands.len() == 1 {
                    inner.push((cands[0].as_str(), 0));
                } else {
                    for cand in cands {
                        inner.push((cand.as_str(), scores[i]));
                        i += 1;
                    }
                }
                results.push(inner);
            }
        }
        results
    }

    /// Returns the start position of this token in characters.
    #[inline]
    pub const fn start(&self) -> usize {
        self.start
    }

    /// Returns the end position of this token in characters.
    #[inline]
    pub const fn end(&self) -> usize {
        self.end
    }
}

/// Iterator returned by [`Sentence::iter_tokens()`].
pub struct TokenIterator<'a, 'b> {
    token: Token<'a, 'b>,
}

impl<'a, 'b> Iterator for TokenIterator<'a, 'b> {
    type Item = Token<'a, 'b>;

    fn next(&mut self) -> Option<Self::Item> {
        self.token.start = self.token.end;
        if let Some(boundaries) = self.token.sentence.boundaries().get(self.token.start..) {
            let mut skip_token = false;
            for (i, &b) in boundaries.iter().enumerate() {
                if b == CharacterBoundary::WordBoundary {
                    if skip_token {
                        self.token.start += i + 1;
                        skip_token = false;
                    } else {
                        self.token.end += i + 1;
                        return Some(self.token);
                    }
                } else if b == CharacterBoundary::Unknown {
                    skip_token = true;
                }
            }
            if skip_token {
                self.token.end = self.token.sentence.boundaries().len() + 1;
                return None;
            }
        } else {
            return None;
        }
        self.token.end = self.token.sentence.boundaries().len() + 1;
        Some(self.token)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::string::ToString;

    use CharacterBoundary::*;
    use CharacterType::*;

    #[test]
    fn test_sentence_from_raw_empty() {
        let s = Sentence::from_raw("");

        assert_eq!(
            "InvalidArgumentError: text: must contain at least one character",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_raw_empty() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_raw("");

        assert_eq!(
            "InvalidArgumentError: text: must contain at least one character",
            &result.err().unwrap().to_string()
        );

        assert_eq!(" ", s.as_raw_text());
        assert_eq!(&[0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 1], s.char_to_str_pos());
        assert_eq!([Other as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_from_raw_null() {
        let s = Sentence::from_raw("A1あ\0ア亜");

        assert_eq!(
            "InvalidArgumentError: text: must not contain NULL",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_raw_null() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_raw("A1あ\0ア亜");

        assert_eq!(
            "InvalidArgumentError: text: must not contain NULL",
            &result.err().unwrap().to_string()
        );

        assert_eq!(" ", s.as_raw_text());
        assert_eq!(&[0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 1], s.char_to_str_pos());
        assert_eq!([Other as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_from_raw_one() {
        let s = Sentence::from_raw("あ").unwrap();

        assert_eq!("あ", s.as_raw_text());
        assert_eq!(&[0, 0, 0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 3], s.char_to_str_pos());
        assert_eq!([Hiragana as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_update_raw_one() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_raw("あ").unwrap();

        assert_eq!("あ", s.as_raw_text());
        assert_eq!(&[0, 0, 0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 3], s.char_to_str_pos());
        assert_eq!([Hiragana as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_from_raw() {
        let s = Sentence::from_raw("Rustで良いプログラミング体験を！").unwrap();

        assert_eq!("Rustで良いプログラミング体験を！", s.as_raw_text());
        assert_eq!(
            &[
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!([Unknown; 17], s.boundaries());
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_update_raw() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_raw("Rustで良いプログラミング体験を！").unwrap();

        assert_eq!("Rustで良いプログラミング体験を！", s.as_raw_text());
        assert_eq!(
            &[
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!([Unknown; 17], s.boundaries());
        assert!(s.boundary_scores().is_empty());
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

        assert_eq!(" ", s.as_raw_text());
        assert_eq!(&[0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 1], s.char_to_str_pos());
        assert_eq!([Other as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
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

        assert_eq!(" ", s.as_raw_text());
        assert_eq!(&[0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 1], s.char_to_str_pos());
        assert_eq!([Other as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
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

        assert_eq!(" ", s.as_raw_text());
        assert_eq!(&[0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 1], s.char_to_str_pos());
        assert_eq!([Other as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
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

        assert_eq!(" ", s.as_raw_text());
        assert_eq!(&[0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 1], s.char_to_str_pos());
        assert_eq!([Other as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
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

        assert_eq!(" ", s.as_raw_text());
        assert_eq!(&[0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 1], s.char_to_str_pos());
        assert_eq!([Other as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_from_tokenized_one() {
        let s = Sentence::from_tokenized("あ").unwrap();

        assert_eq!("あ", s.as_raw_text());
        assert_eq!(&[0, 0, 0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 3], s.char_to_str_pos());
        assert_eq!([Hiragana as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_update_tokenized_one() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("あ").unwrap();

        assert_eq!("あ", s.as_raw_text());
        assert_eq!(&[0, 0, 0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 3], s.char_to_str_pos());
        assert_eq!([Hiragana as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_from_tokenized() {
        let s = Sentence::from_tokenized("Rust で 良い プログラミング 体験 を ！").unwrap();

        assert_eq!("Rustで良いプログラミング体験を！", s.as_raw_text());
        assert_eq!(
            &[
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
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
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_update_tokenized() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("Rust で 良い プログラミング 体験 を ！")
            .unwrap();

        assert_eq!("Rustで良いプログラミング体験を！", s.as_raw_text());
        assert_eq!(
            &[
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
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
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_from_tokenized_with_tags() {
        let s =
            Sentence::from_tokenized("Rust/名詞 で 良い/形容詞 プログラミング 体験 を ！/補助記号")
                .unwrap();

        assert_eq!("Rustで良いプログラミング体験を！", s.as_raw_text());
        assert_eq!(
            &[
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
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
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
        assert_eq!(
            &[
                None,
                None,
                None,
                Some(Cow::Borrowed("名詞")),
                None,
                None,
                Some(Cow::Borrowed("形容詞")),
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
                Some(Cow::Borrowed("補助記号")),
            ],
            s.tags.as_slice()
        );
    }

    #[test]
    fn test_sentence_update_tokenized_with_tags() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("Rust/名詞 で 良い/形容詞 プログラミング 体験 を ！/補助記号")
            .unwrap();

        assert_eq!("Rustで良いプログラミング体験を！", s.as_raw_text());
        assert_eq!(
            &[
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
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
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
        assert_eq!(
            &[
                None,
                None,
                None,
                Some(Cow::Borrowed("名詞")),
                None,
                None,
                Some(Cow::Borrowed("形容詞")),
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
                Some(Cow::Borrowed("補助記号")),
            ],
            s.tags.as_slice()
        );
    }

    #[test]
    fn test_sentence_from_tokenized_with_tags_two_slashes() {
        let s = Sentence::from_tokenized(
            "Rust/名詞 で 良い/形容詞/イイ プログラミング 体験 を ！/補助記号",
        )
        .unwrap();

        assert_eq!("Rustで良いプログラミング体験を！", s.as_raw_text());
        assert_eq!(
            &[
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
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
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
        assert_eq!(
            &[
                None,
                None,
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("名詞")),
                None,
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("形容詞")),
                Some(Cow::Borrowed("イイ")),
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
                Some(Cow::Borrowed("補助記号")),
                None,
            ],
            s.tags.as_slice()
        );
    }

    #[test]
    fn test_sentence_update_tokenized_two_slashes() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("Rust/名詞 で 良い/形容詞/イイ プログラミング 体験 を ！/補助記号")
            .unwrap();

        assert_eq!("Rustで良いプログラミング体験を！", s.as_raw_text());
        assert_eq!(
            &[
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
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
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
        assert_eq!(
            &[
                None,
                None,
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("名詞")),
                None,
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("形容詞")),
                Some(Cow::Borrowed("イイ")),
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
                Some(Cow::Borrowed("補助記号")),
                None,
            ],
            s.tags.as_slice()
        );
    }

    #[test]
    fn test_sentence_from_tokenized_with_tags_empty_slashes() {
        let s = Sentence::from_tokenized(
            "Rust//ラスト で 良い/形容詞/イイ プログラミング 体験 を ！//ビックリ",
        )
        .unwrap();

        assert_eq!("Rustで良いプログラミング体験を！", s.as_raw_text());
        assert_eq!(
            &[
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
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
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
        assert_eq!(
            &[
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("ラスト")),
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("形容詞")),
                Some(Cow::Borrowed("イイ")),
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
                None,
                Some(Cow::Borrowed("ビックリ")),
            ],
            s.tags.as_slice()
        );
    }

    #[test]
    fn test_sentence_update_tokenized_empty_slashes() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("Rust//ラスト で 良い/形容詞/イイ プログラミング 体験 を ！//ビックリ")
            .unwrap();

        assert_eq!("Rustで良いプログラミング体験を！", s.as_raw_text());
        assert_eq!(
            &[
                0, 1, 2, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9, 0, 0, 10, 0, 0, 11, 0,
                0, 12, 0, 0, 13, 0, 0, 14, 0, 0, 15, 0, 0, 16, 0, 0, 17, 0, 0, 18,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 1, 2, 3, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46,],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
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
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
        assert_eq!(
            &[
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("ラスト")),
                None,
                None,
                None,
                None,
                Some(Cow::Borrowed("形容詞")),
                Some(Cow::Borrowed("イイ")),
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
                None,
                Some(Cow::Borrowed("ビックリ")),
            ],
            s.tags.as_slice()
        );
    }

    #[test]
    fn test_sentence_from_tokenized_with_escape_whitespace() {
        let s = Sentence::from_tokenized("火星 猫 の 生態 ( M \\  et\\ al. )").unwrap();

        assert_eq!("火星猫の生態(M et al.)", s.as_raw_text());
        assert_eq!(
            &[
                0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13,
                14, 15, 16,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
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
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_update_tokenized_escape_whitespace() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("火星 猫 の 生態 ( M \\  et\\ al. )")
            .unwrap();

        assert_eq!("火星猫の生態(M et al.)", s.as_raw_text());
        assert_eq!(
            &[
                0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6, 7, 8, 9, 10, 11, 12, 13,
                14, 15, 16,
            ],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!(
            [0, 3, 6, 9, 12, 15, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
            s.char_to_str_pos()
        );
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
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
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_from_tokenized_with_escape_backslash() {
        let s = Sentence::from_tokenized("改行 に \\\\n を 用い る").unwrap();

        assert_eq!("改行に\\nを用いる", s.as_raw_text());
        assert_eq!(
            &[0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 4, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!([0, 3, 6, 9, 10, 11, 14, 17, 20, 23], s.char_to_str_pos());
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
            ],
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_update_tokenized_with_escape_backslash() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("改行 に \\\\n を 用い る").unwrap();

        assert_eq!("改行に\\nを用いる", s.as_raw_text());
        assert_eq!(
            &[0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 4, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8, 0, 0, 9],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!([0, 3, 6, 9, 10, 11, 14, 17, 20, 23], s.char_to_str_pos());
        assert_eq!(
            [
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
            s.char_types()
        );
        assert_eq!(
            [
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
            ],
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_from_tokenized_escape_slash() {
        let s = Sentence::from_tokenized("品詞 に \\/ を 用い る").unwrap();

        assert_eq!("品詞に/を用いる", s.as_raw_text());
        assert_eq!(
            &[0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!([0, 3, 6, 9, 10, 13, 16, 19, 22], s.char_to_str_pos());
        assert_eq!(
            [
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Hiragana as u8,
            ],
            s.char_types()
        );
        assert_eq!(
            [
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
            ],
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_update_tokenized_escape_slash() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_tokenized("品詞 に \\/ を 用い る").unwrap();

        assert_eq!("品詞に/を用いる", s.as_raw_text());
        assert_eq!(
            &[0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 4, 0, 0, 5, 0, 0, 6, 0, 0, 7, 0, 0, 8],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!([0, 3, 6, 9, 10, 13, 16, 19, 22], s.char_to_str_pos());
        assert_eq!(
            [
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Other as u8,
                Hiragana as u8,
                Kanji as u8,
                Hiragana as u8,
                Hiragana as u8,
            ],
            s.char_types()
        );
        assert_eq!(
            [
                NotWordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
                WordBoundary,
            ],
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_to_tokenized_string_unknown() {
        let s = Sentence::from_partial_annotation("火-星 猫|の|生-態").unwrap();
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);

        assert_eq!("の 生態", buf);
    }

    #[test]
    fn test_sentence_to_tokenized_string() {
        let s = Sentence::from_tokenized("Rust で 良い プログラミング 体験 を ！").unwrap();
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);

        assert_eq!("Rust で 良い プログラミング 体験 を ！", buf);
    }

    #[test]
    fn test_sentence_to_tokenized_string_with_tags() {
        let s =
            Sentence::from_tokenized("Rust/名詞 で 良い/形容詞 プログラミング 体験 を ！/補助記号")
                .unwrap();
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);

        assert_eq!(
            "Rust/名詞 で 良い/形容詞 プログラミング 体験 を ！/補助記号",
            buf,
        );
    }

    #[test]
    fn test_sentence_to_tokenized_string_escape() {
        let s = Sentence::from_partial_annotation("火-星-猫|の| |生-態|\\-n").unwrap();
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);

        assert_eq!("火星猫 の \\  生態 \\\\n", buf);
    }

    #[test]
    fn test_sentence_to_tokenized_vec_unknown() {
        let s = Sentence::from_partial_annotation("火-星 猫|の|生-態").unwrap();
        let mut it = s.iter_tokens();

        let token = it.next().unwrap();
        assert_eq!("の", token.surface());

        let token = it.next().unwrap();
        assert_eq!("生態", token.surface());

        assert!(it.next().is_none());
    }

    #[test]
    fn test_sentence_to_tokenized_vec() {
        let s = Sentence::from_tokenized("Rust で 良い プログラミング 体験 を ！").unwrap();
        let mut it = s.iter_tokens();

        let token = it.next().unwrap();
        assert_eq!("Rust", token.surface());

        let token = it.next().unwrap();
        assert_eq!("で", token.surface());

        let token = it.next().unwrap();
        assert_eq!("良い", token.surface());

        let token = it.next().unwrap();
        assert_eq!("プログラミング", token.surface());

        let token = it.next().unwrap();
        assert_eq!("体験", token.surface());

        let token = it.next().unwrap();
        assert_eq!("を", token.surface());

        let token = it.next().unwrap();
        assert_eq!("！", token.surface());

        assert!(it.next().is_none());
    }

    #[test]
    fn test_sentence_to_tokenized_vec_with_tags() {
        let s =
            Sentence::from_tokenized("Rust/名詞 で 良い/形容詞 プログラミング 体験 を ！/補助記号")
                .unwrap();
        let mut it = s.iter_tokens();

        let token = it.next().unwrap();
        assert_eq!("Rust", token.surface());
        assert_eq!("名詞", token.tags()[0].as_ref().unwrap());

        let token = it.next().unwrap();
        assert_eq!("で", token.surface());
        assert!(token.tags()[0].is_none());

        let token = it.next().unwrap();
        assert_eq!("良い", token.surface());
        assert_eq!("形容詞", token.tags()[0].as_ref().unwrap());

        let token = it.next().unwrap();
        assert_eq!("プログラミング", token.surface());
        assert!(token.tags()[0].is_none());

        let token = it.next().unwrap();
        assert_eq!("体験", token.surface());
        assert!(token.tags()[0].is_none());

        let token = it.next().unwrap();
        assert_eq!("を", token.surface());
        assert!(token.tags()[0].is_none());

        let token = it.next().unwrap();
        assert_eq!("！", token.surface());
        assert_eq!("補助記号", token.tags()[0].as_ref().unwrap());

        assert!(it.next().is_none());
    }

    #[test]
    fn test_sentence_from_partial_annotation_empty() {
        let s = Sentence::from_partial_annotation("");

        assert_eq!(
            "InvalidArgumentError: partial_annotation_text: must contain at least one character",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_partial_annotation_empty() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_partial_annotation("");

        assert_eq!(
            "InvalidArgumentError: partial_annotation_text: must contain at least one character",
            &result.err().unwrap().to_string()
        );

        assert_eq!(" ", s.as_raw_text());
        assert_eq!(&[0, 1], s.str_to_char_pos.as_slice());
        assert_eq!([0, 1], s.char_to_str_pos());
        assert_eq!([Other as u8], s.char_types());
        assert!(s.boundaries().is_empty());
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_from_partial_annotation_null() {
        let s = Sentence::from_partial_annotation("A-1-あ-\0-ア-亜");

        assert_eq!(
            "InvalidArgumentError: partial_annotation_text: must not contain NULL",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_partial_annotation_null() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_partial_annotation("A-1-あ-\0-ア-亜");

        assert_eq!(
            "InvalidArgumentError: partial_annotation_text: must not contain NULL",
            &result.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_from_partial_annotation_invalid_length() {
        let result = Sentence::from_partial_annotation("火-星 猫|の|生-態 ");

        assert_eq!(
            "InvalidArgumentError: partial_annotation_text: invalid annotation",
            &result.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_partial_annotation_invalid_length() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_partial_annotation("火-星 猫|の|生-態 ");

        assert_eq!(
            "InvalidArgumentError: partial_annotation_text: invalid annotation",
            &result.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_from_partial_annotation_invalid_boundary_character() {
        let s = Sentence::from_partial_annotation("火-星?猫|の|生-態");

        assert_eq!(
            "InvalidArgumentError: partial_annotation_text: contains an invalid boundary character: '?'",
            &s.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_update_partial_annotation_invalid_boundary_character() {
        let mut s = Sentence::from_raw("12345").unwrap();
        let result = s.update_partial_annotation("火-星?猫|の|生-態");

        assert_eq!(
            "InvalidArgumentError: partial_annotation_text: contains an invalid boundary character: '?'",
            &result.err().unwrap().to_string()
        );
    }

    #[test]
    fn test_sentence_from_partial_annotation_one() {
        let s = Sentence::from_partial_annotation("火-星 猫|の|生-態").unwrap();

        assert_eq!("火星猫の生態", s.as_raw_text());
        assert_eq!(
            &[0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!([0, 3, 6, 9, 12, 15, 18], s.char_to_str_pos());
        assert_eq!(
            [
                Kanji as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Kanji as u8,
                Kanji as u8,
            ],
            s.char_types()
        );
        assert_eq!(
            [
                NotWordBoundary,
                Unknown,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
            ],
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
    }

    #[test]
    fn test_sentence_update_partial_annotation_one() {
        let mut s = Sentence::from_raw("12345").unwrap();
        s.update_partial_annotation("火-星 猫|の|生-態").unwrap();

        assert_eq!("火星猫の生態", s.as_raw_text());
        assert_eq!(
            &[0, 0, 0, 1, 0, 0, 2, 0, 0, 3, 0, 0, 4, 0, 0, 5, 0, 0, 6],
            s.str_to_char_pos.as_slice(),
        );
        assert_eq!([0, 3, 6, 9, 12, 15, 18], s.char_to_str_pos());
        assert_eq!(
            [
                Kanji as u8,
                Kanji as u8,
                Kanji as u8,
                Hiragana as u8,
                Kanji as u8,
                Kanji as u8,
            ],
            s.char_types()
        );
        assert_eq!(
            [
                NotWordBoundary,
                Unknown,
                WordBoundary,
                WordBoundary,
                NotWordBoundary,
            ],
            s.boundaries()
        );
        assert!(s.boundary_scores().is_empty());
    }
}
