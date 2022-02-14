//! # vaporetto_tantivy
//!
//! Vaporetto Tokenizer for Tantivy
//!
//! ## Examples
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::{Read, BufReader};
//!
//! use tantivy::tokenizer::Tokenizer;
//! use vaporetto::Model;
//! use vaporetto_tantivy::VaporettoTokenizer;
//!
//! let mut f = BufReader::new(File::open("model.zst").unwrap());
//! let mut decoder = ruzstd::StreamingDecoder::new(&mut f).unwrap();
//! let mut buff = vec![];
//! decoder.read_to_end(&mut buff).unwrap();
//! let model = Model::read(&mut buff.as_slice()).unwrap();
//!
//! let tokenizer = VaporettoTokenizer::new(model, "DGR").unwrap();
//!
//! let mut stream = tokenizer.token_stream("東京特許許可局");
//!
//! let token = stream.next().unwrap();
//! assert_eq!(token.text, "東京");
//! assert_eq!(token.offset_from, 0);
//! assert_eq!(token.offset_to, 6);
//! assert_eq!(token.position, 0);
//!
//! let token = stream.next().unwrap();
//! assert_eq!(token.text, "特許");
//! assert_eq!(token.offset_from, 6);
//! assert_eq!(token.offset_to, 12);
//! assert_eq!(token.position, 1);
//!
//! let token = stream.next().unwrap();
//! assert_eq!(token.text, "許可");
//! assert_eq!(token.offset_from, 12);
//! assert_eq!(token.offset_to, 18);
//! assert_eq!(token.position, 2);
//!
//! let token = stream.next().unwrap();
//! assert_eq!(token.text, "局");
//! assert_eq!(token.offset_from, 18);
//! assert_eq!(token.offset_to, 21);
//! assert_eq!(token.position, 3);
//!
//! assert!(stream.next().is_none());
/// ```
use std::sync::Arc;

use tantivy::tokenizer::{BoxTokenStream, Token, TokenStream, Tokenizer};
use vaporetto::{BoundaryType, CharacterType, Model, Predictor, Sentence};
use vaporetto_rules::{
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter, SplitLinebreaksFilter},
    string_filters::KyteaFullwidthFilter,
    SentenceFilter, StringFilter,
};

/// Tokenize the text using Vaporetto.
#[derive(Clone)]
pub struct VaporettoTokenizer {
    predictor: Arc<Predictor>,
    prefilter: KyteaFullwidthFilter,
    postfilters: Vec<Arc<dyn SentenceFilter>>,
}

impl VaporettoTokenizer {
    /// Creates a new VaporettoTokenizer.
    ///
    /// # Arguments
    ///
    /// * `model` - A model data of Vaporetto.
    /// * `wsconst` - Character types that the tokenizer does not segment.
    ///               D: Digit, R: Roman, H: Hiragana, T: Katakana, K: Kanji, O: Other,
    ///               G: Grapheme cluster.
    ///
    /// # Errors
    ///
    /// Error is returned when
    ///   - the model is invalid, or
    ///   - `wsconst` contains an invalid character type.
    pub fn new(model: Model, wsconst: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let mut postfilters: Vec<Arc<dyn SentenceFilter>> = vec![Arc::new(SplitLinebreaksFilter)];
        for c in wsconst.chars() {
            postfilters.push(match c {
                'D' => Arc::new(KyteaWsConstFilter::new(CharacterType::Digit)),
                'R' => Arc::new(KyteaWsConstFilter::new(CharacterType::Roman)),
                'H' => Arc::new(KyteaWsConstFilter::new(CharacterType::Hiragana)),
                'T' => Arc::new(KyteaWsConstFilter::new(CharacterType::Katakana)),
                'K' => Arc::new(KyteaWsConstFilter::new(CharacterType::Kanji)),
                'O' => Arc::new(KyteaWsConstFilter::new(CharacterType::Other)),
                'G' => Arc::new(ConcatGraphemeClustersFilter),
                _ => return Err("Could not parse a wsconst value".into()),
            });
        }
        Ok(Self {
            predictor: Arc::new(Predictor::new(model, false)?),
            prefilter: KyteaFullwidthFilter,
            postfilters,
        })
    }
}

pub struct VaporettoTokenStream<'a> {
    text: &'a str,
    token: Token,
    boundary_pos: Vec<usize>,
    offset_to: usize,
    position: usize,
}

impl Tokenizer for VaporettoTokenizer {
    fn token_stream<'a>(&self, text: &'a str) -> BoxTokenStream<'a> {
        if text.is_empty() {
            return BoxTokenStream::from(VaporettoTokenStream {
                text,
                boundary_pos: vec![],
                token: Token::default(),
                offset_to: 0,
                position: 0,
            });
        }

        // pre filter
        let prefiltered_text = self.prefilter.filter(text);
        let prefiltered_sentence = Sentence::from_raw(prefiltered_text).unwrap();

        // tokenize
        let tokenized_sentence = self.predictor.predict(prefiltered_sentence);

        // post filter
        let postfiltered_sentence = self
            .postfilters
            .iter()
            .fold(tokenized_sentence, |s, filter| filter.filter(s));

        let mut char_indices = text.char_indices();
        char_indices.next();
        let mut boundary_pos = Vec::with_capacity(postfiltered_sentence.chars().len());
        for ((i, _), &b) in char_indices.zip(postfiltered_sentence.boundaries()) {
            if b == BoundaryType::WordBoundary {
                boundary_pos.push(i);
            }
        }
        boundary_pos.push(text.len());

        BoxTokenStream::from(VaporettoTokenStream {
            text,
            token: Token::default(),
            boundary_pos,
            offset_to: 0,
            position: 0,
        })
    }
}

impl<'a> TokenStream for VaporettoTokenStream<'a> {
    fn advance(&mut self) -> bool {
        if self.position < self.boundary_pos.len() {
            self.token.offset_from = self.offset_to;
            self.offset_to = self.boundary_pos[self.position];
            self.token.offset_to = self.offset_to;
            self.token.text.clear();
            self.token
                .text
                .push_str(&self.text[self.token.offset_from..self.token.offset_to]);
            self.token.position = self.position;
            self.token.position_length = self.boundary_pos.len();
            self.position += 1;
            true
        } else {
            false
        }
    }

    fn token(&self) -> &Token {
        &self.token
    }

    fn token_mut(&mut self) -> &mut Token {
        &mut self.token
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::io::{Cursor, Read};

    use tantivy::tokenizer::TextAnalyzer;

    fn token_stream_helper(text: &str, wsconst: &str) -> Vec<Token> {
        let mut f = Cursor::new(include_bytes!("../test_model/model.zst"));
        let mut decoder = ruzstd::StreamingDecoder::new(&mut f).unwrap();
        let mut buff = vec![];
        decoder.read_to_end(&mut buff).unwrap();
        let model = Model::read(&mut buff.as_slice()).unwrap();
        let a = TextAnalyzer::from(VaporettoTokenizer::new(model, wsconst).unwrap());
        let mut token_stream = a.token_stream(text);
        let mut tokens: Vec<Token> = vec![];
        let mut add_token = |token: &Token| {
            tokens.push(token.clone());
        };
        token_stream.process(&mut add_token);
        tokens
    }

    #[test]
    fn test_tokenize_empty() {
        let tokens = token_stream_helper("", "");

        assert_eq!(tokens.len(), 0);
    }

    #[test]
    fn test_tokenizer_tokyo() {
        let tokens = token_stream_helper("東京特許許可局", "");

        assert_eq!(tokens.len(), 4);

        let token = &tokens[0];
        assert_eq!(token.text, "東京");
        assert_eq!(token.offset_from, 0);
        assert_eq!(token.offset_to, 6);
        assert_eq!(token.position, 0);
        assert_eq!(token.position_length, 4);

        let token = &tokens[1];
        assert_eq!(token.text, "特許");
        assert_eq!(token.offset_from, 6);
        assert_eq!(token.offset_to, 12);
        assert_eq!(token.position, 1);
        assert_eq!(token.position_length, 4);

        let token = &tokens[2];
        assert_eq!(token.text, "許可");
        assert_eq!(token.offset_from, 12);
        assert_eq!(token.offset_to, 18);
        assert_eq!(token.position, 2);
        assert_eq!(token.position_length, 4);

        let token = &tokens[3];
        assert_eq!(token.text, "局");
        assert_eq!(token.offset_from, 18);
        assert_eq!(token.offset_to, 21);
        assert_eq!(token.position, 3);
        assert_eq!(token.position_length, 4);
    }

    #[test]
    fn test_tokenizer_no_wsconst() {
        let tokens = token_stream_helper("123456円🤌🏿", "");

        assert_eq!(tokens.len(), 9);

        let token = &tokens[0];
        assert_eq!(token.text, "1");
        assert_eq!(token.offset_from, 0);
        assert_eq!(token.offset_to, 1);
        assert_eq!(token.position, 0);
        assert_eq!(token.position_length, 9);

        let token = &tokens[1];
        assert_eq!(token.text, "2");
        assert_eq!(token.offset_from, 1);
        assert_eq!(token.offset_to, 2);
        assert_eq!(token.position, 1);
        assert_eq!(token.position_length, 9);

        let token = &tokens[2];
        assert_eq!(token.text, "3");
        assert_eq!(token.offset_from, 2);
        assert_eq!(token.offset_to, 3);
        assert_eq!(token.position, 2);
        assert_eq!(token.position_length, 9);

        let token = &tokens[3];
        assert_eq!(token.text, "4");
        assert_eq!(token.offset_from, 3);
        assert_eq!(token.offset_to, 4);
        assert_eq!(token.position, 3);
        assert_eq!(token.position_length, 9);

        let token = &tokens[4];
        assert_eq!(token.text, "5");
        assert_eq!(token.offset_from, 4);
        assert_eq!(token.offset_to, 5);
        assert_eq!(token.position, 4);
        assert_eq!(token.position_length, 9);

        let token = &tokens[5];
        assert_eq!(token.text, "6");
        assert_eq!(token.offset_from, 5);
        assert_eq!(token.offset_to, 6);
        assert_eq!(token.position, 5);
        assert_eq!(token.position_length, 9);

        let token = &tokens[6];
        assert_eq!(token.text, "円");
        assert_eq!(token.offset_from, 6);
        assert_eq!(token.offset_to, 9);
        assert_eq!(token.position, 6);
        assert_eq!(token.position_length, 9);

        let token = &tokens[7];
        assert_eq!(token.text, "🤌");
        assert_eq!(token.offset_from, 9);
        assert_eq!(token.offset_to, 13);
        assert_eq!(token.position, 7);
        assert_eq!(token.position_length, 9);

        let token = &tokens[8];
        assert_eq!(token.text, "🏿");
        assert_eq!(token.offset_from, 13);
        assert_eq!(token.offset_to, 17);
        assert_eq!(token.position, 8);
        assert_eq!(token.position_length, 9);
    }

    #[test]
    fn test_tokenize_wsconst_d() {
        let tokens = token_stream_helper("123456円🤌🏿", "D");

        assert_eq!(tokens.len(), 4);

        let token = &tokens[0];
        assert_eq!(token.text, "123456");
        assert_eq!(token.offset_from, 0);
        assert_eq!(token.offset_to, 6);
        assert_eq!(token.position, 0);
        assert_eq!(token.position_length, 4);

        let token = &tokens[1];
        assert_eq!(token.text, "円");
        assert_eq!(token.offset_from, 6);
        assert_eq!(token.offset_to, 9);
        assert_eq!(token.position, 1);
        assert_eq!(token.position_length, 4);

        let token = &tokens[2];
        assert_eq!(token.text, "🤌");
        assert_eq!(token.offset_from, 9);
        assert_eq!(token.offset_to, 13);
        assert_eq!(token.position, 2);
        assert_eq!(token.position_length, 4);

        let token = &tokens[3];
        assert_eq!(token.text, "🏿");
        assert_eq!(token.offset_from, 13);
        assert_eq!(token.offset_to, 17);
        assert_eq!(token.position, 3);
        assert_eq!(token.position_length, 4);
    }

    #[test]
    fn test_tokenizer_wsconst_g() {
        let tokens = token_stream_helper("123456円🤌🏿", "G");

        assert_eq!(tokens.len(), 8);

        let token = &tokens[0];
        assert_eq!(token.text, "1");
        assert_eq!(token.offset_from, 0);
        assert_eq!(token.offset_to, 1);
        assert_eq!(token.position, 0);
        assert_eq!(token.position_length, 8);

        let token = &tokens[1];
        assert_eq!(token.text, "2");
        assert_eq!(token.offset_from, 1);
        assert_eq!(token.offset_to, 2);
        assert_eq!(token.position, 1);
        assert_eq!(token.position_length, 8);

        let token = &tokens[2];
        assert_eq!(token.text, "3");
        assert_eq!(token.offset_from, 2);
        assert_eq!(token.offset_to, 3);
        assert_eq!(token.position, 2);
        assert_eq!(token.position_length, 8);

        let token = &tokens[3];
        assert_eq!(token.text, "4");
        assert_eq!(token.offset_from, 3);
        assert_eq!(token.offset_to, 4);
        assert_eq!(token.position, 3);
        assert_eq!(token.position_length, 8);

        let token = &tokens[4];
        assert_eq!(token.text, "5");
        assert_eq!(token.offset_from, 4);
        assert_eq!(token.offset_to, 5);
        assert_eq!(token.position, 4);
        assert_eq!(token.position_length, 8);

        let token = &tokens[5];
        assert_eq!(token.text, "6");
        assert_eq!(token.offset_from, 5);
        assert_eq!(token.offset_to, 6);
        assert_eq!(token.position, 5);
        assert_eq!(token.position_length, 8);

        let token = &tokens[6];
        assert_eq!(token.text, "円");
        assert_eq!(token.offset_from, 6);
        assert_eq!(token.offset_to, 9);
        assert_eq!(token.position, 6);
        assert_eq!(token.position_length, 8);

        let token = &tokens[7];
        assert_eq!(token.text, "🤌🏿");
        assert_eq!(token.offset_from, 9);
        assert_eq!(token.offset_to, 17);
        assert_eq!(token.position, 7);
        assert_eq!(token.position_length, 8);
    }

    #[test]
    fn test_tokenize_wsconst_dg() {
        let tokens = token_stream_helper("123456円🤌🏿", "DG");

        assert_eq!(tokens.len(), 3);

        let token = &tokens[0];
        assert_eq!(token.text, "123456");
        assert_eq!(token.offset_from, 0);
        assert_eq!(token.offset_to, 6);
        assert_eq!(token.position, 0);
        assert_eq!(token.position_length, 3);

        let token = &tokens[1];
        assert_eq!(token.text, "円");
        assert_eq!(token.offset_from, 6);
        assert_eq!(token.offset_to, 9);
        assert_eq!(token.position, 1);
        assert_eq!(token.position_length, 3);

        let token = &tokens[2];
        assert_eq!(token.text, "🤌🏿");
        assert_eq!(token.offset_from, 9);
        assert_eq!(token.offset_to, 17);
        assert_eq!(token.position, 2);
        assert_eq!(token.position_length, 3);
    }
}
