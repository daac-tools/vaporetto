//! # vaporetto_rules
//!
//! Rule base filters for Vaporetto.
//!
//! ## Examples
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::BufReader;
//! use std::rc::Rc;
//!
//! use vaporetto::{CharacterType, Model, Predictor, Sentence};
//! use vaporetto_rules::{
//!     SentenceFilter, StringFilter,
//!     sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter},
//!     string_filters::KyteaFullwidthFilter,
//! };
//!
//! let f = BufReader::new(File::open("model.bin").unwrap());
//! let model = Model::read(f).unwrap();
//! let mut predictor = Predictor::new(model, false).unwrap();
//!
//! let pre_filters: Vec<Box<dyn StringFilter<String>>> = vec![
//!     Box::new(KyteaFullwidthFilter),
//! ];
//! let post_filters: Vec<Box<dyn SentenceFilter>> = vec![
//!     Box::new(ConcatGraphemeClustersFilter),
//!     Box::new(KyteaWsConstFilter::new(CharacterType::Digit)),
//! ];
//!
//! let input = "Vaporettoは仲良し家族👨‍👨‍👧‍👦を離れ離れにさせません。"
//!     .to_string();
//!
//! let preproc_input = pre_filters.iter().fold(input, |s, filter| filter.filter(s));
//!
//! let mut sentence = Sentence::from_raw(preproc_input).unwrap();
//! predictor.predict(&mut sentence);
//!
//! post_filters.iter().for_each(|filter| filter.filter(&mut sentence));
//!
//! let mut buf = String::new();
//! sentence.write_tokenized_text(&mut buf);
//! assert_eq!(
//!     "Ｖａｐｏｒｅｔｔｏ は 仲良 し 家族 👨‍👨‍👧‍👦 を 離れ離れ に さ せ ま せ ん 。",
//!     buf,
//! );
//! ```

#![no_std]

#[macro_use]
extern crate alloc;

pub mod sentence_filters;
pub mod string_filters;

use alloc::string::String;

use vaporetto::Sentence;

pub trait SentenceFilter: Send + Sync {
    /// Filter a specified sentence using rules.
    fn filter(&self, sentence: &mut Sentence);
}

pub trait StringFilter<S>: Send + Sync
where
    S: AsRef<str>,
{
    /// Filter a specified string using rules.
    fn filter(&self, string: S) -> String;
}
