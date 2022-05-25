//! # Vaporetto
//!
//! Vaporetto is a fast and lightweight pointwise prediction based tokenizer.
//!
//! ## Examples
//!
//! ```
//! use std::fs::File;
//!
//! use vaporetto::{Model, Predictor, Sentence};
//!
//! let f = File::open("../resources/model.bin").unwrap();
//! let model = Model::read(f).unwrap();
//! let predictor = Predictor::new(model, true).unwrap();
//!
//! let mut buf = String::new();
//!
//! let mut s = Sentence::default();
//!
//! s.update_raw("まぁ社長は火星猫だ").unwrap();
//! predictor.predict(&mut s);
//! s.fill_tags();
//! s.write_tokenized_text(&mut buf);
//! assert_eq!(
//!     "まぁ/名詞/マー 社長/名詞/シャチョー は/助詞/ワ 火星/名詞/カセー 猫/名詞/ネコ だ/助動詞/ダ",
//!     buf,
//! );
//!
//! s.update_raw("まぁ良いだろう").unwrap();
//! predictor.predict(&mut s);
//! s.fill_tags();
//! s.write_tokenized_text(&mut buf);
//! assert_eq!(
//!     "まぁ/副詞/マー 良い/形容詞/ヨイ だろう/助動詞/ダロー",
//!     buf,
//! );
//! ```
//!
//! Tag prediction requires **crate feature** `tag-prediction`.
//!
//! Training requires **crate feature** `train`. For more details, see [`Trainer`].

#![deny(missing_docs)]
#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "portable-simd", feature(portable_simd))]

#[cfg(not(feature = "alloc"))]
compile_error!("`alloc` feature is currently required to build this crate");

#[macro_use]
extern crate alloc;

mod char_scorer;
mod dict_model;
mod model;
mod ngram_model;
mod predictor;
mod sentence;
mod type_scorer;
mod utils;

pub mod errors;

#[cfg(feature = "train")]
mod tag_trainer;
#[cfg(feature = "train")]
mod trainer;

#[cfg(feature = "kytea")]
mod kytea_model;

pub use dict_model::WordWeightRecord;
pub use model::Model;
pub use predictor::Predictor;
pub use sentence::{CharacterBoundary, CharacterType, Sentence, Token, TokenIterator};

#[cfg(feature = "train")]
pub use trainer::{SolverType, Trainer};

#[cfg(feature = "kytea")]
pub use kytea_model::KyteaModel;
