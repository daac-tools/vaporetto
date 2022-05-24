//! # Vaporetto
//!
//! Vaporetto is a fast and lightweight pointwise prediction based tokenizer.
//!
//! ## Examples
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::Read;
//!
//! use vaporetto::{Model, Predictor, Sentence};
//!
//! let mut f = File::open("model.bin").unwrap();
//! let mut model_data = vec![];
//! f.read_to_end(&mut model_data).unwrap();
//! let (model, _) = Model::read_slice(&model_data).unwrap();
//! let predictor = Predictor::new(model, false).unwrap();
//!
//! let mut s = Sentence::from_raw("火星猫の生態").unwrap();
//! predictor.predict(&mut s);
//!
//! let mut buf = String::new();
//! s.write_tokenized_text(&mut buf);
//! println!("{}", buf);
//! ```
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
