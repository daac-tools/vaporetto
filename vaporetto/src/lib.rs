#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(feature = "portable-simd", feature(portable_simd))]

//! # Vaporetto
//!
//! Vaporetto is a fast and lightweight pointwise prediction based tokenizer.
//!
//! ## Examples
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::{prelude::*, stdin, BufReader};
//!
//! use vaporetto::{Model, Predictor, Sentence};
//!
//! let mut f = BufReader::new(File::open("model.bin").unwrap());
//! let model = Model::read(&mut f).unwrap();
//! let predictor = Predictor::new(model).unwrap();
//!
//! let s = Sentence::from_raw("火星猫の生態").unwrap();
//! let s = predictor.predict(s);
//!
//! println!("{:?}", s.to_tokenized_vec().unwrap());
//! ```
//!
//! Training requires **crate feature** `train`. For more details, see [`Trainer`].

mod char_scorer;
mod dict_model;
mod model;
mod ngram_model;
mod predictor;
mod sentence;
mod tag_model;
mod type_scorer;

pub mod errors;

#[cfg(feature = "train")]
mod feature;
#[cfg(feature = "train")]
mod tag_trainer;
#[cfg(feature = "train")]
mod trainer;

#[cfg(feature = "kytea")]
mod kytea_model;

pub use dict_model::WordWeightRecord;
pub use model::Model;
pub use predictor::Predictor;
pub use sentence::{BoundaryType, CharacterType, Sentence, Token};

#[cfg(feature = "train")]
pub use trainer::{SolverType, Trainer};

#[cfg(feature = "kytea")]
pub use kytea_model::KyteaModel;
