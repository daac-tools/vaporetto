#![cfg_attr(docsrs, feature(doc_cfg))]
#![cfg_attr(feature = "simd", feature(portable_simd))]

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
//! let predictor = Predictor::new(model);
//!
//! for line in stdin().lock().lines() {
//!     let s = Sentence::from_raw(line.unwrap()).unwrap();
//!     let s = predictor.predict(s);
//!     let toks = s.to_tokenized_string().unwrap();
//!     println!("{}", toks);
//! }
//! ```
//!
//! Training requires **crate feature** `train`. For more details, see [`Trainer`].

#[macro_use]
mod utils;

mod char_scorer;
mod dict_scorer;
mod model;
mod predictor;
mod sentence;
mod type_scorer;

#[cfg(feature = "train")]
mod feature;
#[cfg(feature = "train")]
mod trainer;

#[cfg(feature = "kytea")]
mod kytea_model;

pub use model::Model;
pub use predictor::Predictor;
pub use sentence::{BoundaryType, CharacterType, Sentence};

#[cfg(feature = "train")]
pub use trainer::{Dataset, SolverType, Trainer};

#[cfg(feature = "kytea")]
pub use kytea_model::KyteaModel;
