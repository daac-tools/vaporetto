//! # Vaporetto
//!
//! Vaporetto is a fast and lightweight pointwise prediction based tokenizer.
//!
//! ## Example of training
//!
//! ```no_run
//! use std::fs::File;
//! use std::io::{prelude::*, BufReader, BufWriter};
//!
//! use vaporetto::{Dataset, Sentence, Trainer};
//!
//! let mut train_sents = vec![];
//! let f = BufReader::new(File::open("dataset-train.txt").unwrap());
//! for (i, line) in f.lines().enumerate() {
//!     train_sents.push(Sentence::from_tokenized(line.unwrap()).unwrap());
//! }
//!
//! let dict: Vec<String> = vec![];
//! let mut dataset = Dataset::new(3, 3, 3, 3, &dict, 0).unwrap();
//! for (i, s) in train_sents.iter().enumerate() {
//!     dataset.push_sentence(s);
//! }
//!
//! let trainer = Trainer::new(0.01, 1., 1.);
//! let model = trainer.train(dataset).unwrap();
//! let mut f = BufWriter::new(File::create("model.bin").unwrap());
//! model.write(&mut f).unwrap();
//! ```
//!
//! ## Example of prediction
//! ```no_run
//! use std::fs::File;
//! use std::io::{prelude::*, stdin, BufReader};
//!
//! use vaporetto::{Model, Predictor, Sentence};
//!
//! let mut f = BufReader::new(File::open("model.bin").unwrap());
//! let model = Model::read(&mut f).unwrap();
//! let mut predictor = Predictor::new(model, true).dict_overwrap_size(3);
//!
//! for line in stdin().lock().lines() {
//!     let s = Sentence::from_raw(line.unwrap()).unwrap();
//!     let s = predictor.predict(s);
//!     let toks = s.to_tokenized_string().unwrap();
//!     println!("{}", toks);
//! }
//! ```

#[macro_use]
mod utils;

mod model;
mod predictor;
mod sentence;

#[cfg(feature = "train")]
mod feature;
#[cfg(feature = "train")]
mod trainer;

#[cfg(feature = "kytea")]
mod kytea_model;

pub use model::Model;
pub use predictor::Predictor;
pub use sentence::{BoundaryType, CharacterType, Sentence};

#[cfg(feature = "multithreading")]
pub use predictor::MultithreadPredictor;

#[cfg(feature = "train")]
pub use trainer::{Dataset, Trainer};

#[cfg(feature = "kytea")]
pub use kytea_model::KyteaModel;
