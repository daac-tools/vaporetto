//! # Vaporetto
//!
//! Vaporetto is a fast and lightweight pointwise prediction based tokenizer.

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
pub use predictor::{MultithreadPredictor, Predictor};
pub use sentence::{BoundaryType, CharacterType, Sentence};

#[cfg(feature = "train")]
pub use trainer::{Dataset, Trainer};

#[cfg(feature = "kytea")]
pub use kytea_model::KyteaModel;
