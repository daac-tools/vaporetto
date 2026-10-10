use alloc::vec::Vec;

use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NgramData<T> {
    pub(crate) ngram: T,
    pub(crate) weights: Vec<i32>,
}

#[derive(Default, Debug, Deserialize, Serialize)]
pub struct NgramModel<T: 'static>(pub Vec<NgramData<T>>);

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TagWeight {
    pub(crate) rel_position: u8,
    pub(crate) weights: Vec<i32>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct TagNgramData<T> {
    pub(crate) ngram: T,
    pub(crate) weights: Vec<TagWeight>,
}

#[derive(Default, Debug, Deserialize, Serialize)]
pub struct TagNgramModel<T: 'static>(pub Vec<TagNgramData<T>>);
