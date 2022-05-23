use alloc::vec::Vec;

use bincode::{Decode, Encode};

#[derive(Clone, Debug, Decode, Encode)]
pub struct NgramData<T> {
    pub(crate) ngram: T,
    pub(crate) weights: Vec<i32>,
}

#[derive(Default, Debug, Decode, Encode)]
pub struct NgramModel<T>(pub Vec<NgramData<T>>);

#[derive(Clone, Debug, Decode, Encode)]
pub struct TagNgramData<T> {
    pub(crate) ngram: T,
    pub(crate) weights: Vec<(u8, Vec<i32>)>,
}

#[derive(Default, Debug, Decode, Encode)]
pub struct TagNgramModel<T>(pub Vec<TagNgramData<T>>);
