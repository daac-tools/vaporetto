use bincode::{Decode, Encode};

#[derive(Clone, Decode, Encode)]
pub struct NgramData<T> {
    pub(crate) ngram: T,
    pub(crate) weights: Vec<i32>,
}

#[derive(Default, Decode, Encode)]
pub struct NgramModel<T> {
    pub(crate) data: Vec<NgramData<T>>,
}
