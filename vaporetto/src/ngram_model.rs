use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Clone, Serialize, Deserialize)]
pub struct NgramData<T>
where
    T: Clone,
{
    pub(crate) ngram: T,
    pub(crate) weights: Vec<i32>,
}

#[derive(Serialize, Deserialize)]
pub struct NgramModel<T>
where
    T: Clone,
{
    pub(crate) data: Vec<NgramData<T>>,
    merged: bool,
}

impl<T> NgramModel<T>
where
    T: AsRef<[u8]> + Clone,
{
    #[cfg(any(feature = "train", feature = "kytea", test))]
    pub fn new(data: Vec<NgramData<T>>) -> Self {
        Self {
            data,
            merged: false,
        }
    }

    pub fn merge_weights(&mut self) {
        if self.merged {
            return;
        }
        self.merged = true;
        let ngrams = self
            .data
            .iter()
            .cloned()
            .map(|d| (d.ngram.as_ref().to_vec(), d.weights))
            .collect::<HashMap<_, _>>();
        for NgramData { ngram, weights } in &mut self.data {
            let ngram = ngram.as_ref();
            let mut new_weights: Option<Vec<_>> = None;
            for st in (0..ngram.len()).rev() {
                if let Some(weights) = ngrams.get(&ngram[st..]) {
                    if let Some(new_weights) = new_weights.as_mut() {
                        for (w_new, w) in new_weights.iter_mut().zip(weights) {
                            *w_new += *w;
                        }
                    } else {
                        new_weights.replace(weights.clone());
                    }
                }
            }
            *weights = new_weights.unwrap();
        }
    }
}
