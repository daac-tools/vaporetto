use std::io::Cursor;

use js_sys::{Array, Object};
use vaporetto::{BoundaryType, CharacterType, Model, Predictor, Sentence};
use vaporetto_rules::{
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter},
    SentenceFilter,
};
use wasm_bindgen::{prelude::*, JsValue};

#[wasm_bindgen]
pub struct Vaporetto {
    predictor: Predictor,
    post_filters: Vec<Box<dyn SentenceFilter>>,
}

#[wasm_bindgen]
impl Vaporetto {
    #[wasm_bindgen]
    pub fn new() -> Self {
        let mut f =
            zstd::Decoder::new(Cursor::new(include_bytes!("../../model/kftt.model"))).unwrap();
        let model = Model::read(&mut f).unwrap();
        let predictor = Predictor::new(model);
        let post_filters: Vec<Box<dyn SentenceFilter>> = vec![
            Box::new(ConcatGraphemeClustersFilter::new()),
            Box::new(KyteaWsConstFilter::new(CharacterType::Digit)),
        ];
        Self {
            predictor,
            post_filters,
        }
    }

    #[wasm_bindgen]
    pub fn predict_partial(&self, text: &str, start: usize, end: usize) -> Object {
        let s = if let Ok(s) = Sentence::from_raw(text) {
            s
        } else {
            return JsValue::NULL.into();
        };
        if start >= end {
            return JsValue::NULL.into();
        }
        let s = self.predictor.predict_partial_with_score(s, start..end);
        let s = self
            .post_filters
            .iter()
            .fold(s, |s, filter| filter.filter(s));

        let result = Array::new();
        for (&score, &b) in s.boundary_scores().unwrap()[start..end]
            .iter()
            .zip(&s.boundaries()[start..end])
        {
            let boundary = Array::new();
            boundary.push(&JsValue::from_bool(b == BoundaryType::WordBoundary));
            boundary.push(&JsValue::from_f64(score));
            result.push(&boundary);
        }
        result.into()
    }
}

impl Default for Vaporetto {
    fn default() -> Self {
        Self::new()
    }
}
