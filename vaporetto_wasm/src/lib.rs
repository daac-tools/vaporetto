use std::io::Cursor;

use js_sys::{Array, Object};
use vaporetto::{BoundaryType, Model, Predictor, Sentence};
use wasm_bindgen::{prelude::*, JsValue};

#[wasm_bindgen]
pub struct Vaporetto {
    predictor: Predictor,
}

#[wasm_bindgen]
impl Vaporetto {
    #[wasm_bindgen]
    pub fn new() -> Self {
        let mut f = Cursor::new(include_bytes!("../../model/kftt.model"));
        let model = Model::read(&mut f).unwrap();
        let predictor = Predictor::new(model);
        Self { predictor }
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
        let mut s = self.predictor.predict_partial_with_score(s, start..end);

        let mut str_to_char_pos = vec![0; text.len() + 1];
        let mut utf8_pos = 0;
        let mut char_len = 0;
        for (i, c) in text.chars().enumerate() {
            str_to_char_pos[utf8_pos] = i;
            utf8_pos += c.len_utf8();
            char_len = i;
        }
        *str_to_char_pos.last_mut().unwrap() = char_len + 1;
        for (i, c) in unicode_segmentation::UnicodeSegmentation::grapheme_indices(text, true) {
            for j in str_to_char_pos[i]..str_to_char_pos[i + c.len()] - 1 {
                s.boundaries_mut()[j] = BoundaryType::NotWordBoundary;
            }
        }
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
