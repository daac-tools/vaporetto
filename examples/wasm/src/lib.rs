use std::io::{Cursor, Read};

use js_sys::{Array, Object};
use vaporetto::{CharacterBoundary, CharacterType, Model, Predictor, Sentence};
use vaporetto_rules::{
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter},
    string_filters::KyteaFullwidthFilter,
    SentenceFilter, StringFilter,
};
use wasm_bindgen::{prelude::*, JsValue};

#[global_allocator]
static ALLOC: wee_alloc::WeeAlloc = wee_alloc::WeeAlloc::INIT;

#[wasm_bindgen]
pub struct Vaporetto {
    predictor: Predictor,
    fullwidth_filter: KyteaFullwidthFilter,
    post_filters: Vec<Box<dyn SentenceFilter>>,
}

#[wasm_bindgen]
impl Vaporetto {
    #[wasm_bindgen]
    pub fn new(filters: &str) -> Self {
        let mut f = Cursor::new(include_bytes!(env!("VAPORETTO_MODEL_PATH")));
        let mut decoder = ruzstd::StreamingDecoder::new(&mut f).unwrap();
        let mut buff = vec![];
        decoder.read_to_end(&mut buff).unwrap();
        let model = Model::read(&mut buff.as_slice()).unwrap();
        let predictor = Predictor::new(model, false).unwrap();
        let post_filters: Vec<_> = filters
            .chars()
            .map(|c| {
                let b: Box<dyn SentenceFilter> = match c {
                    'D' => Box::new(KyteaWsConstFilter::new(CharacterType::Digit)),
                    'R' => Box::new(KyteaWsConstFilter::new(CharacterType::Roman)),
                    'H' => Box::new(KyteaWsConstFilter::new(CharacterType::Hiragana)),
                    'T' => Box::new(KyteaWsConstFilter::new(CharacterType::Katakana)),
                    'K' => Box::new(KyteaWsConstFilter::new(CharacterType::Kanji)),
                    'O' => Box::new(KyteaWsConstFilter::new(CharacterType::Other)),
                    'G' => Box::new(ConcatGraphemeClustersFilter),
                    _ => panic!("invalid filter: {}", c),
                };
                b
            })
            .collect();
        Self {
            predictor,
            fullwidth_filter: KyteaFullwidthFilter,
            post_filters,
        }
    }

    #[wasm_bindgen]
    pub fn tokenize(&self, text: &str) -> Object {
        let result = Array::new();
        let mut s = if let Ok(s) = Sentence::from_raw(text) {
            s
        } else {
            return result.into();
        };
        let norm = self.fullwidth_filter.filter(text);
        let mut s_norm = if let Ok(s) = Sentence::from_raw(norm) {
            s
        } else {
            return result.into();
        };
        self.predictor.predict(&mut s_norm);
        s.boundaries_mut().clone_from_slice(s_norm.boundaries());
        self.post_filters
            .iter()
            .for_each(|filter| filter.filter(&mut s));

        for token in s.iter_tokens() {
            result.push(&JsValue::from_str(token.surface()));
        }
        result.into()
    }

    #[wasm_bindgen]
    pub fn predict(&self, text: &str) -> Object {
        let result = Array::new();
        let text = self.fullwidth_filter.filter(text);
        let mut s = if let Ok(s) = Sentence::from_raw(text) {
            s
        } else {
            return result.into();
        };
        self.predictor.predict(&mut s);
        self.post_filters
            .iter()
            .for_each(|filter| filter.filter(&mut s));

        for (&score, &b) in s.boundary_scores().iter().zip(s.boundaries()) {
            let boundary = Array::new();
            boundary.push(&(b == CharacterBoundary::WordBoundary).into());
            boundary.push(&score.into());
            result.push(&boundary);
        }
        result.into()
    }
}
