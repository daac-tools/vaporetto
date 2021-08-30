# vaporetto_rules

Vaporetto is a fast and lightweight pointwise prediction based tokenizer.
vaporetto_rules is rule-base filters for Vaporetto.

## Examples

```rust
use std::fs::File;
use std::io::BufReader;

use vaporetto::{CharacterType, Model, Predictor, Sentence};
use vaporetto_rules::{
    SentenceFilter, StringFilter,
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter},
    string_filters::KyteaFullwidthFilter,
};

let mut f = BufReader::new(File::open("model.bin").unwrap());
let model = Model::read(&mut f).unwrap();
let mut predictor = Predictor::new(model);

let pre_filters: Vec<Box<dyn StringFilter<String>>> = vec![
    Box::new(KyteaFullwidthFilter::new()),
];
let post_filters: Vec<Box<dyn SentenceFilter>> = vec![
    Box::new(ConcatGraphemeClustersFilter::new()),
    Box::new(KyteaWsConstFilter::new(CharacterType::Digit)),
];

let input = "Vaporettoは仲良し家族👨‍👨‍👧‍👦を離れ離れにさせません。"
    .to_string();

let preproc_input = pre_filters.iter().fold(input, |s, filter| filter.filter(s));

let sentence = Sentence::from_raw(preproc_input).unwrap();
let sentence = predictor.predict(sentence);

let postproc_result = post_filters.iter().fold(sentence, |s, filter| filter.filter(s));

assert_eq!(
    "Ｖａｐｏｒｅｔｔｏ は 仲良 し 家族 👨‍👨‍👧‍👦 を 離れ離れ に さ せ ま せ ん 。",
    postproc_result.to_tokenized_string().unwrap(),
);
```

## License

Licensed under either of

 * Apache License, Version 2.0
   ([LICENSE-APACHE](LICENSE-APACHE) or http://www.apache.org/licenses/LICENSE-2.0)
 * MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.
