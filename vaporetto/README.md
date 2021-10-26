# Vaporetto

Vaporetto is a fast and lightweight pointwise prediction based tokenizer.

## Examples

```rust
use std::fs::File;
use std::io::{prelude::*, stdin, BufReader};

use vaporetto::{Model, Predictor, Sentence};

let mut f = BufReader::new(File::open("model.raw").unwrap());
let model = Model::read(&mut f).unwrap();
let predictor = Predictor::new(model);

for line in stdin().lock().lines() {
    let s = Sentence::from_raw(line.unwrap()).unwrap();
    let s = predictor.predict(s);
    let toks = s.to_tokenized_string().unwrap();
    println!("{}", toks);
}
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
