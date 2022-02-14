# vaporetto_tantivy

Vaporetto is a fast and lightweight pointwise prediction based tokenizer.
vaporetto_tantivy is a crate to use Vaporetto in [Tantivy](https://github.com/quickwit-oss/tantivy).

# Example

```rust
use std::fs::File;
use std::io::{Read, BufReader};

use tantivy::schema::{IndexRecordOption, Schema, TextFieldIndexing, TextOptions};
use tantivy::Index;
use vaporetto::Model;
use vaporetto_tantivy::VaporettoTokenizer;

let mut schema_builder = Schema::builder();
let text_field_indexing = TextFieldIndexing::default()
    .set_tokenizer("ja_vaporetto")
    .set_index_option(IndexRecordOption::WithFreqsAndPositions);
let text_options = TextOptions::default()
    .set_indexing_options(text_field_indexing)
    .set_stored();
schema_builder.add_text_field("title", text_options);
let schema = schema_builder.build();
let index = Index::create_in_ram(schema);

// Loads a model with decompression.
let mut f = BufReader::new(File::open("bccwj-suw+unidic.model.zst").unwrap());
let mut decoder = ruzstd::StreamingDecoder::new(&mut f).unwrap();
let mut buff = vec![];
decoder.read_to_end(&mut buff).unwrap();
let model = Model::read(&mut buff.as_slice()).unwrap();

// Creates VaporettoTokenizer with wsconst=DGR.
let tokenizer = VaporettoTokenizer::new(model, "DGR").unwrap();
index
    .tokenizers()
    .register("ja_vaporetto", tokenizer);
```
