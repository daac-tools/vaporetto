use std::env;
use std::fs::File;
use std::io::{BufWriter, Cursor, Read, Write};
use std::path::PathBuf;

use vaporetto::{Model, Predictor};

fn main() {
    let out = &PathBuf::from(env::var_os("OUT_DIR").unwrap());
    File::create(out.join("memory.x"))
        .unwrap()
        .write_all(include_bytes!("memory.x"))
        .unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");

    let mut f = Cursor::new(include_bytes!(env!("VAPORETTO_MODEL_PATH")));
    let mut decoder = ruzstd::StreamingDecoder::new(&mut f).unwrap();
    let mut buff = vec![];
    decoder.read_to_end(&mut buff).unwrap();
    let (model, _) = Model::read_slice(&buff).unwrap();
    let predictor = Predictor::new(model).unwrap();
    let mut buf = BufWriter::new(File::create(out.join("predictor.bin")).unwrap());
    let model_data = predictor.serialize_to_vec().unwrap();
    buf.write_all(&model_data).unwrap();
}
