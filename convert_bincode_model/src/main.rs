//! Converts a model file created by Vaporetto 0.6 or earlier (serialized with bincode) into the
//! current format (serialized with postcard).
//!
//! The structs defined in this file mirror the internal model representation of Vaporetto.
//! bincode and postcard both encode structs as a sequence of their fields, so the field order of
//! these structs must match the internal one.

use std::fs;
use std::io::Read;
use std::path::PathBuf;

use bincode::Decode;
use clap::Parser;
use serde::Serialize;
use vaporetto::Model;

/// Magic number of models serialized with bincode.
const BINCODE_MODEL_MAGIC: &[u8] = b"VaporettoTokenizer 0.5.0\n";

/// Magic number of models serialized with postcard.
const POSTCARD_MODEL_MAGIC: &[u8] = b"VaporettoTokenizer 0.7.0\n";

#[derive(Decode, Serialize)]
struct NgramData<T> {
    ngram: T,
    weights: Vec<i32>,
}

#[derive(Decode, Serialize)]
struct TagWeight {
    rel_position: u8,
    weights: Vec<i32>,
}

#[derive(Decode, Serialize)]
struct TagNgramData<T> {
    ngram: T,
    weights: Vec<TagWeight>,
}

#[derive(Decode, Serialize)]
struct WordWeightRecord {
    word: String,
    weights: Vec<i32>,
    comment: String,
}

#[derive(Decode, Serialize)]
struct TagModel {
    token: String,
    tags: Vec<Vec<String>>,
    char_ngram_model: Vec<TagNgramData<String>>,
    type_ngram_model: Vec<TagNgramData<Vec<u8>>>,
    bias: Vec<i32>,
}

#[derive(Decode, Serialize)]
struct ModelData {
    char_ngram_model: Vec<NgramData<String>>,
    type_ngram_model: Vec<NgramData<Vec<u8>>>,
    dict_model: Vec<WordWeightRecord>,
    bias: i32,
    char_window_size: u8,
    type_window_size: u8,
    tag_models: Vec<TagModel>,
}

#[derive(Parser, Debug)]
#[clap(
    name = "convert_bincode_model",
    about = "A program to convert a model created by Vaporetto 0.6 or earlier into the current format."
)]
struct Args {
    /// Model file created by Vaporetto 0.6 or earlier
    #[arg(long)]
    model_in: PathBuf,

    /// Vaporetto model file
    #[arg(long)]
    model_out: PathBuf,

    /// The number of workers for zstd (0 means multithreaded will be disabled)
    #[arg(long, default_value = "0")]
    zstd_workers: u32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("Loading model file...");
    let mut f = zstd::Decoder::new(fs::File::open(args.model_in)?)?;
    let mut data = vec![];
    f.read_to_end(&mut data)?;
    let data = data
        .strip_prefix(BINCODE_MODEL_MAGIC)
        .ok_or("the input is not a model created by Vaporetto 0.6 or earlier")?;
    let (model_data, size): (ModelData, _) =
        bincode::decode_from_slice(data, bincode::config::standard())?;
    if size != data.len() {
        return Err("the input model contains trailing data".into());
    }

    eprintln!("Converting...");
    let model_data = postcard::to_extend(&model_data, POSTCARD_MODEL_MAGIC.to_vec())?;

    // Verifies that the current version of Vaporetto can read the converted model.
    let (model, rest) = Model::read_slice(&model_data)?;
    if !rest.is_empty() {
        return Err("failed to convert the model".into());
    }

    eprintln!("Saving model file...");
    let mut f = zstd::Encoder::new(fs::File::create(args.model_out)?, 19)?;
    f.multithread(args.zstd_workers)?;
    model.write(&mut f)?;
    f.finish()?;

    Ok(())
}
