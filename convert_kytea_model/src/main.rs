use std::convert::TryFrom;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;

use clap::Parser;
use vaporetto::{KyteaModel, Model};

#[derive(Parser, Debug)]
#[clap(
    name = "convert_kytea_model",
    about = "A program to convert KyTea model."
)]
struct Args {
    /// KyTea model file
    #[clap(long)]
    model_in: PathBuf,

    /// Vespa model file
    #[clap(long)]
    model_out: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("Loading model file...");
    let mut f = BufReader::new(fs::File::open(args.model_in).unwrap());
    let model = KyteaModel::read(&mut f)?;

    eprintln!("Saving model file...");
    let model = Model::try_from(model)?;
    let mut f = zstd::Encoder::new(fs::File::create(args.model_out)?, 19)?;
    model.write(&mut f)?;
    f.finish()?;

    Ok(())
}
