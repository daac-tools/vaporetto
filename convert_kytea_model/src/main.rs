use std::convert::TryFrom;
use std::fs;
use std::io::BufReader;
use std::path::PathBuf;

use structopt::StructOpt;
use vaporetto::{KyteaModel, Model};

#[derive(StructOpt, Debug)]
#[structopt(
    name = "convert_kytea_model",
    about = "A program to convert KyTea model."
)]
struct Opt {
    /// KyTea model file
    #[structopt(long)]
    model_in: PathBuf,

    /// Vespa model file
    #[structopt(long)]
    model_out: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    eprintln!("Loading model file...");
    let mut f = BufReader::new(fs::File::open(opt.model_in).unwrap());
    let model = KyteaModel::read(&mut f)?;

    eprintln!("Saving model file...");
    let model = Model::try_from(model)?;
    let mut f = zstd::Encoder::new(fs::File::create(opt.model_out)?, 19)?;
    model.write(&mut f)?;
    f.finish()?;

    Ok(())
}
