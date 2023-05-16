use std::fs;
use std::path::PathBuf;

use clap::Parser;
use serde::{Deserialize, Serialize};
use vaporetto::{Model, WordWeightRecord};

#[derive(Parser, Debug)]
#[command(about = "A program to manipulate tarined models.")]
struct Args {
    /// Input path of the model file
    #[arg(long)]
    model_in: PathBuf,

    /// Output path of the model file
    #[arg(long)]
    model_out: Option<PathBuf>,

    /// Output a dictionary contained in the model.
    #[arg(long)]
    dump_dict: Option<PathBuf>,

    /// Replace a dictionary if the argument is specified.
    #[arg(long)]
    replace_dict: Option<PathBuf>,
}

#[derive(Deserialize, Serialize)]
struct WordWeightRecordFlatten {
    word: String,
    weights: String,
    comment: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("Loading model file...");
    let mut f = zstd::Decoder::new(fs::File::open(args.model_in)?)?;
    let mut model = Model::read(&mut f)?;

    if let Some(path) = args.dump_dict {
        eprintln!("Saving dictionary file...");
        let file = fs::File::create(path)?;
        let mut wtr = csv::Writer::from_writer(file);
        for data in model.dictionary() {
            let str_weights: Vec<_> = data.get_weights().iter().map(|w| w.to_string()).collect();
            wtr.serialize(WordWeightRecordFlatten {
                word: data.get_word().to_string(),
                weights: str_weights.join(" "),
                comment: data.get_comment().to_string(),
            })?;
        }
    }

    if let Some(path) = args.replace_dict {
        eprintln!("Loading dictionary file...");
        let file = fs::File::open(path)?;
        let mut rdr = csv::Reader::from_reader(file);
        let mut dict = vec![];
        for result in rdr.deserialize() {
            let record: WordWeightRecordFlatten = result?;
            let mut weights = vec![];
            for w in record.weights.split(' ') {
                weights.push(w.parse()?);
            }
            dict.push(WordWeightRecord::new(record.word, weights, record.comment)?);
        }
        model.replace_dictionary(dict);
    }

    if let Some(path) = args.model_out {
        eprintln!("Saving model file...");
        let mut f = zstd::Encoder::new(fs::File::create(path)?, 19)?;
        model.write(&mut f)?;
        f.finish()?;
    }

    Ok(())
}
