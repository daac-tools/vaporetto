use std::fs;
use std::path::PathBuf;

use clap::Parser;
use serde::{Deserialize, Serialize};
use vaporetto::{Model, WordWeightRecord};

#[derive(Parser, Debug)]
#[clap(
    name = "manipulate_model",
    about = "A program to manipulate tarined models."
)]
struct Args {
    /// Input path of the model file
    #[clap(long)]
    model_in: PathBuf,

    /// Output path of the model file
    #[clap(long)]
    model_out: Option<PathBuf>,

    /// Output a dictionary contained in the model.
    #[clap(long)]
    dump_dict: Option<PathBuf>,

    /// Replace a dictionary if the argument is specified.
    #[clap(long)]
    replace_dict: Option<PathBuf>,
}

#[derive(Deserialize, Serialize)]
struct WordWeightRecordFlatten {
    word: String,
    right: i32,
    inside: i32,
    left: i32,
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
            wtr.serialize(WordWeightRecordFlatten {
                word: data.get_word().to_string(),
                right: data.get_right_weight(),
                inside: data.get_inside_weight(),
                left: data.get_left_weight(),
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
            dict.push(WordWeightRecord::new(
                record.word,
                record.right,
                record.inside,
                record.left,
                record.comment,
            ));
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
