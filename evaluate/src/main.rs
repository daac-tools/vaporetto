use std::fs::File;
use std::io::{prelude::*, stdin, BufReader};
use std::path::PathBuf;

use structopt::StructOpt;
use vaporetto::{BoundaryType, Model, Predictor, Sentence};

#[derive(StructOpt, Debug)]
#[structopt(
    name = "evaluate",
    about = "A program to evaluate the accuracy of Vaporetto."
)]
struct Opt {
    /// The model file to use when analyzing text
    #[structopt(long)]
    model: PathBuf,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    eprintln!("Loading model file...");
    let mut f = BufReader::new(File::open(opt.model).unwrap());
    let model = Model::read(&mut f)?;
    let predictor = Predictor::new(model);

    eprintln!("Start tokenization");
    let mut n_true_positive = 0;
    let mut n_false_positive = 0;
    let mut n_false_negative = 0;

    for line in stdin().lock().lines() {
        let s = Sentence::from_tokenized(line?)?;
        let reference = s.boundaries().to_vec();
        let s = predictor.predict(s);
        for (&r, &h) in reference.iter().zip(s.boundaries()) {
            if r == h {
                if h == BoundaryType::WordBoundary {
                    n_true_positive += 1;
                }
            } else if h == BoundaryType::WordBoundary {
                n_false_positive += 1;
            } else {
                n_false_negative += 1;
            }
        }
    }

    let precision = n_true_positive as f64 / (n_true_positive + n_false_positive) as f64;
    let recall = n_true_positive as f64 / (n_true_positive + n_false_negative) as f64;
    let f1 = 2. * precision * recall / (precision + recall);
    println!("Precision: {}", precision);
    println!("Recall: {}", recall);
    println!("F1: {}", f1);

    Ok(())
}
