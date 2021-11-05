use std::fs::File;
use std::io::{prelude::*, stdin};
use std::path::PathBuf;
use std::str::FromStr;

use structopt::StructOpt;
use vaporetto::{BoundaryType, CharacterType, Model, Predictor, Sentence};
use vaporetto_rules::{
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter},
    string_filters::KyteaFullwidthFilter,
    SentenceFilter, StringFilter,
};

#[derive(Debug)]
enum WsConst {
    GraphemeCluster,
    CharType(CharacterType),
}

impl FromStr for WsConst {
    type Err = &'static str;
    fn from_str(wsconst: &str) -> Result<Self, Self::Err> {
        match wsconst {
            "D" => Ok(Self::CharType(CharacterType::Digit)),
            "R" => Ok(Self::CharType(CharacterType::Roman)),
            "H" => Ok(Self::CharType(CharacterType::Hiragana)),
            "T" => Ok(Self::CharType(CharacterType::Katakana)),
            "K" => Ok(Self::CharType(CharacterType::Kanji)),
            "O" => Ok(Self::CharType(CharacterType::Other)),
            "G" => Ok(Self::GraphemeCluster),
            _ => Err("Could not parse a wsconst value"),
        }
    }
}

#[derive(StructOpt, Debug)]
#[structopt(
    name = "evaluate",
    about = "A program to evaluate the accuracy of Vaporetto."
)]
struct Opt {
    /// The model file to use when analyzing text
    #[structopt(long)]
    model: PathBuf,

    /// Do not segment some character types: {D, R, H, T, K, O, G}.
    /// D: Digit, R: Roman, H: Hiragana, T: Katakana, K: Kanji, O: Other, G: Grapheme cluster.
    #[structopt(long)]
    wsconst: Vec<WsConst>,

    /// Do not normalize input strings before prediction.
    #[structopt(long)]
    no_norm: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    let fullwidth_filter = KyteaFullwidthFilter::new();
    let mut post_filters: Vec<Box<dyn SentenceFilter>> = vec![];
    for wsconst in &opt.wsconst {
        match wsconst {
            WsConst::GraphemeCluster => {
                post_filters.push(Box::new(ConcatGraphemeClustersFilter::new()))
            }
            WsConst::CharType(char_type) => {
                post_filters.push(Box::new(KyteaWsConstFilter::new(*char_type)))
            }
        }
    }

    eprintln!("Loading model file...");
    let mut f = zstd::Decoder::new(File::open(opt.model)?)?;
    let model = Model::read(&mut f)?;
    let predictor = Predictor::new(model);

    eprintln!("Start tokenization");
    let mut n_true_positive = 0;
    let mut n_false_positive = 0;
    let mut n_false_negative = 0;

    for line in stdin().lock().lines() {
        let s = Sentence::from_tokenized(line?)?;
        let s = if opt.no_norm {
            s
        } else {
            let new_line = fullwidth_filter.filter(s.to_raw_string());
            let mut new_s = Sentence::from_raw(new_line)?;
            new_s.boundaries_mut().clone_from_slice(s.boundaries());
            new_s
        };
        let reference = s.boundaries().to_vec();
        let s = predictor.predict(s);
        let s = post_filters.iter().fold(s, |s, filter| filter.filter(s));
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
