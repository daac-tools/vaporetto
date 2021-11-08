use std::fs::File;
use std::io::{prelude::*, stdin};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use structopt::StructOpt;
use vaporetto::{CharacterType, Model, Predictor, Sentence};
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
#[structopt(name = "predict", about = "A program to perform word segmentation.")]
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
    let mut n_boundaries = 0;
    let start = Instant::now();
    for line in stdin().lock().lines() {
        let line = line?;
        let s = if opt.no_norm {
            let s = Sentence::from_raw(line)?;
            predictor.predict(s)
        } else {
            let norm = fullwidth_filter.filter(&line);
            let mut s_orig = Sentence::from_raw(line)?;
            let s = Sentence::from_raw(norm)?;
            let s = predictor.predict(s);
            s_orig.boundaries_mut().clone_from_slice(s.boundaries());
            s_orig
        };
        let s = post_filters.iter().fold(s, |s, filter| filter.filter(s));
        n_boundaries += s.boundaries().len();
        let toks = s.to_tokenized_string()?;
        println!("{}", toks);
    }
    let duration = start.elapsed();
    eprintln!("Elapsed: {} [sec]", duration.as_secs_f64());
    eprintln!(
        "Speed: {} [boundaries/sec]",
        n_boundaries as f64 / duration.as_secs_f64()
    );

    Ok(())
}
