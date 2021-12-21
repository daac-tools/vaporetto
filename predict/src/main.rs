use std::fs::File;
use std::io::{prelude::*, stdin};
use std::path::PathBuf;
use std::rc::Rc;
use std::str::FromStr;
use std::time::Instant;

use structopt::StructOpt;
use vaporetto::{errors::VaporettoError, CharacterType, Model, Predictor, Sentence};
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

    /// Prints scores.
    #[structopt(long)]
    scores: bool,

    /// Do not normalize input strings before prediction.
    #[structopt(long)]
    no_norm: bool,
}

fn print_scores(s: &Sentence) {
    if let Some(scores) = s.boundary_scores().as_ref() {
        for (i, score) in scores.iter().enumerate() {
            println!("{}:{}{} {}", i, s.chars()[i], s.chars()[i + 1], score);
        }
        println!();
    }
}

fn tokenize(
    predictor: &Predictor,
    text: impl Into<String>,
    mut buf1: Sentence,
    mut buf2: Sentence,
    pre_filters: &[Box<dyn StringFilter>],
    post_filters: &[Box<dyn SentenceFilter>],
) -> Result<(String, Sentence, Sentence), VaporettoError> {
    let text = text.into();
    if pre_filters.is_empty() {
        buf1.update_raw(text)?;
    } else {
        let text_rc = Rc::new(text);
        let filt_text = Rc::try_unwrap(
            pre_filters
                .iter()
                .fold(Rc::clone(&text_rc), |s, filter| Rc::new(filter.filter(&s))),
        )
        .unwrap();
        let text = Rc::try_unwrap(text_rc).unwrap();
        buf1.update_raw(filt_text)?;
        buf2.update_raw(text)?;
    }
    buf1 = predictor.predict_with_score(buf1);
    buf1 = post_filters.iter().fold(buf1, |s, filter| filter.filter(s));
    let result = if pre_filters.is_empty() {
        buf1.to_tokenized_string()?
    } else {
        buf2.boundaries_mut().copy_from_slice(buf1.boundaries());
        buf2.to_tokenized_string()?
    };
    Ok((result, buf1, buf2))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    let mut pre_filters: Vec<Box<dyn StringFilter>> = vec![];
    if !opt.no_norm {
        pre_filters.push(Box::new(KyteaFullwidthFilter::new()));
    }
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
    let predictor = Predictor::new(model)?;

    eprintln!("Start tokenization");
    let mut n_chars = 0;
    let start = Instant::now();
    let mut buf1 = Sentence::from_raw(" ")?;
    let mut buf2 = Sentence::from_raw(" ")?;
    for line in stdin().lock().lines() {
        let line = line?;
        if line.is_empty() {
            println!();
            continue;
        }
        let ret = tokenize(&predictor, line, buf1, buf2, &pre_filters, &post_filters)?;
        let result = ret.0;
        buf1 = ret.1;
        buf2 = ret.2;
        println!("{}", result);
        if opt.scores {
            print_scores(&buf1);
        }
        n_chars += buf1.chars().len();
    }
    let duration = start.elapsed();
    eprintln!("Elapsed: {} [sec]", duration.as_secs_f64());
    eprintln!(
        "Speed: {} [chars/sec]",
        n_chars as f64 / duration.as_secs_f64()
    );

    Ok(())
}
