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

#[derive(Debug)]
enum EvaluationMetric {
    CharBoundaryAccuracy,
    WordAccuracy,
}

impl FromStr for EvaluationMetric {
    type Err = &'static str;
    fn from_str(metric: &str) -> Result<Self, Self::Err> {
        match metric {
            "char" => Ok(Self::CharBoundaryAccuracy),
            "word" => Ok(Self::WordAccuracy),
            _ => Err("Could not parse a metric value"),
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

    /// Evaluation metric: {char, word}.
    /// char: evaluates each charactor boundary.
    /// word: evaluates each word using Nagata's method.
    #[structopt(long, default_value = "char")]
    metric: EvaluationMetric,
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
    let predictor = Predictor::new(model)?;

    eprintln!("Start tokenization");

    let mut results = vec![];
    for line in stdin().lock().lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let s = Sentence::from_tokenized(line)?;
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
        results.push((reference, s.boundaries().to_vec()));
    }

    match opt.metric {
        EvaluationMetric::CharBoundaryAccuracy => {
            let mut n_tp = 0;
            let mut n_tn = 0;
            let mut n_fp = 0;
            let mut n_fn = 0;
            for (rs, hs) in results {
                for (r, h) in rs.into_iter().zip(hs) {
                    if r == h {
                        if h == BoundaryType::WordBoundary {
                            n_tp += 1;
                        } else {
                            n_tn += 1;
                        }
                    } else if h == BoundaryType::WordBoundary {
                        n_fp += 1;
                    } else {
                        n_fn += 1;
                    }
                }
            }
            let precision = n_tp as f64 / (n_tp + n_fp) as f64;
            let recall = n_tp as f64 / (n_tp + n_fn) as f64;
            let f1 = 2. * precision * recall / (precision + recall);
            println!("Precision: {}", precision);
            println!("Recall: {}", recall);
            println!("F1: {}", f1);
            println!("TP: {}, TN: {}, FP: {}, FN: {}", n_tp, n_tn, n_fp, n_fn);
        }
        EvaluationMetric::WordAccuracy => {
            // Reference:
            // Masaaki Nagata. 1994. A stochastic Japanese morphological analyzer using a forward-DP
            // backward-A* n-best search algorithm. In COLING 1994 Volume 1: The 15th International
            // Conference on Computational Linguistics.
            let mut n_sys = 0;
            let mut n_ref = 0;
            let mut n_cor = 0;
            for (rs, hs) in results {
                let mut matched = true;
                for (r, h) in rs.into_iter().zip(hs) {
                    if r == h {
                        if h == BoundaryType::WordBoundary {
                            if matched {
                                n_cor += 1;
                            }
                            matched = true;
                            n_ref += 1;
                            n_sys += 1;
                        }
                    } else {
                        if h == BoundaryType::WordBoundary {
                            n_sys += 1;
                        } else {
                            n_ref += 1;
                        }
                        matched = false;
                    }
                }
                if matched {
                    n_cor += 1;
                }
                n_sys += 1;
                n_ref += 1;
            }
            let precision = n_cor as f64 / n_sys as f64;
            let recall = n_cor as f64 / n_ref as f64;
            let f1 = 2. * precision * recall / (precision + recall);
            println!("Precision: {}", precision);
            println!("Recall: {}", recall);
            println!("F1: {}", f1);
        }
    }

    Ok(())
}
