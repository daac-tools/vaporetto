use std::fs::File;
use std::io::{prelude::*, stdin};
use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;
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

#[derive(Parser, Debug)]
#[clap(
    name = "evaluate",
    about = "A program to evaluate the accuracy of Vaporetto."
)]
struct Args {
    /// The model file to use when analyzing text
    #[clap(long)]
    model: PathBuf,

    /// Predicts POS tags.
    #[clap(long)]
    predict_tags: bool,

    /// Do not segment some character types: {D, R, H, T, K, O, G}.
    /// D: Digit, R: Roman, H: Hiragana, T: Katakana, K: Kanji, O: Other, G: Grapheme cluster.
    #[clap(long)]
    wsconst: Vec<WsConst>,

    /// Do not normalize input strings before prediction.
    #[clap(long)]
    no_norm: bool,

    /// Evaluation metric: {char, word}.
    /// char: evaluates each charactor boundary.
    /// word: evaluates each word using Nagata's method.
    #[clap(long, default_value = "char")]
    metric: EvaluationMetric,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let fullwidth_filter = KyteaFullwidthFilter;
    let mut post_filters: Vec<Box<dyn SentenceFilter>> = vec![];
    for wsconst in &args.wsconst {
        match wsconst {
            WsConst::GraphemeCluster => post_filters.push(Box::new(ConcatGraphemeClustersFilter)),
            WsConst::CharType(char_type) => {
                post_filters.push(Box::new(KyteaWsConstFilter::new(*char_type)))
            }
        }
    }

    eprintln!("Loading model file...");
    let mut f = zstd::Decoder::new(File::open(args.model)?)?;
    let model = Model::read(&mut f)?;
    let predictor = Predictor::new(model, args.predict_tags)?;

    eprintln!("Start tokenization");

    let mut results = vec![];
    for line in stdin().lock().lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let mut s = Sentence::from_tokenized(line)?;
        let ref_boundaries = s.boundaries().to_vec();
        let ref_tags = s.tags().to_vec();
        if !args.no_norm {
            let new_line = fullwidth_filter.filter(s.to_raw_string());
            s = Sentence::from_raw(new_line)?
        };
        s = predictor.predict(s);
        s = post_filters.iter().fold(s, |s, filter| filter.filter(s));
        s = predictor.fill_tags(s);
        let hyp_boundaries = s.boundaries().to_vec();
        let hyp_tags = s.tags().to_vec();
        results.push((ref_boundaries, ref_tags, hyp_boundaries, hyp_tags));
    }

    match args.metric {
        EvaluationMetric::CharBoundaryAccuracy => {
            let mut n_tp = 0;
            let mut n_tn = 0;
            let mut n_fp = 0;
            let mut n_fn = 0;
            for (rs_b, _, hs_b, _) in results {
                for (r, h) in rs_b.into_iter().zip(hs_b) {
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
            for (rs_b, rs_t, hs_b, hs_t) in results {
                let mut matched = true;
                for (((r_b, r_t), h_b), h_t) in rs_b.iter().zip(&rs_t).zip(&hs_b).zip(&hs_t) {
                    if r_b == h_b {
                        if *h_b == BoundaryType::WordBoundary {
                            if matched && r_t == h_t {
                                n_cor += 1;
                            }
                            matched = true;
                            n_ref += 1;
                            n_sys += 1;
                        }
                    } else {
                        if *h_b == BoundaryType::WordBoundary {
                            n_sys += 1;
                        } else {
                            n_ref += 1;
                        }
                        matched = false;
                    }
                }
                if matched && rs_t.last().unwrap() == hs_t.last().unwrap() {
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
