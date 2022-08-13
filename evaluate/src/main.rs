use std::fs::File;
use std::io::{prelude::*, stdin};
use std::path::PathBuf;
use std::str::FromStr;

use clap::Parser;
use vaporetto::{CharacterBoundary, CharacterType, Model, Predictor, Sentence};
use vaporetto_rules::{
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter, PatternMatchTagger},
    string_filters::KyteaFullwidthFilter,
    SentenceFilter, StringFilter,
};

#[derive(Clone, Debug)]
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

#[derive(clap::ValueEnum, Clone, Debug)]
enum EvaluationMetric {
    Char,
    Word,
}

#[derive(Parser, Debug)]
#[clap(
    name = "evaluate",
    about = "A program to evaluate the accuracy of Vaporetto."
)]
struct Args {
    /// The model file to use when analyzing text
    #[clap(long, action)]
    model: PathBuf,

    /// Predicts POS tags.
    #[clap(long, action)]
    predict_tags: bool,

    /// Do not segment some character types: {D, R, H, T, K, O, G}.
    /// D: Digit, R: Roman, H: Hiragana, T: Katakana, K: Kanji, O: Other, G: Grapheme cluster.
    #[clap(long, action)]
    wsconst: Vec<WsConst>,

    /// Do not normalize input strings before prediction.
    #[clap(long, action)]
    no_norm: bool,

    /// Evaluation metric: {char, word}.
    /// char: evaluates each charactor boundary.
    /// word: evaluates each word using Nagata's method.
    #[clap(long, action, default_value = "char")]
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
    let word_tag_map: Vec<(String, Vec<Option<String>>)> = if args.predict_tags {
        let config = bincode::config::standard();
        bincode::decode_from_std_read(&mut f, config).unwrap_or_else(|_| vec![])
    } else {
        vec![]
    };
    let pattern_match_tagger = (!word_tag_map.is_empty())
        .then(|| PatternMatchTagger::new(word_tag_map.into_iter().collect()));

    eprintln!("Start tokenization");

    let mut results = vec![];
    #[allow(clippy::significant_drop_in_scrutinee)]
    for line in stdin().lock().lines() {
        let line = line?;
        if line.is_empty() {
            continue;
        }
        let mut s = Sentence::from_tokenized(&line)?;
        let ref_boundaries = s.boundaries().to_vec();
        let mut ref_tags = vec![];
        for i in 0..=ref_boundaries.len() {
            ref_tags.push(s.tags()[i * s.n_tags()..(i + 1) * s.n_tags()].to_vec());
        }
        if !args.no_norm {
            let new_line = fullwidth_filter.filter(s.as_raw_text());
            s = Sentence::from_raw(new_line)?
        };
        predictor.predict(&mut s);
        post_filters.iter().for_each(|filter| filter.filter(&mut s));
        if args.predict_tags {
            s.fill_tags();
            if let Some(tagger) = pattern_match_tagger.as_ref() {
                tagger.filter(&mut s);
            }
        }
        let sys_boundaries = s.boundaries().to_vec();
        let mut sys_tags = vec![];
        for i in 0..=sys_boundaries.len() {
            sys_tags.push(s.tags()[i * s.n_tags()..(i + 1) * s.n_tags()].to_vec());
        }
        results.push((ref_boundaries, ref_tags, sys_boundaries, sys_tags));
    }

    match args.metric {
        EvaluationMetric::Char => {
            let mut n_tp = 0;
            let mut n_tn = 0;
            let mut n_fp = 0;
            let mut n_fn = 0;
            for (rs_b, _, hs_b, _) in results {
                for (r, h) in rs_b.into_iter().zip(hs_b) {
                    if r == h {
                        if h == CharacterBoundary::WordBoundary {
                            n_tp += 1;
                        } else {
                            n_tn += 1;
                        }
                    } else if h == CharacterBoundary::WordBoundary {
                        n_fp += 1;
                    } else {
                        n_fn += 1;
                    }
                }
            }
            let precision = f64::from(n_tp) / f64::from(n_tp + n_fp);
            let recall = f64::from(n_tp) / f64::from(n_tp + n_fn);
            let f1 = 2. * precision * recall / (precision + recall);
            println!("Precision: {}", precision);
            println!("Recall: {}", recall);
            println!("F1: {}", f1);
            println!("TP: {}, TN: {}, FP: {}, FN: {}", n_tp, n_tn, n_fp, n_fn);
        }
        EvaluationMetric::Word => {
            // Reference:
            // Masaaki Nagata. 1994. A stochastic Japanese morphological analyzer using a forward-DP
            // backward-A* n-best search algorithm. In COLING 1994 Volume 1: The 15th International
            // Conference on Computational Linguistics.
            let mut n_sys = 0;
            let mut n_ref = 0;
            let mut n_cor = 0;
            for (refs_b, refs_t, syss_b, syss_t) in results {
                let mut matched = true;
                for (((r_b, r_t), s_b), s_t) in refs_b.iter().zip(&refs_t).zip(&syss_b).zip(&syss_t)
                {
                    if r_b == s_b {
                        if *s_b == CharacterBoundary::WordBoundary {
                            if matched && r_t == s_t {
                                n_cor += 1;
                            }
                            matched = true;
                            n_ref += 1;
                            n_sys += 1;
                        }
                    } else {
                        if *s_b == CharacterBoundary::WordBoundary {
                            n_sys += 1;
                        } else {
                            n_ref += 1;
                        }
                        matched = false;
                    }
                }
                if matched && refs_t.last().unwrap() == syss_t.last().unwrap() {
                    n_cor += 1;
                }
                n_sys += 1;
                n_ref += 1;
            }
            let precision = f64::from(n_cor) / f64::from(n_sys);
            let recall = f64::from(n_cor) / f64::from(n_ref);
            let f1 = 2. * precision * recall / (precision + recall);
            println!("Precision: {}", precision);
            println!("Recall: {}", recall);
            println!("F1: {}", f1);
        }
    }

    Ok(())
}
