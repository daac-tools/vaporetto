use std::fs::File;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use clap::Parser;
use vaporetto::{CharacterType, Model, Predictor, Sentence};
use vaporetto_rules::{
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter},
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

#[derive(Parser, Debug)]
#[clap(name = "predict", about = "A program to perform word segmentation.")]
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

    /// Prints scores.
    #[clap(long, action)]
    scores: bool,

    /// Do not normalize input strings before prediction.
    #[clap(long, action)]
    no_norm: bool,

    /// Buffers this tokenizer's output.
    #[clap(long, action)]
    buffered_out: bool,
}

fn print_scores(s: &Sentence, out: &mut dyn Write) -> Result<(), Box<dyn std::error::Error>> {
    let mut chars_iter = s.as_raw_text().chars();
    let mut prev_c = chars_iter.next().unwrap();
    for (i, (c, score)) in chars_iter.zip(s.boundary_scores()).enumerate() {
        writeln!(out, "{}:{}{} {}", i, prev_c, c, score)?;
        prev_c = c;
    }
    writeln!(out)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let pre_filter = KyteaFullwidthFilter;
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
    let start = Instant::now();
    let stdout = io::stdout();
    let mut out: Box<dyn Write> = if args.buffered_out {
        Box::new(BufWriter::new(stdout.lock()))
    } else {
        Box::new(stdout.lock())
    };
    let mut buf = String::new();
    let mut s = Sentence::default();
    if args.no_norm {
        for line in io::stdin().lock().lines() {
            let line = line?;
            if s.update_raw(line).is_ok() {
                predictor.predict(&mut s);
                post_filters.iter().for_each(|filter| filter.filter(&mut s));
                if args.predict_tags {
                    s.fill_tags();
                }
                s.write_tokenized_text(&mut buf);
                writeln!(out, "{}", buf)?;
                if args.scores {
                    print_scores(&s, &mut *out)?;
                }
            } else {
                writeln!(out)?;
            }
        }
    } else {
        let mut s_orig = Sentence::default();
        for line in io::stdin().lock().lines() {
            let line = line?;
            let line_preproc = pre_filter.filter(&line);
            if s.update_raw(line_preproc).is_ok() {
                predictor.predict(&mut s);
                post_filters.iter().for_each(|filter| filter.filter(&mut s));
                if args.predict_tags {
                    s.fill_tags();
                }
                s_orig.update_raw(line)?;
                s_orig.reset_tags(s.n_tags());
                s_orig.boundaries_mut().copy_from_slice(s.boundaries());
                s_orig.tags_mut().clone_from_slice(s.tags());
                s_orig.write_tokenized_text(&mut buf);
                writeln!(out, "{}", buf)?;
                if args.scores {
                    print_scores(&s, &mut *out)?;
                }
            } else {
                writeln!(out)?;
            }
        }
    }
    out.flush()?;

    let duration = start.elapsed();

    eprintln!("Elapsed: {} [sec]", duration.as_secs_f64());

    Ok(())
}
