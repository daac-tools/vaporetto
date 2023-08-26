use std::fs::File;
use std::io::{self, BufRead, BufWriter, Write};
use std::path::PathBuf;
use std::str::FromStr;
use std::time::Instant;

use clap::Parser;
use vaporetto::{CharacterBoundary, CharacterType, Model, Predictor, Sentence};
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
#[command(about = "A program to perform word segmentation.")]
struct Args {
    /// The model file to use when analyzing text
    #[arg(long)]
    model: PathBuf,

    /// Predicts POS tags.
    #[arg(long)]
    predict_tags: bool,

    /// Do not segment some character types: {D, R, H, T, K, O, G}.
    /// D: Digit, R: Roman, H: Hiragana, T: Katakana, K: Kanji, O: Other, G: Grapheme cluster.
    #[arg(long)]
    wsconst: Vec<WsConst>,

    /// Prints boundary scores.
    #[arg(long)]
    scores: bool,

    /// Prints tag scores.
    #[arg(long)]
    tag_scores: bool,

    /// Do not normalize input strings before prediction.
    #[arg(long)]
    no_norm: bool,
}

fn print_scores(s: &Sentence, mut out: impl Write) -> Result<(), Box<dyn std::error::Error>> {
    let mut chars_iter = s.as_raw_text().chars();
    let mut prev_c = chars_iter.next().unwrap();
    for (i, (c, score)) in chars_iter.zip(s.boundary_scores()).enumerate() {
        writeln!(out, "{i}:{prev_c}{c} {score}")?;
        prev_c = c;
    }
    out.write_all(b"\n")?;
    Ok(())
}

fn print_tag_scores_one(
    cands: &[Vec<String>],
    scores: &[i32],
    mut out: impl Write,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut i = 0;
    for cands in cands {
        out.write_all(b"\t")?;
        if cands.len() == 1 {
            write!(out, "{}:0", cands[0])?;
        } else {
            for (j, cand) in cands.iter().enumerate() {
                if j != 0 {
                    out.write_all(b",")?;
                }
                write!(out, "{cand}:{}", scores[i])?;
                i += 1;
            }
        }
    }
    Ok(())
}

fn print_tag_scores(s: &Sentence, mut out: impl Write) -> Result<(), Box<dyn std::error::Error>> {
    let mut chars_iter = s.as_raw_text().chars();
    let mut scores_iter = s.tag_scores().iter();
    for ((b, c), s) in s
        .boundaries()
        .iter()
        .zip(&mut chars_iter)
        .zip(&mut scores_iter)
    {
        write!(out, "{c}")?;
        if *b == CharacterBoundary::WordBoundary {
            if let Some((cands, scores)) = s.as_ref() {
                print_tag_scores_one(cands, scores, &mut out)?;
            }
            out.write_all(b"\n")?;
        }
    }
    let c = chars_iter.next().unwrap();
    let s = scores_iter.next().unwrap();
    write!(out, "{c}")?;
    if let Some((cands, scores)) = s.as_ref() {
        print_tag_scores_one(cands, scores, &mut out)?;
    }
    out.write_all(b"\n")?;
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
    let mut predictor = Predictor::new(model, args.predict_tags)?;
    if args.tag_scores {
        predictor.store_tag_scores(true);
    }

    let is_tty = atty::is(atty::Stream::Stdout);

    eprintln!("Start tokenization");
    let mut out = BufWriter::new(io::stdout().lock());
    let mut buf = String::new();
    let mut s = Sentence::default();

    let start = Instant::now();
    if args.no_norm {
        let lines = io::stdin().lock().lines();
        for line in lines {
            let line = line?;
            if s.update_raw(line).is_ok() {
                predictor.predict(&mut s);
                post_filters.iter().for_each(|filter| filter.filter(&mut s));
                if args.predict_tags {
                    s.fill_tags();
                }
                s.write_tokenized_text(&mut buf);
                out.write_all(buf.as_bytes())?;
                if args.scores {
                    print_scores(&s, &mut out)?;
                }
            }
            out.write_all(b"\n")?;
            if args.tag_scores {
                print_tag_scores(&s, &mut out)?;
            }
            if is_tty {
                out.flush()?;
            }
        }
    } else {
        let mut s_orig = Sentence::default();
        let lines = io::stdin().lock().lines();
        for line in lines {
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
                out.write_all(buf.as_bytes())?;
                out.write_all(b"\n")?;
                if args.scores {
                    print_scores(&s, &mut out)?;
                }
            } else {
                out.write_all(b"\n")?;
            }
            if args.tag_scores {
                print_tag_scores(&s, &mut out)?;
            }
            if is_tty {
                out.flush()?;
            }
        }
    }

    let duration = start.elapsed();

    eprintln!("Elapsed: {} [sec]", duration.as_secs_f64());

    Ok(())
}
