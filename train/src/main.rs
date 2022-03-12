use std::collections::BTreeSet;
use std::fs::File;
use std::io::{prelude::*, stderr, BufReader};
use std::path::PathBuf;

use clap::{ArgGroup, Parser};
use vaporetto::{Sentence, SolverType, Trainer};
use vaporetto_rules::{string_filters::KyteaFullwidthFilter, StringFilter};

#[derive(Parser, Debug)]
#[clap(
    name = "train",
    about = "A program to train models of Vespa.",
    group = ArgGroup::new("dataset").required(true).multiple(true),
)]
struct Args {
    /// A tokenized training corpus
    #[clap(long, group = "dataset")]
    tok: Vec<PathBuf>,

    /// A partially annotated training corpus
    #[clap(long, group = "dataset")]
    part: Vec<PathBuf>,

    /// A word dictionary file
    #[clap(long)]
    dict: Vec<PathBuf>,

    /// The file to write the trained model to
    #[clap(long)]
    model: PathBuf,

    /// The character window to use for word segmentation
    #[clap(long, default_value = "3")]
    charw: u8,

    /// The character n-gram length to use for word segmentation
    #[clap(long, default_value = "3")]
    charn: u8,

    /// The character type window to use for word segmentation
    #[clap(long, default_value = "3")]
    typew: u8,

    /// The character type n-gram length to use for word segmentation
    #[clap(long, default_value = "3")]
    typen: u8,

    /// Dictionary words greater than this value will be grouped together
    #[clap(long, default_value = "4")]
    dictn: u8,

    /// The epsilon stopping criterion for classifier training
    #[clap(long, default_value = "0.01")]
    eps: f64,

    /// The cost hyperparameter for classifier training
    #[clap(long, default_value = "1.0")]
    cost: f64,

    /// The solver. {0, 1, 2, 3, 4, 5, 6, 7} (see LIBLINEAR documentation for more details)
    #[clap(long, default_value = "1")]
    solver: SolverType,

    /// Do not normalize training data.
    #[clap(long)]
    no_norm: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let fullwidth_filter = KyteaFullwidthFilter;

    eprintln!("Loading dataset...");
    let mut train_sents = vec![];

    for path in args.tok {
        eprintln!("Loading {:?} ...", path);
        let f = File::open(path)?;
        let f = BufReader::new(f);
        for (i, line) in f.lines().enumerate() {
            if i % 10000 == 0 {
                eprint!("# of sentences: {}\r", i);
                stderr().flush()?;
            }
            let s = Sentence::from_tokenized(line?)?;
            let s = if args.no_norm {
                s
            } else {
                let new_line = fullwidth_filter.filter(s.to_raw_string());
                let mut new_s = Sentence::from_raw(new_line)?;
                new_s.boundaries_mut().clone_from_slice(s.boundaries());
                new_s.tags_mut().clone_from_slice(s.tags());
                new_s
            };
            train_sents.push(s);
        }
        eprintln!("# of sentences: {}", train_sents.len());
    }
    for path in args.part {
        eprintln!("Loading {:?} ...", path);
        let f = File::open(path)?;
        let f = BufReader::new(f);
        for (i, line) in f.lines().enumerate() {
            if i % 10000 == 0 {
                eprint!("# of sentences: {}\r", i);
                stderr().flush()?;
            }
            let s = Sentence::from_partial_annotation(line?)?;
            let s = if args.no_norm {
                s
            } else {
                let new_line = fullwidth_filter.filter(s.to_raw_string());
                let mut new_s = Sentence::from_raw(new_line)?;
                new_s.boundaries_mut().copy_from_slice(s.boundaries());
                new_s.tags_mut().clone_from_slice(s.tags());
                new_s
            };
            train_sents.push(s);
        }
        eprintln!("# of sentences: {}", train_sents.len());
    }

    let mut dictionary = BTreeSet::new();
    for path in args.dict {
        eprintln!("Loading {:?} ...", path);
        let f = File::open(path)?;
        let f = BufReader::new(f);
        for (i, line) in f.lines().enumerate() {
            if i % 100000 == 0 {
                eprint!("# of words: {}\r", i);
                stderr().flush()?;
            }
            let line = line?;
            let line = if args.no_norm {
                line
            } else {
                fullwidth_filter.filter(&line)
            };
            dictionary.insert(line);
        }
        eprintln!("# of words: {}", dictionary.len());
    }
    let dictionary: Vec<String> = dictionary.into_iter().collect();

    eprintln!("Extracting into features...");
    let mut trainer = Trainer::new(
        args.charn, args.charw, args.typen, args.typew, dictionary, args.dictn,
    )?;
    for (i, s) in train_sents.iter().enumerate() {
        if i % 10000 == 0 {
            eprint!(
                "# of features: {}, # of tag features: {}\r",
                trainer.n_features(),
                trainer.n_tag_features()
            );
            stderr().flush()?;
        }
        trainer.push_sentence(s)?;
    }
    eprintln!(
        "# of features: {}, # of tag features: {}",
        trainer.n_features(),
        trainer.n_tag_features()
    );

    eprintln!("Start training...");
    let model = trainer.train(args.eps, args.cost, args.solver)?;
    eprintln!("Finish training.");

    let mut f = zstd::Encoder::new(File::create(args.model)?, 19)?;
    model.write(&mut f)?;
    f.finish()?;

    Ok(())
}
