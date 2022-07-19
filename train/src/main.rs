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
    about = "A program to train models of Vaporetto.",
    group = ArgGroup::new("dataset").required(true).multiple(true),
)]
struct Args {
    /// A tokenized training corpus
    #[clap(long, action, group = "dataset")]
    tok: Vec<PathBuf>,

    /// A partially annotated training corpus
    #[clap(long, action, group = "dataset")]
    part: Vec<PathBuf>,

    /// A word dictionary file
    #[clap(long, action)]
    dict: Vec<PathBuf>,

    /// The file to write the trained model to
    #[clap(long, action)]
    model: PathBuf,

    /// The character window to use for word segmentation
    #[clap(long, action, default_value = "3")]
    charw: u8,

    /// The character n-gram length to use for word segmentation
    #[clap(long, action, default_value = "3")]
    charn: u8,

    /// The character type window to use for word segmentation
    #[clap(long, action, default_value = "3")]
    typew: u8,

    /// The character type n-gram length to use for word segmentation
    #[clap(long, action, default_value = "3")]
    typen: u8,

    /// Dictionary words longer than this value will be grouped together, where the length is in
    /// characters
    #[clap(long, action, default_value = "4")]
    dictn: u8,

    /// The epsilon stopping criterion for classifier training
    #[clap(long, action, default_value = "0.01")]
    eps: f64,

    /// The cost hyperparameter for classifier training
    #[clap(long, action, default_value = "1.0")]
    cost: f64,

    /// The solver. {0, 1, 2, 3, 4, 5, 6, 7} (see LIBLINEAR documentation for more details)
    #[clap(long, action)]
    solver: SolverType,

    /// Do not normalize training data.
    #[clap(long, action)]
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
            let s = Sentence::from_tokenized(&line?)?;
            let s = if args.no_norm {
                s
            } else {
                let new_line = fullwidth_filter.filter(s.as_raw_text());
                let mut new_s = Sentence::from_raw(new_line)?;
                new_s.boundaries_mut().clone_from_slice(s.boundaries());
                new_s.reset_tags(s.n_tags());
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
            let s = Sentence::from_partial_annotation(&line?)?;
            let s = if args.no_norm {
                s
            } else {
                let new_line = fullwidth_filter.filter(s.as_raw_text());
                let mut new_s = Sentence::from_raw(new_line)?;
                new_s.boundaries_mut().copy_from_slice(s.boundaries());
                new_s.reset_tags(s.n_tags());
                new_s.tags_mut().clone_from_slice(s.tags());
                new_s
            };
            train_sents.push(s);
        }
        eprintln!("# of sentences: {}", train_sents.len());
    }

    let mut tag_dictionary = vec![];
    let mut dictionary = BTreeSet::new();
    for path in args.dict {
        eprintln!("Loading {:?} ...", path);
        let f = File::open(path)?;
        let f = BufReader::new(f);
        for line in f.lines() {
            if dictionary.len() % 10000 == 0 {
                eprint!("# of words: {}\r", dictionary.len());
                stderr().flush()?;
            }
            let s = Sentence::from_tokenized(&line?)?;
            let s = if args.no_norm {
                s
            } else {
                let new_line = fullwidth_filter.filter(s.as_raw_text());
                let mut new_s = Sentence::from_raw(new_line)?;
                new_s.boundaries_mut().clone_from_slice(s.boundaries());
                new_s.reset_tags(s.n_tags());
                new_s.tags_mut().clone_from_slice(s.tags());
                new_s
            };
            for token in s.iter_tokens() {
                dictionary.insert(token.surface().to_string());
            }
            tag_dictionary.push(s);
        }
        eprintln!("# of words: {}", dictionary.len());
    }
    let dictionary = dictionary.into_iter().collect();

    eprintln!("Extracting into features...");
    let mut trainer = Trainer::new(
        args.charw,
        args.charn,
        args.typew,
        args.typen,
        dictionary,
        args.dictn,
        &tag_dictionary,
    )?;
    for (i, s) in train_sents.iter().enumerate() {
        if i % 10000 == 0 {
            eprint!("# of features: {}\r", trainer.n_features(),);
            stderr().flush()?;
        }
        trainer.add_example(s);
    }
    eprintln!("# of features: {}", trainer.n_features(),);

    eprintln!("Start training...");
    let model = trainer.train(args.eps, args.cost, args.solver)?;
    eprintln!("Finish training.");

    let mut f = zstd::Encoder::new(File::create(args.model)?, 19)?;
    model.write(&mut f)?;
    f.finish()?;

    Ok(())
}
