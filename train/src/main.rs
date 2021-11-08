use std::collections::BTreeSet;
use std::fs::File;
use std::io::{prelude::*, stderr, BufReader};
use std::path::PathBuf;

use structopt::{clap::ArgGroup, StructOpt};
use vaporetto::{Dataset, Sentence, SolverType, Trainer};
use vaporetto_rules::{string_filters::KyteaFullwidthFilter, StringFilter};

#[derive(StructOpt, Debug)]
#[structopt(
    name = "train",
    about = "A program to train models of Vespa.",
    group = ArgGroup::with_name("dataset").required(true).multiple(true),
)]
struct Opt {
    /// A tokenized training corpus
    #[structopt(long, group = "dataset")]
    tok: Vec<PathBuf>,

    /// A partially annotated training corpus
    #[structopt(long, group = "dataset")]
    part: Vec<PathBuf>,

    /// A word dictionary file
    #[structopt(long)]
    dict: Vec<PathBuf>,

    /// The file to write the trained model to
    #[structopt(long)]
    model: PathBuf,

    /// The character window to use for word segmentation
    #[structopt(long, default_value = "3")]
    charw: usize,

    /// The character n-gram length to use for word segmentation
    #[structopt(long, default_value = "3")]
    charn: usize,

    /// The character type window to use for word segmentation
    #[structopt(long, default_value = "3")]
    typew: usize,

    /// The character type n-gram length to use for word segmentation
    #[structopt(long, default_value = "3")]
    typen: usize,

    /// Dictionary words greater than this value will be grouped together
    #[structopt(long, default_value = "4")]
    dictn: usize,

    /// The epsilon stopping criterion for classifier training
    #[structopt(long, default_value = "0.01")]
    eps: f64,

    /// The cost hyperparameter for classifier training
    #[structopt(long, default_value = "1.0")]
    cost: f64,

    /// Whether to use a bias value in classifier training
    #[structopt(long)]
    no_bias: bool,

    /// The solver. {0, 1, 2, 3, 4, 5, 6, 7} (see LIBLINEAR documentation for more details)
    #[structopt(long, default_value = "1")]
    solver: SolverType,

    /// Do not normalize training data.
    #[structopt(long)]
    no_norm: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    let fullwidth_filter = KyteaFullwidthFilter::new();

    eprintln!("Loading dataset...");
    let mut train_sents = vec![];

    for path in opt.tok {
        eprintln!("Loading {:?} ...", path);
        let f = File::open(path)?;
        let f = BufReader::new(f);
        for (i, line) in f.lines().enumerate() {
            if i % 10000 == 0 {
                eprint!("# of sentences: {}\r", i);
                stderr().flush()?;
            }
            let s = Sentence::from_tokenized(line?)?;
            let s = if opt.no_norm {
                s
            } else {
                let new_line = fullwidth_filter.filter(s.to_raw_string());
                let mut new_s = Sentence::from_raw(new_line)?;
                new_s.boundaries_mut().clone_from_slice(s.boundaries());
                new_s
            };
            train_sents.push(s);
        }
        eprintln!("# of sentences: {}", train_sents.len());
    }
    for path in opt.part {
        eprintln!("Loading {:?} ...", path);
        let f = File::open(path)?;
        let f = BufReader::new(f);
        for (i, line) in f.lines().enumerate() {
            if i % 10000 == 0 {
                eprint!("# of sentences: {}\r", i);
                stderr().flush()?;
            }
            let s = Sentence::from_partial_annotation(line?)?;
            let s = if opt.no_norm {
                s
            } else {
                let new_line = fullwidth_filter.filter(s.to_raw_string());
                let mut new_s = Sentence::from_raw(new_line)?;
                new_s.boundaries_mut().clone_from_slice(s.boundaries());
                new_s
            };
            train_sents.push(s);
        }
        eprintln!("# of sentences: {}", train_sents.len());
    }

    let mut dictionary = BTreeSet::new();
    for path in opt.dict {
        eprintln!("Loading {:?} ...", path);
        let f = File::open(path)?;
        let f = BufReader::new(f);
        for (i, line) in f.lines().enumerate() {
            if i % 100000 == 0 {
                eprint!("# of words: {}\r", i);
                stderr().flush()?;
            }
            let line = line?;
            let line = if opt.no_norm {
                line
            } else {
                fullwidth_filter.filter(line)
            };
            dictionary.insert(line);
        }
        eprintln!("# of words: {}", dictionary.len());
    }
    let dictionary: Vec<String> = dictionary.into_iter().collect();

    eprintln!("Extracting into features...");
    let mut dataset = Dataset::new(
        opt.charn, opt.charw, opt.typen, opt.typew, dictionary, opt.dictn,
    )?;
    for (i, s) in train_sents.iter().enumerate() {
        if i % 10000 == 0 {
            eprint!("# of features: {}\r", dataset.n_features());
            stderr().flush()?;
        }
        dataset.push_sentence(s);
    }
    eprintln!("# of features: {}", dataset.n_features());

    eprintln!("Start training...");
    let trainer = Trainer::new(opt.eps, opt.cost, if opt.no_bias { 0. } else { 1. });
    let model = trainer.train(dataset, opt.solver)?;
    eprintln!("Finish training.");

    let mut f = zstd::Encoder::new(File::create(opt.model)?, 19)?;
    model.write(&mut f)?;
    f.finish()?;

    Ok(())
}
