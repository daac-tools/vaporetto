use std::collections::BTreeSet;
use std::fs::File;
use std::io::{prelude::*, stderr, BufReader};
use std::path::PathBuf;

use clap::{ArgGroup, Parser, ValueEnum};
#[cfg(feature = "liblinear")]
use vaporetto::LiblinearSolver;
use vaporetto::{Sentence, Solver, Trainer};
use vaporetto_rules::{string_filters::KyteaFullwidthFilter, StringFilter};

#[derive(Parser, Debug)]
#[command(
    about = "A program to train models of Vaporetto.",
    group = ArgGroup::new("dataset").required(true).multiple(true),
)]
struct Args {
    /// A tokenized training corpus
    #[arg(long, group = "dataset")]
    tok: Vec<PathBuf>,

    /// A partially annotated training corpus
    #[arg(long, group = "dataset")]
    part: Vec<PathBuf>,

    /// A word dictionary file
    #[arg(long)]
    dict: Vec<PathBuf>,

    /// The file to write the trained model to
    #[arg(long)]
    model: PathBuf,

    /// The character window to use for word segmentation
    #[arg(long, default_value = "3")]
    charw: u8,

    /// The character n-gram length to use for word segmentation
    #[arg(long, default_value = "3")]
    charn: u8,

    /// The character type window to use for word segmentation
    #[arg(long, default_value = "3")]
    typew: u8,

    /// The character type n-gram length to use for word segmentation
    #[arg(long, default_value = "3")]
    typen: u8,

    /// Dictionary words longer than this value will be grouped together, where the length is in
    /// characters
    #[arg(long, default_value = "4")]
    dictn: u8,

    /// The epsilon stopping criterion for classifier training
    #[arg(long, default_value = "0.01")]
    eps: f64,

    /// The cost hyperparameter for classifier training. For elastic-net, the costs of the L1
    /// and L2 terms separated by a comma (e.g. 1.0,10; the cost of the L2 term may be inf).
    #[arg(long, default_value = "1.0")]
    cost: String,

    /// The solver. The names are solvers in Rust, which need less memory; the numbers are
    /// LIBLINEAR's solvers (see LIBLINEAR documentation for more details).
    #[arg(long, value_enum)]
    solver: SolverArg,

    /// Do not normalize training data.
    #[arg(long)]
    no_norm: bool,

    /// The number of workers for zstd (0 means multithreaded will be disabled)
    #[arg(long, default_value = "0")]
    zstd_workers: u32,
}

/// The solvers that can be given by --solver.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum SolverArg {
    /// L1-regularized L2-loss SVC
    #[value(name = "l1r-l2loss-svc")]
    L1rL2lossSvc,
    /// L2-regularized L2-loss SVC (dual)
    #[value(name = "l2r-l2loss-svc-dual")]
    L2rL2lossSvcDual,
    /// L2-regularized L1-loss SVC (dual)
    #[value(name = "l2r-l1loss-svc-dual")]
    L2rL1lossSvcDual,
    /// L2-regularized logistic regression (dual)
    #[value(name = "l2r-lr-dual")]
    L2rLrDual,
    /// Elastic-net regularized L2-loss SVC: ||w||_1 / C1 + ||w||_2^2 / (2 C2) + Σ L2 loss
    #[value(name = "elastic-net")]
    ElasticNet,
    /// LIBLINEAR: L2-regularized logistic regression (primal)
    #[cfg(feature = "liblinear")]
    #[value(name = "0")]
    Liblinear0,
    /// LIBLINEAR: L2-regularized L2-loss SVC (dual)
    #[cfg(feature = "liblinear")]
    #[value(name = "1")]
    Liblinear1,
    /// LIBLINEAR: L2-regularized L2-loss SVC (primal)
    #[cfg(feature = "liblinear")]
    #[value(name = "2")]
    Liblinear2,
    /// LIBLINEAR: L2-regularized L1-loss SVC (dual)
    #[cfg(feature = "liblinear")]
    #[value(name = "3")]
    Liblinear3,
    /// LIBLINEAR: SVC by Crammer and Singer
    #[cfg(feature = "liblinear")]
    #[value(name = "4")]
    Liblinear4,
    /// LIBLINEAR: L1-regularized L2-loss SVC
    #[cfg(feature = "liblinear")]
    #[value(name = "5")]
    Liblinear5,
    /// LIBLINEAR: L1-regularized logistic regression
    #[cfg(feature = "liblinear")]
    #[value(name = "6")]
    Liblinear6,
    /// LIBLINEAR: L2-regularized logistic regression (dual)
    #[cfg(feature = "liblinear")]
    #[value(name = "7")]
    Liblinear7,
}

/// The solver given by --solver and --cost.
fn parse_solver(solver: SolverArg, cost: &str) -> Result<Solver, String> {
    let parse = |c: &str| {
        c.trim()
            .parse::<f64>()
            .map_err(|e| format!("invalid cost {c:?}: {e}"))
    };
    let rust = |solver: fn(f64) -> Solver| Ok(solver(parse(cost)?));
    #[cfg(feature = "liblinear")]
    let liblinear = |solver| {
        Ok(Solver::Liblinear {
            solver,
            cost: parse(cost)?,
        })
    };
    match solver {
        SolverArg::L1rL2lossSvc => rust(|cost| Solver::L1RegularizedL2LossSVC { cost }),
        SolverArg::L2rL2lossSvcDual => rust(|cost| Solver::L2RegularizedL2LossSVCDual { cost }),
        SolverArg::L2rL1lossSvcDual => rust(|cost| Solver::L2RegularizedL1LossSVCDual { cost }),
        SolverArg::L2rLrDual => rust(|cost| Solver::L2RegularizedLogisticDual { cost }),
        SolverArg::ElasticNet => {
            let (l1_cost, l2_cost) = cost
                .split_once(',')
                .ok_or("elastic-net needs the costs of the L1 and L2 terms: --cost <L1>,<L2>")?;
            Ok(Solver::ElasticNet {
                l1_cost: parse(l1_cost)?,
                l2_cost: parse(l2_cost)?,
            })
        }
        #[cfg(feature = "liblinear")]
        SolverArg::Liblinear0 => liblinear(LiblinearSolver::L2RegularizedLogistic),
        #[cfg(feature = "liblinear")]
        SolverArg::Liblinear1 => liblinear(LiblinearSolver::L2RegularizedL2LossSVCDual),
        #[cfg(feature = "liblinear")]
        SolverArg::Liblinear2 => liblinear(LiblinearSolver::L2RegularizedL2LossSVC),
        #[cfg(feature = "liblinear")]
        SolverArg::Liblinear3 => liblinear(LiblinearSolver::L2RegularizedL1LossSVCDual),
        #[cfg(feature = "liblinear")]
        SolverArg::Liblinear4 => liblinear(LiblinearSolver::CrammerSingerSVC),
        #[cfg(feature = "liblinear")]
        SolverArg::Liblinear5 => liblinear(LiblinearSolver::L1RegularizedL2LossSVC),
        #[cfg(feature = "liblinear")]
        SolverArg::Liblinear6 => liblinear(LiblinearSolver::L1RegularizedLogistic),
        #[cfg(feature = "liblinear")]
        SolverArg::Liblinear7 => liblinear(LiblinearSolver::L2RegularizedLogisticDual),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let solver = parse_solver(args.solver, &args.cost)?;

    let fullwidth_filter = KyteaFullwidthFilter;

    eprintln!("Loading dataset...");
    let mut train_sents = vec![];

    for path in args.tok {
        eprintln!("Loading {path:?} ...");
        let f = BufReader::new(File::open(&path)?);
        for (i, line) in f.lines().enumerate() {
            if i % 10000 == 0 {
                eprint!("# of sentences: {i}\r");
                stderr().flush()?;
            }
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let s = Sentence::from_tokenized(&line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
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
        eprintln!("Loading {path:?} ...");
        let f = BufReader::new(File::open(&path)?);
        for (i, line) in f.lines().enumerate() {
            if i % 10000 == 0 {
                eprint!("# of sentences: {i}\r");
                stderr().flush()?;
            }
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let s = Sentence::from_partial_annotation(&line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
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
        eprintln!("Loading {path:?} ...");
        let f = BufReader::new(File::open(&path)?);
        for (i, line) in f.lines().enumerate() {
            if dictionary.len() % 10000 == 0 {
                eprint!("# of words: {}\r", dictionary.len());
                stderr().flush()?;
            }
            let line = line?;
            if line.is_empty() {
                continue;
            }
            let s = Sentence::from_tokenized(&line)
                .map_err(|e| format!("{}:{}: {e}", path.display(), i + 1))?;
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
    let model = trainer.train(args.eps, solver)?;
    eprintln!("Finish training.");

    let mut f = zstd::Encoder::new(File::create(args.model)?, 19)?;
    f.multithread(args.zstd_workers)?;
    model.write(&mut f)?;
    f.finish()?;

    Ok(())
}
