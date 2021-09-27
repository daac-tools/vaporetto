use std::fs::File;
use std::io::{prelude::*, stdin, BufReader};
use std::path::PathBuf;
use std::time::Instant;

use structopt::StructOpt;
use vaporetto::{Model, Predictor, Sentence};

#[derive(StructOpt, Debug)]
#[structopt(name = "predict", about = "A program to perform word segmentation.")]
struct Opt {
    /// The model file to use when analyzing text
    #[structopt(long)]
    model: PathBuf,

    /// Number of threads
    #[structopt(long, default_value = "0")]
    n_threads: usize,

    /// Chunk size of each thread
    #[structopt(long, default_value = "10")]
    mt_chunk_size: usize,

    /// Window size for dictionary words
    #[structopt(long, default_value = "3")]
    chunk_dict_window: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opt = Opt::from_args();

    eprintln!("Loading model file...");
    let mut f = BufReader::new(File::open(opt.model).unwrap());
    let model = Model::read(&mut f)?;
    let predictor = Predictor::new(model).dict_window_size(opt.chunk_dict_window);

    eprintln!("Start tokenization");
    let mut n_boundaries = 0;
    let start = Instant::now();
    if opt.n_threads == 0 {
        for line in stdin().lock().lines() {
            let s = Sentence::from_raw(line?)?;
            let s = predictor.predict(s);
            let toks = s.to_tokenized_string()?;
            n_boundaries += s.boundaries().len();
            println!("{}", toks);
        }
    } else {
        let predictor = predictor.multithreading(opt.n_threads, opt.mt_chunk_size);
        for line in stdin().lock().lines() {
            let s = Sentence::from_raw(line?)?;
            let s = predictor.predict(s);
            let toks = s.to_tokenized_string()?;
            n_boundaries += s.boundaries().len();
            println!("{}", toks);
        }
    }
    let duration = start.elapsed();
    eprintln!("Elapsed: {} [sec]", duration.as_secs_f64());
    eprintln!(
        "Speed: {} [boundaries/sec]",
        n_boundaries as f64 / duration.as_secs_f64()
    );

    Ok(())
}
