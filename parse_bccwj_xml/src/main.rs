use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;

use clap::Parser;
use quick_xml::events::Event;
use vaporetto::{CharacterBoundary, Sentence};

#[derive(Parser, Debug)]
#[clap(
    name = "parse_bccwj_xml",
    about = "A program to parse XML files of BCCWJ."
)]
struct Args {
    /// Generates long-unit-word corpus.
    #[clap(long)]
    luw: bool,

    /// Attributes to be added a tag information.
    ///
    /// If multiple attributes are specified, separated by `|`, the value of the next attribute is
    /// used if the previous one does not exist.
    #[clap(long)]
    attr: Vec<String>,

    /// XML files to be parsed.
    xml_files: Vec<PathBuf>,
}

struct Token {
    surface: String,
    tags: Vec<Option<(String, usize)>>,
}

fn parse_xml(
    rdr: impl BufRead,
    luw: bool,
    attrs: &HashMap<Vec<u8>, (usize, usize)>,
    n_attrs: usize,
) -> quick_xml::Result<Vec<String>> {
    let mut results = vec![];
    let mut rdr = quick_xml::Reader::from_reader(rdr);
    let mut buf = vec![];
    let mut word = String::new();
    let mut tokens: Vec<Token> = vec![];
    let mut token = None;
    let word_tag = if luw { b"LUW" } else { b"SUW" };
    loop {
        buf.clear();
        match rdr.read_event_into(&mut buf) {
            Ok(Event::Start(e)) => {
                if e.name().as_ref() == b"sentence" {
                    tokens.clear();
                } else if e.name().as_ref() == word_tag {
                    word.clear();
                    let mut tags = vec![None; n_attrs];
                    for attr in e.attributes() {
                        let attr = attr?;
                        if let Some(&(idx, priority)) = attrs.get(attr.key.as_ref()) {
                            if let Some(&(_, p)) = tags[idx].as_ref() {
                                if p < priority {
                                    continue;
                                }
                            }
                            tags[idx].replace((String::from_utf8(attr.value.to_vec())?, priority));
                        }
                    }
                    token.replace(Token {
                        surface: String::new(),
                        tags,
                    });
                }
            }
            Ok(Event::End(e)) => {
                if e.name().as_ref() == b"sentence" {
                    let mut boundaries = vec![];
                    let mut tags = vec![];
                    let mut text = String::new();
                    for token in &tokens {
                        for _ in 1..token.surface.chars().count() {
                            boundaries.push(CharacterBoundary::NotWordBoundary);
                            for _ in 0..n_attrs {
                                tags.push(None);
                            }
                        }
                        for tag in &token.tags {
                            tags.push(tag.as_ref().map(|(t, _)| t.into()));
                        }
                        boundaries.push(CharacterBoundary::WordBoundary);
                        text.push_str(&token.surface);
                    }
                    if let Ok(mut s) = Sentence::from_raw(text) {
                        s.boundaries_mut()
                            .copy_from_slice(&boundaries[..boundaries.len() - 1]);
                        s.reset_tags(n_attrs);
                        s.tags_mut().clone_from_slice(&tags);
                        let mut buf = String::new();
                        s.write_tokenized_text(&mut buf);
                        results.push(buf);
                    }
                } else if e.name().as_ref() == word_tag {
                    let mut token = token.take().unwrap();
                    token.surface = word.replace("\r\n", "");
                    word.clear();
                    tokens.push(token);
                }
            }
            Ok(Event::Text(e)) => {
                word.push_str(e.unescape()?.as_ref());
            }
            Ok(Event::Eof) => break,
            Err(e) => return Err(e),
            _ => {}
        }
    }
    Ok(results)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    let mut attrs = HashMap::new();
    for (i, attr) in args.attr.iter().enumerate() {
        for (j, attr) in attr.split('|').enumerate() {
            if attrs.contains_key(attr.as_bytes()) {
                return Err(format!("`{}` is used as another tag.", attr).into());
            }
            attrs.insert(attr.as_bytes().to_vec(), (i, j));
        }
    }

    for path in args.xml_files {
        let rdr = BufReader::new(File::open(path)?);
        let results = parse_xml(rdr, args.luw, &attrs, args.attr.len())?;
        for result in results {
            println!("{}", result);
        }
    }
    Ok(())
}
