pub mod text_input;
pub mod token_view;

use std::cell::RefCell;
use std::io::{Cursor, Read};
use std::rc::Rc;

use gloo_worker::{Bridge, Bridged, HandlerId, Public, WorkerLink};
use serde::{Deserialize, Serialize};
use yew::{html, Component, Context, Html};

use once_cell::sync::Lazy;
use vaporetto::{CharacterType, Model, Predictor, Sentence};
use vaporetto_rules::{
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter, PatternMatchTagger},
    string_filters::KyteaFullwidthFilter,
    SentenceFilter, StringFilter,
};

use crate::text_input::TextInput;
use crate::token_view::TokenView;

static PREDICTOR: Lazy<(Predictor, PatternMatchTagger)> = Lazy::new(|| {
    let mut f = Cursor::new(include_bytes!("bccwj-suw+unidic+tag-huge.model.zst"));
    let mut decoder = ruzstd::StreamingDecoder::new(&mut f).unwrap();
    let mut buff = vec![];
    decoder.read_to_end(&mut buff).unwrap();
    let (model, rest) = Model::read_slice(&buff).unwrap();
    let config = bincode::config::standard();
    let word_tag_map: Vec<(String, Vec<Option<String>>)> = bincode::decode_from_slice(rest, config).unwrap().0;
    (Predictor::new(model, true).unwrap(), PatternMatchTagger::new(word_tag_map.into_iter().collect()))
});

pub enum Message {
    SetText(String),
    WorkerMessage(WorkerOutput),
}

pub struct Worker {
    link: WorkerLink<Self>,
    sentence1: RefCell<Sentence<'static, 'static>>,
    sentence2: RefCell<Sentence<'static, 'static>>,
}

#[derive(Serialize, Deserialize)]
pub struct WorkerInput {
    pub text: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct WorkerOutput {
    pub tokens: Vec<(String, Vec<String>)>,
    pub n_tags: usize,
}

impl gloo_worker::Worker for Worker {
    type Input = WorkerInput;
    type Message = ();
    type Output = WorkerOutput;
    type Reach = Public<Self>;

    fn create(link: WorkerLink<Self>) -> Self {
        Lazy::force(&PREDICTOR);
        Self {
            link,
            sentence1: RefCell::new(Sentence::default()),
            sentence2: RefCell::new(Sentence::default()),
        }
    }

    fn update(&mut self, _msg: Self::Message) {}

    fn handle_input(&mut self, msg: Self::Input, id: HandlerId) {
        let pre_filter = KyteaFullwidthFilter;

        let sentence_orig = &mut self.sentence1.borrow_mut();
        let sentence_filtered = &mut self.sentence2.borrow_mut();

        if msg.text.is_empty() {
            sentence_orig.update_raw(" ").unwrap();
        } else {
            sentence_orig.update_raw(msg.text).unwrap();
        }
        let filtered_text = pre_filter.filter(sentence_orig.as_raw_text());
        sentence_filtered.update_raw(filtered_text).unwrap();

        PREDICTOR.0.predict(sentence_filtered);

        let wsconst_g = ConcatGraphemeClustersFilter;
        let wsconst_d = KyteaWsConstFilter::new(CharacterType::Digit);
        wsconst_g.filter(sentence_filtered);
        wsconst_d.filter(sentence_filtered);

        sentence_filtered.fill_tags();
        PREDICTOR.1.filter(sentence_filtered);
        let n_tags = sentence_filtered.n_tags();

        sentence_orig
            .boundaries_mut()
            .copy_from_slice(sentence_filtered.boundaries());
        sentence_orig.reset_tags(n_tags);
        sentence_orig
            .tags_mut()
            .clone_from_slice(sentence_filtered.tags());

        let tokens = sentence_orig
            .iter_tokens()
            .map(|token| {
                (
                    token.surface().to_string(),
                    token
                        .tags()
                        .iter()
                        .map(|tag| {
                            tag.as_ref()
                                .map(|tag| tag.to_string())
                                .unwrap_or_else(String::new)
                        })
                        .collect(),
                )
            })
            .collect();

        let output = Self::Output { tokens, n_tags };
        self.link.respond(id, output);
    }

    fn name_of_resource() -> &'static str {
        "worker.js"
    }

    fn resource_path_is_relative() -> bool {
        true
    }
}

pub struct App {
    text: String,
    worker: Box<dyn Bridge<Worker>>,
    worker_out: WorkerOutput,
}

impl Component for App {
    type Message = Message;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let cb = {
            let link = ctx.link().clone();
            move |e| link.send_message(Self::Message::WorkerMessage(e))
        };
        let mut worker = Worker::bridge(Rc::new(cb));
        worker.send(WorkerInput {
            text: " ".to_string(),
        });
        Self {
            text: String::new(),
            worker,
            worker_out: WorkerOutput {
                tokens: vec![],
                n_tags: 0,
            },
        }
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Message::SetText(text) => {
                self.text = text.clone();
                self.worker.send(WorkerInput { text });
            }
            Message::WorkerMessage(message) => {
                self.worker_out = message;
            }
        };
        true
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let input_cb = ctx.link().callback(Message::SetText);
        let WorkerOutput { tokens, n_tags } = self.worker_out.clone();
        html! {
            <>
                <header>
                    <h1>{"Vaporetto Wasm Demo"}</h1>
                    <p class="header-link"><a href="https://github.com/daac-tools/vaporetto">{"[Project Page]"}</a></p>
                </header>
                <main>
                    <div class="entry">
                        {
                            if tokens.is_empty() {
                                html!{
                                    <input type="text" disabled=true />
                                }
                            } else {
                                html!{
                                    <TextInput {input_cb} value={self.text.clone()} />
                                }
                            }
                        }
                    </div>
                    {
                        if tokens.is_empty() {
                            html!{
                                <div id="loading">{"Loading..."}</div>
                            }
                        } else {
                            html!{
                                <div class="results">
                                    <TokenView {tokens} {n_tags} />
                                </div>
                            }
                        }
                    }
                </main>
            </>
        }
    }
}
