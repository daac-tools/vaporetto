pub mod text_input;
pub mod token_view;

use serde::{Deserialize, Serialize};
use yew_agent::{Bridge, Bridged, HandlerId, Public, WorkerLink};

use std::cell::RefCell;
use std::io::{Cursor, Read};
use std::rc::Rc;

use once_cell::sync::Lazy;
use vaporetto::{CharacterType, Model, Predictor, Sentence};
use vaporetto_rules::{
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter},
    string_filters::KyteaFullwidthFilter,
    SentenceFilter, StringFilter,
};
use yew::prelude::*;

use crate::text_input::TextInput;
use crate::token_view::TokenView;

static PREDICTOR: Lazy<Predictor> = Lazy::new(|| {
    let mut f = Cursor::new(include_bytes!("bccwj-suw+unidic+tag.model.zst"));
    let mut decoder = ruzstd::StreamingDecoder::new(&mut f).unwrap();
    let mut buff = vec![];
    decoder.read_to_end(&mut buff).unwrap();
    let (model, _) = Model::read_slice(&buff).unwrap();
    Predictor::new(model, true).unwrap()
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

impl yew_agent::Worker for Worker {
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

    fn update(&mut self, _msg: Self::Message) {
        // no messaging
    }

    fn handle_input(&mut self, msg: Self::Input, id: HandlerId) {
        println!("test");
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

        PREDICTOR.predict(sentence_filtered);

        let wsconst_g = ConcatGraphemeClustersFilter;
        let wsconst_d = KyteaWsConstFilter::new(CharacterType::Digit);
        wsconst_g.filter(sentence_filtered);
        wsconst_d.filter(sentence_filtered);

        sentence_filtered.fill_tags();
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
        "vaporetto/worker.js"
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
        let on_change = ctx.link().callback(Message::SetText);
        let WorkerOutput { tokens, n_tags } = self.worker_out.clone();
        html! {
            <>
            <header>{"🛥 Vaporetto Demo"}</header>
            <main>
                <div class="entry">
                    {
                        if tokens.is_empty() {
                            html!{
                                <input type="text" disabled=true />
                            }
                        } else {
                            html!{
                                <TextInput {on_change} value={self.text.clone()} />
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
