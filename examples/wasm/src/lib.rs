pub mod i18n;

pub mod text_input;
pub mod token_view;

use std::borrow::Cow;
use std::io::Read;
use std::rc::Rc;

use gloo_worker::{HandlerId, Spawnable, Worker, WorkerBridge, WorkerScope};
use serde::{Deserialize, Serialize};
use vaporetto::{CharacterType, Model, Predictor, Sentence};
use vaporetto_rules::{
    sentence_filters::{ConcatGraphemeClustersFilter, KyteaWsConstFilter},
    string_filters::KyteaFullwidthFilter,
    SentenceFilter, StringFilter,
};
use web_sys::UrlSearchParams;
use yew::{html, Component, Context, Html};

use crate::text_input::TextInput;
use crate::token_view::TokenView;

#[derive(Clone, PartialEq, Serialize, Deserialize)]
pub struct Token {
    pub surface: String,
    pub tags: Vec<String>,
}

pub struct WorkerMessage {
    pub id: HandlerId,
    pub output: (Vec<Token>, usize),
}

#[ouroboros::self_referencing]
pub struct VaporettoWorker {
    predictor: Predictor,
    wsconst_g: ConcatGraphemeClustersFilter,
    wsconst_d: KyteaWsConstFilter,

    #[borrows(predictor)]
    #[covariant]
    sentence_orig: Sentence<'static, 'this>,
    #[borrows(predictor)]
    #[covariant]
    sentence_filtered: Sentence<'static, 'this>,
}

impl Worker for VaporettoWorker {
    type Input = String;
    type Message = WorkerMessage;
    type Output = (Vec<Token>, usize);

    fn create(_scope: &WorkerScope<Self>) -> Self {
        let model_data = include_bytes!("bccwj-suw+unidic_pos+pron.model.zst");
        let mut decoder = ruzstd::StreamingDecoder::new(model_data.as_slice()).unwrap();
        let mut buff = vec![];
        decoder.read_to_end(&mut buff).unwrap();
        let (model, _) = Model::read_slice(&buff).unwrap();
        VaporettoWorkerBuilder {
            predictor: Predictor::new(model, true).unwrap(),
            wsconst_g: ConcatGraphemeClustersFilter,
            wsconst_d: KyteaWsConstFilter::new(CharacterType::Digit),
            sentence_orig_builder: |_| Sentence::default(),
            sentence_filtered_builder: |_| Sentence::default(),
        }
        .build()
    }

    fn update(&mut self, scope: &WorkerScope<Self>, msg: Self::Message) {
        let WorkerMessage { id, output } = msg;
        scope.respond(id, output);
    }

    fn received(&mut self, scope: &WorkerScope<Self>, msg: Self::Input, id: HandlerId) {
        let pre_filter = KyteaFullwidthFilter;
        let filtered_text = pre_filter.filter(&msg);

        if msg.is_empty() {
            scope.send_message(WorkerMessage {
                id,
                output: (vec![], 0),
            });
            return;
        }

        self.with_mut(|fields| {
            fields.sentence_filtered.update_raw(filtered_text).unwrap();
            fields.predictor.predict(fields.sentence_filtered);
            fields.wsconst_g.filter(fields.sentence_filtered);
            fields.wsconst_d.filter(fields.sentence_filtered);
            fields.sentence_filtered.fill_tags();

            fields.sentence_orig.update_raw(msg).unwrap();

            let n_tags = fields.sentence_filtered.n_tags();
            fields
                .sentence_orig
                .boundaries_mut()
                .copy_from_slice(fields.sentence_filtered.boundaries());
            fields.sentence_orig.reset_tags(n_tags);
            for (d, s) in fields
                .sentence_orig
                .tags_mut()
                .iter_mut()
                .zip(fields.sentence_filtered.tags())
            {
                *d = s.as_ref().map(|x| Cow::Owned(x.to_string()));
            }
        });

        let tokens = self
            .borrow_sentence_orig()
            .iter_tokens()
            .map(|token| Token {
                surface: token.surface().to_string(),
                tags: token
                    .tags()
                    .iter()
                    .map(|tag| {
                        tag.as_ref()
                            .map(|tag| tag.to_string())
                            .unwrap_or_else(String::new)
                    })
                    .collect(),
            })
            .collect();
        let n_tags = self.borrow_sentence_orig().n_tags();

        let output = (tokens, n_tags);
        scope.send_message(WorkerMessage { id, output })
    }
}

pub enum Msg {
    SetText(String),
    WorkerResult((Vec<Token>, usize)),
}

pub struct App {
    bridge: WorkerBridge<VaporettoWorker>,
    text: Rc<String>,
    tokens: Option<Rc<Vec<Token>>>,
    n_tags: usize,
}

impl Component for App {
    type Message = Msg;
    type Properties = ();

    fn create(ctx: &Context<Self>) -> Self {
        let link = ctx.link().clone();
        let bridge = VaporettoWorker::spawner()
            .callback(move |m| {
                link.send_message(Msg::WorkerResult(m));
            })
            .spawn("./vaporetto_worker.js");

        let text = web_sys::window()
            .unwrap()
            .location()
            .search()
            .ok()
            .and_then(|s| UrlSearchParams::new_with_str(&s).ok())
            .and_then(|q| q.get("text"))
            .unwrap_or_else(String::new);
        bridge.send(text.clone());

        Self {
            bridge,
            text: text.into(),
            tokens: None,
            n_tags: 0,
        }
    }

    fn update(&mut self, _ctx: &Context<Self>, msg: Self::Message) -> bool {
        match msg {
            Msg::SetText(text) => {
                self.text = Rc::new(text);
                self.bridge.send(self.text.to_string());
            }
            Msg::WorkerResult((tokens, n_tags)) => {
                self.tokens.replace(Rc::new(tokens));
                self.n_tags = n_tags;
            }
        };
        true
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        html! {
            <>
                <header>
                    <h1>{ fl!("title") }</h1>
                    <p class="header-link">
                        <a href="https://github.com/daac-tools/vaporetto">{ fl!("project-page") }</a>
                    </p>
                </header>
                <main>
                    {
                        html! {
                            <TextInput
                                callback={ctx.link().callback(Msg::SetText)}
                                value={self.tokens.is_some().then(|| Rc::clone(&self.text))}
                            />
                        }
                    }
                    {
                        if let Some(tokens) = &self.tokens {
                            html! {
                                <TokenView tokens={Rc::clone(tokens)} n_tags={self.n_tags} />
                            }
                        } else {
                            html! {
                                <div id="loading">{ fl!("loading") }</div>
                            }
                        }
                    }
                </main>
            </>
        }
    }
}
