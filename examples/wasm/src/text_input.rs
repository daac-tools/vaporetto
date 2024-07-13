use std::rc::Rc;

use gloo_timers::callback::Timeout;
use wasm_bindgen_futures::JsFuture;
use web_sys::HtmlInputElement;
use yew::{html, platform::spawn_local, Callback, Component, Context, Html, NodeRef, Properties};

use crate::fl;

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub value: Option<Rc<String>>,
    pub callback: Callback<String>,
}

pub struct TextInput {
    input_ref: NodeRef,
    button_ref: NodeRef,
}

impl Component for TextInput {
    type Message = ();
    type Properties = Props;

    fn create(_ctx: &Context<Self>) -> Self {
        Self {
            input_ref: NodeRef::default(),
            button_ref: NodeRef::default(),
        }
    }

    fn view(&self, ctx: &Context<Self>) -> Html {
        let Props { value, callback } = ctx.props();
        let callback = callback.clone();

        let input_ref = self.input_ref.clone();
        let oninput = Callback::from(move |_| {
            let input = input_ref.cast::<HtmlInputElement>().unwrap();
            callback.emit(input.value());
        });

        let input_ref = self.input_ref.clone();
        let button_ref = self.button_ref.clone();
        let clipboard_click = Callback::from(move |_| {
            let input = input_ref.cast::<HtmlInputElement>().unwrap();
            if let Some(clipboard) = web_sys::window().unwrap().navigator().clipboard() {
                let loc = web_sys::window().unwrap().location();
                let content = format!(
                    "{}{}?text={}",
                    loc.origin().unwrap(),
                    loc.pathname().unwrap(),
                    js_sys::encode_uri_component(&input.value())
                );
                let promise = clipboard.write_text(&content);
                let button_ref = button_ref.clone();
                spawn_local(async move {
                    JsFuture::from(promise).await.unwrap();
                    let button = button_ref.cast::<HtmlInputElement>().unwrap();
                    button.set_class_name("copied");
                    let button_ref = button_ref.clone();
                    let timeout = Timeout::new(1000, move || {
                        let button = button_ref.cast::<HtmlInputElement>().unwrap();
                        button.set_class_name("");
                    });
                    timeout.forget();
                });
            }
        });
        if let Some(value) = value.as_ref() {
            html! {
                <div class="input-bar">
                    <input
                        ref={self.input_ref.clone()}
                        type="text"
                        placeholder={ fl!("place-holder") }
                        value={value.to_string()}
                        {oninput}
                    />
                    <button
                        ref={self.button_ref.clone()}
                        onclick={clipboard_click}
                    >
                        {"🔗"}
                    </button>
                </div>
            }
        } else {
            html! {
                <div class="input-bar">
                    <input
                        type="text"
                        placeholder={ fl!("place-holder") }
                        disabled=true
                    />
                    <button
                        disabled=true
                    >
                        {"🔗"}
                    </button>
                </div>
            }
        }
    }

    fn changed(&mut self, ctx: &Context<Self>, old_props: &Self::Properties) -> bool {
        ctx.props().value != old_props.value
    }

    fn rendered(&mut self, _ctx: &Context<Self>, first_render: bool) {
        if first_render {
            if let Some(input) = self.input_ref.cast::<HtmlInputElement>() {
                input.focus().unwrap();
            }
        }
    }
}
