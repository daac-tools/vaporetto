//! Copied from an example of Yew.
//! https://github.com/yewstack/yew/blob/475cf20a86b237694d31fc06a99e9013540bb915/examples/password_strength/src/text_input.rs
//! 
//! Author: Philip Peterson <pc.peterso@gmail.com>
//! Licensed under either of MIT or Apache-2.0.

use wasm_bindgen::{JsCast, UnwrapThrowExt};
use web_sys::{Event, HtmlInputElement, InputEvent};
use yew::prelude::*;

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub value: String,
    pub on_change: Callback<String>,
}

fn get_value_from_input_event(e: InputEvent) -> String {
    let event: Event = e.dyn_into().unwrap_throw();
    let event_target = event.target().unwrap_throw();
    let target: HtmlInputElement = event_target.dyn_into().unwrap_throw();
    target.value()
}

#[function_component(TextInput)]
pub fn text_input(props: &Props) -> Html {
    let Props { value, on_change } = props.clone();

    let oninput = Callback::from(move |input_event: InputEvent| {
        on_change.emit(get_value_from_input_event(input_event));
    });

    html! {
        <input type="text" placeholder="Enter Japanese here" {value} {oninput} />
    }
}
