use std::rc::Rc;

use yew::{function_component, html, Html, Properties};

use crate::Token;

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub tokens: Rc<Vec<Token>>,
    pub n_tags: usize,
}

#[function_component(TokenView)]
pub fn token_view(props: &Props) -> Html {
    let Props { tokens, n_tags } = &props;

    html! {
        <table>
            <thead>
                <tr>
                    <th>{"Surface"}</th>
                    {
                        for (1..*n_tags + 1).map(|i| html! {
                            <th>{"Tag "}{i.to_string()}</th>
                        })
                    }
                </tr>
            </thead>
            <tbody>
                {
                    for tokens.iter().map(|token| html! {
                        <tr>
                            <td>{&token.surface}</td>
                            {
                                for token.tags.iter().map(|tag| html! {
                                    <td>{tag}</td>
                                })
                            }
                        </tr>
                    })
                }
            </tbody>
        </table>
    }
}
