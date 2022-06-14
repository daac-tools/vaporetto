use yew::prelude::*;

#[derive(Clone, PartialEq, Properties)]
pub struct Props {
    pub tokens: Vec<(String, Vec<String>)>,
    pub n_tags: usize,
}

#[function_component(TokenView)]
pub fn token_view(props: &Props) -> Html {
    let Props { tokens, n_tags } = props.clone();

    html! {
        <pre>
            <table>
                <thead>
                    <tr>
                        <td>{"Surface"}</td>
                        {
                            for (1..n_tags + 1).map(|i| html! {
                                <td>{"Tag "}{i.to_string()}</td>
                            })
                        }

                    </tr>
                </thead>
                <tbody>
                    {
                        for tokens.into_iter().map(|(surface, tags)| html! {
                            <tr>
                                <td>{surface}</td>
                                {
                                    for tags.iter().map(|tag| html! {
                                        <td>{tag}</td>
                                    })
                                }
                            </tr>
                        })
                    }
                </tbody>
            </table>
        </pre>
    }
}
