use fluent::{bundle::FluentBundle, FluentResource};
use fluent_langneg::NegotiationStrategy;
use intl_memoizer::concurrent::IntlLangMemoizer;
use once_cell::sync::Lazy;
use unic_langid::{langid, LanguageIdentifier};

pub static FLUENT_BUNDLE_DATA: Lazy<FluentBundle<FluentResource, IntlLangMemoizer>> =
    Lazy::new(|| {
        let mut available_locales = std::collections::HashMap::new();

        // Register locales
        available_locales.insert(langid!("ja"), include_str!("../locales/en.ftl"));
        available_locales.insert(langid!("en"), include_str!("../locales/ja.ftl"));

        let requested: Vec<LanguageIdentifier> = web_sys::window()
            .unwrap()
            .navigator()
            .languages()
            .iter()
            .map(|id| id.as_string().unwrap().parse().unwrap())
            .collect();
        let available: Vec<LanguageIdentifier> = available_locales.keys().cloned().collect();
        let default: LanguageIdentifier = "en-US".parse().unwrap();
        let supported = fluent_langneg::negotiate_languages(
            &requested,
            &available,
            Some(&default),
            NegotiationStrategy::Filtering,
        );
        let supported_lang = supported[0].clone();
        let res =
            FluentResource::try_new(available_locales.get(&supported_lang).unwrap().to_string())
                .unwrap();
        let mut bundle = FluentBundle::new_concurrent(vec![supported_lang]);
        bundle.add_resource(res).unwrap();
        bundle
    });

#[macro_export]
macro_rules! fluent_format {
    ( $id:expr ) => {
        fluent_format!( $id, )
    };
    ( $id:expr, $($key:expr => $value:expr),* ) => {
        {
            let bundle = &$crate::locales::FLUENT_BUNDLE_DATA;
            let msg = bundle.get_message($id).unwrap();
            let mut errors = vec![];
            let pattern = msg.value().unwrap();
            #[allow(unused_mut)]
            let mut args: fluent::FluentArgs = fluent::FluentArgs::new();
            $(
                args.set($key, $value);
            )*
            $crate::locales::FLUENT_BUNDLE_DATA.format_pattern(&pattern, Some(&args), &mut errors).to_string()
        }
    };
}
