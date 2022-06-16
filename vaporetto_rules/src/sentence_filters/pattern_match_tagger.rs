use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;
use vaporetto::Sentence;

use crate::SentenceFilter;

pub struct PatternMatchTagger {
    rules: HashMap<String, Vec<Option<String>>>,
}

impl PatternMatchTagger {
    pub const fn new(rules: HashMap<String, Vec<Option<String>>>) -> Self {
        Self { rules }
    }
}

impl SentenceFilter for PatternMatchTagger {
    fn filter(&self, sentence: &mut Sentence) {
        let n_tags = sentence.n_tags();
        let mut tag_queue: Vec<(usize, usize, Option<Cow<'static, str>>)> = vec![];
        for token in sentence.iter_tokens() {
            for (j, tag_ref) in token.tags().iter().enumerate() {
                if tag_ref.is_none() {
                    if let Some(tags) = self.rules.get(token.surface()) {
                        let tag = tags
                            .get(j)
                            .and_then(|tag| tag.as_ref().map(|tag| Cow::Owned(tag.clone())));
                        tag_queue.push((token.end() - 1, j, tag));
                    }
                }
            }
        }
        for (i, j, tag) in tag_queue {
            sentence.tags_mut()[i * n_tags + j] = tag;
        }
    }
}
