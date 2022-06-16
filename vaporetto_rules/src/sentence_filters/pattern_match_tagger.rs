use alloc::borrow::Cow;
use alloc::string::String;
use alloc::vec::Vec;

use hashbrown::HashMap;
use vaporetto::Sentence;

use crate::SentenceFilter;

/// Rule based tagger using pattern matching.
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

#[cfg(test)]
mod tests {
    use super::*;

    use alloc::string::String;

    #[test]
    fn test_concat_grapheme_clusters_no_boundary() {
        let mut s = Sentence::from_tokenized("これ/名詞/ソレ は テスト/名詞 です//デス").unwrap();
        let mut rules = HashMap::new();
        rules.insert("これ".into(), vec![Some("代名詞".into()), Some("コレ".into())]);
        rules.insert("は".into(), vec![Some("助詞".into()), Some("ワ".into())]);
        rules.insert("テスト".into(), vec![Some("名詞".into()), Some("テスト".into())]);
        rules.insert("です".into(), vec![Some("助動詞".into()), Some("デス".into())]);
        let filter = PatternMatchTagger::new(rules);
        filter.filter(&mut s);
        let mut buf = String::new();
        s.write_tokenized_text(&mut buf);
        assert_eq!("これ/名詞/ソレ は/助詞/ワ テスト/名詞/テスト です/助動詞/デス", buf);
    }
}
