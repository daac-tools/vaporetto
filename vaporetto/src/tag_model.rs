use crate::ngram_model::NgramModel;

use bincode::{Decode, Encode};

#[derive(Decode, Encode)]
pub struct TagClassInfo {
    pub(crate) name: String,
    pub(crate) bias: i32,
}

// Left and right weight arrays of the TagModel are ordered as follows:
//
//      tok1 tok2 tok3 ...
//
// tag1   1    5    9
// tag2   2    6    .
// tag3   3    7    .
// ...    4    8    .
#[derive(Default, Decode, Encode)]
pub struct TagModel {
    pub(crate) class_info: Vec<TagClassInfo>,
    pub(crate) left_char_model: NgramModel<String>,
    pub(crate) right_char_model: NgramModel<String>,
    pub(crate) self_char_model: NgramModel<String>,
}
