#[cfg(feature = "train")]
use std::collections::HashMap;

#[cfg(feature = "train")]
use crate::feature::Feature;

#[cfg(feature = "train")]
pub struct FeatureIDManager<'a> {
    pub(crate) map: HashMap<Feature<'a>, u32>,
}

#[cfg(feature = "train")]
impl<'a> FeatureIDManager<'a> {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn get_id(&mut self, feature: Feature<'a>) -> u32 {
        self.map.get(&feature).copied().unwrap_or_else(|| {
            let new_id = self.map.len() as u32;
            self.map.insert(feature, new_id);
            new_id
        })
    }
}

#[cfg(feature = "train")]
impl<'a> Default for FeatureIDManager<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "train")]
pub struct StringIdManager {
    pub(crate) map: HashMap<Vec<u8>, usize>,
}

#[cfg(feature = "train")]
impl StringIdManager {
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
        }
    }

    pub fn get_id(&mut self, key: &[u8]) -> usize {
        self.map.get(key).copied().unwrap_or_else(|| {
            let new_id = self.map.len();
            self.map.insert(key.into(), new_id);
            new_id
        })
    }
}

#[cfg(test)]
macro_rules! ct2u8 {
    ( $( $v:path ),* ) => {
        ct2u8!( $( $v, )* )
    };
    ( $( $v:path, )* ) => {
        [
            $(
                $v as u8,
            )*
        ]
    };
}

#[cfg(test)]
macro_rules! ct2u8vec {
    ( $( $v:path ),* ) => {
        ct2u8vec!( $( $v, )* )
    };
    ( $( $v:path, )* ) => {
        vec![
            $(
                $v as u8,
            )*
        ]
    };
}
