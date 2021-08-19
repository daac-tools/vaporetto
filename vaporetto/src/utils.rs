#[cfg(feature = "train")]
use std::collections::{BTreeMap, HashMap};

#[cfg(feature = "train")]
use crate::feature::Feature;

#[cfg(feature = "train")]
pub(crate) struct FeatureIDManager<'a> {
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
        if let Some(&id) = self.map.get(&feature) {
            id
        } else {
            let new_id = self.map.len() as u32;
            self.map.insert(feature, new_id);
            new_id
        }
    }
}

#[cfg(feature = "train")]
impl<'a> Default for FeatureIDManager<'a> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(feature = "train")]
pub(crate) struct LazyIndexSort {
    pub(crate) map: BTreeMap<Vec<u8>, u64>,
}

#[cfg(feature = "train")]
impl LazyIndexSort {
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }

    pub fn get_id(&mut self, key: &[u8]) -> u64 {
        if let Some(&id) = self.map.get(key) {
            id
        } else {
            let new_id = self.map.len() as u64;
            self.map.insert(key.into(), new_id);
            new_id
        }
    }

    pub fn sort(&mut self) -> Vec<usize> {
        let mut result = Vec::with_capacity(self.map.len());
        for (new_id, (_, id)) in self.map.iter_mut().enumerate() {
            result.push(*id as usize);
            *id = new_id as u64;
        }
        result
    }
}

#[cfg(test)]
macro_rules! ct2u8 {
    ( $( $v:path ),* ) => {
        ct2u8!( $( $v, )* );
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
        ct2u8vec!( $( $v, )* );
    };
    ( $( $v:path, )* ) => {
        vec![
            $(
                $v as u8,
            )*
        ]
    };
}
