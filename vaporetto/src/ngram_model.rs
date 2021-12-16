use std::collections::HashMap;
use std::io::{Read, Write};
use std::mem;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};

use crate::errors::Result;

#[derive(Clone)]
pub struct NgramData<T>
where
    T: Clone,
{
    pub(crate) ngram: T,
    pub(crate) weights: Vec<i32>,
}

impl<T> NgramData<T>
where
    T: AsRef<[u8]> + Clone,
{
    pub fn serialize<W>(&self, mut buf: W) -> Result<usize>
    where
        W: Write,
    {
        let ngram = self.ngram.as_ref();
        let ngram_size = ngram.len();
        let weights_size = self.weights.len();
        buf.write_u32::<LittleEndian>(ngram_size.try_into().unwrap())?;
        buf.write_u32::<LittleEndian>(weights_size.try_into().unwrap())?;
        buf.write_all(ngram)?;
        for &w in &self.weights {
            buf.write_i32::<LittleEndian>(w)?;
        }
        Ok(mem::size_of::<u32>() * 2 + ngram_size + mem::size_of::<i32>() * weights_size)
    }
}

impl NgramData<String> {
    pub fn deserialize<R>(mut buf: R) -> Result<Self>
    where
        R: Read,
    {
        let ngram_size = buf.read_u32::<LittleEndian>()?;
        let weights_size = buf.read_u32::<LittleEndian>()?;
        let mut ngram_bytes = vec![0; ngram_size.try_into().unwrap()];
        buf.read_exact(&mut ngram_bytes)?;
        let ngram = String::from_utf8(ngram_bytes)?;
        let mut weights = vec![];
        for _ in 0..weights_size {
            weights.push(buf.read_i32::<LittleEndian>()?);
        }
        Ok(Self { ngram, weights })
    }
}

impl NgramData<Vec<u8>> {
    pub fn deserialize<R>(mut buf: R) -> Result<Self>
    where
        R: Read,
    {
        let ngram_size = buf.read_u32::<LittleEndian>()?;
        let weights_size = buf.read_u32::<LittleEndian>()?;
        let mut ngram = vec![0; ngram_size.try_into().unwrap()];
        buf.read_exact(&mut ngram)?;
        let mut weights = Vec::with_capacity(weights_size.try_into().unwrap());
        for _ in 0..weights_size {
            weights.push(buf.read_i32::<LittleEndian>()?);
        }
        Ok(Self { ngram, weights })
    }
}

pub struct NgramModel<T>
where
    T: Clone,
{
    pub(crate) data: Vec<NgramData<T>>,
    merged: bool,
}

impl<T> NgramModel<T>
where
    T: AsRef<[u8]> + Clone,
{
    #[cfg(any(feature = "train", feature = "kytea", test))]
    pub fn new(data: Vec<NgramData<T>>) -> Self {
        Self {
            data,
            merged: false,
        }
    }

    pub fn merge_weights(&mut self) {
        if self.merged {
            return;
        }
        self.merged = true;
        let ngrams = self
            .data
            .iter()
            .cloned()
            .map(|d| (d.ngram.as_ref().to_vec(), d.weights))
            .collect::<HashMap<_, _>>();
        for NgramData { ngram, weights } in &mut self.data {
            let ngram = ngram.as_ref();
            let mut new_weights: Option<Vec<_>> = None;
            for st in (0..ngram.len()).rev() {
                if let Some(weights) = ngrams.get(&ngram[st..]) {
                    if let Some(new_weights) = new_weights.as_mut() {
                        for (w_new, w) in new_weights.iter_mut().zip(weights) {
                            *w_new += *w;
                        }
                    } else {
                        new_weights.replace(weights.clone());
                    }
                }
            }
            *weights = new_weights.unwrap();
        }
    }

    pub fn serialize<W>(&self, mut buf: W) -> Result<usize>
    where
        W: Write,
    {
        let data_size = self.data.len();
        buf.write_u32::<LittleEndian>(data_size.try_into().unwrap())?;
        let mut total_size = mem::size_of::<u32>();
        for d in &self.data {
            total_size += d.serialize(&mut buf)?;
        }
        buf.write_u8(self.merged.into())?;
        Ok(total_size + mem::size_of::<u8>())
    }
}

impl NgramModel<String> {
    pub fn deserialize<R>(mut buf: R) -> Result<Self>
    where
        R: Read,
    {
        let data_size = buf.read_u32::<LittleEndian>()?;
        let mut data = Vec::with_capacity(data_size.try_into().unwrap());
        for _ in 0..data_size {
            data.push(NgramData::<String>::deserialize(&mut buf)?);
        }
        let merged_u8 = buf.read_u8()?;
        Ok(Self {
            data,
            merged: merged_u8 != 0,
        })
    }
}

impl NgramModel<Vec<u8>> {
    pub fn deserialize<R>(mut buf: R) -> Result<Self>
    where
        R: Read,
    {
        let data_size = buf.read_u32::<LittleEndian>()?;
        let mut data = Vec::with_capacity(data_size.try_into().unwrap());
        for _ in 0..data_size {
            data.push(NgramData::<Vec<u8>>::deserialize(&mut buf)?);
        }
        let merged_u8 = buf.read_u8()?;
        Ok(Self {
            data,
            merged: merged_u8 != 0,
        })
    }
}
