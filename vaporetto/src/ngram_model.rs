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
        let mut check = vec![false; self.data.len()];
        let ngram_ids: HashMap<_, _> = self
            .data
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, d)| (d.ngram.as_ref().to_vec(), i))
            .collect();
        let mut stack = vec![];
        for i in 0..self.data.len() {
            if check[i] {
                continue;
            }
            stack.push(i);
            let ngram = self.data[i].ngram.as_ref();
            for j in 1..ngram.len() {
                if let Some(&k) = ngram_ids.get(&ngram[j..]) {
                    stack.push(k);
                    if check[k] {
                        break;
                    }
                }
            }
            let mut idx_from = stack.pop().unwrap();
            check[idx_from] = true;
            while let Some(idx_to) = stack.pop() {
                let mut new_weights = self.data[idx_from].weights.clone();
                for (w1, w2) in new_weights.iter_mut().zip(&self.data[idx_to].weights) {
                    *w1 += w2;
                }
                self.data[idx_to].weights = new_weights;
                idx_from = idx_to;
                check[idx_to] = true;
            }
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
