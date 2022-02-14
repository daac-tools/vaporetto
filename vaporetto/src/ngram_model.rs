use std::io::{Read, Write};
use std::mem;

use crate::errors::Result;
use crate::utils;

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
    pub fn serialize<W>(&self, mut wtr: W) -> Result<usize>
    where
        W: Write,
    {
        let ngram = self.ngram.as_ref();
        let ngram_size = ngram.len();
        let weights_size = self.weights.len();
        utils::write_u32(&mut wtr, ngram_size.try_into().unwrap())?;
        utils::write_u32(&mut wtr, weights_size.try_into().unwrap())?;
        wtr.write_all(ngram)?;
        for &w in &self.weights {
            utils::write_i32(&mut wtr, w)?;
        }
        Ok(mem::size_of::<u32>() * 2 + ngram_size + mem::size_of::<i32>() * weights_size)
    }
}

impl NgramData<String> {
    pub fn deserialize<R>(mut rdr: R) -> Result<Self>
    where
        R: Read,
    {
        let ngram_size = utils::read_u32(&mut rdr)?;
        let weights_size = utils::read_u32(&mut rdr)?;
        let mut ngram_bytes = vec![0; ngram_size.try_into().unwrap()];
        rdr.read_exact(&mut ngram_bytes)?;
        let ngram = String::from_utf8(ngram_bytes)?;
        let mut weights = vec![];
        for _ in 0..weights_size {
            weights.push(utils::read_i32(&mut rdr)?);
        }
        Ok(Self { ngram, weights })
    }
}

impl NgramData<Vec<u8>> {
    pub fn deserialize<R>(mut rdr: R) -> Result<Self>
    where
        R: Read,
    {
        let ngram_size = utils::read_u32(&mut rdr)?;
        let weights_size = utils::read_u32(&mut rdr)?;
        let mut ngram = vec![0; ngram_size.try_into().unwrap()];
        rdr.read_exact(&mut ngram)?;
        let mut weights = Vec::with_capacity(weights_size.try_into().unwrap());
        for _ in 0..weights_size {
            weights.push(utils::read_i32(&mut rdr)?);
        }
        Ok(Self { ngram, weights })
    }
}

#[derive(Default)]
pub struct NgramModel<T>
where
    T: Clone,
{
    pub(crate) data: Vec<NgramData<T>>,
}

impl<T> NgramModel<T>
where
    T: AsRef<[u8]> + Clone,
{
    #[cfg(any(feature = "train", feature = "kytea", test))]
    pub fn new(data: Vec<NgramData<T>>) -> Self {
        Self { data }
    }

    pub fn serialize<W>(&self, mut wtr: W) -> Result<usize>
    where
        W: Write,
    {
        let data_size = self.data.len();
        utils::write_u32(&mut wtr, data_size.try_into().unwrap())?;
        let mut total_size = mem::size_of::<u32>();
        for d in &self.data {
            total_size += d.serialize(&mut wtr)?;
        }
        Ok(total_size + mem::size_of::<u8>())
    }
}

impl NgramModel<String> {
    pub fn deserialize<R>(mut rdr: R) -> Result<Self>
    where
        R: Read,
    {
        let data_size = utils::read_u32(&mut rdr)?;
        let mut data = Vec::with_capacity(data_size.try_into().unwrap());
        for _ in 0..data_size {
            data.push(NgramData::<String>::deserialize(&mut rdr)?);
        }
        Ok(Self { data })
    }
}

impl NgramModel<Vec<u8>> {
    pub fn deserialize<R>(mut rdr: R) -> Result<Self>
    where
        R: Read,
    {
        let data_size = utils::read_u32(&mut rdr)?;
        let mut data = Vec::with_capacity(data_size.try_into().unwrap());
        for _ in 0..data_size {
            data.push(NgramData::<Vec<u8>>::deserialize(&mut rdr)?);
        }
        Ok(Self { data })
    }
}
