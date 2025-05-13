use core::hash::{BuildHasher, Hash, Hasher};
use core::ops::{Deref, DerefMut};

use alloc::vec::Vec;

#[cfg(feature = "kytea")]
use std::io::{self, Read};

use bincode::{
    de::Decoder,
    enc::{write::Writer, Encoder},
    error::{DecodeError, EncodeError},
    Decode, Encode,
};
use hashbrown::{DefaultHashBuilder, HashMap};

#[cfg(feature = "fix-weight-length")]
#[inline(always)]
pub const fn trim_end_zeros(mut w: &[i32]) -> &[i32] {
    while let Some((&last, rest)) = w.split_last() {
        if last != 0 {
            break;
        }
        w = rest;
    }
    w
}

pub struct VecWriter(pub Vec<u8>);

impl Writer for VecWriter {
    fn write(&mut self, bytes: &[u8]) -> Result<(), EncodeError> {
        self.0.extend_from_slice(bytes);
        Ok(())
    }
}

#[derive(Clone, Debug, Default)]
pub struct SerializableHashMap<K, V, S = DefaultHashBuilder>(pub HashMap<K, V, S>);

impl<K, V, S> Deref for SerializableHashMap<K, V, S> {
    type Target = HashMap<K, V, S>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<K, V, S> DerefMut for SerializableHashMap<K, V, S> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<K, V, S, Context> Decode<Context> for SerializableHashMap<K, V, S>
where
    K: Decode<Context> + Eq + Hash,
    V: Decode<Context>,
    S: BuildHasher + Default,
{
    fn decode<D: Decoder<Context = Context>>(decoder: &mut D) -> Result<Self, DecodeError> {
        let mut result = HashMap::with_hasher(S::default());
        let size: u64 = Decode::decode(decoder)?;
        for _ in 0..size {
            let k = Decode::decode(decoder)?;
            let v = Decode::decode(decoder)?;
            result.insert(k, v);
        }
        Ok(Self(result))
    }
}

impl<K, V, S> Encode for SerializableHashMap<K, V, S>
where
    K: Encode,
    V: Encode,
{
    fn encode<E: Encoder>(&self, encoder: &mut E) -> Result<(), EncodeError> {
        let size = u64::try_from(self.0.len()).unwrap();
        Encode::encode(&size, encoder)?;
        for (k, v) in &self.0 {
            Encode::encode(k, encoder)?;
            Encode::encode(v, encoder)?;
        }
        Ok(())
    }
}

// Copied from https://prng.di.unimi.it/splitmix64.c
pub struct SplitMix64 {
    x: u64,
}

impl SplitMix64 {
    const fn add(&mut self, i: u64) {
        self.x ^= i;
        self.x = self.x.wrapping_add(0x9e3779b97f4a7c15);
        self.x = (self.x ^ (self.x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        self.x = (self.x ^ (self.x >> 27)).wrapping_mul(0x94d049bb133111eb);
        self.x = self.x ^ (self.x >> 31);
    }
}

impl Hasher for SplitMix64 {
    #[inline(always)]
    fn finish(&self) -> u64 {
        self.x
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        for &i in bytes {
            self.add(u64::from(i));
        }
    }

    #[inline(always)]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }

    #[inline(always)]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }

    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }

    #[inline(always)]
    fn write_i8(&mut self, i: i8) {
        self.add(i as u64);
    }

    #[inline(always)]
    fn write_i16(&mut self, i: i16) {
        self.add(i as u64);
    }

    #[inline(always)]
    fn write_i32(&mut self, i: i32) {
        self.add(i as u64);
    }

    #[inline(always)]
    fn write_i64(&mut self, i: i64) {
        self.add(i as u64);
    }
}

#[derive(Clone, Copy, Default)]
pub struct SplitMix64Builder;

impl BuildHasher for SplitMix64Builder {
    type Hasher = SplitMix64;

    #[inline(always)]
    fn build_hasher(&self) -> Self::Hasher {
        SplitMix64 { x: 0 }
    }
}

#[cfg(feature = "kytea")]
pub fn read_u8<R>(mut rdr: R) -> io::Result<u8>
where
    R: Read,
{
    let mut buf = [0];
    rdr.read_exact(&mut buf)?;
    Ok(buf[0])
}

#[cfg(feature = "kytea")]
pub fn read_u16<R>(mut rdr: R) -> io::Result<u16>
where
    R: Read,
{
    let mut buf = [0; 2];
    rdr.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

#[cfg(feature = "kytea")]
pub fn read_i16<R>(mut rdr: R) -> io::Result<i16>
where
    R: Read,
{
    let mut buf = [0; 2];
    rdr.read_exact(&mut buf)?;
    Ok(i16::from_le_bytes(buf))
}

#[cfg(feature = "kytea")]
pub fn read_u32<R>(mut rdr: R) -> io::Result<u32>
where
    R: Read,
{
    let mut buf = [0; 4];
    rdr.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

#[cfg(feature = "kytea")]
pub fn read_i32<R>(mut rdr: R) -> io::Result<i32>
where
    R: Read,
{
    let mut buf = [0; 4];
    rdr.read_exact(&mut buf)?;
    Ok(i32::from_le_bytes(buf))
}

#[cfg(feature = "kytea")]
pub fn read_f64<R>(mut rdr: R) -> io::Result<f64>
where
    R: Read,
{
    let mut buf = [0; 8];
    rdr.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}
