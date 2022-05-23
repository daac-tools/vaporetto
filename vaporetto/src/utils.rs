use core::hash::{BuildHasher, Hasher};
use core::num::Wrapping;

use alloc::vec::Vec;

#[cfg(feature = "kytea")]
use std::io::{self, Read};

use bincode::enc::write::Writer;
use bincode::error::EncodeError;

pub struct VecWriter(pub Vec<u8>);

impl Writer for VecWriter {
    fn write(&mut self, bytes: &[u8]) -> Result<(), EncodeError> {
        self.0.extend_from_slice(bytes);
        Ok(())
    }
}

pub struct SplitMix64 {
    x: Wrapping<u64>,
}

impl Hasher for SplitMix64 {
    #[inline(always)]
    fn finish(&self) -> u64 {
        let mut z = self.x;
        z = (z ^ (z >> 30)) * Wrapping(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)) * Wrapping(0x94d049bb133111eb);
        (z ^ (z >> 31)).0
    }

    #[inline(always)]
    fn write(&mut self, bytes: &[u8]) {
        for &i in bytes {
            self.x ^= u64::from(i);
            self.x += 0x9e3779b97f4a7c15;
        }
    }

    #[inline(always)]
    fn write_u8(&mut self, i: u8) {
        self.x ^= u64::from(i);
        self.x += 0x9e3779b97f4a7c15;
    }

    #[inline(always)]
    fn write_u16(&mut self, i: u16) {
        self.x ^= u64::from(i);
        self.x += 0x9e3779b97f4a7c15;
    }

    #[inline(always)]
    fn write_u32(&mut self, i: u32) {
        self.x ^= u64::from(i);
        self.x += 0x9e3779b97f4a7c15;
    }

    #[inline(always)]
    fn write_u64(&mut self, i: u64) {
        self.x ^= i;
        self.x += 0x9e3779b97f4a7c15;
    }

    #[inline(always)]
    fn write_i8(&mut self, i: i8) {
        self.x ^= i as u64;
        self.x += 0x9e3779b97f4a7c15;
    }

    #[inline(always)]
    fn write_i16(&mut self, i: i16) {
        self.x ^= i as u64;
        self.x += 0x9e3779b97f4a7c15;
    }

    #[inline(always)]
    fn write_i32(&mut self, i: i32) {
        self.x ^= i as u64;
        self.x += 0x9e3779b97f4a7c15;
    }

    #[inline(always)]
    fn write_i64(&mut self, i: i64) {
        self.x ^= i as u64;
        self.x += 0x9e3779b97f4a7c15;
    }
}

#[derive(Clone, Copy, Default)]
pub struct SplitMix64Builder;

impl BuildHasher for SplitMix64Builder {
    type Hasher = SplitMix64;

    #[inline(always)]
    fn build_hasher(&self) -> Self::Hasher {
        SplitMix64 { x: Wrapping(0) }
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
