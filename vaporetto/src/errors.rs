//! Definition of errors.

use core::fmt;

use alloc::string::String;

#[cfg(feature = "std")]
use std::error::Error;

/// A specialized Result type for Vaporetto.
pub type Result<T, E = VaporettoError> = core::result::Result<T, E>;

/// The error type for Vaporetto.
#[derive(Debug)]
pub enum VaporettoError {
    /// The error variant for [`InvalidModelError`].
    InvalidModel(InvalidModelError),

    /// The error variant for [`InvalidArgumentError`].
    InvalidArgument(InvalidArgumentError),

    /// The error variant for [`FromUtf8Error`](alloc::string::FromUtf8Error).
    UTF8Error(alloc::string::FromUtf8Error),

    /// The error variant for [`TryFromIntError`](core::num::TryFromIntError).
    CastError(core::num::TryFromIntError),

    /// The error variant for [`DecodeError`](bincode::error::DecodeError).
    DecodeError(bincode::error::DecodeError),

    /// The error variant for [`EncodeError`](bincode::error::EncodeError).
    EncodeError(bincode::error::EncodeError),

    /// The error variant for [`std::io::Error`].
    #[cfg(feature = "std")]
    #[cfg_attr(docsrs, doc(cfg(feature = "std")))]
    IOError(std::io::Error),
}

impl VaporettoError {
    pub(crate) fn invalid_model<S>(msg: S) -> Self
    where
        S: Into<String>,
    {
        Self::InvalidModel(InvalidModelError { msg: msg.into() })
    }

    pub(crate) fn invalid_argument<S>(arg: &'static str, msg: S) -> Self
    where
        S: Into<String>,
    {
        Self::InvalidArgument(InvalidArgumentError {
            arg,
            msg: msg.into(),
        })
    }
}

impl fmt::Display for VaporettoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidModel(e) => e.fmt(f),
            Self::InvalidArgument(e) => e.fmt(f),
            Self::UTF8Error(e) => e.fmt(f),
            Self::CastError(e) => e.fmt(f),
            Self::DecodeError(e) => e.fmt(f),
            Self::EncodeError(e) => e.fmt(f),

            #[cfg(feature = "std")]
            Self::IOError(e) => e.fmt(f),
        }
    }
}

#[cfg(feature = "std")]
impl Error for VaporettoError {}

/// Error used when the model is invalid.
#[derive(Debug)]
pub struct InvalidModelError {
    /// Error message.
    pub(crate) msg: String,
}

impl fmt::Display for InvalidModelError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "InvalidModelError: {}", self.msg)
    }
}

#[cfg(feature = "std")]
impl Error for InvalidModelError {}

/// Error used when the argument is invalid.
#[derive(Debug)]
pub struct InvalidArgumentError {
    /// Name of the argument.
    pub(crate) arg: &'static str,

    /// Error message.
    pub(crate) msg: String,
}

impl fmt::Display for InvalidArgumentError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "InvalidArgumentError: {}: {}", self.arg, self.msg)
    }
}

#[cfg(feature = "std")]
impl Error for InvalidArgumentError {}

impl From<alloc::string::FromUtf8Error> for VaporettoError {
    fn from(error: alloc::string::FromUtf8Error) -> Self {
        Self::UTF8Error(error)
    }
}

impl From<core::num::TryFromIntError> for VaporettoError {
    fn from(error: core::num::TryFromIntError) -> Self {
        Self::CastError(error)
    }
}

impl From<bincode::error::DecodeError> for VaporettoError {
    fn from(error: bincode::error::DecodeError) -> Self {
        Self::DecodeError(error)
    }
}

impl From<bincode::error::EncodeError> for VaporettoError {
    fn from(error: bincode::error::EncodeError) -> Self {
        Self::EncodeError(error)
    }
}

#[cfg(feature = "std")]
impl From<std::io::Error> for VaporettoError {
    fn from(error: std::io::Error) -> Self {
        Self::IOError(error)
    }
}
