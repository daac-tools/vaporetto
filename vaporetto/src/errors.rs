//! Definition of errors.

use std::error::Error;
use std::fmt;

pub type Result<T, E = VaporettoError> = std::result::Result<T, E>;

#[derive(Debug)]
pub enum VaporettoError {
    InvalidModel(InvalidModelError),
    InvalidSentence(InvalidSentenceError),
    InvalidArgument(InvalidArgumentError),
    IOError(std::io::Error),
    UTF8Error(std::string::FromUtf8Error),
    CastError(std::num::TryFromIntError),
}

impl VaporettoError {
    pub(crate) fn invalid_model<S>(msg: S) -> Self
    where
        S: Into<String>,
    {
        Self::InvalidModel(InvalidModelError { msg: msg.into() })
    }

    pub(crate) fn invalid_sentence<S>(msg: S) -> Self
    where
        S: Into<String>,
    {
        Self::InvalidSentence(InvalidSentenceError { msg: msg.into() })
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
            Self::InvalidSentence(e) => e.fmt(f),
            Self::InvalidArgument(e) => e.fmt(f),
            Self::IOError(e) => e.fmt(f),
            Self::UTF8Error(e) => e.fmt(f),
            Self::CastError(e) => e.fmt(f),
        }
    }
}

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

impl Error for InvalidModelError {}

/// Error used when the sentence is invalid.
#[derive(Debug)]
pub struct InvalidSentenceError {
    /// Error message.
    pub(crate) msg: String,
}

impl fmt::Display for InvalidSentenceError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "InvalidSentenceError: {}", self.msg)
    }
}

impl Error for InvalidSentenceError {}

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

impl Error for InvalidArgumentError {}

impl From<std::io::Error> for VaporettoError {
    fn from(error: std::io::Error) -> Self {
        Self::IOError(error)
    }
}

impl From<std::string::FromUtf8Error> for VaporettoError {
    fn from(error: std::string::FromUtf8Error) -> Self {
        Self::UTF8Error(error)
    }
}

impl From<std::num::TryFromIntError> for VaporettoError {
    fn from(error: std::num::TryFromIntError) -> Self {
        Self::CastError(error)
    }
}
