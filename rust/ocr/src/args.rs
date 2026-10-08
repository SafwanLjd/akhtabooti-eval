//! Command-line flags.

use std::fmt;

pub const USAGE: &str = "usage: akh-ocr --det <model.mnn> --rec <model.mnn> --keys <keys.txt> \
[--rec <model.mnn> --keys <keys.txt> ...] [--order engine|reading] [--version]

Reads JSON lines {\"id\", \"path\"} on stdin and writes one JSON line per image on stdout.
--order engine (default) keeps the Core's output; --order reading merges boxes into visual rows.
Repeat --rec/--keys to run several recognisers on each detected box and keep the higher score.";

/// Output order of the recognised boxes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Order {
    /// The order ocr-rs returns, which is what the Core joins.
    Engine,
    /// One line per visual row, rows top to bottom, boxes in a row left to right.
    Reading,
}

impl Order {
    fn parse(value: &str) -> Result<Self, ArgsError> {
        match value {
            "engine" => Ok(Self::Engine),
            "reading" => Ok(Self::Reading),
            other => Err(ArgsError(format!("--order must be engine or reading, got {other}"))),
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Engine => "engine",
            Self::Reading => "reading",
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct ArgsError(pub String);

impl fmt::Display for ArgsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct Args {
    pub det: Option<String>,
    pub recs: Vec<String>,
    pub keys: Vec<String>,
    pub order: Order,
    pub version: bool,
}

impl Args {
    pub fn parse(argv: impl IntoIterator<Item = String>) -> Result<Self, ArgsError> {
        let mut args = Self {
            det: None,
            recs: Vec::new(),
            keys: Vec::new(),
            order: Order::Engine,
            version: false,
        };
        let mut argv = argv.into_iter();
        while let Some(flag) = argv.next() {
            match flag.as_str() {
                "--version" => args.version = true,
                "--det" => args.det = Some(value(&flag, argv.next())?),
                "--rec" => args.recs.push(value(&flag, argv.next())?),
                "--keys" => args.keys.push(value(&flag, argv.next())?),
                "--order" => args.order = Order::parse(&value(&flag, argv.next())?)?,
                other => return Err(ArgsError(format!("unknown argument {other}"))),
            }
        }
        if args.recs.len() != args.keys.len() {
            return Err(ArgsError(format!(
                "got {} --rec and {} --keys; pass them in pairs",
                args.recs.len(),
                args.keys.len()
            )));
        }
        Ok(args)
    }

    /// Checks the flags needed to run recognition.
    pub fn require_models(&self) -> Result<&str, ArgsError> {
        let det = self
            .det
            .as_deref()
            .ok_or_else(|| ArgsError("--det is required".into()))?;
        if self.recs.is_empty() {
            return Err(ArgsError("at least one --rec/--keys pair is required".into()));
        }
        Ok(det)
    }
}

fn value(flag: &str, next: Option<String>) -> Result<String, ArgsError> {
    next.ok_or_else(|| ArgsError(format!("{flag} needs a value")))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(line: &str) -> Result<Args, ArgsError> {
        Args::parse(line.split_whitespace().map(str::to_owned))
    }

    #[test]
    fn parses_repeated_recognisers() {
        let args = parse("--det d --rec a --keys ka --rec b --keys kb --order reading").unwrap();
        assert_eq!(args.det.as_deref(), Some("d"));
        assert_eq!(args.recs, ["a", "b"]);
        assert_eq!(args.keys, ["ka", "kb"]);
        assert_eq!(args.order, Order::Reading);
    }

    #[test]
    fn rejects_unpaired_keys() {
        assert!(parse("--det d --rec a").is_err());
    }

    #[test]
    fn rejects_unknown_order() {
        assert!(parse("--order columns").is_err());
    }

    #[test]
    fn version_needs_no_models() {
        let args = parse("--version").unwrap();
        assert!(args.version);
        assert!(args.require_models().is_err());
    }
}
