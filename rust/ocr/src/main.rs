//! akh-ocr reads JSON lines `{"id", "path"}` on stdin and writes one JSON result per image on stdout.
//! After loading the models it writes `{"ready": true, "seconds": <load time>}` as the first line.
//!
//! With one recogniser and the default `--order engine` it calls `OcrEngine::recognize` and builds the text the
//! way the Akhtabooti Core does: each recognised item followed by a newline, in engine order. With several
//! recognisers it detects and crops boxes once, runs every recogniser on every crop and keeps the result with the
//! higher confidence. The crops go through `RecModel::recognize_batch`, so pixel sampling differs slightly from the
//! engine's single-recogniser path.

mod args;
mod order;

use std::any::Any;
use std::fs::File;
use std::io::{self, BufRead, BufWriter, Write};
use std::os::unix::io::FromRawFd;
use std::panic::{self, AssertUnwindSafe};
use std::process::ExitCode;
use std::time::Instant;

use image::DynamicImage;
use ocr_rs::{DetOnlyEngine, OcrEngine, OcrEngineConfig, OcrResult, OcrResult_, RecOnlyEngine};
use serde::{Deserialize, Serialize};
use serde_json::json;

use args::{Args, Order, USAGE};
use order::Line;

#[derive(Deserialize)]
struct Request {
    id: String,
    path: String,
}

#[derive(Serialize)]
struct Response {
    id: String,
    text: String,
    lines: Vec<Line>,
    seconds: f64,
    error: Option<String>,
}

enum Recognizer {
    /// One detector and one recogniser through `OcrEngine`, the call the Core makes.
    Engine(Box<OcrEngine>),
    /// One detection pass; every recogniser reads every box.
    Multi {
        det: DetOnlyEngine,
        recs: Vec<RecOnlyEngine>,
        min_confidence: f32,
    },
}

impl Recognizer {
    fn load(det: &str, recs: &[String], keys: &[String]) -> OcrResult<Self> {
        if let ([rec], [key]) = (recs, keys) {
            return Ok(Self::Engine(Box::new(OcrEngine::new(det, rec, key, None)?)));
        }
        let recs = recs
            .iter()
            .zip(keys)
            .map(|(rec, key)| OcrEngine::rec_only(rec, key, None))
            .collect::<OcrResult<_>>()?;
        Ok(Self::Multi {
            det: OcrEngine::det_only(det, None)?,
            recs,
            min_confidence: OcrEngineConfig::default().min_result_confidence,
        })
    }

    fn recognize(&self, image: &DynamicImage) -> OcrResult<Vec<OcrResult_>> {
        match self {
            Self::Engine(engine) => engine.recognize(image),
            Self::Multi {
                det,
                recs,
                min_confidence,
            } => recognize_best(det, recs, *min_confidence, image),
        }
    }
}

/// Keeps, per detected box, the recognition with the highest confidence; drops the rest as the engine does.
fn recognize_best(
    det: &DetOnlyEngine,
    recs: &[RecOnlyEngine],
    min_confidence: f32,
    image: &DynamicImage,
) -> OcrResult<Vec<OcrResult_>> {
    let (crops, boxes): (Vec<_>, Vec<_>) = det.detect_and_crop(image)?.into_iter().unzip();
    let mut best = recs[0].recognize_batch(&crops)?;
    for rec in &recs[1..] {
        for (kept, candidate) in best.iter_mut().zip(rec.recognize_batch(&crops)?) {
            if candidate.confidence > kept.confidence {
                *kept = candidate;
            }
        }
    }
    Ok(best
        .into_iter()
        .zip(boxes)
        .filter(|(rec, _)| !rec.text.is_empty() && rec.confidence >= min_confidence)
        .map(|(rec, bbox)| OcrResult_::new(rec.text, rec.confidence, bbox))
        .collect())
}

fn read_image(recognizer: &Recognizer, order: Order, path: &str) -> Result<Vec<Line>, String> {
    let image = image::open(path).map_err(|e| format!("cannot open image {path}: {e}"))?;
    let items = recognizer
        .recognize(&image)
        .map_err(|e| format!("ocr failed on {path}: {e}"))?;
    Ok(match order {
        Order::Engine => order::engine_lines(items),
        Order::Reading => order::reading_lines(items),
    })
}

fn panic_message(payload: Box<dyn Any + Send>) -> String {
    let detail = payload
        .downcast_ref::<&str>()
        .map(|s| (*s).to_owned())
        .or_else(|| payload.downcast_ref::<String>().cloned())
        .unwrap_or_default();
    format!("ocr panicked: {detail}")
}

/// Builds the text the way the Core's `ocr_rec` does: every line followed by a newline.
fn core_text(lines: &[Line]) -> String {
    let mut text = String::new();
    for line in lines {
        text.push_str(&line.text);
        text.push('\n');
    }
    text
}

fn respond(recognizer: &Recognizer, order: Order, line: &str) -> Response {
    let start = Instant::now();
    let (id, outcome) = match serde_json::from_str::<Request>(line) {
        Ok(request) => {
            let result = panic::catch_unwind(AssertUnwindSafe(|| read_image(recognizer, order, &request.path)));
            (request.id, result.unwrap_or_else(|payload| Err(panic_message(payload))))
        }
        Err(e) => (String::new(), Err(format!("invalid request line: {e}"))),
    };
    let (lines, error) = match outcome {
        Ok(lines) => (lines, None),
        Err(message) => (Vec::new(), Some(message)),
    };
    let text = core_text(&lines);
    Response {
        id,
        text,
        lines,
        seconds: start.elapsed().as_secs_f64(),
        error,
    }
}

fn version_json(args: &Args) -> serde_json::Value {
    json!({
        "akh_ocr": env!("CARGO_PKG_VERSION"),
        "ocr_rs": ocr_rs::version(),
        "mnn": ocr_rs::mnn_version(),
        "image": env!("AKH_IMAGE_VERSION"),
        "det": args.det,
        "rec": args.recs,
        "keys": args.keys,
        "order": args.order.name(),
    })
}

/// Points fd 1 at stderr so native log output cannot corrupt the protocol, and returns the original stdout.
fn protocol_stdout() -> io::Result<File> {
    // SAFETY: dup/dup2 on the process's own standard descriptors; the duplicate is owned by the returned File.
    unsafe {
        let fd = libc::dup(libc::STDOUT_FILENO);
        if fd < 0 || libc::dup2(libc::STDERR_FILENO, libc::STDOUT_FILENO) < 0 {
            return Err(io::Error::last_os_error());
        }
        Ok(File::from_raw_fd(fd))
    }
}

fn serve(recognizer: &Recognizer, order: Order, out: &mut impl Write) -> io::Result<()> {
    for line in io::stdin().lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        write_line(out, &respond(recognizer, order, &line))?;
    }
    Ok(())
}

fn write_line(out: &mut impl Write, value: &impl Serialize) -> io::Result<()> {
    serde_json::to_writer(&mut *out, value)?;
    out.write_all(b"\n")?;
    out.flush()
}

fn run(args: &Args) -> Result<(), String> {
    if args.version {
        println!("{}", version_json(args));
        return Ok(());
    }
    let det = args.require_models().map_err(|e| e.to_string())?;
    let mut out = BufWriter::new(protocol_stdout().map_err(|e| format!("cannot redirect stdout: {e}"))?);
    let start = Instant::now();
    let recognizer = Recognizer::load(det, &args.recs, &args.keys).map_err(|e| format!("cannot load models: {e}"))?;
    let ready = json!({"ready": true, "seconds": start.elapsed().as_secs_f64()});
    write_line(&mut out, &ready)
        .and_then(|()| serve(&recognizer, args.order, &mut out))
        .map_err(|e| format!("i/o error: {e}"))
}

fn main() -> ExitCode {
    let args = match Args::parse(std::env::args().skip(1)) {
        Ok(args) => args,
        Err(e) => {
            eprintln!("{e}\n{USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(message) => {
            eprintln!("akh-ocr: {message}");
            ExitCode::FAILURE
        }
    }
}
