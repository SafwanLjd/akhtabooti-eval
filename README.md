# akhtabooti-eval

Evaluation harness for [Akhtabooti](https://github.com/wlaith/akhtabooti-desktop), JOSA's offline PII scanner. It measures how well the Akhtabooti Core and candidate models find personal data in documents.

## How it works

1. **Dataset.** `akheval synth` generates Evaluation Dataset v1 in `datasets/akhtabooti-v1/` from `configs/dataset-v1.toml` and one seed. It writes 432 synthetic Arabic, English and mixed documents, each with its PII marked as character spans plus look-alike negatives. Each document gets a native file (PDF, DOCX, XLSX and others). Every identity document and part of the rest also get phone photos, scans or chat screenshots with capture damage. `akheval freeze` checksums every file.
2. **Detectors** take text and return PII spans. `core-rules` compiles the rule code of a pinned Core commit (`rust/core-rules/`). `nym`, `dialogpii` and `hf-token` run the Nym, DialogPII and CAMeLBERT models on the CPU. `hybrid` combines rules and a model.
3. **Extractors** turn a file into text: extractous (Apache Tika), tesseract, or PaddleOCR through `akh-ocr` (`rust/ocr/`), the OCR library the Core uses.
4. **Runs** send the dev or test split through an extractor and a detector. Each run scores the output against the marked values: precision, recall, F1 and F2 at span, value and document level, by label, language, channel and capture, with bootstrap intervals over documents.
5. **Comparisons** pair two runs document by document and list regressions.

`configs/eval-v1.toml` defines every detector, extractor, pipeline, run and comparison. `configs/models/` pins the model downloads by revision and sha256.

## Setup

Requires Python 3.11 or newer and cargo 1.84 or newer. Building the dataset also needs LibreOffice, poppler, tesseract with `ara` and `eng`, and the Amiri, Liberation, Noto and FreeFont fonts.

```sh
python -m venv .venv
.venv/bin/pip install -e ".[synth,extract,models,dev]"
rust/core-rules/build.sh 68d1749     # Core main, 7 Oct 2026; also 937e6ca and 97c7afb
rust/ocr/build.sh
akheval fetch                        # models listed in configs/models/
```

DialogPII needs more than 4 GB of free RAM.

## Run

From the repository root:

```sh
akheval validate                     # check the dataset
akheval suite                        # every run, then the comparisons and an index
akheval run --run text-core-main     # one run
akheval tune --detector nym-small    # threshold sweep on the dev split
akheval score-app-report --report scan-report.json --app-version 0.1.0
```

Results go to `reports/v1/`, which git ignores: one folder per run (`run.json`, `predictions.jsonl`, `summary.json`, `errors.jsonl`, `report.md`), plus comparisons, threshold sweeps and `index.md`. `run.json` records the dataset checksum, the Core commit, model checksums, settings and the machine.

## Licence

Apache-2.0
