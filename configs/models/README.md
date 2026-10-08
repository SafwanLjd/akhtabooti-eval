# Model manifests

`akheval fetch` reads every `*.toml` file in this directory and downloads what it lists; `--only <name>` limits it to one entry.

| Entry | Keys | Behaviour |
|---|---|---|
| `[[file]]` | `name`, `url`, `path`, `sha256` | Downloads `url` to `path` and checks the sha256. A file already at `path` with that checksum is kept. |
| `[[hf]]` | `name`, `repo`, `revision`, `path`, `allow` | Calls `huggingface_hub.snapshot_download` for `repo` at commit `revision` into `path`, limited to the `allow` glob patterns. It then writes `path/FETCHED.json` with the repository, revision and the sha256 of every file. |

Set `revision` to a commit sha so a rerun fetches the same files. Names must be unique across all files, and paths are relative to the repository root.
