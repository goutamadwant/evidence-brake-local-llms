# An External Brake? Retrieved evidence against false notes and false peers in local LLMs

Code, data and analysis outputs for the paper *An External Brake? Retrieved Evidence Against False Notes and False Peers
in Local LLMs* (Goutam Adwant).

A false rationale in shared text can turn a right answer into a wrong one. The study measures whether an evidence block in
the prompt reduces that loss for three local open-weight models (Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B, Gemma 4 31B QAT,
served by LM Studio) on FEVER and SciFact claims with validated false rationales. Misinformation arrives in two channels:
unattributed notes, and three scripted peers. The contrasts compare the accuracy drop under a retrieved block with no context,
and under a retrieved or gold block with a distractor block matched in length and structure, under a decision rule frozen
before any study run and calibrated by simulation.

## Layout

```
artifact/      run_cell.py (one experiment cell), analyze.py (every table, figure and reported number),
               analysis_validation.py (calibration of the decision rule), conformance_test.py, and the scripts that
               built the frames, wrote and validated the rationales, audited the blocks and checked the pilot
protocol/      protocol.yaml: pre-registered design, decision rules, statistics and kill criteria
data/          frames/ (claims, retrieved and gold blocks, matched distractors, rationales, block audits, SHA256SUMS),
               instances.json (model to server instance map), cache/ (every model answer, keyed by request)
experiments/   registry.json (every run), runs/<cell>/ (result.json, predictions.jsonl with full responses, run.json, log),
               model_manifest.json (model file hashes and server settings), serving_receipts.jsonl, pilot and selection records
analysis/      results_manifest.json, generated_numbers.tex, tables/, figures/, analysis_validation.json
```

## Setup

Python 3.9 or newer. The runner uses only the standard library; the analysis needs numpy and matplotlib. The reported
outputs were made with CPython 3.9.6, numpy 2.0.2 and matplotlib 3.9.4 (pinned in `artifact/requirements.txt`); other
versions reproduce every table entry and decision but can change the last digit of a few macros:

```bash
pip install -r artifact/requirements.txt
export GROUND_TRUTH_DIR="$PWD/data"
```

## Reproduce the analysis (no model calls)

```bash
cd artifact
python3 analyze.py
python3 conformance_test.py
```

`analyze.py` reads `experiments/registry.json`, the run outputs and the frames, and rewrites `analysis/` (10,000
bootstrap resamples, a few minutes on one machine). `conformance_test.py` checks the decision rule, the table and the
macros against fixed decision pairs.

## Run one cell

```bash
cd artifact
python3 run_cell.py --method no_context --dataset fever --seed 1 --model qwen3-coder-30b-a3b-instruct --out /tmp/cell
```

Answers already in `data/cache/` are not re-sent, so a registered cell reproduces its predictions exactly, but every cell
checks the serving provenance before it starts and so needs LM Studio (`PAPER_LMSTUDIO_HOST`, default `http://127.0.0.1:11435`) with the instance named in `data/instances.json`
loaded at context 8192, and the model files whose hashes are in `experiments/model_manifest.json`; it fails closed
otherwise. To check the reported numbers without a server, use the analysis replay above. Every cell of the study and its exact command is in `experiments/registry.json`; group methods (`group_none`,
`group_retrieved`, model `fleet`) are served one model at a time with `warm_group.py` before their cell runs.

Methods: `no_context`, `instruction`, `retrieved`, `distractor_retrieved`, `gold`, `distractor_gold`, the controls
`reasoned`, `reasoned_gold`, `reasoned_distractor_gold`, `gold_last`, `distractor_gold_last`, `skeptic`, and the group
methods; datasets `fever`, `scifact`.

## Rebuild the frames

`prepare_frames.py` rebuilds `data/frames/` from the public sources (FEVER `shared_task_dev.jsonl` and the June 2017
Wikipedia pages, indexed with SQLite FTS5; SciFact `claims_train.jsonl`, `claims_dev.jsonl` and `corpus.jsonl`) placed in
`GROUND_TRUTH_DIR`. `make_rationales.py` and `audit_blocks.py` call the writer and judge models; their outputs are in
`data/frames/` and `data/cache/`, so the study can be replayed without them.

## Protocol

`protocol/protocol.yaml` is the protocol frozen on 4 November 2026 before any study run. It was frozen in a private
repository (its sha256 is recorded in every run record) and not registered with a third party. The released copy differs
from the frozen file only as follows; no value, threshold, rule, list or code-read field changed:

- removed keys: `lane_id`, `topic_id`, `planned_evidence`, `planned_evidence_note`, `venue_reconciliation`,
  `required_disclosures` (internal planning and the disclosure checklist that the paper itself satisfies) and the empty
  `amendments` list (there were no amendments);
- renamed key: `headroom.engine_note` to `headroom.pilot_rule` (same text, with the name of the internal pilot step made
  generic);
- reworded text, references to internal build tooling, scheduling and approvals made generic: two instance notes in
  `models_or_configs`, `serving_readiness.manifest`, `serving_readiness.fail_closed`, `serving_readiness.isolation`,
  `serving_readiness.older_note`, three sentences of `kill_criteria` (now "the study is stopped"),
  `matrix.cut_order` (the approval clause of an amendment that never happened) and `compute.where`.

The frozen file's sha256 is `1b31bf62f63a154a42e451e9b6b5d85ac4451de40442f43fbc6eed2b7f37483d`; every run record in
`experiments/registry.json` carries it.

## Licences

Code: MIT (`LICENSE`). Generated outputs and rationales: MIT, except rationales written for FEVER claims and any file
that quotes FEVER text (frames, cache, predictions), which are CC BY-SA 3.0; SciFact text keeps its own licences. See
`DATA_LICENSES.md`. Model weights are not included; the GGUF builds are the LM Studio community builds
`lmstudio-community/Qwen3.6-35B-A3B-GGUF` (Q4_K_M), `lmstudio-community/Qwen3-Coder-30B-A3B-Instruct-GGUF` (Q4_K_M) and
`lmstudio-community/gemma-4-31B-it-QAT-GGUF` (Q4_0), with the judge `mistralai/Mistral-Small-3.2` (MLX 4-bit); file
hashes and server versions are in `experiments/model_manifest.json`.

## Citation

See `CITATION.cff`.
