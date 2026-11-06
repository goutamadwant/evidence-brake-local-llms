# An External Brake? Retrieved evidence against false notes and false peers in local LLMs

Code, data and analysis outputs for the paper *An External Brake? Retrieved Evidence Against False Notes and False Peers
in Local LLMs* (Goutam Adwant).

A false rationale in shared text can turn a right answer into a wrong one. The study measures whether an evidence block in
the prompt reduces that loss for three local open-weight models (Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B, Gemma 4 31B QAT,
served by LM Studio) on FEVER and SciFact claims with validated false rationales. Misinformation arrives in two channels:
unattributed notes, and three scripted peers. Each contrast compares the accuracy drop under a retrieved or gold block with
no context, or with a distractor block matched in length and structure, under a pre-registered and calibrated decision rule.

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

Python 3.9 or newer. The runner uses only the standard library; the analysis needs numpy and matplotlib:

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

## Licences

Code, rationales and generated outputs: MIT (`LICENSE`). The frames carry FEVER and SciFact text under their own licences;
see `DATA_LICENSES.md`. Model weights are not included.

## Citation

See `CITATION.cff`.
