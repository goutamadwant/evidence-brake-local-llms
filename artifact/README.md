# Evidence against false notes and false peers: replication artifact

Measures, for three local open-weight models on FEVER and SciFact claims, whether evidence blocks reduce the accuracy lost to false
rationales, compared with no context and with a distractor block matched in length and structure.

## Contents

| File | Purpose |
|---|---|
| `prepare_frames.py` | Builds the claim frames, retrieved blocks and matched distractors from the public datasets |
| `make_rationales.py` | Writes and validates the false rationales (the stimuli) |
| `audit_blocks.py` | Judge audit of every evidence and distractor block |
| `selection_check.py` | No-context accuracy on all frame claims, by validation outcome |
| `feasibility.py` | Pre-freeze feasibility run |
| `pilot_checks.py` | Values for the pilot kill criteria |
| `run_cell.py` | One matrix cell: one method, dataset, seed and model |
| `warm_group.py` | Serves the group methods one model at a time into the answer cache |
| `capture_manifest.py` | Pins model files, server settings and frame hashes |
| `analyze.py` | Results manifest, generated numbers, table and figure |
| `analysis_validation.py` | Calibration of the decision rule, run before any study cell |
| `reproduce.sh` | Regenerates the analysis outputs from the registry |

The runner uses only the Python standard library; `requirements.txt` lists the analysis packages.

## One cell

```bash
python3 run_cell.py --method no_context --dataset fever --seed 1 --model qwen3-coder-30b-a3b-instruct --out /tmp/cell
```

A cell needs an LM Studio server with the named instance of `instances.json` in the data folder loaded at context 8192, and it fails
closed otherwise. Answers already in the cache are not re-sent, so a registered cell reproduces its predictions exactly, but the provenance check still
needs the server; `analyze.py` replays every reported number without one.

## Analysis

```bash
./reproduce.sh
```

## Data

FEVER (CC BY-SA 3.0) and SciFact (claims and annotations CC BY 4.0) are obtained from their official sources. The rationales are
stimuli, not evidence, and are published with the artifact.
