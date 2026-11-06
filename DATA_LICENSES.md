# Data licences

The files in `data/frames/`, `data/cache/` and the `predictions.jsonl` files under `experiments/runs/` contain text from
two public datasets, which keeps its original licence:

- **FEVER** (claims and Wikipedia sentences from the June 2017 pre-processed dump): Creative Commons Attribution-ShareAlike
  3.0, with Wikipedia material under the Wikipedia copyright terms. Source and licence:
  https://fever.ai/dataset/fever.html and https://fever.ai/download/fever/license.html.
  Thorne et al., "FEVER: a Large-scale Dataset for Fact Extraction and VERification", NAACL 2018.
- **SciFact** claims and evidence annotations: Creative Commons Attribution 4.0. Corpus abstracts (from S2ORC): Open Data
  Commons Attribution License 1.0. Source and licence: https://github.com/allenai/scifact (LICENSE.md).
  Wadden et al., "Fact or Fiction: Verifying Scientific Claims", EMNLP 2020.

The false rationales, block audits and model answers were produced for this study. Rationales written for FEVER claims,
and any file quoting FEVER text, are released under CC BY-SA 3.0, as the FEVER licence requires for adapted material; the
other rationales and outputs are released under the MIT licence of the repository, except where they quote dataset text.
The rationales are deliberately false stimuli (see `data/README.md`).

Model weights are not included. The study used Qwen3.6-35B-A3B, Qwen3-Coder-30B-A3B, Gemma 4 31B QAT and Mistral Small 3.2
(judge), all released under Apache-2.0; their file hashes are in `experiments/model_manifest.json`.
