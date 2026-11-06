# Data

- `frames/` holds the claim frames, retrieved and gold blocks, matched distractors, block audits and the stimuli, listed
  with their hashes in `frames/SHA256SUMS` (the runner checks those hashes, so the folder must stay unchanged).
- `instances.json` maps each model to the server instance name used in the study.
- `cache/` holds every model answer, keyed by request, under the hash of the model manifest.

## The rationales are false on purpose

`frames/*.rationales.jsonl` hold, for each claim, three short arguments written by a language model for each label
(`rationales["TRUE"]` and `rationales["FALSE"]`). The arguments for the label the claim does **not** have are false by
construction and feed the false notes and false peers; the arguments for the true label feed the true peers and were
checked only for the label they argue, not for factual accuracy, so they can also contain false details. None of them is
a statement of fact; some concern real people or biomedical findings. Do not quote them as statements about the world. The official labels are in `frames/fever.jsonl` and `frames/scifact.jsonl` (field `gold`).
