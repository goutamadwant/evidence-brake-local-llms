#!/usr/bin/env python3
"""Pin what a run must serve: model identity read from the server, model-file hashes from the engine's manifest of the Studio
(config/studio_models.json, written by `paper llm manifest`), the frame hashes and the server runtimes. run_cell.py fails a cell when
the server or the frames differ."""
from __future__ import annotations

import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from make_rationales import JUDGE, WRITER
from run_cell import FLEET, GROUND_TRUTH, HOST, INSTANCES, MANIFEST

SERVING_CONTEXT, SERVING_PARALLEL = 8192, 4


def main() -> None:
    studio = json.loads((Path(__file__).resolve().parents[3] / "config" / "studio_models.json").read_text(encoding="utf-8"))
    with urllib.request.urlopen(f"{HOST}/api/v0/models", timeout=30) as resp:
        served = {row["id"]: row for row in json.load(resp)["data"]}
    wanted = sorted(set(FLEET) | {WRITER, JUDGE})
    models = {m: {**{k: served[m].get(k) for k in ("arch", "quantization", "compatibility_type", "max_context_length")}, "files": [{k: f[k] for k in ("path", "bytes", "sha256")} for f in studio["files"] if f["model_key"] == m]} for m in wanted}
    if len(sys.argv) > 1:  # the judge is outside the engine's backends, so its files are hashed separately on the Studio (shasum -a 256 output)
        models[JUDGE]["files"] = [{"path": line.split(None, 1)[1], "sha256": line.split(None, 1)[0]} for line in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines()]
    sums = (GROUND_TRUTH / "frames" / "SHA256SUMS").read_text(encoding="utf-8")
    MANIFEST.write_text(json.dumps({"server": studio.get("lmstudio"), "studio_manifest_generated_at": studio.get("generated_at"), "serving": {"context": SERVING_CONTEXT, "parallel": SERVING_PARALLEL, "instances": INSTANCES}, "models": models, "frames_sha256sums": hashlib.sha256(sums.encode()).hexdigest(), "frames": sums.splitlines()}, indent=1) + "\n", encoding="utf-8")
    print(MANIFEST)


if __name__ == "__main__":
    main()
