#!/usr/bin/env python3
"""Serve the group methods one model at a time on a machine that cannot hold all three: fill the cache for one model and one round, with
only its instance loaded. Round 1 runs for every model first, then round 2, since round 2 prompts quote the round 1 answers. Prompts and
request seeds are the ones run_cell.py sends, so the cell then reads every answer back from the cache and sends no request."""
from __future__ import annotations

import argparse

from run_cell import load_study_frames, run_group


def main() -> None:
    ap = argparse.ArgumentParser()
    for name in ("method", "dataset", "model"):
        ap.add_argument(f"--{name}", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--round", type=int, choices=(1, 2), required=True)
    args = ap.parse_args()
    frames, _ = load_study_frames(args.method, args.dataset)
    run_group(frames, args.method, args.seed, args.dataset, warm=(args.model, args.round))


if __name__ == "__main__":
    main()
