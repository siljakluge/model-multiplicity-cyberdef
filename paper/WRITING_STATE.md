# Paper Writing State

This file is a checkpoint so the paper work can resume cleanly after aborted Codex/OpenClaw turns.

## Current Status

- LaTeX paper skeleton exists in `paper/main.tex`.
- Section files exist under `paper/sections/`.
- `introduction.tex` contains a first complete draft.
- `methodology.tex` contains a first complete draft of the implemented pipeline.
- `experiments.tex`, `results.tex`, `discussion.tex`, and `conclusion.tex` are placeholders.
- `references.bib` contains initial references from the Bachelor thesis bibliography.

## Next Small Chunks

1. Expand `paper/sections/experiments.tex` with the concrete NSL-KDD configuration from `configs/default.yaml` and `configs/smoke.yaml`.
2. Add citations into `introduction.tex` and `methodology.tex` where the current text mentions model multiplicity, Rashomon sets, SHAP, and explanation variability.
3. Add a short related-work section if the target venue expects it.
4. Run final experiments and write `results.tex`.
5. Write `discussion.tex` and finalize `conclusion.tex`.

## Robust Workflow

- Make one small paper change at a time.
- Commit and push after each finished section or subsection.
- Do not combine long PDF extraction, writing, verification, and pushing in one large turn.
