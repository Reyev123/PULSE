# Architecture

The single-lead pipeline replaces the PULSE image VLM with **signal models for
facts** and a **small local LLM for narrative**.

```
single-lead signal ─┬─► ecg_analysis  (HR, PAC/PVC)           ┐
                    └─► RhythmCNN     (N / AF / Other / Noisy) ├─► facts ─► Ollama LLM ─► report
ECG image ─► ecg_digitize ─► signal ──────────────────────────┘
```

## Components

| Layer | Module | Role |
|---|---|---|
| Facts — beats | [GUI/ecg_analysis.py](../GUI/ecg_analysis.py) | HR, PAC/PVC (neurokit2 + morphology) |
| Facts — rhythm | [models/rhythm_cnn.py](../models/rhythm_cnn.py), [models/rhythm_infer.py](../models/rhythm_infer.py) | 1-D CNN: N/AF/Other/Noisy (optional, checkpoint-gated) |
| Image → signal | [GUI/ecg_digitize.py](../GUI/ecg_digitize.py) | best-effort trace digitization |
| Narrative | [GUI/narrative.py](../GUI/narrative.py) | Ollama LLM writes prose from facts |
| GUI | [GUI/app.py](../GUI/app.py) | Dash UI (single + batch) |

## Local LLM (narrative)

- Uses **Ollama** at `http://127.0.0.1:11434`, model `llama3.2:3b` (2 GB).
- Override with env `OLLAMA_MODEL` / `OLLAMA_URL`.
- A small (3B) model is sufficient; the facts are pre-computed, the LLM only
  phrases them and is instructed not to invent 12-lead findings.

## Training the RhythmCNN

```bash
python tools/fetch_physionet2017.py --out-dir data/physionet2017   # labeled, single-lead, 300 Hz
python models/train_rhythm_cnn.py  --data-dir data/physionet2017 --epochs 30
# -> models/rhythm_cnn.pt ; the GUI picks it up automatically for a Rhythm card.
```

The shipped `models/rhythm_cnn.pt` was trained on the PhysioNet/CinC 2017 set
(8,528 records, class-weighted loss): validation accuracy **0.73**, macro-F1
**0.68** (N F1 0.83, AF recall 0.82). See [VALIDATION.md](VALIDATION.md).

## Retiring PULSE

PULSE is **no longer used** by the GUI (narrative now comes from Ollama). The
PULSE loader, the `LLaVA/` subproject, the vendored flash-attn wheel, and the
ECGBench `evaluation/` scripts have all been **removed** from the codebase.
- The `models--PULSE-ECG--PULSE-7B` weights (HF cache) are no longer required;
  the app never loads them.
- The original 12-lead PULSE code lives upstream at
  [AIMedLab/PULSE](https://github.com/AIMedLab/PULSE) if ever needed again.

> Research use only - not an FDA-cleared diagnostic.
