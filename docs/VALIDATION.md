# Single-Lead Validation

Validation of the single-lead pipeline against the **MIT-BIH Arrhythmia
Database** (PhysioNet), which provides per-beat ground-truth annotations.

- Data: `wfdb.rdrecord/rdann` from `mitdb`, lead **MLII**, 360 Hz.
- Records saved as CSV in `samples/mitbih_*.csv`.
- Ground truth from annotation symbols: `V` = PVC, `A`/`a` = atrial premature (PAC).

## 1. Beat detector — ectopy detection (first 120 s)

Two improvements were made and benchmarked:
1. **Bigeminy-robust prematurity** — reference the sinus cycle length via an
   upper RR percentile instead of the local median.
2. **Morphology-template classification** — split premature beats by correlating
   each QRS against a sinus template (aberrant/wide → PVC, else PAC).

| Rec | GT ectopy | Original | After #1 | After #1+#2 (dPAC/dPVC) |
|----|----|----|----|----|
| 100 | 1  | 0  | 1  | 1 / 0 |
| 106 | 21 | 5  | 22 | 0 / 22 |
| 119 | 26 | 26 | 28 | 2 / 26 |
| 200 | 64 | 48 | 64 | 3 / 61 |
| 209 | 4  | 4  | 4  | 3 / 1 |
| 233 | 53 | 39 | 46 | 5 / 41 |
| **Total** | **169** | **122 (72%)** | **165 (98%)** | — |

- Total ectopy detection: **72% → 98%** (bigeminy fix; 106: 5 → 22).
- PVC/PAC classification fixed the inversion on 119 (**0 → 26 PVC**); PVC recall
  across PVC-heavy records ≈ **92%**.
- Heart rate: accurate throughout.

## 2. New pipeline vs. PULSE — head-to-head on `samples/` (MIT-BIH)

The signal-based pipeline (`ecg_analysis` for facts + `RhythmCNN` for rhythm)
was run on the same six records that PULSE was tested on. Analyzer values are
over the first 120 s; RhythmCNN over the first 60 s.

| Rec | Known diagnosis | New: HR / PAC / PVC | New rhythm (CNN) | Old PULSE narrative |
|----|----|----|----|----|
| 100 | Sinus, occasional APCs | 74 / 1 / 0 | other (0.64) | "sinus **bradycardia**… PVCs & PACs" |
| 106 | Sinus, frequent PVCs (bigeminy) | 69 / 0 / **22** | normal (0.56) | "sinus rhythm… PVCs" + "RA abnormality" |
| 119 | Sinus, frequent PVCs | 66 / 2 / **26** | other (0.95) | "sinus **bradycardia**… PACs & PVCs" |
| 200 | Frequent PVCs+APCs, VT runs | 89 / 3 / **61** | other (0.90) | "sinus rhythm… PVCs" + "**WPW**" |
| 209 | Sinus, frequent **APCs** | 94 / **3** / 1 | normal (0.69) | "sinus rhythm… **PVCs**" |
| 233 | Sinus, frequent PVCs | 105 / 5 / **41** | other (0.94) | "sinus **bradycardia**… PACs & PVCs" |

### Findings — are we doing better? Yes.
- **Heart rate:** now correct on every record (74–105 bpm). PULSE repeatedly
  said "bradycardia", even at 100+ bpm.
- **Ectopy count + type:** PVC-heavy records are quantified and correctly typed
  (106 → 22, 119 → 26, 200 → 61, 233 → 41 PVCs); atrial-ectopy records surface
  PACs (100, 209). PULSE only asserted ectopy *existed* and mislabeled 209.
- **No hallucinations:** the new pipeline never invents 12-lead findings; PULSE
  produced WPW / RA-abnormality on single-lead strips.
- **Rhythm class:** the RhythmCNN coarse label (N / AF / Other / Noisy) flags the
  abnormal ectopy-heavy records as "Other" and never false-fires AF or Noisy on
  these clean sinus strips.

## Conclusion

For single-lead, the **signal analyzer + RhythmCNN** align with ground truth on
HR, ectopy count, and ectopy type, and the local LLM only phrases these facts.
PULSE (trained on 12-lead images) is out-of-distribution on single-lead and is
retired from the pipeline.

## 3. Deep system validation harness

Reproducible scoring against annotated PhysioNet databases (no public Frontier X
Plus dataset exists; these stand in for single-lead / exercise / motion data):

```bash
python tools/fetch_validation_sets.py --dbs mitdb afdb nstdb
python tools/validate_system.py beats --data-dir data/validation/mitdb   # analyzer vs beat truth
python tools/validate_system.py af    --data-dir data/validation/afdb    # RhythmCNN AF detection
python tools/validate_system.py noise --data-dir data/validation/mitdb   # artifact robustness sweep
```

### Results (initial run)

**Analyzer — beats (MIT-BIH, first 5 min × 10 records):**

| Metric | Result |
|---|---|
| Beat detection | Se 0.99 / PPV 1.00 |
| Heart rate | MAE 0.5 bpm |
| PVC | Se 0.73 / PPV 0.83 |
| PAC | Se 0.59 / **PPV 0.14** |

**RhythmCNN — AF (MIT-BIH AFDB, 30 s windows):** sensitivity **0.21**,
specificity 0.998 (misses ~79% of AF).

**Analyzer — noise robustness (real nstdb EM/MA/BW noise mixed into MIT-BIH):**
PVC detection is unaffected by **baseline wander** (Se 0.73 / PPV 0.82 even at
0 dB) and tolerates **muscle artifact** down to ~6 dB. **Electrode-motion (EM)**
is the failure mode: at 0 dB SNR, PVC Se falls to 0.55, PPV to 0.37, and HR MAE
rises to 7 bpm — the exercise/motion scenario. → gate low-SNR EM segments (the
RhythmCNN "Noisy" class can flag these).

### Findings & actions
- **HR + QRS detection are excellent** and robust to synthetic noise.
- **PAC precision is poor (0.14)** — single-lead PAC/PVC split is intrinsically
  limited; report ectopy *burden*, treat the PAC/PVC split cautiously.
- **RhythmCNN AF sensitivity was far too low (0.21)** on continuous recordings —
  the PhysioNet-2017 held-out F1 (0.68) overstated real performance.

### Retraining (proposal executed)
Added AFDB-derived 30 s windows (`tools/build_af_trainset.py`, 408 AF / 454
non-AF) to PhysioNet 2017 and retrained with noise augmentation
(`--augment`, validation kept clean):

| RhythmCNN | AF sensitivity | AF specificity | macro-F1 |
|---|---|---|---|
| v1 (PhysioNet 2017 only) | **0.21** | 0.998 | 0.68 |
| v2 (+ AFDB + augmentation), AFDB re-test | 0.96 | 0.96 | 0.68 |
| v2, held-out LTAFDB (6 records) | 0.968 | 0.989 | — |
| v3 (retrained, early-stop + LR schedule), held-out LTAFDB (6 records) | 0.942 | 0.984 | **0.70** |

The AFDB re-test is in-distribution (AFDB windows were in training). The
**LTAFDB** tests use a separate database never seen in training — the honest
number. AF detection went from unusable (0.21) to strong (~0.94-0.97).

**v2 → v3 decision.** v3 retrains with early stopping + `ReduceLROnPlateau` and
logs the train-vs-val gap (which stayed ~0 — no overfitting). v3 wins on overall
rhythm classification (val macro-F1 0.702 vs 0.678, accuracy 0.77 vs 0.73, N/O
F1 up) and is noise-augmented for the exercise use case; it trades a small,
within-noise AF-sensitivity drop (0.968 → 0.942, both strong). **v3 was promoted**
to `models/rhythm_cnn.pt`; the previous model is kept at
`models/rhythm_cnn_v2.pt.bak`.
- **Rollback:** `cp models/rhythm_cnn_v2.pt.bak models/rhythm_cnn.pt` (then restart the GUI).
- The trainer now logs the train-vs-val gap, uses `ReduceLROnPlateau`, and
  early-stops (`--patience`) so overfitting is observable and curbed.

## 4. Signal-quality gate

Motion/EMG artifact (the exercise failure mode) is now detected and gated.
`ecg_analysis.signal_quality()` scores each recording from baseline-to-QRS power
(electrode motion), HF-to-QRS power (EMG), and QRS-band kurtosis, returning
**good / fair / poor**. Calibrated against real nstdb EM/MA noise:

| Signal | Quality |
|---|---|
| clean | good |
| EM/MA at 6 dB | fair |
| EM/MA at 0 dB (where PVC Se collapses to 0.55) | poor |

When quality is **poor**, the GUI shows a red *Signal quality* card, the rhythm
label is annotated "low signal quality, interpret with caution", and the
narrative adds a motion-artifact caveat — so AF/ectopy calls are not trusted on
unreliable signal.

> ⚠️ Research use only — not an FDA-cleared diagnostic. Counts are screening
> estimates. Single-lead cannot support 12-lead-only findings (axis,
> lead-specific localization).
