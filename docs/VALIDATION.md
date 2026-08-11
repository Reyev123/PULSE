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

## 2. PULSE narrative vs. known diagnosis (10 s strip)

PULSE-7B was run on rendered single-lead images and compared to the known
diagnosis and the signal analyzer.

| Rec | Known diagnosis | Analyzer | PULSE narrative | Verdict |
|----|----|----|----|----|
| 100 | Sinus, occasional APCs | HR 76, PAC 1 | "sinus **bradycardia**… PVCs & PACs" | rate wrong, false PVC |
| 106 | Sinus, frequent PVCs (bigeminy) | HR 60 | "sinus rhythm… **PVCs**" + "RA abnormality" | PVC ✓, hallucination |
| 119 | Sinus, frequent PVCs | HR 66, PVC 2 | "sinus **bradycardia**… PACs & PVCs" | PVC ✓, rate wrong |
| 200 | Frequent PVCs+APCs, VT runs | HR 94, PVC 7 | "sinus rhythm… PVCs" + "**WPW**" | PVC ✓, hallucination |
| 209 | Sinus, frequent **APCs** | HR 94 | "sinus rhythm… **PVCs**" | wrong (atrial called PVC) |
| 233 | Sinus, frequent PVCs | HR 100, PVC 5 | "sinus **bradycardia**… PACs & PVCs" | PVC ✓, rate badly wrong |

### Findings
- PULSE usually detects *that* ectopy exists, but on single-lead images it:
  - gets **rate wrong** (repeatedly says "bradycardia", even at 100 bpm),
  - **hallucinates** specific diagnoses (WPW, RA abnormality),
  - **confuses PAC vs PVC** (209).
- The deterministic signal analyzer aligns better with ground truth for HR and
  ectopy type.

## Conclusion

For single-lead, **trust the signal analyzer for quantitative facts (HR, PAC/PVC)**
and treat PULSE's single-lead prose as unreliable. PULSE was trained on 12-lead
images; single-lead is out-of-distribution.

> ⚠️ Research use only — not an FDA-cleared diagnostic. Counts are screening
> estimates. Single-lead cannot support 12-lead-only findings (axis,
> lead-specific localization).
