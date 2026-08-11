# PULSE Prompt Field — Options & Examples

The **Prompt** field in the GUI is free text sent to the PULSE-7B model. The
app automatically prepends the measured **HR / PAC / PVC** so PULSE's narrative
stays consistent with the counts — you don't need to restate them.

PULSE is instruction-tuned on ECG image tasks (ECGInstruct / ECGBench), so it
handles report generation, diagnosis, yes/no, and multiple-choice questions.

## Report / summary
- `Please write a clinical report based on this ECG image.` *(default)*
- `Summarize the key findings in 2–3 sentences.`
- `Explain the ECG findings step by step for teaching.`

## Rhythm & rate — best fit for single-lead
- `What is the cardiac rhythm?`
- `Is the rhythm regular or irregular, and what is the heart rate?`
- `Is there evidence of atrial fibrillation or atrial flutter?`
- `Are there ectopic/premature beats? Classify them if possible.`

## Targeted yes/no or diagnosis
- `Does this ECG show a bundle branch block? If so, which type?`
- `Are there signs of ischemia or prior infarction?`
- `Is the QT interval prolonged?`
- `List all abnormalities you can identify.`

## Intervals / measurements
- `Comment on the PR, QRS, and QT intervals.`
- `Estimate the heart rate and comment on whether it is bradycardic or tachycardic.`

## Multiple-choice (ECGBench style)
```
Question: What is the most likely rhythm?
Options: A) Sinus rhythm  B) Atrial fibrillation  C) Atrial flutter  D) SVT
Answer with the option letter and a brief justification.
```

## Triage
- `Are there any critical findings that require urgent attention?`

## Interpret the measured metrics
Because the measured HR/PAC/PVC are injected into the prompt, you can ask PULSE
to reason about them:
- `Given the measured ectopy, comment on its clinical significance.`
- `Is the measured heart rate appropriate for the observed rhythm?`

## Limitations for single-lead use
Questions that require the full **12-lead layout** are **not reliable** from a
single lead — avoid or interpret with caution:
- cardiac **axis**
- lead-specific **localization** ("which wall is affected?")
- **R-wave progression**

Prefer **rhythm, rate, ectopy, and AF-screening** prompts.

> ⚠️ Research use only — not an FDA-cleared diagnostic. Single-lead output and
> the automated PAC/PVC counts are screening estimates, not a diagnosis.
