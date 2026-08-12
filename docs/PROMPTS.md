# Prompt Field — Options & Examples

The **Prompt** field in the GUI is free text sent to the local **Ollama
narrative model** (default `llama3.1:8b`). It is **not** an image/diagnosis
model — the diagnosis-grade facts come from the signal pipeline, and the LLM
only phrases them.

The app automatically injects the measured facts (**heart rate, total beats,
PACs, PVCs, and the RhythmCNN rhythm class**) into the prompt, and the system
prompt forces the model to (a) state every measurement, (b) write flowing prose,
and (c) not invent 12-lead findings. You don't need to restate the numbers.

## Report / summary
- `Please write a clinical report based on this single-lead ECG.` *(default)*
- `Summarize the key findings in 2–3 sentences.`
- `Write a patient-friendly explanation of these results.`

## Interpret the measured metrics
Because HR / PAC / PVC / rhythm are injected, ask the model to reason **about
those facts** (it will not re-measure the trace):
- `Given the measured ectopy, comment on its clinical significance.`
- `Is the measured heart rate appropriate for the classified rhythm?`
- `Explain what PACs and PVCs are and whether this burden is notable.`

## Rhythm & rate framing
- `Describe the rhythm and rate in one short paragraph.`
- `Does the ectopy burden warrant follow-up? Frame as screening, not diagnosis.`

## Triage tone
- `Highlight anything a clinician should review, based only on these measurements.`

## What NOT to ask
The narrative model has **no access to the raw waveform** and must not infer
image/12-lead findings. Avoid prompts that request measurements the system did
not provide:
- cardiac **axis**, lead-specific **localization** ("which wall?"),
  **R-wave progression**, **QT/PR/QRS intervals**, ST/ischemia, bundle branch
  block — none of these are measured by the current pipeline.
- Asking these will either be ignored or invite hallucination.

If you need a new measurement (e.g. QT interval), add it to the **signal
analyzer** first; then it becomes available to the prompt.

## Changing the model
Set `OLLAMA_MODEL` (and optionally `OLLAMA_URL`) before launching the GUI, e.g.
`OLLAMA_MODEL=meditron:7b python GUI/app.py`. See
[ARCHITECTURE.md](ARCHITECTURE.md) for the model rationale.

> ⚠️ Research use only — not an FDA-cleared diagnostic. Single-lead output and
> the automated PAC/PVC counts are screening estimates, not a diagnosis.
