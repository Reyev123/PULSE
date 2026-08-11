# Teach Multimodal LLMs to Comprehend Electrocardiographic Images
The code, data, and models for "Teach Multimodal LLMs to Comprehend Electrocardiographic Images".

## Dataset and Model
#### 🌐 Project Page: [Page](https://aimedlab.github.io/PULSE/)
#### 📄 Paper: [Arxiv](https://arxiv.org/abs/2410.19008)
#### 🤗 Model: [PULSE-7B](https://huggingface.co/PULSE-ECG/PULSE-7B)
#### 👩‍⚕️ Training data: [ECGInstruct](https://huggingface.co/datasets/PULSE-ECG/ECGInstruct)
#### ⚖️ Evaluation data: [ECGBench](https://huggingface.co/datasets/PULSE-ECG/ECGBench) 
#### 🩺 Demo: [Demo](https://huggingface.co/spaces/aidhlab/PULSE-7B)


> ## Single-Lead Fork (Frontier X Plus)
>
> This repository has been extended for **single-lead** wearable ECG (e.g.
> Frontier X Plus). The single-lead workflow **does not use the PULSE image
> VLM** — it is out-of-distribution on single-lead data (see
> [docs/VALIDATION.md](docs/VALIDATION.md)). Instead it uses:
>
> - **Signal analysis** ([GUI/ecg_analysis.py](GUI/ecg_analysis.py)) — heart
>   rate, PAC/PVC counts and percentages.
> - **RhythmCNN** ([models/rhythm_cnn.py](models/rhythm_cnn.py)) — 1-D CNN
>   rhythm classifier (Normal / AF / Other / Noisy), trained on PhysioNet/CinC
>   2017 (`models/rhythm_cnn.pt`).
> - **Local LLM narrative** ([GUI/narrative.py](GUI/narrative.py)) — a small
>   Ollama model (`llama3.1:8b`) that phrases the measured facts into a report.
> - **Dash GUI** ([GUI/app.py](GUI/app.py)) — upload raw signal or image, view
>   the trace, run analysis, read the report, export input + report as PDF.
>
> **Run the GUI:**
> ```shell
> conda activate pulse-llava
> python GUI/app.py           # http://127.0.0.1:8050  (open in a real browser)
> # Ollama must be running:  ollama serve  &&  ollama pull llama3.1:8b
> ```
>
> **Train the RhythmCNN:**
> ```shell
> python tools/fetch_physionet2017.py --out-dir data/physionet2017
> python models/train_rhythm_cnn.py  --data-dir data/physionet2017 --epochs 30
> ```
>
> Design and validation details: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
> [docs/VALIDATION.md](docs/VALIDATION.md), [docs/PROMPTS.md](docs/PROMPTS.md).
>
> The sections below document the **original 12-lead PULSE** model (training,
> ECGBench evaluation), retained for reference.




## Installation

```shell
conda create -n pulse-llava python=3.10 -y
conda activate pulse-llava
pip install -r GUI/requirements.txt
```

For the narrative reports, install [Ollama](https://ollama.com) and pull the
model:

```shell
ollama serve &
ollama pull llama3.1:8b
```

## Original 12-lead PULSE

The 12-lead PULSE image VLM (LLaVA subproject, ECGBench evaluation, and training
scripts) is **not part of this single-lead fork** — it was out-of-distribution
on single-lead data and has been removed here. The full 12-lead code, model, and
benchmarks remain available upstream:
[github.com/AIMedLab/PULSE](https://github.com/AIMedLab/PULSE) ·
[PULSE-7B](https://huggingface.co/PULSE-ECG/PULSE-7B) ·
[ECGBench](https://huggingface.co/datasets/PULSE-ECG/ECGBench).

## Citation
If you find this work helpful, please cite our paper:
```
@article{liu2024teach,
  title={Teach Multimodal LLMs to Comprehend Electrocardiographic Images},
  author={Ruoqi Liu, Yuelin Bai, Xiang Yue, Ping Zhang},
  journal={arXiv preprint arXiv:2410.19008},
  year={2024}
}
```

