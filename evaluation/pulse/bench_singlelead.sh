#!/bin/bash
# Batch inference for single-lead ECG (e.g. Frontier X Plus) with PULSE.
#
# Handles TWO input types:
#   -t images  INPUT is a folder of already-rendered ECG images.
#   -t raw     INPUT is a folder of Frontier X Plus signal files (csv/txt/npy),
#              rendered to images first.
#
# Reuses the PULSE batch harness (model_ecg_resume.py) — only the manifest and
# image folder differ from ECGBench. Output is a JSONL of generated reports.
#
# Usage:
#   bash bench_singlelead.sh -t images -i /path/to/images
#   bash bench_singlelead.sh -t raw    -i /path/to/signals -r 500 -c 0 -u mv
set -e

MODEL="PULSE-ECG/PULSE-7B"       # or a fine-tuned checkpoint dir
TYPE="images"                    # images | raw
INPUT=""
OUTDIR="outputs/single_lead"
PROMPT="Please write a clinical report based on this single-lead ECG image."
CONV="llava_v1"
FS=500; COLUMN=0; UNIT="mv"      # raw-only options

while getopts m:t:i:o:q:r:c:u:h opt; do
  case $opt in
    m) MODEL=$OPTARG;;
    t) TYPE=$OPTARG;;
    i) INPUT=$OPTARG;;
    o) OUTDIR=$OPTARG;;
    q) PROMPT=$OPTARG;;
    r) FS=$OPTARG;;
    c) COLUMN=$OPTARG;;
    u) UNIT=$OPTARG;;
    h) grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
  esac
done

if [[ -z "$INPUT" ]]; then echo "Error: -i INPUT is required. Use -h."; exit 1; fi

# Resolve to repo root (this script lives in evaluation/pulse/).
cd "$(dirname "$0")/../.."
mkdir -p "$OUTDIR"
MANIFEST="$OUTDIR/manifest.json"

if [[ "$TYPE" == "images" ]]; then
    IMAGE_FOLDER="$INPUT"
    python tools/render_single_lead.py prep \
        --input "$INPUT" --type images \
        --json-out "$MANIFEST" --prompt "$PROMPT"
elif [[ "$TYPE" == "raw" ]]; then
    IMAGE_FOLDER="$OUTDIR/images"
    python tools/render_single_lead.py prep \
        --input "$INPUT" --type raw --image-out "$IMAGE_FOLDER" \
        --json-out "$MANIFEST" --prompt "$PROMPT" \
        --fs "$FS" --column "$COLUMN" --unit "$UNIT"
else
    echo "Error: -t must be 'images' or 'raw'."; exit 1
fi

python LLaVA/llava/eval/model_ecg_resume.py \
    --model-path "$MODEL" \
    --image-folder "$IMAGE_FOLDER" \
    --question-file "$MANIFEST" \
    --answers-file "$OUTDIR/results.jsonl" \
    --conv-mode "$CONV"

echo "Reports written to $OUTDIR/results.jsonl"
