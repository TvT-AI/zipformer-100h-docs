# Zipformer ASR Benchmark — WER Results

Word Error Rate (%) across Vietnamese ASR benchmarks.

> **Note:** This table is a hand-curated aggregated summary that includes both local empirical evaluations and external paper references (e.g., GigaSpeech 2, Common Voice 17.0). Running `python src/main.py` will generate a fresh local evaluation report in `results/metrics/` containing only the datasets you have locally available.

Per-sample logs: `results/logs/{model}/{dataset}.jsonl` — run `python src/main.py` to generate.

## Source Classification

| Model | Params | Source | Engine |
| :--- | :---: | :---: | :--- |
| Zipformer 30M 6000h | 30M | Empirical | sherpa-onnx |
| Zipformer SSL 100h | 68M | Empirical | sherpa-onnx |
| Whisper Tiny / Small / Medium / Large-v3 | 74M–1542M | Empirical | HF Transformers |

## Benchmark WER (%)

| Model | Params | GigaSpeech 2 | Common Voice 17.0 | FLEURS | VIVOS | VLSP 2020 | Custom 10h |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Whisper Tiny | 74M | N/A | N/A | 79.12 | 79.15 | 80.76 | 152.71 ‡ |
| Whisper Small | 244M | 23.40 [1] | 17.80 [1] | 21.15 | 22.22 | 29.47 | 20.77 |
| Whisper Medium | 769M | 19.10 [1] | 14.50 [1] | 12.23 | 18.98 | 24.07 | 12.17 |
| Whisper Large-v3 | 1542M | 17.94 [1] | 13.74 [1] | **7.86** | 16.78 | 32.12 | 19.63 |
| **Zipformer SSL 100h** | **68M** | N/A | N/A | 10.71 | **6.23** | **10.47** | **9.37** |
| **Zipformer 30M 6000h** | **30M** | N/A | N/A | **9.23** | **4.64** | **9.98** | **6.91** |

**Notes:**
- `N/A` = Not evaluated locally as the dataset is not available in this study.
- `‡` = WER > 100% due to hallucination — Whisper Tiny repeats/fabricates tokens on compressed meeting audio.
- `[1]` = Benchmark results referenced from Yang et al., GigaSpeech 2 (2024), Table 3, Vietnamese baseline. These models were evaluated by the authors of the paper.

# Models
This directory contains inference-ready model files.
## Zipformer SSL 100h (not included)
Our custom model, fine-tuned via HuBERT self-supervised learning on 100h of Vietnamese data.
ONNX files are already included in this repo under `zipformer_ssl_100h/`.
> **Note:**  zipformer_ssl_100h model must be placed here manually (not distributed in this repo).
> Contact the authors or generate from the training recipe.
```
models/zipformer_ssl_100h/
├── encoder-epoch-9-avg-5.onnx   
├── decoder-epoch-9-avg-5.onnx   
├── joiner-epoch-9-avg-5.onnx    
├── bpe.model                    
└── tokens.txt                   
```
## Zipformer 30M 6000h (download from HuggingFace)
```bash
python -c "
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='hynt/Zipformer-30M-RNNT-6000h',
    local_dir='models/Zipformer-30M-RNNT-6000h',
    ignore_patterns=['*.pt']   # skip jit checkpoint, only need ONNX
)
"
```
## Whisper models (auto-downloaded)
Whisper models are **automatically downloaded** from HuggingFace when you run the benchmark.
No manual setup needed. Models are cached by the `transformers` library.
