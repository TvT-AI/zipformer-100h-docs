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
