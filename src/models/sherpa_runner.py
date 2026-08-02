import os
import gc
import json
import soundfile as sf
import librosa
from datetime import datetime
from tqdm import tqdm
import sherpa_onnx

from utils.metrics import normalize_text, calculate_wer

def find_model_file(model_dir: str, prefix: str) -> str:
    # First try to find the standard (non-int8) ONNX file
    for f in os.listdir(model_dir):
        if f.startswith(prefix) and f.endswith(".onnx") and "int8" not in f:
            return os.path.join(model_dir, f)
    # Fallback to any ONNX file if full precision is not found
    for f in os.listdir(model_dir):
        if f.startswith(prefix) and f.endswith(".onnx"):
            return os.path.join(model_dir, f)
    return ""

def evaluate_sherpa(model_name: str, model_dir: str, ds_name: str, samples: list, logs_dir: str, num_threads: int = 4):
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("~", "").replace("-", "_").lower()
    safe_ds_name = ds_name.lower()
    model_log_dir = os.path.join(logs_dir, safe_model_name)
    os.makedirs(model_log_dir, exist_ok=True)
    log_file = os.path.join(model_log_dir, f"{safe_ds_name}.jsonl")
    
    tokens_path = os.path.join(model_dir, "tokens.txt")
    encoder_path = find_model_file(model_dir, "encoder")
    decoder_path = find_model_file(model_dir, "decoder")
    joiner_path = find_model_file(model_dir, "joiner")

    if not all([os.path.exists(tokens_path), encoder_path, decoder_path, joiner_path]):
        print(f"[ERROR] Incomplete ONNX model in {model_dir}")
        return None

    if hasattr(sherpa_onnx.OfflineRecognizer, "from_transducer"):
        recognizer = sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder_path, decoder=decoder_path, joiner=joiner_path,
            tokens=tokens_path, num_threads=num_threads, sample_rate=16000,
            feature_dim=80, decoding_method="greedy_search"
        )
    else:
        config = sherpa_onnx.OfflineRecognizerConfig(
            feat_config=sherpa_onnx.FeatureConfig(sample_rate=16000, feature_dim=80),
            model_config=sherpa_onnx.OfflineModelConfig(
                transducer=sherpa_onnx.OfflineTransducerModelConfig(
                    encoder=encoder_path, decoder=decoder_path, joiner=joiner_path
                ),
                tokens=tokens_path, num_threads=num_threads, provider="cpu"
            ),
            decoding_method="greedy_search"
        )
        recognizer = sherpa_onnx.OfflineRecognizer(config)

    references, predictions = [], []
    
    with open(log_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(samples, desc=f"Sherpa: {model_name}")):
            try:
                if "audio_path" in item:
                    audio, sample_rate = sf.read(item["audio_path"], dtype="float32")
                else:
                    audio = item["audio_array"].astype("float32")
                    sample_rate = item["sampling_rate"]

                if audio.ndim > 1: audio = audio.mean(axis=1)
                if sample_rate != 16000: audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

                stream = recognizer.create_stream()
                stream.accept_waveform(16000, audio)
                recognizer.decode_stream(stream)
                
                ref_norm = normalize_text(item["text"])
                pred_norm = normalize_text(stream.result.text)
                
                references.append(ref_norm)
                predictions.append(pred_norm)
                
                wer_sample = calculate_wer([ref_norm], [pred_norm])
                log_data = {
                    "id": item.get("id", f"sample_{i}"),
                    "ref": ref_norm,
                    "pred": pred_norm,
                    "wer": wer_sample,
                    "timestamp": datetime.now().isoformat()
                }
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[WARN] Failed predicting for sample: {e}")
                ref_norm = normalize_text(item["text"])
                pred_norm = ""
                references.append(ref_norm)
                predictions.append(pred_norm)
                log_data = {
                    "id": item.get("id", f"sample_{i}"),
                    "ref": ref_norm,
                    "pred": pred_norm,
                    "wer": 100.0,
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                f.write(json.dumps(log_data, ensure_ascii=False) + "\n")
            
    del recognizer
    gc.collect()
    
    return calculate_wer(references, predictions)
