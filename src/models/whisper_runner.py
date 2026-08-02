import os
import gc
import json
import torch
import soundfile as sf
import librosa
from datetime import datetime
from tqdm import tqdm
from transformers import pipeline

from utils.metrics import normalize_text, calculate_wer

def evaluate_whisper(model_name: str, model_path: str, ds_name: str, samples: list, logs_dir: str):
    safe_model_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("~", "").replace("-", "_").lower()
    safe_ds_name = ds_name.lower()
    model_log_dir = os.path.join(logs_dir, safe_model_name)
    os.makedirs(model_log_dir, exist_ok=True)
    log_file = os.path.join(model_log_dir, f"{safe_ds_name}.jsonl")
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    print(f"Loading {model_name} on {device}...")
    
    try:
        pipe = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            torch_dtype=torch_dtype,
            device=device,
            framework="pt",
        )
    except Exception as e:
        print(f"[ERROR] Failed to load {model_name}: {e}")
        return None

    references, predictions = [], []
    
    with open(log_file, "w", encoding="utf-8") as f:
        for i, item in enumerate(tqdm(samples, desc=f"Whisper: {model_name}")):
            try:
                if "audio_path" in item:
                    audio, sample_rate = sf.read(item["audio_path"], dtype="float32")
                else:
                    audio = item["audio_array"].astype("float32")
                    sample_rate = item["sampling_rate"]

                if audio.ndim > 1: audio = audio.mean(axis=1)
                if sample_rate != 16000: audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                
                result = pipe(audio, generate_kwargs={"language": "vietnamese", "task": "transcribe"})
                
                ref_norm = normalize_text(item["text"])
                pred_norm = normalize_text(result["text"])
                
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

    del pipe
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    gc.collect()
    
    if not references: 
        return None
    
    return calculate_wer(references, predictions)
