import os
import json
from datasets import load_from_disk


def _load_local_paths(base_dir: str) -> dict:
    """Load optional local path overrides from configs/local_paths.json.
    Users should copy configs/local_paths.example.json -> configs/local_paths.json
    and set their local dataset paths. This file is gitignored."""
    local_paths_file = os.path.join(base_dir, "configs", "local_paths.json")
    if os.path.exists(local_paths_file):
        with open(local_paths_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {k: v for k, v in data.items() if not k.startswith("_") and v}
    return {}

def load_vivos(data_dir: str):
    vivos_root = os.path.join(data_dir, "vivos")
    split_dir = os.path.join(vivos_root, "test")
    prompts_file = os.path.join(split_dir, "prompts.txt")
    waves_dir = os.path.join(split_dir, "waves")

    if not os.path.exists(prompts_file):
        return None

    # Precompute a map of all wav files to their absolute paths to avoid O(N^2) scanning
    wav_files = {}
    for root, _, files in os.walk(waves_dir):
        for f in files:
            if f.endswith(".wav"):
                wav_files[f] = os.path.join(root, f)

    samples = []
    with open(prompts_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            parts = line.split(maxsplit=1)
            if len(parts) < 2: continue
            utt_id, text = parts[0], parts[1]
            spk_id = utt_id.split("_")[0]
            audio_path = os.path.join(waves_dir, spk_id, f"{utt_id}.wav")
            
            if os.path.exists(audio_path):
                samples.append({"id": utt_id, "audio_path": audio_path, "text": text})
            else:
                fallback_path = wav_files.get(f"{utt_id}.wav")
                if fallback_path:
                    samples.append({"id": utt_id, "audio_path": fallback_path, "text": text})
    return samples

def load_lhotse_custom(dataset_dir: str):
    recordings_file = os.path.join(dataset_dir, "recordings.jsonl")
    supervisions_file = os.path.join(dataset_dir, "supervisions.jsonl")

    if not os.path.exists(recordings_file) or not os.path.exists(supervisions_file):
        return None

    recordings = {}
    with open(recordings_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            recordings[data["id"]] = data["sources"][0]["source"]

    samples = []
    with open(supervisions_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            rec_id = data["recording_id"]
            text = data.get("text", "")
            if rec_id in recordings:
                audio_path = recordings[rec_id]
                if not os.path.isabs(audio_path):
                    audio_path = os.path.join(dataset_dir, audio_path)
                samples.append({"id": rec_id, "audio_path": audio_path, "text": text})
    return samples
def load_metadata_jsonl(dataset_dir: str):
    metadata_path = os.path.join(dataset_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        return None
        
    samples = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            data = json.loads(line)
            audio_path = data.get("audio_filepath", data.get("path", ""))
            text = data.get("text", data.get("transcription", ""))
            
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(dataset_dir, audio_path)
                
            if os.path.exists(audio_path):
                samples.append({"id": data.get("id", f"sample_{len(samples)}"), "audio_path": audio_path, "text": text})
    return samples

def load_hf_local(dataset_dir: str):
    if not os.path.exists(dataset_dir) or not os.listdir(dataset_dir):
        return None
    try:
        from datasets import Audio
        import io
        import soundfile as sf
        
        ds = load_from_disk(dataset_dir)
        # Prevent datasets library from using torchcodec by disabling auto-decode
        ds = ds.cast_column("audio", Audio(decode=False))
        
        samples = []
        for i, item in enumerate(ds):
            sample_id = item.get("id", item.get("path", f"sample_{i}"))
            text = item.get("transcription", item.get("sentence", item.get("text", "")))
            
            # Read bytes using soundfile directly
            audio_bytes = item["audio"]["bytes"]
            if audio_bytes is not None:
                audio_array, sampling_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")
            else:
                # Fallback if path is provided instead of bytes
                audio_path = item["audio"]["path"]
                if not audio_path or not os.path.exists(audio_path):
                    print(f"[WARN] Audio path not found: {audio_path}, skipping sample")
                    continue
                audio_array, sampling_rate = sf.read(audio_path, dtype="float32")
                
            samples.append({
                "id": sample_id,
                "audio_array": audio_array,
                "sampling_rate": sampling_rate,
                "text": text
            })
        return samples
    except Exception as e:
        print(f"[WARN] Failed to load local HF dataset from {dataset_dir}: {e}")
        return None

def get_dataset(data_dir: str, ds_config: dict):
    ds_type = ds_config["type"]
    ds_path_key = ds_config["path"]
    ds_path = os.path.join(data_dir, ds_path_key)

    # Allow per-dataset path overrides via configs/local_paths.json (gitignored)
    # Copy configs/local_paths.example.json -> configs/local_paths.json to set overrides.
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    local_paths = _load_local_paths(base_dir)
    if ds_path_key in local_paths:
        override = local_paths[ds_path_key]
        if os.path.exists(override):
            ds_path = override

    if ds_type == "vivos":
        return load_vivos(data_dir)
    elif ds_type == "lhotse":
        return load_lhotse_custom(ds_path)
    elif ds_type == "nemo_jsonl":
        return load_metadata_jsonl(ds_path)
    elif ds_type == "hf_local":
        return load_hf_local(ds_path)
    return None
