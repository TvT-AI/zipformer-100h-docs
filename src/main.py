import os
import sys
import json
from datetime import datetime

import argparse

from data.loaders import get_dataset
from models.sherpa_runner import evaluate_sherpa
from models.whisper_runner import evaluate_whisper

def main():
    parser = argparse.ArgumentParser(description="Run ASR benchmarks")
    parser.add_argument("--force", action="store_true", help="Force re-evaluate all datasets, ignoring progress")
    args = parser.parse_args()

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    base_dir = os.path.dirname(os.path.dirname(__file__))
    config_file = os.path.join(base_dir, "configs", "benchmark_config.json")
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")
    results_dir = os.path.join(base_dir, "results")
    metrics_dir = os.path.join(results_dir, "metrics")
    logs_dir = os.path.join(results_dir, "logs")
    
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    if not os.path.exists(config_file):
        print(f"[ERROR] Config file not found at {config_file}")
        return
        
    with open(config_file, "r", encoding="utf-8") as f:
        config = json.load(f)
        
    print("=" * 60)
    print("LOADING DATASETS (ML Research Architecture)")
    print("=" * 60)
    
    loaded_datasets = {}
    for ds_name, ds_config in config.get("datasets", {}).items():
        print(f"Attempting to load {ds_name}...")
        samples = get_dataset(data_dir, ds_config)
        if samples:
            loaded_datasets[ds_name] = samples
            print(f" -> Loaded {ds_name}: {len(samples)} samples.")
        else:
            print(f" -> Skipped {ds_name} (data not found or empty).")

    if not loaded_datasets:
        print("[ERROR] No datasets loaded. Exiting.")
        return

    models = config.get("models", {})
    

    progress_file = os.path.join(results_dir, "benchmark_progress.json")
    if os.path.exists(progress_file) and not args.force:
        print(f"\nResuming from {progress_file}")
        with open(progress_file, "r", encoding="utf-8") as f:
            results = json.load(f)
    else:
        results = {m: {d: "N/A" for d in loaded_datasets} for m in models}

    for m in models:
        if m not in results: results[m] = {}
        for d in loaded_datasets:
            if d not in results[m]: results[m][d] = "N/A"

    print("\n" + "=" * 60)
    print("STARTING BENCHMARK EXECUTION")
    print("=" * 60)

    for model_name, info in models.items():
        print(f"\n--- Model: {model_name} ---")
        # Ensure path is absolute for local models
        model_path = info["path"]
        if info["type"] == "sherpa" and not os.path.isabs(model_path):
            model_path = os.path.join(models_dir, model_path)
            
        for ds_name, samples in loaded_datasets.items():
            if not args.force and results[model_name].get(ds_name) not in ["N/A", "Error"] and results[model_name].get(ds_name) != None:
                print(f"Skipping {model_name} on {ds_name}, already evaluated: {results[model_name][ds_name]}")
                continue
            
            try:
                if info["type"] == "whisper":
                    wer = evaluate_whisper(model_name, model_path, ds_name, samples, logs_dir)
                elif info["type"] == "sherpa":
                    wer = evaluate_sherpa(model_name, model_path, ds_name, samples, logs_dir)
                else:
                    wer = None
                    print(f"[ERROR] Unknown model type: {info['type']}")
                
                if wer is not None:
                    results[model_name][ds_name] = f"{wer:.2f}%"
                    print(f"Result {ds_name}: {wer:.2f}% WER")
            except Exception as e:
                print(f"[ERROR] Failed to evaluate {model_name} on {ds_name}: {e}")
                results[model_name][ds_name] = "Error"
            
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=4)

    # Save Markdown Report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(metrics_dir, f"benchmark_report_{timestamp}.md")
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write("# Final Evaluation Report\n\n")
        f.write("| Model | " + " | ".join(loaded_datasets.keys()) + " |\n")
        f.write("|" + "|".join(["---"] * (len(loaded_datasets) + 1)) + "|\n")
        
        for model_name, ds_results in results.items():
            row = [model_name] + [str(ds_results.get(ds, "N/A")) for ds in loaded_datasets]
            f.write("| " + " | ".join(row) + " |\n")

    print(f"\nEvaluation complete. Full report saved to: {out_file}")

if __name__ == "__main__":
    main()
