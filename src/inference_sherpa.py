"""
Zipformer ASR Inference Script (sherpa-onnx)
=============================================
Usage:
    python src/inference_sherpa.py --audio path/to/audio.wav
    python src/inference_sherpa.py --audio path/to/audio.wav --model_dir models/Zipformer-30M-RNNT-6000h

Supported model directories:
    models/zipformer_ssl_100h        -- 68M Zipformer fine-tuned on 100h Vietnamese (epoch-9-avg-5)
    models/Zipformer-30M-RNNT-6000h  -- 30M Zipformer trained on 6000h Vietnamese
"""

import os
import argparse
import soundfile as sf
import librosa

try:
    import sherpa_onnx
except ImportError:
    raise ImportError(
        "sherpa-onnx is not installed. Run: pip install sherpa-onnx soundfile librosa"
    )


def find_onnx_file(model_dir: str, prefix: str) -> str:
    """Find ONNX file by prefix, ensuring we load the full precision model for fair comparison."""
    # First try to find the standard (non-int8) ONNX file
    for fname in os.listdir(model_dir):
        if fname.startswith(prefix) and fname.endswith(".onnx") and "int8" not in fname:
            return os.path.join(model_dir, fname)
    # Fallback to any ONNX file if full precision is not found
    for fname in os.listdir(model_dir):
        if fname.startswith(prefix) and fname.endswith(".onnx"):
            return os.path.join(model_dir, fname)
    return ""


def create_recognizer(model_dir: str, num_threads: int = 4) -> sherpa_onnx.OfflineRecognizer:
    """Load a Zipformer Transducer recognizer from ONNX model files."""
    tokens_path  = os.path.join(model_dir, "tokens.txt")
    encoder_path = find_onnx_file(model_dir, "encoder")
    decoder_path = find_onnx_file(model_dir, "decoder")
    joiner_path  = find_onnx_file(model_dir, "joiner")

    missing = [p for p in [tokens_path, encoder_path, decoder_path, joiner_path]
               if not p or not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(
            f"Missing required model files in {model_dir}.\n"
            f"Expected: encoder-*.onnx, decoder-*.onnx, joiner-*.onnx, tokens.txt\n"
            f"Missing: {missing}"
        )

    if hasattr(sherpa_onnx.OfflineRecognizer, "from_transducer"):
        return sherpa_onnx.OfflineRecognizer.from_transducer(
            encoder=encoder_path,
            decoder=decoder_path,
            joiner=joiner_path,
            tokens=tokens_path,
            num_threads=num_threads,
            sample_rate=16000,
            feature_dim=80,
            decoding_method="greedy_search",
        )
    else:
        config = sherpa_onnx.OfflineRecognizerConfig(
            feat_config=sherpa_onnx.FeatureConfig(sample_rate=16000, feature_dim=80),
            model_config=sherpa_onnx.OfflineModelConfig(
                transducer=sherpa_onnx.OfflineTransducerModelConfig(
                    encoder=encoder_path,
                    decoder=decoder_path,
                    joiner=joiner_path,
                ),
                tokens=tokens_path,
                num_threads=num_threads,
                provider="cpu",
            ),
            decoding_method="greedy_search",
        )
        return sherpa_onnx.OfflineRecognizer(config)


def transcribe_file(recognizer: sherpa_onnx.OfflineRecognizer, audio_path: str) -> str:
    """Transcribe a single audio file."""
    audio, sample_rate = sf.read(audio_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)

    stream = recognizer.create_stream()
    stream.accept_waveform(16000, audio)
    recognizer.decode_stream(stream)
    return stream.result.text


if __name__ == "__main__":
    # Default model directory relative to project root
    _default_model = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "models", "zipformer_ssl_100h"
    )

    parser = argparse.ArgumentParser(
        description="Zipformer ASR inference via sherpa-onnx"
    )
    parser.add_argument(
        "--audio", type=str, required=True,
        help="Path to input audio file (.wav, .flac, .mp3)"
    )
    parser.add_argument(
        "--model_dir", type=str, default=_default_model,
        help="Directory containing ONNX model files (default: models/zipformer_ssl_100h)"
    )
    parser.add_argument(
        "--threads", type=int, default=4,
        help="Number of CPU threads for inference"
    )

    args = parser.parse_args()

    if not os.path.isabs(args.model_dir):
        args.model_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), args.model_dir
        )

    print(f"Model  : {args.model_dir}")
    print(f"Audio  : {args.audio}")
    recognizer = create_recognizer(args.model_dir, num_threads=args.threads)
    transcript = transcribe_file(recognizer, args.audio)

    print("\n" + "=" * 60)
    print("TRANSCRIPTION RESULT")
    print("=" * 60)
    print(transcript)
    print("=" * 60)
