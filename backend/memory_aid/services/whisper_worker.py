from __future__ import annotations

import argparse
import contextlib
import io

import librosa
import numpy as np
from scipy.io import wavfile
import whisper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio")
    parser.add_argument("--model", default="base")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.self_test:
        with contextlib.redirect_stdout(io.StringIO()):
            whisper.load_model(args.model)
        print("ok")
        return 0

    if not args.audio:
        return 2

    audio = load_wav_audio(args.audio)

    # Keep stdout clean for the parent process parser.
    with contextlib.redirect_stdout(io.StringIO()):
        model = whisper.load_model(args.model)
        result = model.transcribe(
            audio,
            fp16=False,
            language="en",
            task="transcribe",
            verbose=False,
        )

    text = (result.get("text", "") or "").strip()
    print(text)
    return 0


def load_wav_audio(path: str) -> np.ndarray:
    sample_rate, audio = wavfile.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    if np.issubdtype(audio.dtype, np.integer):
        max_value = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / float(max_value)
    else:
        audio = audio.astype(np.float32)

    if sample_rate != 16000:
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
    return audio


if __name__ == "__main__":
    raise SystemExit(main())
