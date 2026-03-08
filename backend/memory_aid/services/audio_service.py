from __future__ import annotations

import base64
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import librosa
import numpy as np
from scipy.io import wavfile

from memory_aid.services.speaker_id import SpeakerIdentifier


logger = logging.getLogger(__name__)


def decode_pcm16_audio(base64_audio: str) -> np.ndarray:
    if not base64_audio:
        return np.empty(0, dtype=np.float32)

    raw_bytes = base64.b64decode(base64_audio)
    pcm = np.frombuffer(raw_bytes, dtype=np.int16)
    if pcm.size == 0:
        return np.empty(0, dtype=np.float32)
    return pcm.astype(np.float32) / 32768.0


class WhisperAudioService:
    def __init__(self) -> None:
        self.target_sample_rate = int(os.getenv("WHISPER_SAMPLE_RATE", "16000"))
        self.model_name = os.getenv("WHISPER_MODEL", "base")
        self.minimum_transcribe_seconds = float(
            os.getenv("MIN_TRANSCRIBE_SECONDS", "2.5")
        )
        self.maximum_buffer_seconds = float(os.getenv("MAX_AUDIO_BUFFER_SECONDS", "6.0"))
        self.silence_threshold = float(os.getenv("AUDIO_SILENCE_THRESHOLD", "0.01"))
        self.speaker_identifier = SpeakerIdentifier()
        self.buffer = np.empty(0, dtype=np.float32)
        self.backend_dir = Path(__file__).resolve().parents[2]
        self.worker_python = os.getenv("AI_WORKER_PYTHON", sys.executable)
        self.transcription_supported = self._check_transcription_support()

    def reset(self) -> None:
        self.buffer = np.empty(0, dtype=np.float32)
        self.speaker_identifier.clear()

    def clear_buffer(self) -> None:
        self.buffer = np.empty(0, dtype=np.float32)

    def set_reference_speaker(self, base64_audio: str, sample_rate: int) -> None:
        samples = decode_pcm16_audio(base64_audio)
        self.speaker_identifier.set_reference(samples, sample_rate)

    def process_chunk(
        self,
        base64_audio: str,
        sample_rate: int,
        allow_capture: bool,
    ) -> str | None:
        samples = decode_pcm16_audio(base64_audio)
        if samples.size == 0:
            return None

        if self.speaker_identifier.is_patient_voice(samples, sample_rate):
            return None

        if not self.transcription_supported:
            return None

        if not allow_capture:
            self.clear_buffer()
            return None

        normalized = self._resample(samples, sample_rate)
        if self._rms(normalized) < self.silence_threshold:
            return None

        self.buffer = np.concatenate([self.buffer, normalized])
        max_length = int(self.maximum_buffer_seconds * self.target_sample_rate)
        if self.buffer.size > max_length:
            self.buffer = self.buffer[-max_length:]

        if self.buffer.size < int(self.minimum_transcribe_seconds * self.target_sample_rate):
            return None

        transcript = self.transcribe(self.buffer.copy())
        self.clear_buffer()
        cleaned = transcript.strip()
        return cleaned or None

    def transcribe_clip(
        self,
        base64_audio: str,
        sample_rate: int,
        exclude_reference_speaker: bool = False,
    ) -> str | None:
        samples = decode_pcm16_audio(base64_audio)
        if samples.size == 0 or not self.transcription_supported:
            return None

        if exclude_reference_speaker and self.speaker_identifier.is_patient_voice(
            samples, sample_rate
        ):
            return None

        normalized = self._resample(samples, sample_rate)
        if self._rms(normalized) < self.silence_threshold:
            return None

        transcript = self.transcribe(normalized)
        cleaned = transcript.strip()
        return cleaned or None

    def transcribe(self, samples: np.ndarray) -> str:
        with tempfile.TemporaryDirectory(prefix="whisper_") as temp_dir:
            temp_path = Path(temp_dir)
            wav_path = temp_path / "chunk.wav"

            pcm16 = np.clip(samples, -1.0, 1.0)
            pcm16 = (pcm16 * 32767).astype(np.int16)
            wavfile.write(wav_path, self.target_sample_rate, pcm16)

            command = [
                self.worker_python,
                "-m",
                "memory_aid.services.whisper_worker",
                "--audio",
                str(wav_path),
                "--model",
                self.model_name,
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
                cwd=str(self.backend_dir),
                env=self._worker_env(),
            )
            if result.returncode != 0:
                logger.warning(
                    "Whisper worker failed (code=%s): %s",
                    result.returncode,
                    result.stderr.strip(),
                )
                return ""
            return result.stdout.strip()

    def _check_transcription_support(self) -> bool:
        command = [
            self.worker_python,
            "-m",
            "memory_aid.services.whisper_worker",
            "--self-test",
            "--model",
            self.model_name,
        ]
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=20,
                cwd=str(self.backend_dir),
                env=self._worker_env(),
            )
        except subprocess.TimeoutExpired:
            logger.warning("Whisper self-test timed out; transcription disabled.")
            return False

        if result.returncode != 0:
            logger.warning(
                "Whisper self-test failed; transcription disabled: %s",
                result.stderr.strip(),
            )
            return False

        return True

    def _worker_env(self) -> dict:
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        backend_path = str(self.backend_dir)
        if existing_pythonpath:
            env["PYTHONPATH"] = f"{backend_path}{os.pathsep}{existing_pythonpath}"
        else:
            env["PYTHONPATH"] = backend_path
        return env

    def _resample(self, samples: np.ndarray, sample_rate: int) -> np.ndarray:
        if sample_rate == self.target_sample_rate:
            return samples.astype(np.float32)

        resampled = librosa.resample(
            samples.astype(np.float32),
            orig_sr=sample_rate,
            target_sr=self.target_sample_rate,
        )
        return resampled.astype(np.float32)

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        if samples.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(samples))))
