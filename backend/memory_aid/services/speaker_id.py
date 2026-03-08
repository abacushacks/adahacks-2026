from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np


@dataclass
class SpeakerSignature:
    vector: np.ndarray


class SpeakerIdentifier:
    def __init__(self, similarity_threshold: float = 0.84) -> None:
        self.similarity_threshold = similarity_threshold
        self.reference_signature: SpeakerSignature | None = None

    def clear(self) -> None:
        self.reference_signature = None

    def set_reference(self, samples: np.ndarray, sample_rate: int) -> None:
        signature = self._build_signature(samples, sample_rate)
        self.reference_signature = signature

    def is_patient_voice(self, samples: np.ndarray, sample_rate: int) -> bool:
        if self.reference_signature is None:
            return False

        probe_signature = self._build_signature(samples, sample_rate)
        if probe_signature is None:
            return False

        similarity = self._cosine_similarity(
            self.reference_signature.vector, probe_signature.vector
        )
        return similarity >= self.similarity_threshold

    def _build_signature(
        self, samples: np.ndarray, sample_rate: int
    ) -> SpeakerSignature | None:
        if samples.size < int(sample_rate * 0.25):
            return None

        normalized = librosa.util.normalize(samples.astype(np.float32))
        if self._rms(normalized) < 0.01:
            return None

        mfcc = librosa.feature.mfcc(y=normalized, sr=sample_rate, n_mfcc=20)
        delta = librosa.feature.delta(mfcc)
        centroid = librosa.feature.spectral_centroid(y=normalized, sr=sample_rate)
        rolloff = librosa.feature.spectral_rolloff(y=normalized, sr=sample_rate)

        vector = np.concatenate(
            [
                np.mean(mfcc, axis=1),
                np.mean(delta, axis=1),
                np.mean(centroid, axis=1),
                np.mean(rolloff, axis=1),
            ]
        ).astype(np.float32)
        return SpeakerSignature(vector=vector)

    @staticmethod
    def _cosine_similarity(lhs: np.ndarray, rhs: np.ndarray) -> float:
        denominator = np.linalg.norm(lhs) * np.linalg.norm(rhs)
        if denominator == 0:
            return 0.0
        return float(np.dot(lhs, rhs) / denominator)

    @staticmethod
    def _rms(samples: np.ndarray) -> float:
        return float(np.sqrt(np.mean(np.square(samples))))
