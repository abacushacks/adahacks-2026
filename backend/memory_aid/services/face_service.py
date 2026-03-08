from __future__ import annotations

import os
from typing import Iterable

import numpy as np


class FaceService:
    def __init__(self) -> None:
        self.match_threshold = float(os.getenv("FACE_MATCH_THRESHOLD", "0.6"))
        self.mouth_activity_threshold = float(
            os.getenv("MOUTH_ACTIVITY_THRESHOLD", "0.05")
        )

    @staticmethod
    def normalize_face_payload(face: dict) -> dict:
        frame_width = max(int(face.get("frameWidth", 0)), 1)
        frame_height = max(int(face.get("frameHeight", 0)), 1)

        x = max(int(face.get("x", 0)), 0)
        y = max(int(face.get("y", 0)), 0)
        width = max(int(face.get("width", 0)), 0)
        height = max(int(face.get("height", 0)), 0)

        if x + width > frame_width:
            width = max(0, frame_width - x)
        if y + height > frame_height:
            height = max(0, frame_height - y)

        return {
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "frameWidth": frame_width,
            "frameHeight": frame_height,
        }

    @staticmethod
    def normalize_descriptor(descriptor: list[float]) -> list[float]:
        return [float(value) for value in descriptor]

    def find_best_match(self, probe_descriptor: list[float], people: Iterable):
        best_person = None
        best_distance = None

        probe = np.asarray(probe_descriptor, dtype=np.float32)
        for person in people:
            candidate = np.asarray(person.face_embedding_data, dtype=np.float32)
            if candidate.size == 0 or candidate.size != probe.size:
                continue

            distance = float(np.linalg.norm(probe - candidate))
            if best_distance is None or distance < best_distance:
                best_distance = distance
                best_person = person

        if best_person is None or best_distance is None:
            return None

        if best_distance > self.match_threshold:
            return None

        return best_person
