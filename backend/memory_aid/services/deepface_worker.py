from __future__ import annotations

import argparse
import base64
import contextlib
import importlib
import io
import json
import sys

import cv2
import numpy as np

DeepFace = importlib.import_module("deepface.DeepFace")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--model", default="Facenet512")
    parser.add_argument("--detector", default="opencv")
    parser.add_argument("--serve", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Suppress library chatter so stdout stays protocol-safe.
    with contextlib.redirect_stdout(io.StringIO()):
        DeepFace.build_model(task="facial_recognition", model_name=args.model)
        DeepFace.build_model(task="face_detector", model_name=args.detector)

    if args.serve:
        print(json.dumps({"status": "ready"}), flush=True)
        for line in sys.stdin:
            try:
                request = json.loads(line)
                payload = analyze_image(
                    image_base64=request["image"],
                    model_name=args.model,
                    detector_backend=args.detector,
                )
                print(json.dumps({"ok": True, "result": payload}), flush=True)
            except Exception as exc:  # pragma: no cover - worker process guard
                print(
                    json.dumps(
                        {
                            "ok": False,
                            "error": f"{type(exc).__name__}: {exc}",
                        }
                    ),
                    flush=True,
                )
        return 0

    if not args.image:
        return 2

    payload = analyze_image_path(
        image_path=args.image,
        model_name=args.model,
        detector_backend=args.detector,
    )
    print(json.dumps(payload))
    return 0


def analyze_image_path(image_path: str, model_name: str, detector_backend: str) -> dict:
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            faces = DeepFace.extract_faces(
                img_path=image_path,
                detector_backend=detector_backend,
                enforce_detection=True,
                align=True,
            )

            if not faces:
                return None

            primary = max(
                faces,
                key=lambda face: face["facial_area"]["w"] * face["facial_area"]["h"],
            )
            face_region = np.clip(primary["face"] * 255, 0, 255).astype("uint8")
            representation = DeepFace.represent(
                img_path=face_region,
                model_name=model_name,
                detector_backend="skip",
                enforce_detection=False,
                normalization="base",
            )
    except Exception as exc:
        if _is_face_not_detected(exc):
            return None
        raise

    return {
        "facial_area": primary["facial_area"],
        "embedding": representation[0]["embedding"],
    }


def analyze_image(image_base64: str, model_name: str, detector_backend: str) -> dict | None:
    raw_bytes = base64.b64decode(image_base64)
    image_array = np.frombuffer(raw_bytes, dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    if image is None:
        return None

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            faces = DeepFace.extract_faces(
                img_path=image,
                detector_backend=detector_backend,
                enforce_detection=True,
                align=True,
            )

            if not faces:
                return None

            primary = max(
                faces,
                key=lambda face: face["facial_area"]["w"] * face["facial_area"]["h"],
            )
            face_region = np.clip(primary["face"] * 255, 0, 255).astype("uint8")
            representation = DeepFace.represent(
                img_path=face_region,
                model_name=model_name,
                detector_backend="skip",
                enforce_detection=False,
                normalization="base",
            )
    except Exception as exc:
        if _is_face_not_detected(exc):
            return None
        raise

    return {
        "facial_area": primary["facial_area"],
        "embedding": representation[0]["embedding"],
    }


def _is_face_not_detected(exc: Exception) -> bool:
    message = str(exc)
    return type(exc).__name__ == "FaceNotDetected" or "Face could not be detected" in message


if __name__ == "__main__":
    raise SystemExit(main())
