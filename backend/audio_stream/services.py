import json
import numpy as np
import whisper
import logging
from typing import List, Optional, Dict, Any
from scipy.signal import resample_poly
from asgiref.sync import sync_to_async

from .models import Face
from . import constants

logger = logging.getLogger(__name__)

class AudioProcessingService:
    """
    Service for audio signal processing including normalization and resampling.
    """

    @staticmethod
    def normalize(audio_np: np.ndarray) -> np.ndarray:
        """
        Applies peak normalization to the audio signal.
        """
        max_val = np.max(np.abs(audio_np))
        if max_val > constants.MIN_SIGNAL_FOR_NORMALIZATION:
            return audio_np / max_val
        
        logger.debug(f"Audio signal too weak for normalization (max: {max_val:.5f})")
        return audio_np

    @staticmethod
    def resample(audio_np: np.ndarray, source_sr: int) -> np.ndarray:
        """
        Resamples the audio signal from source_sr to TARGET_SAMPLE_RATE.
        """
        if source_sr == constants.TARGET_SAMPLE_RATE:
            return audio_np
            
        return resample_poly(audio_np, constants.TARGET_SAMPLE_RATE, source_sr)

class TranscriptionService:
    """
    Service for transcribing audio using the Whisper model.
    """
    _model = None

    @classmethod
    def get_model(cls):
        """
        Lazy-loads the Whisper model to save resources if transcription is not needed immediately.
        """
        if cls._model is None:
            logger.info(f"Loading Whisper model '{constants.WHISPER_MODEL_NAME}'")
            cls._model = whisper.load_model(constants.WHISPER_MODEL_NAME)
        return cls._model

    @classmethod
    def transcribe(
        cls, 
        audio_np: np.ndarray, 
        history: List[str] = None
    ) -> Dict[str, Any]:
        """
        Transcribes the audio signal using Whisper.
        
        Args:
            audio_np: The audio signal to transcribe (should be 16kHz mono).
            history: Previous transcription segments for context.
            
        Returns:
            A dictionary containing the transcription result.
        """
        model = cls.get_model()
        
        # Build prompt from history
        history_prompt = None
        if history:
            history_prompt = " ".join(history[-constants.MAX_HISTORY_PROMPT_SIZE:])

        return model.transcribe(
            audio=audio_np.astype(np.float32),
            fp16=constants.WHISPER_FP16,
            language=constants.WHISPER_LANGUAGE,
            beam_size=constants.WHISPER_BEAM_SIZE,
            condition_on_previous_text=constants.WHISPER_CONDITION_ON_PREVIOUS_TEXT,
            initial_prompt=history_prompt
        )

class FaceService:
    """
    Service for handling face descriptor operations.
    """

    @staticmethod
    async def find_existing_face(incoming_descriptor: List[float]) -> Optional[Face]:
        """
        Searches for an existing face that matches the given descriptor.
        """
        faces = await sync_to_async(list)(Face.objects.all())
        incoming_np = np.array(incoming_descriptor)
        
        for face in faces:
            stored_descriptor = np.array(face.get_descriptor())
            distance = np.linalg.norm(incoming_np - stored_descriptor)
            if distance < constants.FACE_DISTANCE_THRESHOLD:
                return face
        
        return None

    @staticmethod
    async def create_face(label: str, descriptor: List[float]) -> Face:
        """
        Stores a new face in the database.
        """
        return await sync_to_async(Face.objects.create)(
            label=label,
            descriptor=json.dumps(descriptor)
        )
