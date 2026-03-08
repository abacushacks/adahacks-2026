import json
import logging
import asyncio
import numpy as np
from typing import List, Optional, Dict, Any
from concurrent.futures import ThreadPoolExecutor
from channels.generic.websocket import AsyncWebsocketConsumer
from asgiref.sync import sync_to_async

from . import constants
from .services import AudioProcessingService, TranscriptionService, FaceService
from .zen_service import ZenService
from .models import Face

logger = logging.getLogger(__name__)

# Executor for running CPU-bound transcription tasks
executor = ThreadPoolExecutor(max_workers=2)

class AudioConsumer(AsyncWebsocketConsumer):
    """
    WebSocket consumer for real-time audio streaming and transcription.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_buffer: List[np.ndarray] = []
        self.processing: bool = False
        self.sample_rate: int = constants.DEFAULT_SOURCE_SAMPLE_RATE
        self.last_transcribed_text: str = ""
        self.consecutive_empty_count: int = 0
        self.transcription_history: List[str] = []
        self.active_faces: List[str] = []
        self.label_to_face_id: Dict[str, int] = {}

    async def connect(self) -> None:
        """
        Handles WebSocket connection.
        """
        # Join face_updates group
        await self.channel_layer.group_add(
            'face_updates',
            self.channel_name
        )
        await self.accept()
        logger.info("Audio WebSocket connected")

    async def disconnect(self, close_code: int) -> None:
        """
        Handles WebSocket disconnection.
        """
        # Leave face_updates group
        await self.channel_layer.group_discard(
            'face_updates',
            self.channel_name
        )
        logger.info(f"Audio WebSocket disconnected: {close_code}")

    async def receive(self, text_data: Optional[str] = None, bytes_data: Optional[bytes] = None) -> None:
        """
        Handles incoming data from the WebSocket.
        """
        if bytes_data:
            await self._handle_audio_bytes(bytes_data)
        elif text_data:
            await self._handle_text_data(text_data)

    async def _handle_audio_bytes(self, bytes_data: bytes) -> None:
        """
        Processes incoming raw audio bytes.
        """
        # Convert incoming Float32 bytes to numpy array (little-endian)
        audio_chunk = np.frombuffer(bytes_data, dtype='<f4')
        self.audio_buffer.append(audio_chunk)
        
        # Check if buffer reached threshold for processing
        if len(self.audio_buffer) >= constants.BUFFER_THRESHOLD and not self.processing:
            logger.debug(f"Buffer reached threshold ({len(self.audio_buffer)} chunks). Starting processing.")
            await self.process_audio()

    async def _handle_text_data(self, text_data: str) -> None:
        """
        Processes incoming JSON text data.
        """
        try:
            data = json.loads(text_data)
            msg_type = data.get('type')

            if msg_type == 'config':
                self.sample_rate = data.get('sampleRate', self.sample_rate)
                logger.info(f"Set sample rate to {self.sample_rate}")
            elif msg_type == 'flush':
                if self.audio_buffer and not self.processing:
                    logger.info(f"Flush requested. Processing {len(self.audio_buffer)} chunks.")
                    await self.process_audio(is_flush=True)
            elif msg_type == 'face_descriptor':
                await self.handle_face_descriptor(data)
            elif msg_type == 'active_faces':
                self.active_faces = data.get('labels', [])
                logger.info(f"Active faces updated: {self.active_faces}")
        except json.JSONDecodeError:
            logger.error("Failed to decode incoming JSON text data")
        except Exception as e:
            logger.exception(f"Error handling text data: {e}")

    async def process_audio(self, is_flush: bool = False) -> None:
        """
        Concatenates, filters, and transcribes the accumulated audio buffer.
        """
        self.processing = True
        try:
            if not self.audio_buffer:
                return

            # Prepare audio data for processing
            all_audio = np.concatenate(self.audio_buffer)
            
            # Identify the "new" part of the buffer for silence detection
            new_data = all_audio
            if not is_flush and len(self.audio_buffer) >= constants.BUFFER_THRESHOLD:
                new_data = np.concatenate(self.audio_buffer[-constants.NEW_DATA_SIZE:])

            new_max = np.max(np.abs(new_data)) if len(new_data) > 0 else 0
            
            # Manage rolling buffer
            self._manage_buffer(is_flush)

            # Silence detection logic
            if not is_flush and new_max < constants.SILENCE_THRESHOLD:
                await self._handle_silence(new_max)
                return

            self.consecutive_empty_count = 0

            # Offload processing and transcription to a thread pool
            text = await self._transcribe_audio(all_audio)
            
            # Send result if it's meaningful and new
            if text and text != self.last_transcribed_text:
                await self._send_transcription(text)
            elif not text:
                # Clear buffer if transcription is empty to prevent accumulation of noise
                self.audio_buffer = []
                self.last_transcribed_text = ""
                
        except Exception as e:
            logger.exception(f"Error during audio processing: {e}")
        finally:
            self.processing = False

    def _manage_buffer(self, is_flush: bool) -> None:
        """
        Updates the audio buffer according to the processing state.
        """
        if is_flush:
            self.audio_buffer = []
        elif len(self.audio_buffer) > constants.ROLLING_TAIL_SIZE:
            self.audio_buffer = self.audio_buffer[-constants.ROLLING_TAIL_SIZE:]
        else:
            self.audio_buffer = []

    async def _handle_silence(self, new_max: float) -> None:
        """
        Handles detected silence in the audio stream.
        """
        logger.debug(f"Silence detected (max: {new_max:.5f}). Skipping transcription.")
        self.consecutive_empty_count += 1
        if self.consecutive_empty_count >= 2:
            self.audio_buffer = []
            self.last_transcribed_text = ""

    async def _transcribe_audio(self, audio_data: np.ndarray) -> str:
        """
        Runs the full transcription pipeline: normalize -> resample -> transcribe.
        """
        loop = asyncio.get_event_loop()
        
        def pipeline():
            # 1. Normalize
            normalized_audio = AudioProcessingService.normalize(audio_data)
            # 2. Resample
            resampled_audio = AudioProcessingService.resample(normalized_audio, self.sample_rate)
            # 3. Transcribe
            result = TranscriptionService.transcribe(resampled_audio, self.transcription_history)
            return result.get('text', '').strip()

        return await loop.run_in_executor(executor, pipeline)

    async def _send_transcription(self, text: str) -> None:
        """
        Sends the transcription result back via WebSocket and updates history.
        """
        logger.info(f"Final transcription: {text}")
        await self.send(text_data=json.dumps({
            'type': 'transcription',
            'text': text
        }))
        self.last_transcribed_text = text
        
        self.transcription_history.append(text)
        if len(self.transcription_history) > constants.MAX_TRANSCRIPTION_HISTORY:
            self.transcription_history.pop(0)

        # Extract name from speech and update face
        await self._check_for_name_introduction(text)

    async def _check_for_name_introduction(self, text: str) -> None:
        """
        Checks transcription for name introductions and updates the face in DB using Zen LLM.
        """
        if len(self.active_faces) != 1:
            if len(self.active_faces) > 1:
                logger.info(f"Multiple faces on screen ({self.active_faces}). Skipping name update.")
            return

        target_label = self.active_faces[0]
        face_id = self.label_to_face_id.get(target_label)
        if not face_id:
            logger.warning(f"No face ID found for active label '{target_label}'")
            return

        parsed_data = await ZenService.parse_name_from_text(text)
        name = parsed_data.get('name')
        
        if not name:
            return

        logger.info(f"Name detected: '{name}' for face '{target_label}' (ID: {face_id})")

        try:
            # Find the face by ID and update its name
            face = await sync_to_async(lambda: Face.objects.filter(id=face_id).first())()
            if not face:
                return

            face.name = name
            await sync_to_async(face.save)(update_fields=['name'])
            logger.info(f"Updated face '{target_label}' name to '{name}'")

            # Re-run Zen with the new name for better context
            zen_context = await ZenService.get_context(face)
            relationship = zen_context.get('relationship', face.relationship or 'Known Person')
            context = zen_context.get('context', 'No recent conversations recorded.')

            # Save relationship if updated
            if relationship and relationship != face.relationship:
                face.relationship = relationship
                await sync_to_async(face.save)(update_fields=['relationship'])

            # Push the update to the frontend
            await self.send(text_data=json.dumps({
                'type': 'face_recognized',
                'label': target_label,
                'name': name,
                'metadata': face.metadata or [],
                'relationship': relationship,
                'context': context
            }))
        except Exception as e:
            logger.exception(f"Error updating face name: {e}")

    async def handle_face_descriptor(self, data: Dict[str, Any]) -> None:
        """
        Handles incoming face descriptors for registration or recognition.
        """
        label = data.get('label')
        descriptor = data.get('descriptor')
        
        if not label or not descriptor:
            return

        try:
            existing_face = await FaceService.find_existing_face(descriptor)
            
            if existing_face:
                self.label_to_face_id[label] = existing_face.id
                # Use stored relationship if available, otherwise call Zen and save it
                if existing_face.relationship:
                    # Use stored relationship, still get fresh context from Zen
                    zen_context = await ZenService.get_context(existing_face)
                    relationship = existing_face.relationship
                    context = zen_context.get('context', 'No recent conversations recorded.')
                else:
                    # First time — call Zen and save relationship to DB
                    zen_context = await ZenService.get_context(existing_face)
                    relationship = zen_context.get('relationship', 'Known Person')
                    context = zen_context.get('context', 'No recent conversations recorded.')
                    # Persist relationship to DB
                    existing_face.relationship = relationship
                    await sync_to_async(existing_face.save)(update_fields=['relationship'])
                
                await self.send(text_data=json.dumps({
                    'type': 'face_recognized',
                    'label': label,
                    'name': existing_face.name or label,
                    'metadata': existing_face.metadata or [],
                    'relationship': relationship,
                    'context': context
                }))
            else:
                new_face = await FaceService.create_face(label, descriptor)
                self.label_to_face_id[label] = new_face.id
                logger.info(f"New face stored: {label}")
        except Exception as e:
            logger.exception(f"Error handling face descriptor: {e}")

    async def face_update(self, event: Dict[str, Any]) -> None:
        """
        Handles face updates broadcasted from the database signal.
        """
        await self.send(text_data=json.dumps({
            'type': 'face_recognized',
            'label': event['label'],
            'name': event['name'],
            'metadata': event['metadata'],
            'relationship': event.get('relationship', 'Known Person'),
            'context': event.get('context', 'No recent conversations recorded.')
        }))
