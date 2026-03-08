import json
import numpy as np
import whisper
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import resample_poly
from .models import Face
from asgiref.sync import sync_to_async

# Load Whisper model once (using 'base.en' for better accuracy)
model = whisper.load_model("base.en")
executor = ThreadPoolExecutor(max_workers=2)

class AudioConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_buffer = []
        self.processing = False
        self.sample_rate = 48000 # Default fallback
        self.last_transcribed_text = ""
        self.consecutive_empty_count = 0
        self.transcription_history = []

    async def connect(self):
        await self.accept()
        # print("[AudioConsumer] WebSocket connected")

    async def disconnect(self, close_code):
        # print(f"[AudioConsumer] WebSocket disconnected: {close_code}")
        pass

    async def receive(self, text_data=None, bytes_data=None):
        if bytes_data:
            # Convert incoming Float32 bytes to numpy array
            # Specify little-endian to be safe, as browsers typically use it
            audio_chunk = np.frombuffer(bytes_data, dtype='<f4')
            self.audio_buffer.append(audio_chunk)
            
            # Log every chunk received for debugging
            # if len(self.audio_buffer) % 10 == 0:
            #     print(f"[AudioConsumer] Received {len(self.audio_buffer)} audio chunks...")
            
            # Accumulate slightly more context for better transcription
            # 2048 samples at 48kHz is ~0.042 seconds per chunk
            # 100 chunks at 48kHz is ~4.2 seconds (increased from 45 for more context)
            if len(self.audio_buffer) >= 100 and not self.processing:
                print(f"[AudioConsumer] Buffer filled with {len(self.audio_buffer)} chunks. Starting processing.")
                await self.process_audio()

        elif text_data:
            try:
                data = json.loads(text_data)
                if data.get('type') == 'config':
                    self.sample_rate = data.get('sampleRate', self.sample_rate)
                    # print(f"[AudioConsumer] Set sample rate to {self.sample_rate}")
                elif data.get('type') == 'flush':
                    if self.audio_buffer and not self.processing:
                        print(f"[AudioConsumer] Flush requested. Processing {len(self.audio_buffer)} chunks.")
                        await self.process_audio(is_flush=True)
                elif data.get('type') == 'face_descriptor':
                    await self.handle_face_descriptor(data)
                else:
                    # print(f"[AudioConsumer] Received text data: {data}")
                    pass
            except Exception as e:
                # print(f"[AudioConsumer] Error parsing text data: {e}")
                pass

    async def process_audio(self, is_flush=False):
        self.processing = True
        try:
            # Concatenate chunks
            audio_data = np.concatenate(self.audio_buffer)
            
            # Identify the "new" part of the buffer (excluding the rolling tail)
            # We keep the last 25 chunks as tail, so the first 75 chunks are relatively new 
            # (if we reached 100).
            new_data = audio_data
            if not is_flush and len(self.audio_buffer) >= 100:
                # Roughly the last 75 chunks added
                new_data = np.concatenate(self.audio_buffer[-75:])

            # Check for signal in the new data
            new_max = np.max(np.abs(new_data)) if len(new_data) > 0 else 0
            
            # Use a rolling buffer: keep the last 25 chunks (~1.0s) for context in the next run
            # but only if we have enough chunks and it's NOT a flush.
            # If it's a flush, we want to clear the buffer because speech has ended.
            if is_flush:
                self.audio_buffer = []
            elif len(self.audio_buffer) > 25:
                self.audio_buffer = self.audio_buffer[-25:]
            else:
                self.audio_buffer = []

            # If the NEW data is extremely quiet, it's likely silence following speech.
            # We clear the buffer tail to prevent re-transcribing the previous speech.
            # However, if it's a flush, we SHOULD process even if it's quiet, as it might be
            # the tail end of a word.
            if not is_flush and new_max < 0.005:
                print(f"[AudioConsumer] Silence detected (max: {new_max:.5f}). Skipping transcription.")
                self.consecutive_empty_count += 1
                if self.consecutive_empty_count >= 2:
                    self.audio_buffer = []
                    self.last_transcribed_text = ""
                self.processing = False
                return

            # Reset count if there is signal
            self.consecutive_empty_count = 0

            # Run whisper in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            print(f"[AudioConsumer] Transcribing {len(audio_data)} samples...")
            result = await loop.run_in_executor(executor, self.transcribe, audio_data)
            
            text = result['text'].strip()
            print(f"[AudioConsumer] Raw transcription result: '{text}'")
            
            # Only send if the text is non-empty and DIFFERENT from the last one (if we had one)
            # or if it's much longer, implying it's a continuation.
            if text and text != self.last_transcribed_text:
                # Basic check for partial overlap: if the new text starts with the old text,
                # we might want to only send the new part, but for now we send the whole thing 
                # as it might be a corrected version.
                
                print(f"[AudioConsumer] Final transcription: {text}")
                await self.send(text_data=json.dumps({
                    'type': 'transcription',
                    'text': text
                }))
                self.last_transcribed_text = text
                
                # Keep a small history for context
                self.transcription_history.append(text)
                if len(self.transcription_history) > 5:
                    self.transcription_history.pop(0)
            elif not text:
                # If transcription is empty, we don't need to keep the buffer for context
                self.audio_buffer = []
                self.last_transcribed_text = ""
        except Exception as e:
            print(f"Error processing audio: {e}")
        finally:
            self.processing = False

    def transcribe(self, audio_np):
        # Browser sends audio at its internal sample rate (e.g., 44.1kHz or 48kHz).
        # Whisper expects 16kHz mono. 
        
        target_sr = 16000
        source_sr = self.sample_rate

        # Peak Normalization
        max_val = np.max(np.abs(audio_np))
        if max_val > 0.001: 
            audio_np = audio_np / max_val
        else:
            print(f"[AudioConsumer] Audio signal too weak for normalization (max: {max_val:.5f})")
        
        # Resample to 16kHz
        # Use resample_poly for better performance and quality
        resampled_audio = resample_poly(audio_np, target_sr, source_sr)
        
        # Use decode instead of transcribe for faster processing on chunks if possible
        # but transcribe is generally more robust for handling the 30s window.
        # We'll stick with transcribe but tune parameters.
        # beam_size=2 for better accuracy, condition_on_previous_text=True for continuity.
        
        # Build prompt from history
        history_prompt = " ".join(self.transcription_history[-3:]) if self.transcription_history else None

        return model.transcribe(
            resampled_audio.astype(np.float32), 
            fp16=False, 
            language='en', 
            beam_size=2, 
            best_of=None,
            condition_on_previous_text=True,
            initial_prompt=history_prompt
        )

    async def handle_face_descriptor(self, data):
        label = data.get('label')
        descriptor = data.get('descriptor')
        
        if not label or not descriptor:
            return

        # Check if face already exists (simple Euclidean distance)
        faces = await sync_to_async(list)(Face.objects.all())
        
        is_already_stored = False
        incoming_descriptor = np.array(descriptor)
        
        for face in faces:
            stored_descriptor = np.array(face.get_descriptor())
            dist = np.linalg.norm(incoming_descriptor - stored_descriptor)
            if dist < 0.6: # Same threshold as frontend FaceMatcher default
                is_already_stored = True
                break
        
        if is_already_stored:
            await self.send(text_data=json.dumps({
                'type': 'debug',
                'message': f"Face '{label}' is already stored in the database."
            }))
        else:
            await sync_to_async(Face.objects.create)(
                label=label,
                descriptor=json.dumps(descriptor)
            )
            print(f"[AudioConsumer] New face stored: {label}")
