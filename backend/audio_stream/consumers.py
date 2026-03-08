import json
import numpy as np
import whisper
import asyncio
from channels.generic.websocket import AsyncWebsocketConsumer
from concurrent.futures import ThreadPoolExecutor
from scipy.signal import resample

# Load Whisper model once (using 'tiny.en' for maximum speed)
model = whisper.load_model("tiny.en")
executor = ThreadPoolExecutor(max_workers=2)

class AudioConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audio_buffer = []
        self.processing = False
        self.sample_rate = 48000 # Default fallback

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
            
            # Accumulate less context for faster updates (e.g., ~1.5 seconds worth)
            # 2048 samples at 48kHz is ~0.042 seconds per chunk
            # 35 chunks at 48kHz is ~1.5 seconds
            if len(self.audio_buffer) >= 35 and not self.processing:
                await self.process_audio()

        elif text_data:
            try:
                data = json.loads(text_data)
                if data.get('type') == 'config':
                    self.sample_rate = data.get('sampleRate', self.sample_rate)
                    # print(f"[AudioConsumer] Set sample rate to {self.sample_rate}")
                else:
                    # print(f"[AudioConsumer] Received text data: {data}")
                    pass
            except Exception as e:
                # print(f"[AudioConsumer] Error parsing text data: {e}")
                pass

    async def process_audio(self):
        self.processing = True
        try:
            # Concatenate chunks
            audio_data = np.concatenate(self.audio_buffer)
            
            # Use a rolling buffer: keep the last 10 chunks (~0.8s) for context in the next run
            # but only if we have enough chunks.
            if len(self.audio_buffer) > 10:
                self.audio_buffer = self.audio_buffer[-10:]
            else:
                self.audio_buffer = []

            # print(f"[AudioConsumer] Processing audio: {len(audio_data)} samples at {self.sample_rate}Hz...")

            # Run whisper in a thread to avoid blocking the event loop
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, self.transcribe, audio_data)
            
            text = result['text'].strip()
            # print(f"[AudioConsumer] Transcription result: '{text}'")
            if text:
                print(f"[AudioConsumer] Final transcription: {text}")
                await self.send(text_data=json.dumps({
                    'type': 'transcription',
                    'text': text
                }))
        except Exception as e:
            print(f"Error processing audio: {e}")
        finally:
            self.processing = False

    def transcribe(self, audio_np):
        # Browser sends audio at its internal sample rate (e.g., 44.1kHz or 48kHz).
        # Whisper expects 16kHz mono. 
        
        target_sr = 16000
        source_sr = self.sample_rate

        # Simple Peak Normalization
        max_val = np.max(np.abs(audio_np))
        if max_val > 0.01: # Only normalize if there's significant signal
            audio_np = audio_np / max_val
            # print(f"[AudioConsumer] Normalized audio (max was {max_val:.4f})")
        else:
            # print(f"[AudioConsumer] Audio signal very low (max {max_val:.4f}), skipping normalization")
            pass
        
        # Calculate number of samples after resampling
        num_samples = int(len(audio_np) * target_sr / source_sr)
        resampled_audio = resample(audio_np, num_samples)
        
        # Return results explicitly in English
        # Using task='transcribe' is default but good to be explicit
        return model.transcribe(resampled_audio.astype(np.float32), fp16=False, language='en', beam_size=1, best_of=1)
