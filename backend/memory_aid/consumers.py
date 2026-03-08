import logging

from asgiref.sync import sync_to_async
from channels.generic.websocket import AsyncJsonWebsocketConsumer

from memory_aid.services.session_state import MemoryAidSession


logger = logging.getLogger(__name__)


class MemoryAidConsumer(AsyncJsonWebsocketConsumer):
    async def connect(self) -> None:
        self.session = MemoryAidSession()
        await self.accept()
        await self.send_json({"type": "session_ready"})

    async def disconnect(self, close_code: int) -> None:
        await sync_to_async(self.session.reset)()

    async def receive_json(self, content: dict, **kwargs) -> None:
        message_type = content.get("type")

        try:
            if message_type == "session.start":
                await self.send_json({"type": "session_ack"})
                if not self.session.transcription_supported():
                    await self.send_json(
                        {
                            "type": "error",
                            "message": (
                                "Whisper is unavailable in this Python runtime. "
                                "Face tracking continues, but audio transcription is disabled."
                            ),
                        }
                    )
                return

            if message_type == "session.stop":
                await sync_to_async(self.session.reset)()
                await self.send_json({"type": "session_stopped"})
                return

            if message_type == "speaker.reference":
                await sync_to_async(self.session.set_speaker_reference)(
                    content["audio"], content["sampleRate"]
                )
                await self.send_json({"type": "speaker_reference_ready"})
                return

            if message_type == "profile.enroll":
                messages = await sync_to_async(self.session.enroll_current_face)(
                    content["name"]
                )
                await self._emit(messages)
                return

            if message_type == "profile.voice_note":
                messages = await sync_to_async(self.session.process_voice_note)(
                    content["audio"], content["sampleRate"]
                )
                await self._emit(messages)
                return

            if message_type == "media.face":
                messages = await sync_to_async(self.session.process_face_observation)(
                    content["descriptor"],
                    content["face"],
                    content.get("mouthActivity", 0.0),
                    content.get("timestamp"),
                )
                await self._emit(messages)
                return

            if message_type == "media.face_lost":
                messages = await sync_to_async(self.session.process_face_lost)()
                await self._emit(messages)
                return

            if message_type == "media.audio":
                messages = await sync_to_async(self.session.process_audio_chunk)(
                    content["audio"],
                    content["sampleRate"],
                    content.get("timestamp"),
                )
                await self._emit(messages)
                return

            await self.send_json(
                {
                    "type": "error",
                    "message": f"Unsupported message type: {message_type}",
                }
            )
        except KeyError as exc:
            logger.exception("Missing websocket payload field: %s", exc)
            await self.send_json(
                {"type": "error", "message": f"Missing websocket field: {exc}"}
            )
        except ValueError as exc:
            await self.send_json({"type": "error", "message": str(exc)})
        except Exception:  # pragma: no cover - defensive runtime guard
            logger.exception("Realtime session processing failed")
            await self.send_json(
                {"type": "error", "message": "Realtime session processing failed."}
            )

    async def _emit(self, messages: list[dict]) -> None:
        for message in messages:
            await self.send_json(message)
