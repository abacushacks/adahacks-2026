from __future__ import annotations

import time
from dataclasses import dataclass, field

from django.db.models import Prefetch

from memory_aid.models import Memory, Person
from memory_aid.services.audio_service import WhisperAudioService
from memory_aid.services.face_service import FaceService
from memory_aid.services.speech_parser import ExtractedFact, extract_memory_facts, extract_name_candidate


@dataclass
class PendingProfile:
    embedding: list[float] | None = None
    name_votes: dict[str, int] = field(default_factory=dict)

    def register(self, candidate_name: str) -> int:
        current_votes = self.name_votes.get(candidate_name, 0) + 1
        self.name_votes[candidate_name] = current_votes
        return current_votes


class MemoryAidSession:
    mouth_capture_window_seconds = 1.5

    def __init__(self) -> None:
        self.face_service = FaceService()
        self.audio_service = WhisperAudioService()
        self.pending_profile = PendingProfile()
        self.active_person_id: int | None = None
        self.face_visible = False
        self.last_face_box: dict | None = None
        self.last_face_seen_at = 0.0
        self.last_mouth_motion_at = 0.0

    def reset(self) -> None:
        self.audio_service.reset()
        self._reset_tracking_state()

    def _reset_tracking_state(self) -> None:
        self.audio_service.clear_buffer()
        self.pending_profile = PendingProfile()
        self.active_person_id = None
        self.face_visible = False
        self.last_face_box = None
        self.last_face_seen_at = 0.0
        self.last_mouth_motion_at = 0.0

    def set_speaker_reference(self, base64_audio: str, sample_rate: int) -> None:
        self.audio_service.set_reference_speaker(base64_audio, sample_rate)

    def transcription_supported(self) -> bool:
        return self.audio_service.transcription_supported

    def process_face_observation(
        self,
        descriptor: list[float],
        face: dict,
        mouth_activity: float,
        timestamp_ms: int | None,
    ) -> list[dict]:
        event_time = self._normalize_timestamp(timestamp_ms)

        self.face_visible = True
        self.last_face_seen_at = event_time
        if mouth_activity >= self.face_service.mouth_activity_threshold:
            self.last_mouth_motion_at = event_time

        normalized_descriptor = self.face_service.normalize_descriptor(descriptor)
        self.last_face_box = self.face_service.normalize_face_payload(face)

        people = self._all_people()
        matched_person = self.face_service.find_best_match(normalized_descriptor, people)

        if matched_person is not None:
            self.active_person_id = matched_person.id
            self.pending_profile = PendingProfile()
            return [
                {
                    "type": "tracking_update",
                    "trackingStatus": "recognized",
                    "face": self.last_face_box,
                    "metadata": self._serialize_person(matched_person),
                }
            ]

        self.active_person_id = None
        self.pending_profile.embedding = normalized_descriptor
        return [
            {
                "type": "tracking_update",
                "trackingStatus": "unknown",
                "face": self.last_face_box,
                "metadata": None,
            }
        ]

    def process_face_lost(self) -> list[dict]:
        if not self.face_visible:
            return []
        self._reset_tracking_state()
        return [{"type": "tracking_lost"}]

    def process_audio_chunk(
        self,
        base64_audio: str,
        sample_rate: int,
        timestamp_ms: int | None,
    ) -> list[dict]:
        if not self.face_visible:
            return []

        event_time = self._normalize_timestamp(timestamp_ms)
        allow_capture = (
            self.last_mouth_motion_at > 0
            and (event_time - self.last_mouth_motion_at) <= self.mouth_capture_window_seconds
        )

        transcript = self.audio_service.process_chunk(
            base64_audio=base64_audio,
            sample_rate=sample_rate,
            allow_capture=allow_capture,
        )
        if transcript is None:
            return []

        if self.active_person_id is not None:
            return self._update_known_person(transcript)

        return self._attempt_new_person_creation(transcript)

    def process_voice_note(self, base64_audio: str, sample_rate: int) -> list[dict]:
        if not self.face_visible or self.last_face_box is None:
            raise ValueError("Keep one face visible while recording voice.")

        transcript = self.audio_service.transcribe_clip(
            base64_audio=base64_audio,
            sample_rate=sample_rate,
            exclude_reference_speaker=False,
        )
        if transcript is None:
            raise ValueError("No recognizable speech was captured.")

        if self.active_person_id is None:
            if self.pending_profile.embedding is None:
                raise ValueError("No face embedding is available for enrollment yet.")

            name_candidate = extract_name_candidate(transcript)
            if name_candidate is None:
                raise ValueError(
                    "Say a phrase like 'My name is John' while the face is visible."
                )

            person = Person.objects.create(
                name=name_candidate,
                face_embedding_data=self.pending_profile.embedding,
            )
            self.active_person_id = person.id
            self.pending_profile = PendingProfile()
            self._apply_transcript_updates(person, transcript, allow_name_update=False)
            person = self._person_with_memories(person.id)

            return [
                {"type": "profile_learned", "personId": person.id, "name": person.name},
                {
                    "type": "tracking_update",
                    "trackingStatus": "recognized",
                    "face": self.last_face_box,
                    "metadata": self._serialize_person(person),
                },
            ]

        person = self._person_with_memories(self.active_person_id)
        changed = self._apply_transcript_updates(person, transcript, allow_name_update=True)
        if not changed:
            raise ValueError(
                "No supported name or memory detail was found in that recording."
            )

        person = self._person_with_memories(person.id)
        return [
            {
                "type": "tracking_update",
                "trackingStatus": "recognized",
                "face": self.last_face_box,
                "metadata": self._serialize_person(person),
            }
        ]

    def enroll_current_face(self, name: str) -> list[dict]:
        normalized_name = self._normalize_manual_name(name)
        if normalized_name is None:
            raise ValueError("Enter a name before enrolling the current face.")

        if self.pending_profile.embedding is None or self.last_face_box is None:
            raise ValueError("A face must be visible before it can be enrolled.")

        person = Person.objects.create(
            name=normalized_name,
            face_embedding_data=self.pending_profile.embedding,
        )
        self.active_person_id = person.id
        self.pending_profile = PendingProfile()

        person = self._person_with_memories(person.id)
        return [
            {"type": "profile_learned", "personId": person.id, "name": person.name},
            {
                "type": "tracking_update",
                "trackingStatus": "recognized",
                "face": self.last_face_box,
                "metadata": self._serialize_person(person),
            },
        ]

    def _attempt_new_person_creation(self, transcript: str) -> list[dict]:
        if self.pending_profile.embedding is None or self.last_face_box is None:
            return []

        name_candidate = extract_name_candidate(transcript)
        if name_candidate is None:
            return []

        vote_count = self.pending_profile.register(name_candidate)
        if vote_count < 2:
            return []

        person = Person.objects.create(
            name=name_candidate,
            face_embedding_data=self.pending_profile.embedding,
        )
        self.active_person_id = person.id
        self.pending_profile = PendingProfile()

        person = self._person_with_memories(person.id)
        return [
            {"type": "profile_learned", "personId": person.id, "name": person.name},
            {
                "type": "tracking_update",
                "trackingStatus": "recognized",
                "face": self.last_face_box,
                "metadata": self._serialize_person(person),
            },
        ]

    def _update_known_person(self, transcript: str) -> list[dict]:
        person = self._person_with_memories(self.active_person_id)
        changed = self._apply_transcript_updates(person, transcript, allow_name_update=False)

        if not changed or self.last_face_box is None:
            return []

        person = self._person_with_memories(person.id)
        return [
            {
                "type": "tracking_update",
                "trackingStatus": "recognized",
                "face": self.last_face_box,
                "metadata": self._serialize_person(person),
            }
        ]

    def _write_memory_fact(self, person: Person, fact: ExtractedFact) -> bool:
        exists = Memory.objects.filter(
            person=person,
            fact_type__iexact=fact.fact_type,
            fact_value__iexact=fact.fact_value,
        ).exists()
        if exists:
            return False
        Memory.objects.create(
            person=person,
            fact_type=fact.fact_type,
            fact_value=fact.fact_value,
        )
        return True

    def _apply_transcript_updates(
        self,
        person: Person,
        transcript: str,
        allow_name_update: bool,
    ) -> bool:
        changed = False

        if allow_name_update:
            name_candidate = extract_name_candidate(transcript)
            if name_candidate is not None and person.name != name_candidate:
                person.name = name_candidate
                person.save(update_fields=["name"])
                changed = True

        facts = extract_memory_facts(transcript)
        for fact in facts:
            changed |= self._write_memory_fact(person, fact)

        return changed

    @staticmethod
    def _normalize_timestamp(timestamp_ms: int | None) -> float:
        if timestamp_ms is None:
            return time.time()
        return float(timestamp_ms) / 1000.0

    @staticmethod
    def _normalize_manual_name(name: str) -> str | None:
        collapsed = " ".join(name.strip().split())
        if not collapsed:
            return None
        return collapsed[:255]

    @staticmethod
    def _serialize_person(person: Person) -> dict:
        memories = list(person.memories.all())
        relationship = ""
        details: list[str] = []

        for memory in memories:
            if memory.fact_type.lower() == "relationship":
                relationship = memory.fact_value
            else:
                details.append(f"{memory.fact_type}: {memory.fact_value}")

        return {
            "personId": person.id,
            "name": person.name,
            "relationship": relationship,
            "details": details,
        }

    @staticmethod
    def _all_people():
        return Person.objects.prefetch_related(
            Prefetch("memories", queryset=Memory.objects.order_by("id"))
        ).all()

    @staticmethod
    def _person_with_memories(person_id: int) -> Person:
        return (
            Person.objects.prefetch_related(
                Prefetch("memories", queryset=Memory.objects.order_by("id"))
            )
            .get(id=person_id)
        )
