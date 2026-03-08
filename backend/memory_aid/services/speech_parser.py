from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ExtractedFact:
    fact_type: str
    fact_value: str


NAME_STOP_WORDS = {
    "married",
    "engaged",
    "dating",
    "divorced",
    "widowed",
    "your",
    "friend",
    "daughter",
    "son",
    "wife",
    "husband",
    "sister",
    "brother",
}

NAME_PATTERNS = [
    re.compile(r"\bmy name is (?P<name>[a-zA-Z][a-zA-Z' -]{0,40})\b", re.IGNORECASE),
    re.compile(r"\bthis is (?P<name>[a-zA-Z][a-zA-Z' -]{0,40})\b", re.IGNORECASE),
    re.compile(r"\bi am (?P<name>[a-zA-Z][a-zA-Z' -]{0,40})\b", re.IGNORECASE),
    re.compile(r"\bi'm (?P<name>[a-zA-Z][a-zA-Z' -]{0,40})\b", re.IGNORECASE),
]

RELATIONSHIP_TO_PATIENT_PATTERNS = [
    re.compile(
        r"\b(?:i am|i'm|this is|it's me[,]? your)\s+(?P<value>daughter|son|wife|husband|sister|brother|friend|neighbor|caregiver|nurse|doctor|granddaughter|grandson)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\byour (?P<value>daughter|son|wife|husband|sister|brother|friend|neighbor|caregiver|nurse|doctor|granddaughter|grandson)\b",
        re.IGNORECASE,
    ),
]

BIRTHDAY_PATTERNS = [
    re.compile(
        r"\b(?:my birthday is|i was born on|my birth date is)\s+(?P<value>[^.!?]+)",
        re.IGNORECASE,
    ),
]

RELATIONSHIP_STATUS_PATTERNS = [
    re.compile(
        r"\b(?:i am|i'm|we are)\s+(?P<value>married|engaged|dating|divorced|widowed)\b",
        re.IGNORECASE,
    ),
]

LIFE_EVENT_PATTERNS = [
    re.compile(
        r"\b(?:i moved to|we moved to)\s+(?P<value>[^.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i started working at|i work at|i got a new job at)\s+(?P<value>[^.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i retired from|i graduated from)\s+(?P<value>[^.!?]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i had a baby|i bought a house|i got promoted|i got engaged)\b[^.!?]*",
        re.IGNORECASE,
    ),
]


def extract_name_candidate(transcript: str) -> str | None:
    for pattern in NAME_PATTERNS:
        match = pattern.search(transcript)
        if match is None:
            continue

        candidate = _normalize_name(match.group("name"))
        if candidate is None:
            continue
        return candidate

    return None


def extract_memory_facts(transcript: str) -> list[ExtractedFact]:
    facts: list[ExtractedFact] = []
    seen = set()

    for pattern in RELATIONSHIP_TO_PATIENT_PATTERNS:
        for match in pattern.finditer(transcript):
            value = match.group("value").strip().title()
            fact = ExtractedFact("Relationship", value)
            if fact not in seen:
                facts.append(fact)
                seen.add(fact)

    for pattern in BIRTHDAY_PATTERNS:
        for match in pattern.finditer(transcript):
            value = _clean_fact_value(match.group("value"))
            fact = ExtractedFact("Birthday", value)
            if fact not in seen:
                facts.append(fact)
                seen.add(fact)

    for pattern in RELATIONSHIP_STATUS_PATTERNS:
        for match in pattern.finditer(transcript):
            value = match.group("value").strip().title()
            fact = ExtractedFact("Relationship", value)
            if fact not in seen:
                facts.append(fact)
                seen.add(fact)

    for pattern in LIFE_EVENT_PATTERNS:
        for match in pattern.finditer(transcript):
            value = match.groupdict().get("value")
            sentence = _clean_fact_value(value or match.group(0))
            fact = ExtractedFact("LifeEvent", sentence)
            if fact not in seen:
                facts.append(fact)
                seen.add(fact)

    return facts


def _normalize_name(raw_name: str) -> str | None:
    stripped = re.sub(r"[^a-zA-Z' -]", "", raw_name).strip()
    if not stripped:
        return None

    tokens = [token for token in stripped.split() if token]
    if not 1 <= len(tokens) <= 3:
        return None

    if tokens[0].lower() in NAME_STOP_WORDS:
        return None

    return " ".join(token.capitalize() for token in tokens)


def _clean_fact_value(raw_value: str) -> str:
    cleaned = raw_value.strip(" .,!?\n\t")
    if not cleaned:
        return raw_value
    return cleaned[0].upper() + cleaned[1:]
