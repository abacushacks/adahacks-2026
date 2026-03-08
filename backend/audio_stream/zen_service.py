import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

logger = logging.getLogger(__name__)

_zen_executor = ThreadPoolExecutor(max_workers=1)


class ZenService:
    """
    Service for enriching face metadata using the Zen LLM (OpenAI compatible).
    Takes raw metadata from the Face model and returns a structured
    response with relationship and context summary.
    """
    _client = None

    @classmethod
    def _get_client(cls):
        """Lazy-loads the OpenAI client."""
        if cls._client is None:
            try:
                api_key = os.environ.get('ZEN_KEY', '')
                if not api_key:
                    logger.warning("ZEN_KEY not set. Zen enrichment will be skipped.")
                    return None
                cls._client = OpenAI(
                    api_key=api_key,
                    base_url="https://opencode.ai/zen/v1"
                )
                logger.info("Zen OpenAI client initialized successfully.")
            except Exception as e:
                logger.exception(f"Failed to initialize Zen OpenAI client: {e}")
                return None
        return cls._client

    @classmethod
    def _build_prompt(cls, name: str, metadata: Any, label: str) -> str:
        """Builds the prompt for the Zen API."""
        metadata_str = json.dumps(metadata, indent=2) if metadata else "No metadata available"
        return f"""You are an assistant for a memory aid application for people with dementia.
Given the following information about a recognized person, provide a brief, warm summary.

Person name: {name or label}
Raw metadata from database:
{metadata_str}

Respond ONLY with valid JSON in this exact format, no markdown, no code fences:
{{
  "relationship": "<one or two word relationship label, e.g. Son, Wife, Neighbor, Caregiver, Friend. Use 'Known Person' if unclear>",
  "context": "<1-2 friendly sentences about what you last talked about or recent interactions. If no data, say 'No recent conversations recorded.'>"
}}"""

    @classmethod
    def _call_zen(cls, name: str, metadata: Any, label: str) -> Optional[Dict[str, str]]:
        """Synchronous call to Zen API."""
        client = cls._get_client()
        if client is None:
            return None

        prompt = cls._build_prompt(name, metadata, label)
        try:
            response = client.chat.completions.create(
                model="kimi-k2.5",  # Or whatever model is supported by Zen
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds in JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            text = response.choices[0].message.content.strip()
            result = json.loads(text)
            return {
                'relationship': result.get('relationship', 'Known Person'),
                'context': result.get('context', 'No recent conversations recorded.')
            }
        except json.JSONDecodeError:
            logger.error(f"Zen returned non-JSON response: {text[:200]}")
            return None
        except Exception as e:
            logger.exception(f"Zen API call failed: {e}")
            return None

    @classmethod
    def _build_name_parsing_prompt(cls, text: str) -> str:
        """Builds the prompt for Zen to parse name and relationship from text."""
        return (
            "Analyze this transcription for two things:\n"
            "1. Is someone introducing themselves by name? (e.g. 'My name is...', 'I\\'m...', 'Call me...')\n"
            "2. Is someone stating their relationship? (e.g. 'I\\'m your father', 'I am your son', 'We are friends', 'I\\'m your doctor')\n"
            "\n"
            "Respond ONLY with valid JSON, no markdown:\n"
            "{\"name\": \"extracted name or empty string\", \"relationship\": \"extracted relationship or empty string\"}\n"
            "\n"
            "Examples:\n"
            "- 'My name is John' -> {\"name\": \"John\", \"relationship\": \"\"}\n"
            "- 'I am your father' -> {\"name\": \"\", \"relationship\": \"Father\"}\n"
            "- 'Hi I\\'m Sarah, your nurse' -> {\"name\": \"Sarah\", \"relationship\": \"Nurse\"}\n"
            "- 'How are you today' -> {\"name\": \"\", \"relationship\": \"\"}\n"
            "\n"
            f"Transcription: \"{text}\""
        )

    @classmethod
    def _call_zen_to_parse_name(cls, text: str) -> Optional[Dict[str, str]]:
        """Synchronous call to Zen API to parse a name."""
        client = cls._get_client()
        if client is None:
            return None

        prompt = cls._build_name_parsing_prompt(text)
        try:
            response = client.chat.completions.create(
                model="kimi-k2.5",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds in JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content.strip()
            result = json.loads(response_text)
            return {
                'name': result.get('name', ''),
                'relationship': result.get('relationship', '')
            }
        except json.JSONDecodeError:
            logger.error(f"Zen returned non-JSON response for name parsing: {response_text[:200]}")
            return None
        except Exception as e:
            logger.exception(f"Zen API call failed for name parsing: {e}")
            return None

    @classmethod
    async def parse_name_from_text(cls, text: str) -> Dict[str, str]:
        """Async wrapper that calls Zen in a thread pool to parse a name."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                _zen_executor,
                cls._call_zen_to_parse_name,
                text
            )
            if result:
                return result
        except Exception as e:
            logger.exception(f"Error parsing name from Zen: {e}")

        return {'name': '', 'relationship': ''}

    @classmethod
    async def get_context(cls, face) -> Dict[str, str]:
        """
        Async wrapper that calls Zen in a thread pool.
        Returns enriched context or fallback values.
        """
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                _zen_executor,
                cls._call_zen,
                face.name or face.label,
                face.metadata,
                face.label
            )
            if result:
                return result
        except Exception as e:
            logger.exception(f"Error getting Zen context: {e}")

        # Fallback
        return {
            'relationship': 'Known Person',
            'context': 'No recent conversations recorded.'
        }

    @classmethod
    def _build_key_info_prompt(cls, text: str, existing_items: list) -> str:
        """Builds prompt to extract key info from a transcription."""
        existing_str = json.dumps(existing_items) if existing_items else "[]"
        return f"""You are a real-time memory aid for a person with dementia. You are listening to their conversation.

Your job: maintain a list of important, memorable details. You will be given what was just said AND the current list of recorded items.

IMPORTANT RULES:
- If new info UPDATES or CORRECTS an existing item, REPLACE the old one (e.g. if birthday changes, remove the old birthday and add the new one)
- Add genuinely new important details (dates, people, places, events, health, plans)
- Remove nothing unless it is being corrected/updated by new info
- Keep each item to ONE short sentence (under 12 words)
- Do NOT add greetings, filler, small talk, or trivial chat
- Use warm, simple language

Current recorded items: {existing_str}

What was just said: "{text}"

Respond ONLY with valid JSON containing the FULL updated list:
{{"items": ["item 1", "item 2", ...]}}
If nothing changed, return the existing list as-is.
If nothing important and list is empty: {{"items": []}}"""

    @classmethod
    def _call_zen_extract_key_info(cls, text: str, existing_items: list) -> Optional[list]:
        """Synchronous call to extract key info from transcription."""
        client = cls._get_client()
        if client is None:
            return None

        prompt = cls._build_key_info_prompt(text, existing_items)
        try:
            response = client.chat.completions.create(
                model="kimi-k2.5",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that responds in JSON."},
                    {"role": "user", "content": prompt}
                ],
                response_format={"type": "json_object"}
            )
            response_text = response.choices[0].message.content.strip()
            result = json.loads(response_text)
            items = result.get('items', [])
            # Only return non-empty strings
            return [item for item in items if item and isinstance(item, str)]
        except json.JSONDecodeError:
            logger.error(f"Zen returned non-JSON for key info: {response_text[:200]}")
            return None
        except Exception as e:
            logger.exception(f"Zen key info extraction failed: {e}")
            return None

    @classmethod
    async def extract_key_info(cls, text: str, existing_items: list = None) -> list:
        """Async wrapper for key info extraction."""
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                _zen_executor,
                cls._call_zen_extract_key_info,
                text,
                existing_items or []
            )
            if result:
                return result
        except Exception as e:
            logger.exception(f"Error extracting key info: {e}")
        return []
