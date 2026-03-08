import os
import json
import logging
import asyncio
from typing import Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

_gemini_executor = ThreadPoolExecutor(max_workers=1)


class GeminiService:
    """
    Service for enriching face metadata using the Gemini LLM.
    Takes raw metadata from the Face model and returns a structured
    response with relationship and context summary.
    """
    _model = None

    @classmethod
    def _get_model(cls):
        """Lazy-loads the Gemini generative model."""
        if cls._model is None:
            try:
                import google.generativeai as genai
                api_key = os.environ.get('GEMINI_API_KEY', '')
                if not api_key:
                    logger.warning("GEMINI_API_KEY not set. Gemini enrichment will be skipped.")
                    return None
                genai.configure(api_key=api_key)
                cls._model = genai.GenerativeModel('gemini-2.0-flash')
                logger.info("Gemini model loaded successfully.")
            except Exception as e:
                logger.exception(f"Failed to initialize Gemini model: {e}")
                return None
        return cls._model

    @classmethod
    def _build_prompt(cls, name: str, metadata: Any, label: str) -> str:
        """Builds the prompt for the Gemini API."""
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
    def _call_gemini(cls, name: str, metadata: Any, label: str) -> Optional[Dict[str, str]]:
        """Synchronous call to Gemini API."""
        model = cls._get_model()
        if model is None:
            return None

        prompt = cls._build_prompt(name, metadata, label)
        try:
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Strip markdown code fences if present
            if text.startswith('```'):
                text = text.split('\n', 1)[1] if '\n' in text else text[3:]
                if text.endswith('```'):
                    text = text[:-3].strip()
            result = json.loads(text)
            return {
                'relationship': result.get('relationship', 'Known Person'),
                'context': result.get('context', 'No recent conversations recorded.')
            }
        except json.JSONDecodeError:
            logger.error(f"Gemini returned non-JSON response: {text[:200]}")
            return None
        except Exception as e:
            logger.exception(f"Gemini API call failed: {e}")
            return None

    @classmethod
    async def get_context(cls, face) -> Dict[str, str]:
        """
        Async wrapper that calls Gemini in a thread pool.
        Returns enriched context or fallback values.
        """
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                _gemini_executor,
                cls._call_gemini,
                face.name or face.label,
                face.metadata,
                face.label
            )
            if result:
                return result
        except Exception as e:
            logger.exception(f"Error getting Gemini context: {e}")

        # Fallback
        return {
            'relationship': 'Known Person',
            'context': 'No recent conversations recorded.'
        }
