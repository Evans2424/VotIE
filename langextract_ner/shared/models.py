"""
Gemini 2.0 Flash Model for Portuguese Voting NER

Simple implementation using LangExtract.
"""

import os
import time
import logging
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

import langextract as lx

logger = logging.getLogger(__name__)


@dataclass
class CharInterval:
    """Character interval for extractions"""
    start_pos: int
    end_pos: int


@dataclass
class SimpleExtraction:
    """Extraction with text and entity class"""
    text: str
    extraction_class: str
    char_interval: CharInterval


@dataclass
class ExtractionResult:
    """Result container for entity extractions"""
    document_id: str
    text: str
    extractions: List[SimpleExtraction]
    processing_time: float = 0.0
    model_id: str = "gemini-2.0-flash-exp"
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


class GeminiModel:
    """Gemini 2.0 Flash model for voting entity extraction"""

    def __init__(self, model_id: str = "gemini-2.0-flash-exp", api_key: Optional[str] = None):
        """
        Initialize Gemini model

        Args:
            model_id: Gemini model ID (default: gemini-2.0-flash-exp)
            api_key: Google API key (or set LANGEXTRACT_API_KEY/GOOGLE_API_KEY env var)
        """
        self.model_id = model_id
        self.api_key = api_key or os.getenv('LANGEXTRACT_API_KEY') or os.getenv('GOOGLE_API_KEY')

        if not self.api_key:
            logger.warning("No API key configured. Set LANGEXTRACT_API_KEY or GOOGLE_API_KEY.")

    def extract(self,
                text: str,
                prompt: str,
                examples: List[lx.data.ExampleData],
                document_id: str = "unknown",
                max_retries: int = 3,
                retry_delay: int = 60) -> ExtractionResult:
        """
        Extract voting entities from text

        Args:
            text: Document text
            prompt: Extraction prompt
            examples: Few-shot examples
            document_id: Document identifier
            max_retries: Retry attempts for quota errors
            retry_delay: Delay between retries (seconds)

        Returns:
            ExtractionResult with entities or error
        """
        start_time = time.time()

        for attempt in range(max_retries + 1):
            try:
                # Extract using LangExtract
                result = lx.extract(
                    text_or_documents=text,
                    prompt_description=prompt,
                    examples=examples,
                    model_id=self.model_id,
                    api_key=self.api_key,
                    max_char_buffer=1500,
                    max_workers=1
                )

                # Convert to SimpleExtraction format
                extractions = []
                for ext in result.extractions:
                    # Skip extractions without valid char_interval
                    if not hasattr(ext, 'char_interval') or ext.char_interval is None:
                        logger.warning(f"Skipping extraction without char_interval: {ext}")
                        continue

                    if not hasattr(ext.char_interval, 'start_pos') or not hasattr(ext.char_interval, 'end_pos'):
                        logger.warning(f"Skipping extraction with invalid char_interval: {ext.char_interval}")
                        continue

                    char_interval = CharInterval(
                        start_pos=ext.char_interval.start_pos,
                        end_pos=ext.char_interval.end_pos
                    )

                    extracted_text = text[ext.char_interval.start_pos:ext.char_interval.end_pos]

                    extractions.append(SimpleExtraction(
                        text=extracted_text,
                        extraction_class=ext.extraction_class,
                        char_interval=char_interval
                    ))

                processing_time = time.time() - start_time
                logger.info(f"Extracted {len(extractions)} entities from {document_id}")

                return ExtractionResult(
                    document_id=document_id,
                    text=text,
                    extractions=extractions,
                    processing_time=processing_time,
                    model_id=self.model_id
                )

            except Exception as e:
                error_str = str(e).lower()

                # Check for quota/rate limit errors
                is_quota_error = any(pattern in error_str for pattern in [
                    'quota', 'rate limit', 'rate_limit_exceeded',
                    'resource_exhausted', 'resource has been exhausted', '429'
                ])

                # Retry on quota errors
                if is_quota_error and attempt < max_retries:
                    logger.warning(f"Quota limit for {document_id}. Waiting {retry_delay}s...")
                    time.sleep(retry_delay)
                    continue

                # Failed - return error
                processing_time = time.time() - start_time
                error_msg = f"Extraction failed for {document_id}: {str(e)}"
                logger.error(error_msg)

                return ExtractionResult(
                    document_id=document_id,
                    text=text,
                    extractions=[],
                    processing_time=processing_time,
                    model_id=self.model_id,
                    error=error_msg
                )
