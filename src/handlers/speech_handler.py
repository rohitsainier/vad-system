# src/handlers/speech_handler.py
"""
Speech segment handler for processing detected speech
"""
import numpy as np
import asyncio
from typing import Optional, Callable, List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import soundfile as sf
import io
import structlog

from ..core.state_machine import SpeechSegment

logger = structlog.get_logger(__name__)


@dataclass
class ProcessedSpeech:
    """Processed speech segment with additional data"""
    segment: SpeechSegment
    audio_data: np.ndarray
    sample_rate: int
    transcript: Optional[str] = None
    metadata: Dict[str, Any] = None


class SpeechHandler:
    """
    Handler for processing speech segments
    
    Features:
    - Save speech to files
    - Send to transcription service
    - Custom processing callbacks
    - Batch processing
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        output_dir: Optional[str] = None,
        save_format: str = "wav",
        min_duration_ms: float = 500,
        max_duration_ms: float = 30000
    ):
        self.sample_rate = sample_rate
        self.output_dir = Path(output_dir) if output_dir else None
        self.save_format = save_format
        self.min_duration_ms = min_duration_ms
        self.max_duration_ms = max_duration_ms
        
        self._processors: List[Callable] = []
        self._segment_count = 0
        
        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def add_processor(self, processor: Callable[[ProcessedSpeech], None]):
        """Add a speech processor callback"""
        self._processors.append(processor)
    
    def handle(self, segment: SpeechSegment) -> Optional[ProcessedSpeech]:
        """
        Handle a speech segment
        
        Args:
            segment: Detected speech segment
            
        Returns:
            ProcessedSpeech object or None if filtered
        """
        # Validate segment
        if segment.audio_data is None:
            logger.warning("Segment has no audio data")
            return None
        
        duration = segment.duration_ms
        if duration is None:
            logger.warning("Segment has no duration")
            return None
        
        # Filter by duration
        if duration < self.min_duration_ms:
            logger.debug("Segment too short", duration_ms=duration)
            return None
        
        if duration > self.max_duration_ms:
            logger.warning("Segment too long, truncating", duration_ms=duration)
            max_samples = int(self.max_duration_ms / 1000 * self.sample_rate)
            audio_data = segment.audio_data[:max_samples]
        else:
            audio_data = segment.audio_data
        
        self._segment_count += 1
        
        processed = ProcessedSpeech(
            segment=segment,
            audio_data=audio_data,
            sample_rate=self.sample_rate,
            metadata={
                "segment_id": self._segment_count,
                "original_duration_ms": duration
            }
        )
        
        # Save to file if configured
        if self.output_dir:
            self._save_segment(processed)
        
        # Call processors
        for processor in self._processors:
            try:
                processor(processed)
            except Exception as e:
                logger.error("Processor error", error=str(e))
        
        return processed
    
    async def handle_async(self, segment: SpeechSegment) -> Optional[ProcessedSpeech]:
        """Handle segment asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.handle, segment)
    
    def _save_segment(self, processed: ProcessedSpeech):
        """Save segment to file"""
        filename = f"speech_{self._segment_count:06d}_{int(processed.segment.start_time_ms)}.{self.save_format}"
        filepath = self.output_dir / filename
        
        try:
            sf.write(
                str(filepath),
                processed.audio_data,
                self.sample_rate,
                format=self.save_format.upper()
            )
            
            processed.metadata["saved_path"] = str(filepath)
            logger.info("Speech saved", path=str(filepath))
            
        except Exception as e:
            logger.error("Failed to save speech", error=str(e))
    
    def get_audio_bytes(self, processed: ProcessedSpeech, format: str = "wav") -> bytes:
        """
        Get speech audio as bytes
        
        Args:
            processed: Processed speech object
            format: Output format
            
        Returns:
            Audio bytes
        """
        buffer = io.BytesIO()
        sf.write(buffer, processed.audio_data, self.sample_rate, format=format)
        buffer.seek(0)
        return buffer.read()
    
    @property
    def segment_count(self) -> int:
        return self._segment_count