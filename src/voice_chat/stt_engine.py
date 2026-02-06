# src/voice_chat/stt_engine.py
"""
Speech-to-Text Engine using faster-whisper
GPU accelerated with CTranslate2
"""
import numpy as np
import time
from typing import Optional, Tuple
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Result from speech-to-text"""
    text: str
    language: str
    confidence: float
    duration_ms: float
    processing_time_ms: float
    
    def __repr__(self):
        return f'TranscriptionResult("{self.text[:50]}...", lang={self.language}, conf={self.confidence:.2f})'


class STTEngine:
    """
    Speech-to-Text using faster-whisper
    
    Features:
    - GPU accelerated (CTranslate2)
    - Multiple model sizes
    - Language detection
    - Word-level timestamps
    """
    
    # Model sizes and approximate VRAM usage
    MODELS = {
        "tiny": "~1GB",
        "base": "~1GB",
        "small": "~2GB",
        "medium": "~5GB",
        "large-v3": "~10GB",
        "distil-large-v3": "~6GB (faster)",
    }
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = "auto",
        compute_type: str = "float16",
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = False  # We already have our own VAD
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.beam_size = beam_size
        self.vad_filter = vad_filter
        
        self._model = None
        self._is_initialized = False
    
    def initialize(self):
        """Load Whisper model"""
        if self._is_initialized:
            return
        
        logger.info(
            "Loading Whisper model",
            model=self.model_size,
            device=self.device,
            compute_type=self.compute_type
        )
        
        from faster_whisper import WhisperModel
        
        # Auto-detect device
        if self.device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            device = self.device
        
        # Adjust compute type for CPU
        compute_type = self.compute_type
        if device == "cpu" and compute_type == "float16":
            compute_type = "int8"
        
        self._model = WhisperModel(
            self.model_size,
            device=device,
            compute_type=compute_type
        )
        
        self._is_initialized = True
        logger.info("Whisper model loaded", model=self.model_size, device=device)
    
    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000
    ) -> TranscriptionResult:
        """
        Transcribe audio to text
        
        Args:
            audio: Float32 audio array
            sample_rate: Sample rate of audio
            
        Returns:
            TranscriptionResult with text and metadata
        """
        if not self._is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        # Ensure float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)
        
        # Ensure mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        
        # Resample to 16kHz if needed (Whisper expects 16kHz)
        if sample_rate != 16000:
            from scipy.signal import resample
            new_length = int(len(audio) * 16000 / sample_rate)
            audio = resample(audio, new_length)
        
        duration_ms = len(audio) / 16000 * 1000
        
        # Transcribe
        segments, info = self._model.transcribe(
            audio,
            language=self.language,
            beam_size=self.beam_size,
            vad_filter=self.vad_filter,
            without_timestamps=True
        )
        
        # Collect text from segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())
        
        full_text = " ".join(text_parts).strip()
        
        processing_time = (time.time() - start_time) * 1000
        
        result = TranscriptionResult(
            text=full_text,
            language=info.language if info.language else "en",
            confidence=info.language_probability if info.language_probability else 0.0,
            duration_ms=duration_ms,
            processing_time_ms=processing_time
        )
        
        logger.info(
            "Transcription complete",
            text=full_text[:80],
            language=result.language,
            duration_ms=duration_ms,
            processing_ms=processing_time
        )
        
        return result
    
    def cleanup(self):
        """Release model resources"""
        self._model = None
        self._is_initialized = False
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()