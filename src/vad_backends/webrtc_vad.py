# src/vad_backends/webrtc_vad.py
"""
WebRTC VAD implementation - Fast and lightweight
"""
import webrtcvad
import numpy as np
from typing import List
import structlog
from .base_vad import BaseVAD, VADResult

logger = structlog.get_logger(__name__)


class WebRTCVAD(BaseVAD):
    """
    WebRTC VAD - Google's voice activity detector
    
    Features:
    - Very fast (CPU only, no GPU needed)
    - Low memory footprint
    - Adjustable aggressiveness
    - Best for real-time applications
    """
    
    SUPPORTED_SAMPLE_RATES = [8000, 16000, 32000, 48000]
    SUPPORTED_FRAME_SIZES_MS = [10, 20, 30]
    
    def __init__(
        self,
        sample_rate: int = 16000,
        aggressiveness: int = 3
    ):
        super().__init__(sample_rate)
        
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Sample rate must be one of {self.SUPPORTED_SAMPLE_RATES}")
        
        if not 0 <= aggressiveness <= 3:
            raise ValueError("Aggressiveness must be between 0 and 3")
        
        self.aggressiveness = aggressiveness
        self._vad = None
    
    def initialize(self):
        """Initialize WebRTC VAD"""
        if self._is_initialized:
            return
        
        self._vad = webrtcvad.Vad(self.aggressiveness)
        self._is_initialized = True
        
        logger.info(
            "WebRTC VAD initialized",
            sample_rate=self.sample_rate,
            aggressiveness=self.aggressiveness
        )
    
    def process_frame(self, audio_frame: np.ndarray, timestamp_ms: float = 0.0) -> VADResult:
        """Process audio frame through WebRTC VAD"""
        if not self._is_initialized:
            self.initialize()
        
        # WebRTC VAD expects int16 audio
        if audio_frame.dtype == np.float32:
            audio_int16 = (audio_frame * 32767).astype(np.int16)
        else:
            audio_int16 = audio_frame.astype(np.int16)
        
        duration_ms = len(audio_frame) / self.sample_rate * 1000
        
        # Validate frame size
        expected_samples = int(self.sample_rate * duration_ms / 1000)
        if len(audio_int16) != expected_samples:
            logger.warning(
                "Frame size mismatch",
                expected=expected_samples,
                actual=len(audio_int16)
            )
        
        # Convert to bytes
        audio_bytes = audio_int16.tobytes()
        
        try:
            is_speech = self._vad.is_speech(audio_bytes, self.sample_rate)
        except Exception as e:
            logger.error("WebRTC VAD error", error=str(e))
            is_speech = False
        
        # WebRTC VAD doesn't provide probability, use binary confidence
        confidence = 0.9 if is_speech else 0.9
        
        return VADResult(
            is_speech=is_speech,
            confidence=confidence,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            raw_probability=1.0 if is_speech else 0.0
        )
    
    def reset(self):
        """Reset VAD state (WebRTC VAD is stateless)"""
        pass
    
    @property
    def supported_frame_sizes_ms(self) -> List[int]:
        return self.SUPPORTED_FRAME_SIZES_MS
    
    def cleanup(self):
        """Cleanup resources"""
        self._vad = None
        self._is_initialized = False