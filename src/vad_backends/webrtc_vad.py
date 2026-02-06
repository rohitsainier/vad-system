# src/vad_backends/webrtc_vad.py
"""
WebRTC VAD implementation - Fast and lightweight
FIXED: Robust frame size handling for any sample rate conversion
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
    
    IMPORTANT: WebRTC VAD requires EXACT frame sizes:
    - 10ms = 160 samples at 16kHz (320 bytes)
    - 20ms = 320 samples at 16kHz (640 bytes)
    - 30ms = 480 samples at 16kHz (960 bytes)
    
    This implementation handles any input size by:
    1. Accumulating audio in an internal buffer
    2. Extracting exact 30ms frames from the buffer
    3. Processing each exact frame through webrtcvad
    """
    
    SUPPORTED_SAMPLE_RATES = [8000, 16000, 32000, 48000]
    SUPPORTED_FRAME_SIZES_MS = [10, 20, 30]
    
    # Exact samples per 30ms frame for each sample rate
    FRAME_SAMPLES = {
        8000: 240,
        16000: 480,
        32000: 960,
        48000: 1440,
    }
    
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
        
        # Exact frame size for this sample rate
        self._frame_samples = self.FRAME_SAMPLES[sample_rate]
        self._frame_bytes = self._frame_samples * 2  # int16 = 2 bytes per sample
        
        # Internal buffer to accumulate audio for exact frame sizes
        self._buffer = np.array([], dtype=np.float32)
    
    def initialize(self):
        """Initialize WebRTC VAD"""
        if self._is_initialized:
            return
        
        self._vad = webrtcvad.Vad(self.aggressiveness)
        self._buffer = np.array([], dtype=np.float32)
        self._is_initialized = True
        
        logger.info(
            "WebRTC VAD initialized",
            sample_rate=self.sample_rate,
            aggressiveness=self.aggressiveness,
            frame_samples=self._frame_samples,
            frame_bytes=self._frame_bytes
        )
    
    def process_frame(self, audio_frame: np.ndarray, timestamp_ms: float = 0.0) -> VADResult:
        """
        Process audio frame through WebRTC VAD
        
        Handles any input size by buffering and extracting
        exact 30ms frames (480 samples at 16kHz).
        """
        if not self._is_initialized:
            self.initialize()
        
        # Ensure 1D float32
        if len(audio_frame.shape) > 1:
            audio_frame = audio_frame.flatten()
        
        if audio_frame.dtype != np.float32:
            audio_frame = audio_frame.astype(np.float32)
        
        duration_ms = len(audio_frame) / self.sample_rate * 1000
        
        # Add incoming audio to buffer
        self._buffer = np.concatenate([self._buffer, audio_frame])
        
        # Extract and process exact 30ms frames
        speech_results = []
        
        while len(self._buffer) >= self._frame_samples:
            # Take exactly frame_samples from buffer
            exact_frame = self._buffer[:self._frame_samples]
            self._buffer = self._buffer[self._frame_samples:]
            
            # Convert to int16
            is_speech = self._process_exact_frame(exact_frame)
            speech_results.append(is_speech)
        
        # Aggregate results
        if speech_results:
            speech_count = sum(1 for r in speech_results if r)
            total = len(speech_results)
            speech_ratio = speech_count / total
            
            is_speech = speech_ratio >= 0.5
            confidence = speech_ratio if is_speech else (1.0 - speech_ratio)
            raw_probability = speech_ratio
        else:
            # Not enough audio accumulated yet for a full frame
            is_speech = False
            confidence = 0.5
            raw_probability = 0.0
        
        return VADResult(
            is_speech=is_speech,
            confidence=confidence,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            raw_probability=raw_probability
        )
    
    def _process_exact_frame(self, frame: np.ndarray) -> bool:
        """
        Process an exact-sized frame through webrtcvad
        
        Args:
            frame: Exactly self._frame_samples float32 samples
            
        Returns:
            True if speech detected
        """
        assert len(frame) == self._frame_samples, \
            f"Frame must be exactly {self._frame_samples} samples, got {len(frame)}"
        
        # Convert float32 to int16 with proper scaling
        peak = np.max(np.abs(frame))
        if peak > 1e-10:
            # Scale to use ~80% of int16 range
            scale = min(26000.0 / peak, 32767.0)
            audio_int16 = (frame * scale).astype(np.int16)
        else:
            # Silent frame
            audio_int16 = np.zeros(self._frame_samples, dtype=np.int16)
        
        # Convert to bytes
        audio_bytes = audio_int16.tobytes()
        
        # Verify byte count (safety check)
        assert len(audio_bytes) == self._frame_bytes, \
            f"Expected {self._frame_bytes} bytes, got {len(audio_bytes)}"
        
        try:
            return self._vad.is_speech(audio_bytes, self.sample_rate)
        except Exception as e:
            logger.debug("WebRTC VAD frame error", 
                        error=str(e),
                        frame_len=len(frame),
                        bytes_len=len(audio_bytes))
            return False
    
    def reset(self):
        """Reset VAD state"""
        self._buffer = np.array([], dtype=np.float32)
    
    @property
    def supported_frame_sizes_ms(self) -> List[int]:
        return self.SUPPORTED_FRAME_SIZES_MS
    
    def cleanup(self):
        """Cleanup resources"""
        self._vad = None
        self._buffer = np.array([], dtype=np.float32)
        self._is_initialized = False