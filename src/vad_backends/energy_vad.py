# src/vad_backends/energy_vad.py
"""
Energy-based VAD - Simple and fast
"""
import numpy as np
from typing import List
from collections import deque
import structlog
from .base_vad import BaseVAD, VADResult

logger = structlog.get_logger(__name__)


class EnergyVAD(BaseVAD):
    """
    Energy-based VAD with adaptive threshold
    
    Features:
    - Very fast, minimal CPU usage
    - Works with any sample rate
    - Adaptive noise floor estimation
    - Good for clean audio environments
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold_db: float = -35.0,
        adaptive: bool = True,
        adaptation_rate: float = 0.01
    ):
        super().__init__(sample_rate)
        
        self.base_threshold_db = threshold_db
        self.adaptive = adaptive
        self.adaptation_rate = adaptation_rate
        
        # Adaptive threshold state
        self._noise_floor_db = -60.0
        self._noise_history = deque(maxlen=100)
        self._speech_history = deque(maxlen=50)
    
    def initialize(self):
        """Initialize energy VAD"""
        if self._is_initialized:
            return
        
        self._is_initialized = True
        self.reset()
        
        logger.info(
            "Energy VAD initialized",
            sample_rate=self.sample_rate,
            threshold_db=self.base_threshold_db,
            adaptive=self.adaptive
        )
    
    def process_frame(self, audio_frame: np.ndarray, timestamp_ms: float = 0.0) -> VADResult:
        """Process audio frame using energy detection"""
        if not self._is_initialized:
            self.initialize()
        
        duration_ms = len(audio_frame) / self.sample_rate * 1000
        
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio_frame ** 2))
        if rms > 1e-10:
            rms_db = 20 * np.log10(rms)
        else:
            rms_db = -100.0
        
        # Calculate zero crossing rate (helps distinguish speech from noise)
        zcr = self._calculate_zcr(audio_frame)
        
        # Get adaptive threshold
        threshold_db = self._get_adaptive_threshold(rms_db)
        
        # Detection logic
        is_speech = rms_db > threshold_db
        
        # Additional check: speech typically has moderate ZCR
        if is_speech and (zcr < 0.01 or zcr > 0.5):
            # Very low or very high ZCR suggests non-speech
            is_speech = False
        
        # Calculate confidence based on how far above threshold
        if is_speech:
            db_margin = rms_db - threshold_db
            confidence = min(1.0, 0.5 + db_margin / 20)
        else:
            db_margin = threshold_db - rms_db
            confidence = min(1.0, 0.5 + db_margin / 20)
        
        # Update adaptive state
        if self.adaptive:
            self._update_adaptive_state(rms_db, is_speech)
        
        return VADResult(
            is_speech=is_speech,
            confidence=confidence,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            raw_probability=1.0 if is_speech else 0.0
        )
    
    def _calculate_zcr(self, audio_frame: np.ndarray) -> float:
        """Calculate zero crossing rate"""
        signs = np.sign(audio_frame)
        sign_changes = np.abs(np.diff(signs))
        return float(np.mean(sign_changes) / 2)
    
    def _get_adaptive_threshold(self, current_db: float) -> float:
        """Get current adaptive threshold"""
        if not self.adaptive:
            return self.base_threshold_db
        
        # Threshold is noise floor + margin
        margin = 10.0  # dB above noise floor
        adaptive_threshold = self._noise_floor_db + margin
        
        # Don't go below base threshold
        return max(adaptive_threshold, self.base_threshold_db)
    
    def _update_adaptive_state(self, rms_db: float, is_speech: bool):
        """Update adaptive noise floor estimation"""
        if is_speech:
            self._speech_history.append(rms_db)
        else:
            self._noise_history.append(rms_db)
            
            # Update noise floor (use lower percentile to be robust)
            if len(self._noise_history) >= 10:
                self._noise_floor_db = np.percentile(list(self._noise_history), 20)
    
    def reset(self):
        """Reset adaptive state"""
        self._noise_floor_db = -60.0
        self._noise_history.clear()
        self._speech_history.clear()
    
    @property
    def supported_frame_sizes_ms(self) -> List[int]:
        # Energy VAD works with any frame size
        return [10, 20, 30, 50, 100]