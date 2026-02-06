# src/vad_backends/base_vad.py
"""
Base class for VAD backends
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


@dataclass
class VADResult:
    """Result from VAD processing"""
    is_speech: bool
    confidence: float  # 0.0 to 1.0
    timestamp_ms: float
    duration_ms: float
    raw_probability: Optional[float] = None
    
    def __repr__(self):
        return f"VADResult(speech={self.is_speech}, conf={self.confidence:.2f}, t={self.timestamp_ms:.0f}ms)"


class BaseVAD(ABC):
    """Abstract base class for VAD implementations"""
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._is_initialized = False
    
    @abstractmethod
    def initialize(self):
        """Initialize the VAD model/resources"""
        pass
    
    @abstractmethod
    def process_frame(self, audio_frame: np.ndarray, timestamp_ms: float = 0.0) -> VADResult:
        """
        Process a single audio frame
        
        Args:
            audio_frame: Audio samples (float32, mono)
            timestamp_ms: Timestamp of the frame
            
        Returns:
            VADResult with speech detection info
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset internal state"""
        pass
    
    @property
    @abstractmethod
    def supported_frame_sizes_ms(self) -> List[int]:
        """Return list of supported frame sizes in milliseconds"""
        pass
    
    def cleanup(self):
        """Cleanup resources"""
        pass
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()