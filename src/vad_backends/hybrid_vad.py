# src/vad_backends/hybrid_vad.py
"""
Hybrid VAD - Combines multiple VAD backends for best results
"""
import numpy as np
from typing import List, Dict, Optional
import structlog
from .base_vad import BaseVAD, VADResult
from .silero_vad import SileroVAD
from .webrtc_vad import WebRTCVAD
from .energy_vad import EnergyVAD

logger = structlog.get_logger(__name__)


class HybridVAD(BaseVAD):
    """
    Hybrid VAD combining multiple backends
    
    Features:
    - Weighted combination of Silero, WebRTC, and Energy VAD
    - Configurable weights for different use cases
    - Fallback mechanism if one backend fails
    - Best overall accuracy
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        silero_weight: float = 0.6,
        webrtc_weight: float = 0.3,
        energy_weight: float = 0.1,
        threshold: float = 0.5,
        enable_silero: bool = True,
        enable_webrtc: bool = True,
        enable_energy: bool = True,
        silero_config: Optional[Dict] = None,
        webrtc_config: Optional[Dict] = None,
        energy_config: Optional[Dict] = None
    ):
        super().__init__(sample_rate)
        
        # Normalize weights
        total_weight = 0
        self.weights = {}
        
        if enable_silero:
            self.weights['silero'] = silero_weight
            total_weight += silero_weight
        if enable_webrtc:
            self.weights['webrtc'] = webrtc_weight
            total_weight += webrtc_weight
        if enable_energy:
            self.weights['energy'] = energy_weight
            total_weight += energy_weight
        
        # Normalize
        if total_weight > 0:
            for k in self.weights:
                self.weights[k] /= total_weight
        
        self.threshold = threshold
        
        # Backend configurations
        self.silero_config = silero_config or {}
        self.webrtc_config = webrtc_config or {}
        self.energy_config = energy_config or {}
        
        self.enable_silero = enable_silero
        self.enable_webrtc = enable_webrtc
        self.enable_energy = enable_energy
        
        # Backend instances
        self._silero: Optional[SileroVAD] = None
        self._webrtc: Optional[WebRTCVAD] = None
        self._energy: Optional[EnergyVAD] = None
    
    def initialize(self):
        """Initialize all enabled backends"""
        if self._is_initialized:
            return
        
        errors = []
        
        if self.enable_silero:
            try:
                self._silero = SileroVAD(
                    sample_rate=self.sample_rate,
                    **self.silero_config
                )
                self._silero.initialize()
            except Exception as e:
                logger.warning("Failed to initialize Silero VAD", error=str(e))
                errors.append(('silero', e))
                self._silero = None
                if 'silero' in self.weights:
                    del self.weights['silero']
        
        if self.enable_webrtc:
            try:
                self._webrtc = WebRTCVAD(
                    sample_rate=self.sample_rate,
                    **self.webrtc_config
                )
                self._webrtc.initialize()
            except Exception as e:
                logger.warning("Failed to initialize WebRTC VAD", error=str(e))
                errors.append(('webrtc', e))
                self._webrtc = None
                if 'webrtc' in self.weights:
                    del self.weights['webrtc']
        
        if self.enable_energy:
            try:
                self._energy = EnergyVAD(
                    sample_rate=self.sample_rate,
                    **self.energy_config
                )
                self._energy.initialize()
            except Exception as e:
                logger.warning("Failed to initialize Energy VAD", error=str(e))
                errors.append(('energy', e))
                self._energy = None
                if 'energy' in self.weights:
                    del self.weights['energy']
        
        # Re-normalize weights after failures
        if self.weights:
            total = sum(self.weights.values())
            for k in self.weights:
                self.weights[k] /= total
        
        if not self.weights:
            raise RuntimeError("All VAD backends failed to initialize")
        
        self._is_initialized = True
        
        logger.info(
            "Hybrid VAD initialized",
            active_backends=list(self.weights.keys()),
            weights=self.weights
        )
    
    def process_frame(self, audio_frame: np.ndarray, timestamp_ms: float = 0.0) -> VADResult:
        """Process frame through all backends and combine results"""
        if not self._is_initialized:
            self.initialize()
        
        duration_ms = len(audio_frame) / self.sample_rate * 1000
        results = {}
        
        # Process through each backend
        if self._silero is not None:
            try:
                results['silero'] = self._silero.process_frame(audio_frame, timestamp_ms)
            except Exception as e:
                logger.error("Silero VAD error", error=str(e))
        
        if self._webrtc is not None:
            try:
                # WebRTC needs specific frame sizes, may need to pad/trim
                webrtc_frame = self._prepare_webrtc_frame(audio_frame)
                results['webrtc'] = self._webrtc.process_frame(webrtc_frame, timestamp_ms)
            except Exception as e:
                logger.error("WebRTC VAD error", error=str(e))
        
        if self._energy is not None:
            try:
                results['energy'] = self._energy.process_frame(audio_frame, timestamp_ms)
            except Exception as e:
                logger.error("Energy VAD error", error=str(e))
        
        # Combine results
        weighted_probability = 0.0
        total_weight = 0.0
        
        for backend, result in results.items():
            if backend in self.weights:
                prob = result.raw_probability if result.raw_probability is not None else (1.0 if result.is_speech else 0.0)
                weighted_probability += prob * self.weights[backend]
                total_weight += self.weights[backend]
        
        if total_weight > 0:
            weighted_probability /= total_weight
        
        is_speech = weighted_probability >= self.threshold
        confidence = weighted_probability if is_speech else (1.0 - weighted_probability)
        
        return VADResult(
            is_speech=is_speech,
            confidence=confidence,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            raw_probability=weighted_probability
        )
    
    def _prepare_webrtc_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        """Prepare frame for WebRTC VAD (needs exact frame sizes)"""
        # WebRTC supports 10, 20, 30 ms frames
        samples_30ms = int(self.sample_rate * 0.03)
        
        if len(audio_frame) == samples_30ms:
            return audio_frame
        elif len(audio_frame) > samples_30ms:
            return audio_frame[:samples_30ms]
        else:
            # Pad with zeros
            padded = np.zeros(samples_30ms, dtype=audio_frame.dtype)
            padded[:len(audio_frame)] = audio_frame
            return padded
    
    def reset(self):
        """Reset all backends"""
        if self._silero:
            self._silero.reset()
        if self._webrtc:
            self._webrtc.reset()
        if self._energy:
            self._energy.reset()
    
    @property
    def supported_frame_sizes_ms(self) -> List[int]:
        return [30, 60, 100]
    
    def cleanup(self):
        """Cleanup all backends"""
        if self._silero:
            self._silero.cleanup()
        if self._webrtc:
            self._webrtc.cleanup()
        if self._energy:
            self._energy.cleanup()
        self._is_initialized = False