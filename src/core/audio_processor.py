# src/core/audio_processor.py
"""
Audio preprocessing pipeline for VAD
"""
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass
import scipy.signal as signal
from collections import deque
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AudioFrame:
    """Represents a processed audio frame"""
    data: np.ndarray
    timestamp_ms: float
    sample_rate: int
    duration_ms: float
    is_processed: bool = False
    
    @property
    def rms_db(self) -> float:
        """Calculate RMS in decibels"""
        rms = np.sqrt(np.mean(self.data ** 2))
        if rms > 0:
            return 20 * np.log10(rms)
        return -100.0
    
    @property
    def peak_amplitude(self) -> float:
        """Get peak amplitude"""
        return float(np.max(np.abs(self.data)))


class AudioProcessor:
    """
    Production-grade audio preprocessing pipeline
    
    Features:
    - Resampling to target sample rate
    - Noise gate
    - Normalization
    - High-pass filtering
    - DC offset removal
    """
    
    def __init__(
        self,
        target_sample_rate: int = 16000,
        target_channels: int = 1,
        apply_noise_gate: bool = True,
        noise_gate_threshold_db: float = -50.0,
        apply_highpass: bool = True,
        highpass_cutoff_hz: float = 80.0,
        normalize: bool = True,
        remove_dc_offset: bool = True
    ):
        self.target_sample_rate = target_sample_rate
        self.target_channels = target_channels
        self.apply_noise_gate = apply_noise_gate
        self.noise_gate_threshold_db = noise_gate_threshold_db
        self.apply_highpass = apply_highpass
        self.highpass_cutoff_hz = highpass_cutoff_hz
        self.normalize = normalize
        self.remove_dc_offset = remove_dc_offset
        
        # Pre-compute highpass filter coefficients
        self._highpass_b, self._highpass_a = signal.butter(
            4, 
            highpass_cutoff_hz / (target_sample_rate / 2), 
            btype='high'
        )
        
        # Filter state for continuous processing
        self._filter_state = signal.lfilter_zi(
            self._highpass_b, self._highpass_a
        )
        
        # Stats for adaptive processing
        self._noise_floor_db = -60.0
        self._noise_samples = deque(maxlen=100)
        
        logger.info(
            "AudioProcessor initialized",
            sample_rate=target_sample_rate,
            channels=target_channels
        )
    
    def process(
        self,
        audio_data: np.ndarray,
        source_sample_rate: int,
        source_channels: int = 1,
        timestamp_ms: float = 0.0
    ) -> AudioFrame:
        """
        Process raw audio data through the pipeline
        
        Args:
            audio_data: Raw audio samples (int16 or float32)
            source_sample_rate: Sample rate of input audio
            source_channels: Number of channels in input
            timestamp_ms: Timestamp of this frame
            
        Returns:
            Processed AudioFrame
        """
        # Convert to float32 if needed
        if audio_data.dtype == np.int16:
            audio = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            audio = audio_data.astype(np.float32) / 2147483648.0
        else:
            audio = audio_data.astype(np.float32)
        
        # Convert to mono if needed
        if source_channels > 1:
            audio = self._to_mono(audio, source_channels)
        
        # Resample if needed
        if source_sample_rate != self.target_sample_rate:
            audio = self._resample(audio, source_sample_rate)
        
        # DC offset removal
        if self.remove_dc_offset:
            audio = audio - np.mean(audio)
        
        # High-pass filter
        if self.apply_highpass:
            audio, self._filter_state = signal.lfilter(
                self._highpass_b,
                self._highpass_a,
                audio,
                zi=self._filter_state * audio[0] if len(audio) > 0 else self._filter_state
            )
        
        # Noise gate
        if self.apply_noise_gate:
            audio = self._apply_noise_gate(audio)
        
        # Normalize
        if self.normalize:
            audio = self._normalize(audio)
        
        duration_ms = len(audio) / self.target_sample_rate * 1000
        
        return AudioFrame(
            data=audio,
            timestamp_ms=timestamp_ms,
            sample_rate=self.target_sample_rate,
            duration_ms=duration_ms,
            is_processed=True
        )
    
    def _to_mono(self, audio: np.ndarray, channels: int) -> np.ndarray:
        """Convert multi-channel audio to mono"""
        if len(audio.shape) == 1:
            # Already mono, might be interleaved
            if len(audio) % channels == 0:
                audio = audio.reshape(-1, channels)
                return np.mean(audio, axis=1)
            return audio
        return np.mean(audio, axis=1)
    
    def _resample(self, audio: np.ndarray, source_rate: int) -> np.ndarray:
        """Resample audio to target sample rate"""
        if source_rate == self.target_sample_rate:
            return audio
        
        # Calculate new length
        duration = len(audio) / source_rate
        new_length = int(duration * self.target_sample_rate)
        
        # Use scipy resample for high quality
        return signal.resample(audio, new_length)
    
    def _apply_noise_gate(self, audio: np.ndarray) -> np.ndarray:
        """Apply noise gate to reduce background noise"""
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 0:
            rms_db = 20 * np.log10(rms)
        else:
            rms_db = -100.0
        
        # Update noise floor estimate
        self._noise_samples.append(rms_db)
        if len(self._noise_samples) >= 10:
            self._noise_floor_db = np.percentile(list(self._noise_samples), 10)
        
        # Apply gate
        if rms_db < self.noise_gate_threshold_db:
            # Soft gate - reduce volume instead of zeroing
            gate_ratio = 0.1
            return audio * gate_ratio
        
        return audio
    
    def _normalize(self, audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """Normalize audio to target dB level"""
        peak = np.max(np.abs(audio))
        if peak > 0:
            target_amplitude = 10 ** (target_db / 20)
            return audio * (target_amplitude / peak)
        return audio
    
    def reset(self):
        """Reset processor state"""
        self._filter_state = signal.lfilter_zi(
            self._highpass_b, self._highpass_a
        )
        self._noise_samples.clear()
        self._noise_floor_db = -60.0