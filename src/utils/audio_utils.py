# src/utils/audio_utils.py
"""
Audio utility functions
"""
import numpy as np
from typing import Tuple, Optional
import scipy.signal as signal
import structlog

logger = structlog.get_logger(__name__)


class AudioUtils:
    """Collection of audio utility functions"""
    
    @staticmethod
    def resample(
        audio: np.ndarray,
        source_rate: int,
        target_rate: int
    ) -> np.ndarray:
        """
        Resample audio to target sample rate
        
        Args:
            audio: Audio samples
            source_rate: Original sample rate
            target_rate: Target sample rate
            
        Returns:
            Resampled audio
        """
        if source_rate == target_rate:
            return audio
        
        duration = len(audio) / source_rate
        target_length = int(duration * target_rate)
        
        return signal.resample(audio, target_length)
    
    @staticmethod
    def to_mono(audio: np.ndarray, channels: int = 2) -> np.ndarray:
        """
        Convert multi-channel audio to mono
        
        Args:
            audio: Audio samples (possibly interleaved or 2D)
            channels: Number of channels
            
        Returns:
            Mono audio
        """
        if len(audio.shape) == 1:
            if channels == 1:
                return audio
            # Interleaved
            audio = audio.reshape(-1, channels)
        
        return np.mean(audio, axis=1)
    
    @staticmethod
    def normalize(
        audio: np.ndarray,
        target_db: float = -3.0,
        headroom: float = 0.1
    ) -> np.ndarray:
        """
        Normalize audio to target level
        
        Args:
            audio: Audio samples
            target_db: Target peak level in dB
            headroom: Headroom to prevent clipping
            
        Returns:
            Normalized audio
        """
        peak = np.max(np.abs(audio))
        if peak < 1e-10:
            return audio
        
        target_amplitude = 10 ** (target_db / 20) * (1 - headroom)
        return audio * (target_amplitude / peak)
    
    @staticmethod
    def calculate_rms(audio: np.ndarray) -> float:
        """Calculate RMS energy"""
        return float(np.sqrt(np.mean(audio ** 2)))
    
    @staticmethod
    def calculate_rms_db(audio: np.ndarray) -> float:
        """Calculate RMS energy in decibels"""
        rms = AudioUtils.calculate_rms(audio)
        if rms > 1e-10:
            return 20 * np.log10(rms)
        return -100.0
    
    @staticmethod
    def calculate_zcr(audio: np.ndarray) -> float:
        """Calculate zero crossing rate"""
        signs = np.sign(audio)
        sign_changes = np.abs(np.diff(signs))
        return float(np.mean(sign_changes) / 2)
    
    @staticmethod
    def apply_fade(
        audio: np.ndarray,
        fade_in_ms: float = 10,
        fade_out_ms: float = 10,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Apply fade in/out to audio
        
        Args:
            audio: Audio samples
            fade_in_ms: Fade in duration
            fade_out_ms: Fade out duration
            sample_rate: Sample rate
            
        Returns:
            Audio with fades applied
        """
        result = audio.copy()
        
        # Fade in
        fade_in_samples = int(fade_in_ms / 1000 * sample_rate)
        if fade_in_samples > 0 and fade_in_samples < len(result):
            fade_in = np.linspace(0, 1, fade_in_samples)
            result[:fade_in_samples] *= fade_in
        
        # Fade out
        fade_out_samples = int(fade_out_ms / 1000 * sample_rate)
        if fade_out_samples > 0 and fade_out_samples < len(result):
            fade_out = np.linspace(1, 0, fade_out_samples)
            result[-fade_out_samples:] *= fade_out
        
        return result
    
    @staticmethod
    def remove_silence(
        audio: np.ndarray,
        threshold_db: float = -40.0,
        min_silence_ms: float = 100,
        sample_rate: int = 16000
    ) -> np.ndarray:
        """
        Remove leading and trailing silence
        
        Args:
            audio: Audio samples
            threshold_db: Silence threshold in dB
            min_silence_ms: Minimum silence duration to trim
            sample_rate: Sample rate
            
        Returns:
            Audio with silence removed
        """
        threshold = 10 ** (threshold_db / 20)
        
        # Find non-silent regions
        amplitude = np.abs(audio)
        is_sound = amplitude > threshold
        
        # Find first and last non-silent sample
        sound_indices = np.where(is_sound)[0]
        
        if len(sound_indices) == 0:
            return audio
        
        start = max(0, sound_indices[0] - int(min_silence_ms / 1000 * sample_rate))
        end = min(len(audio), sound_indices[-1] + int(min_silence_ms / 1000 * sample_rate))
        
        return audio[start:end]
    
    @staticmethod
    def split_on_silence(
        audio: np.ndarray,
        threshold_db: float = -40.0,
        min_silence_ms: float = 500,
        min_speech_ms: float = 250,
        sample_rate: int = 16000
    ) -> list:
        """
        Split audio on silence
        
        Args:
            audio: Audio samples
            threshold_db: Silence threshold in dB
            min_silence_ms: Minimum silence duration to split on
            min_speech_ms: Minimum speech segment duration
            sample_rate: Sample rate
            
        Returns:
            List of (start_sample, end_sample) tuples
        """
        threshold = 10 ** (threshold_db / 20)
        min_silence_samples = int(min_silence_ms / 1000 * sample_rate)
        min_speech_samples = int(min_speech_ms / 1000 * sample_rate)
        
        # Calculate amplitude
        amplitude = np.abs(audio)
        is_sound = amplitude > threshold
        
        segments = []
        in_speech = False
        speech_start = 0
        silence_count = 0
        
        for i, sound in enumerate(is_sound):
            if sound:
                if not in_speech:
                    in_speech = True
                    speech_start = i
                silence_count = 0
            else:
                if in_speech:
                    silence_count += 1
                    if silence_count >= min_silence_samples:
                        # End of speech
                        speech_end = i - silence_count
                        if speech_end - speech_start >= min_speech_samples:
                            segments.append((speech_start, speech_end))
                        in_speech = False
        
        # Handle final segment
        if in_speech:
            if len(audio) - speech_start >= min_speech_samples:
                segments.append((speech_start, len(audio)))
        
        return segments
    
    @staticmethod
    def convert_sample_format(
        audio: np.ndarray,
        from_format: str,
        to_format: str
    ) -> np.ndarray:
        """
        Convert between sample formats
        
        Args:
            audio: Audio samples
            from_format: Source format ('int16', 'int32', 'float32')
            to_format: Target format
            
        Returns:
            Converted audio
        """
        # First convert to float32
        if from_format == 'int16':
            audio_float = audio.astype(np.float32) / 32768.0
        elif from_format == 'int32':
            audio_float = audio.astype(np.float32) / 2147483648.0
        elif from_format == 'float32':
            audio_float = audio.astype(np.float32)
        else:
            raise ValueError(f"Unknown source format: {from_format}")
        
        # Convert to target format
        if to_format == 'int16':
            return (audio_float * 32767).clip(-32768, 32767).astype(np.int16)
        elif to_format == 'int32':
            return (audio_float * 2147483647).astype(np.int32)
        elif to_format == 'float32':
            return audio_float
        else:
            raise ValueError(f"Unknown target format: {to_format}")