# config/settings.py
"""
Production VAD System Configuration
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Literal, Optional
from enum import Enum


class VADBackend(str, Enum):
    SILERO = "silero"
    WEBRTC = "webrtc"
    ENERGY = "energy"
    HYBRID = "hybrid"


class AudioSettings(BaseSettings):
    """Audio processing configuration"""
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    channels: int = Field(default=1, ge=1, le=2)
    # Use 32ms for Silero compatibility (512 samples at 16kHz)
    chunk_duration_ms: int = Field(default=32, ge=10, le=100)
    bit_depth: int = Field(default=16, ge=8, le=32)
    
    @property
    def chunk_size(self) -> int:
        """Calculate chunk size in samples"""
        return int(self.sample_rate * self.chunk_duration_ms / 1000)
    
    @property
    def bytes_per_sample(self) -> int:
        return self.bit_depth // 8
    
    class Config:
        env_prefix = "VAD_AUDIO_"


class VADSettings(BaseSettings):
    """VAD engine configuration"""
    backend: VADBackend = VADBackend.SILERO
    
    # Detection thresholds
    speech_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    silence_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    
    # Timing parameters (in milliseconds)
    min_speech_duration_ms: int = Field(default=250, ge=50)
    min_silence_duration_ms: int = Field(default=300, ge=50)
    speech_pad_ms: int = Field(default=30, ge=0)
    
    # Pre-speech buffer to capture speech onset
    pre_speech_buffer_ms: int = Field(default=300, ge=0)
    
    # Maximum speech duration before forced split
    max_speech_duration_ms: int = Field(default=30000, ge=1000)
    
    # WebRTC VAD specific
    webrtc_aggressiveness: int = Field(default=3, ge=0, le=3)
    
    # Energy VAD specific
    energy_threshold_db: float = Field(default=-35.0, ge=-60.0, le=0.0)
    
    # Hybrid VAD weights
    silero_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    webrtc_weight: float = Field(default=0.3, ge=0.0, le=1.0)
    energy_weight: float = Field(default=0.1, ge=0.0, le=1.0)
    
    class Config:
        env_prefix = "VAD_"


class ServerSettings(BaseSettings):
    """Server configuration"""
    host: str = "0.0.0.0"
    port: int = 8000
    websocket_port: int = 8001
    max_connections: int = 100
    connection_timeout: int = 300
    
    class Config:
        env_prefix = "VAD_SERVER_"


class Settings(BaseSettings):
    """Main settings container"""
    audio: AudioSettings = Field(default_factory=AudioSettings)
    vad: VADSettings = Field(default_factory=VADSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    
    # Logging
    log_level: str = "INFO"
    enable_metrics: bool = True
    metrics_port: int = 9090
    
    class Config:
        env_prefix = "VAD_"
        env_nested_delimiter = "__"


# Global settings instance
settings = Settings()