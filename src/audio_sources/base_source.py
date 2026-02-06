# src/audio_sources/base_source.py
"""
Base class for audio sources
"""
from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator, Iterator
import numpy as np
from dataclasses import dataclass


@dataclass
class AudioChunk:
    """Represents a chunk of audio data"""
    data: np.ndarray
    sample_rate: int
    channels: int
    timestamp_ms: float
    duration_ms: float
    
    @property
    def samples(self) -> int:
        return len(self.data)


class BaseAudioSource(ABC):
    """
    Abstract base class for audio sources
    
    All audio sources must implement this interface to work with the VAD system
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self._is_open = False
    
    @abstractmethod
    def open(self) -> None:
        """Open the audio source"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """Close the audio source"""
        pass
    
    @abstractmethod
    def read(self) -> Optional[np.ndarray]:
        """Read a chunk of audio (blocking)"""
        pass
    
    @abstractmethod
    async def read_async(self) -> Optional[np.ndarray]:
        """Read a chunk of audio (async)"""
        pass
    
    def stream(self) -> Iterator[np.ndarray]:
        """Synchronous iterator for audio chunks"""
        self.open()
        try:
            while self._is_open:
                chunk = self.read()
                if chunk is not None:
                    yield chunk
        finally:
            self.close()
    
    async def stream_async(self) -> AsyncIterator[np.ndarray]:
        """Asynchronous iterator for audio chunks"""
        self.open()
        try:
            while self._is_open:
                chunk = await self.read_async()
                if chunk is not None:
                    yield chunk
        finally:
            self.close()
    
    @property
    def is_open(self) -> bool:
        return self._is_open
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    async def __aenter__(self):
        self.open()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.close()