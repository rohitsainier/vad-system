# src/audio_sources/stream_source.py
"""
Generic stream audio source for handling various audio streams
"""
import numpy as np
import asyncio
from typing import Optional, Callable, AsyncIterator
from queue import Queue, Empty
from threading import Thread, Event
import structlog

from .base_source import BaseAudioSource

logger = structlog.get_logger(__name__)


class StreamSource(BaseAudioSource):
    """
    Generic streaming audio source
    
    Accepts audio data pushed from external sources (e.g., network streams)
    
    Features:
    - Thread-safe audio buffering
    - Configurable buffer size
    - Overflow handling
    - Format conversion
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,
        buffer_size: int = 100,
        overflow_strategy: str = "drop_oldest"  # "drop_oldest", "drop_newest", "block"
    ):
        super().__init__(sample_rate, channels, chunk_duration_ms)
        
        self.buffer_size = buffer_size
        self.overflow_strategy = overflow_strategy
        
        self._buffer: Queue = Queue(maxsize=buffer_size)
        self._stop_event = Event()
        self._total_samples_received = 0
        self._total_samples_dropped = 0
    
    def open(self) -> None:
        """Open the stream source"""
        if self._is_open:
            return
        
        self._stop_event.clear()
        self._is_open = True
        
        logger.info(
            "StreamSource opened",
            sample_rate=self.sample_rate,
            buffer_size=self.buffer_size
        )
    
    def close(self) -> None:
        """Close the stream source"""
        self._is_open = False
        self._stop_event.set()
        
        # Clear buffer
        while not self._buffer.empty():
            try:
                self._buffer.get_nowait()
            except Empty:
                break
        
        logger.info(
            "StreamSource closed",
            samples_received=self._total_samples_received,
            samples_dropped=self._total_samples_dropped
        )
    
    def push(self, audio_data: np.ndarray) -> bool:
        """
        Push audio data into the stream
        
        Args:
            audio_data: Audio samples to add to stream
            
        Returns:
            True if data was added, False if dropped
        """
        if not self._is_open:
            return False
        
        self._total_samples_received += len(audio_data)
        
        # Handle overflow
        if self._buffer.full():
            if self.overflow_strategy == "drop_oldest":
                try:
                    self._buffer.get_nowait()
                    self._total_samples_dropped += self.chunk_size
                except Empty:
                    pass
            elif self.overflow_strategy == "drop_newest":
                self._total_samples_dropped += len(audio_data)
                return False
            # "block" strategy will wait
        
        try:
            self._buffer.put(audio_data, block=(self.overflow_strategy == "block"), timeout=1.0)
            return True
        except:
            self._total_samples_dropped += len(audio_data)
            return False
    
    def push_bytes(self, audio_bytes: bytes, dtype=np.int16) -> bool:
        """
        Push raw audio bytes into the stream
        
        Args:
            audio_bytes: Raw audio bytes
            dtype: Data type of the audio
            
        Returns:
            True if data was added
        """
        audio_data = np.frombuffer(audio_bytes, dtype=dtype)
        
        # Convert to float32
        if dtype == np.int16:
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif dtype == np.int32:
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        
        return self.push(audio_data)
    
    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read next audio chunk (blocking)"""
        if not self._is_open or self._stop_event.is_set():
            return None
        
        try:
            return self._buffer.get(timeout=timeout)
        except Empty:
            return None
    
    async def read_async(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Read next audio chunk (async)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.read, timeout)
    
    @property
    def available(self) -> int:
        """Number of chunks available in buffer"""
        return self._buffer.qsize()
    
    @property
    def stats(self) -> dict:
        """Get stream statistics"""
        return {
            "samples_received": self._total_samples_received,
            "samples_dropped": self._total_samples_dropped,
            "buffer_usage": self._buffer.qsize() / self.buffer_size,
            "drop_rate": self._total_samples_dropped / max(1, self._total_samples_received)
        }