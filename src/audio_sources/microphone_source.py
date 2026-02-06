# src/audio_sources/microphone_source.py
"""
Microphone audio source for real-time VAD
"""
import numpy as np
import sounddevice as sd
import asyncio
from typing import Optional, AsyncIterator, Callable
from queue import Queue
from threading import Thread, Event
import structlog

logger = structlog.get_logger(__name__)


class MicrophoneSource:
    """
    Real-time microphone audio capture
    
    Features:
    - Low-latency audio capture
    - Configurable sample rate and chunk size
    - Async iterator interface
    - Device selection
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,
        device: Optional[int] = None,
        dtype: np.dtype = np.float32
    ):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_duration_ms = chunk_duration_ms
        self.device = device
        self.dtype = dtype
        
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        
        self._stream: Optional[sd.InputStream] = None
        self._audio_queue: Queue = Queue()
        self._stop_event = Event()
        self._is_running = False
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for audio stream"""
        if status:
            logger.warning("Audio stream status", status=str(status))
        
        # Copy data to queue
        self._audio_queue.put(indata.copy().flatten())
    
    def start(self):
        """Start audio capture"""
        if self._is_running:
            return
        
        self._stop_event.clear()
        
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.chunk_size,
            device=self.device,
            callback=self._audio_callback
        )
        
        self._stream.start()
        self._is_running = True
        
        logger.info(
            "Microphone started",
            sample_rate=self.sample_rate,
            chunk_size=self.chunk_size,
            device=self.device
        )
    
    def stop(self):
        """Stop audio capture"""
        self._stop_event.set()
        self._is_running = False
        
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        
        # Clear queue
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except:
                pass
        
        logger.info("Microphone stopped")
    
    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read next audio chunk (blocking)"""
        try:
            return self._audio_queue.get(timeout=timeout)
        except:
            return None
    
    async def read_async(self) -> Optional[np.ndarray]:
        """Read next audio chunk (async)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.read, 0.1)
    
    async def stream(self) -> AsyncIterator[np.ndarray]:
        """Async iterator for audio chunks"""
        self.start()
        
        try:
            while self._is_running and not self._stop_event.is_set():
                audio = await self.read_async()
                if audio is not None:
                    yield audio
                else:
                    await asyncio.sleep(0.001)
        finally:
            self.stop()
    
    @staticmethod
    def list_devices() -> list:
        """List available audio devices"""
        return sd.query_devices()
    
    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop()