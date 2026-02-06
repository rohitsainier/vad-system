# src/audio_sources/file_source.py
"""
File audio source for processing audio files
"""
import numpy as np
import soundfile as sf
from typing import Iterator, Optional, Tuple
from pathlib import Path
import structlog

logger = structlog.get_logger(__name__)


class FileSource:
    """
    Audio file source for VAD processing
    
    Features:
    - Supports multiple formats (WAV, FLAC, MP3, etc.)
    - Chunked reading for large files
    - Automatic resampling info
    """
    
    def __init__(
        self,
        file_path: str,
        chunk_duration_ms: int = 30,
        target_sample_rate: Optional[int] = None
    ):
        self.file_path = Path(file_path)
        self.chunk_duration_ms = chunk_duration_ms
        self.target_sample_rate = target_sample_rate
        
        if not self.file_path.exists():
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        # Get file info
        info = sf.info(str(self.file_path))
        self.source_sample_rate = info.samplerate
        self.channels = info.channels
        self.duration_seconds = info.duration
        self.frames = info.frames
        
        # Calculate chunk size
        sr = target_sample_rate or self.source_sample_rate
        self.chunk_size = int(sr * chunk_duration_ms / 1000)
        
        logger.info(
            "FileSource initialized",
            file=str(self.file_path),
            sample_rate=self.source_sample_rate,
            channels=self.channels,
            duration=self.duration_seconds
        )
    
    def read_all(self) -> Tuple[np.ndarray, int]:
        """Read entire file into memory"""
        audio, sr = sf.read(str(self.file_path), dtype='float32')
        
        # Convert to mono if needed
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        return audio, sr
    
    def stream(self) -> Iterator[np.ndarray]:
        """Stream audio in chunks"""
        chunk_frames = int(self.source_sample_rate * self.chunk_duration_ms / 1000)
        
        with sf.SoundFile(str(self.file_path)) as f:
            while True:
                audio = f.read(chunk_frames, dtype='float32')
                
                if len(audio) == 0:
                    break
                
                # Convert to mono if needed
                if len(audio.shape) > 1:
                    audio = np.mean(audio, axis=1)
                
                yield audio
    
    async def stream_async(self):
        """Async stream audio in chunks"""
        for chunk in self.stream():
            yield chunk
    
    @property
    def sample_rate(self) -> int:
        return self.source_sample_rate