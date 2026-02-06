# src/core/vad_engine.py
"""
Main VAD Engine - Orchestrates all components
"""
import asyncio
from typing import Optional, Callable, AsyncIterator, Dict, Any
from dataclasses import dataclass
from enum import Enum
import numpy as np
import structlog
import time

from ..vad_backends.base_vad import BaseVAD, VADResult
from ..vad_backends.silero_vad import SileroVAD
from ..vad_backends.webrtc_vad import WebRTCVAD
from ..vad_backends.energy_vad import EnergyVAD
from ..vad_backends.hybrid_vad import HybridVAD
from .audio_processor import AudioProcessor, AudioFrame
from .state_machine import SpeechStateMachine, SpeechSegment, SpeechState
from ..utils.ring_buffer import RingBuffer
from config.settings import VADSettings, AudioSettings, VADBackend

logger = structlog.get_logger(__name__)


@dataclass
class VADEvent:
    """Event emitted by VAD engine"""
    event_type: str  # "speech_start", "speech_end", "speech_probability"
    timestamp_ms: float
    data: Dict[str, Any]


class VADEngine:
    """
    Production VAD Engine
    
    Features:
    - Multiple backend support (Silero, WebRTC, Energy, Hybrid)
    - Audio preprocessing pipeline
    - State machine for robust detection
    - Async event emission
    - Thread-safe audio buffering
    - Metrics collection
    """
    
    def __init__(
        self,
        audio_settings: Optional[AudioSettings] = None,
        vad_settings: Optional[VADSettings] = None
    ):
        self.audio_settings = audio_settings or AudioSettings()
        self.vad_settings = vad_settings or VADSettings()
        
        # Components
        self._audio_processor: Optional[AudioProcessor] = None
        self._vad_backend: Optional[BaseVAD] = None
        self._state_machine: Optional[SpeechStateMachine] = None
        
        # Audio buffer
        self._audio_buffer: Optional[RingBuffer] = None
        
        # Event callbacks
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None
        self._on_probability: Optional[Callable] = None
        
        # State
        self._is_initialized = False
        self._is_running = False
        self._current_timestamp_ms = 0.0
        
        # Metrics
        self._frames_processed = 0
        self._speech_segments_detected = 0
        self._total_speech_duration_ms = 0.0
        
    def initialize(self):
        """Initialize all components"""
        if self._is_initialized:
            return
        
        logger.info("Initializing VAD Engine", backend=self.vad_settings.backend.value)
        
        # Initialize audio processor
        self._audio_processor = AudioProcessor(
            target_sample_rate=self.audio_settings.sample_rate,
            target_channels=self.audio_settings.channels
        )
        
        # Initialize VAD backend
        self._vad_backend = self._create_backend()
        self._vad_backend.initialize()
        
        # Initialize state machine
        self._state_machine = SpeechStateMachine(
            sample_rate=self.audio_settings.sample_rate,
            min_speech_duration_ms=self.vad_settings.min_speech_duration_ms,
            min_silence_duration_ms=self.vad_settings.min_silence_duration_ms,
            speech_pad_ms=self.vad_settings.speech_pad_ms,
            pre_speech_buffer_ms=self.vad_settings.pre_speech_buffer_ms,
            max_speech_duration_ms=self.vad_settings.max_speech_duration_ms,
            speech_threshold=self.vad_settings.speech_threshold,
            silence_threshold=self.vad_settings.silence_threshold
        )
        
        # Register state machine callbacks
        self._state_machine.on_speech_start(self._handle_speech_start)
        self._state_machine.on_speech_end(self._handle_speech_end)
        
        # Initialize audio buffer (5 seconds capacity)
        buffer_capacity = self.audio_settings.sample_rate * 5
        self._audio_buffer = RingBuffer(buffer_capacity)
        
        self._is_initialized = True
        logger.info("VAD Engine initialized successfully")
    
    def _create_backend(self) -> BaseVAD:
        """Create VAD backend based on settings"""
        if self.vad_settings.backend == VADBackend.SILERO:
            return SileroVAD(
                sample_rate=self.audio_settings.sample_rate,
                threshold=self.vad_settings.speech_threshold
            )
        elif self.vad_settings.backend == VADBackend.WEBRTC:
            return WebRTCVAD(
                sample_rate=self.audio_settings.sample_rate,
                aggressiveness=self.vad_settings.webrtc_aggressiveness
            )
        elif self.vad_settings.backend == VADBackend.ENERGY:
            return EnergyVAD(
                sample_rate=self.audio_settings.sample_rate,
                threshold_db=self.vad_settings.energy_threshold_db
            )
        elif self.vad_settings.backend == VADBackend.HYBRID:
            return HybridVAD(
                sample_rate=self.audio_settings.sample_rate,
                silero_weight=self.vad_settings.silero_weight,
                webrtc_weight=self.vad_settings.webrtc_weight,
                energy_weight=self.vad_settings.energy_weight,
                threshold=self.vad_settings.speech_threshold
            )
        else:
            raise ValueError(f"Unknown backend: {self.vad_settings.backend}")
    
    def process_audio(
        self,
        audio_data: np.ndarray,
        source_sample_rate: int,
        source_channels: int = 1
    ) -> VADResult:
        """
        Process audio data and return VAD result
        
        Args:
            audio_data: Raw audio samples
            source_sample_rate: Sample rate of input
            source_channels: Number of channels
            
        Returns:
            VADResult with speech detection info
        """
        if not self._is_initialized:
            self.initialize()
        
        # Process audio
        frame = self._audio_processor.process(
            audio_data,
            source_sample_rate,
            source_channels,
            self._current_timestamp_ms
        )
        
        # Run VAD
        result = self._vad_backend.process_frame(
            frame.data,
            self._current_timestamp_ms
        )
        
        # Update state machine
        self._state_machine.process(
            result.raw_probability or (1.0 if result.is_speech else 0.0),
            frame.data,
            self._current_timestamp_ms
        )
        
        # Update timestamp
        self._current_timestamp_ms += frame.duration_ms
        self._frames_processed += 1
        
        # Emit probability event
        if self._on_probability:
            self._on_probability(VADEvent(
                event_type="speech_probability",
                timestamp_ms=self._current_timestamp_ms,
                data={
                    "probability": result.raw_probability,
                    "is_speech": result.is_speech,
                    "state": self._state_machine.state.name
                }
            ))
        
        return result
    
    async def process_stream(
        self,
        audio_stream: AsyncIterator[np.ndarray],
        sample_rate: int,
        channels: int = 1
    ) -> AsyncIterator[VADEvent]:
        """
        Process audio stream asynchronously
        
        Args:
            audio_stream: Async iterator of audio chunks
            sample_rate: Sample rate of stream
            channels: Number of channels
            
        Yields:
            VADEvent for each detection event
        """
        if not self._is_initialized:
            self.initialize()
        
        self._is_running = True
        event_queue = asyncio.Queue()
        
        # Set up callbacks to queue events
        def queue_event(segment: SpeechSegment, event_type: str):
            asyncio.create_task(event_queue.put(VADEvent(
                event_type=event_type,
                timestamp_ms=segment.start_time_ms if event_type == "speech_start" else segment.end_time_ms,
                data={
                    "segment": segment,
                    "duration_ms": segment.duration_ms
                }
            )))
        
        self._on_speech_start = lambda s: queue_event(s, "speech_start")
        self._on_speech_end = lambda s: queue_event(s, "speech_end")
        
        try:
            async for audio_chunk in audio_stream:
                if not self._is_running:
                    break
                
                self.process_audio(audio_chunk, sample_rate, channels)
                
                # Yield any queued events
                while not event_queue.empty():
                    yield await event_queue.get()
                    
        finally:
            self._is_running = False
    
    def _handle_speech_start(self, segment: SpeechSegment):
        """Handle speech start event"""
        logger.info(
            "Speech started",
            timestamp_ms=segment.start_time_ms,
            confidence=segment.confidence
        )
        if self._on_speech_start:
            self._on_speech_start(segment)
    
    def _handle_speech_end(self, segment: SpeechSegment):
        """Handle speech end event"""
        self._speech_segments_detected += 1
        if segment.duration_ms:
            self._total_speech_duration_ms += segment.duration_ms
        
        logger.info(
            "Speech ended",
            start_ms=segment.start_time_ms,
            end_ms=segment.end_time_ms,
            duration_ms=segment.duration_ms
        )
        if self._on_speech_end:
            self._on_speech_end(segment)
    
    def on_speech_start(self, callback: Callable[[SpeechSegment], None]):
        """Register speech start callback"""
        self._on_speech_start = callback
    
    def on_speech_end(self, callback: Callable[[SpeechSegment], None]):
        """Register speech end callback"""
        self._on_speech_end = callback
    
    def on_probability(self, callback: Callable[[VADEvent], None]):
        """Register probability update callback"""
        self._on_probability = callback
    
    @property
    def is_speaking(self) -> bool:
        """Check if currently detecting speech"""
        if self._state_machine:
            return self._state_machine.is_speaking
        return False
    
    @property
    def state(self) -> Optional[SpeechState]:
        """Get current state"""
        if self._state_machine:
            return self._state_machine.state
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get engine metrics"""
        return {
            "frames_processed": self._frames_processed,
            "speech_segments_detected": self._speech_segments_detected,
            "total_speech_duration_ms": self._total_speech_duration_ms,
            "current_state": self._state_machine.state.name if self._state_machine else None,
            "is_speaking": self.is_speaking
        }
    
    def reset(self):
        """Reset engine state"""
        if self._vad_backend:
            self._vad_backend.reset()
        if self._state_machine:
            self._state_machine.reset()
        if self._audio_processor:
            self._audio_processor.reset()
        
        self._current_timestamp_ms = 0.0
        logger.info("VAD Engine reset")
    
    def cleanup(self):
        """Cleanup resources"""
        self._is_running = False
        if self._vad_backend:
            self._vad_backend.cleanup()
        self._is_initialized = False
        logger.info("VAD Engine cleaned up")
    
    def __enter__(self):
        self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()