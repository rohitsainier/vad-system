# src/core/state_machine.py
"""
State machine for managing speech detection states with smoothing
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, List, Callable
from collections import deque
import time
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class SpeechState(Enum):
    """Speech detection states"""
    SILENCE = auto()
    SPEECH_START = auto()
    SPEAKING = auto()
    SPEECH_END = auto()


@dataclass
class SpeechSegment:
    """Represents a detected speech segment"""
    start_time_ms: float
    end_time_ms: Optional[float] = None
    audio_data: Optional[np.ndarray] = None
    confidence: float = 0.0
    
    @property
    def duration_ms(self) -> Optional[float]:
        if self.end_time_ms is not None:
            return self.end_time_ms - self.start_time_ms
        return None
    
    @property
    def is_complete(self) -> bool:
        return self.end_time_ms is not None


@dataclass
class StateTransition:
    """Represents a state transition event"""
    from_state: SpeechState
    to_state: SpeechState
    timestamp_ms: float
    segment: Optional[SpeechSegment] = None


class SpeechStateMachine:
    """
    State machine for robust speech detection with hysteresis
    
    Features:
    - Minimum speech/silence duration requirements
    - Pre-speech buffering to capture onset
    - Hangover time to avoid choppy detection
    - Confidence-weighted decision making
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        min_speech_duration_ms: float = 250,
        min_silence_duration_ms: float = 300,
        speech_pad_ms: float = 30,
        pre_speech_buffer_ms: float = 300,
        max_speech_duration_ms: float = 30000,
        speech_threshold: float = 0.5,
        silence_threshold: float = 0.35,
        smoothing_window: int = 5
    ):
        self.sample_rate = sample_rate
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_pad_ms = speech_pad_ms
        self.pre_speech_buffer_ms = pre_speech_buffer_ms
        self.max_speech_duration_ms = max_speech_duration_ms
        self.speech_threshold = speech_threshold
        self.silence_threshold = silence_threshold
        self.smoothing_window = smoothing_window
        
        # State
        self._state = SpeechState.SILENCE
        self._state_start_time_ms = 0.0
        self._current_segment: Optional[SpeechSegment] = None
        
        # Buffers
        self._pre_speech_buffer: deque = deque()
        self._speech_buffer: List[np.ndarray] = []
        self._probability_history: deque = deque(maxlen=smoothing_window)
        
        # Timing
        self._last_speech_time_ms = 0.0
        self._speech_start_time_ms = 0.0
        
        # Callbacks
        self._on_speech_start: Optional[Callable] = None
        self._on_speech_end: Optional[Callable] = None
        self._on_state_change: Optional[Callable] = None
        
        logger.info(
            "SpeechStateMachine initialized",
            min_speech_ms=min_speech_duration_ms,
            min_silence_ms=min_silence_duration_ms
        )
    
    def on_speech_start(self, callback: Callable[[SpeechSegment], None]):
        """Register callback for speech start"""
        self._on_speech_start = callback
    
    def on_speech_end(self, callback: Callable[[SpeechSegment], None]):
        """Register callback for speech end"""
        self._on_speech_end = callback
    
    def on_state_change(self, callback: Callable[[StateTransition], None]):
        """Register callback for any state change"""
        self._on_state_change = callback
    
    def process(
        self,
        probability: float,
        audio_frame: np.ndarray,
        timestamp_ms: float
    ) -> Optional[StateTransition]:
        """
        Process a VAD result and update state
        
        Args:
            probability: Speech probability (0-1)
            audio_frame: Audio data for this frame
            timestamp_ms: Timestamp of this frame
            
        Returns:
            StateTransition if state changed, None otherwise
        """
        # Add to probability history for smoothing
        self._probability_history.append(probability)
        smoothed_prob = np.mean(list(self._probability_history))
        
        # Update pre-speech buffer
        frame_duration_ms = len(audio_frame) / self.sample_rate * 1000
        self._update_pre_speech_buffer(audio_frame, frame_duration_ms)
        
        # Determine if this frame is speech
        is_speech = smoothed_prob >= self.speech_threshold
        
        # State machine logic
        previous_state = self._state
        transition = None
        
        if self._state == SpeechState.SILENCE:
            if is_speech:
                self._state = SpeechState.SPEECH_START
                self._speech_start_time_ms = timestamp_ms
                self._state_start_time_ms = timestamp_ms
                self._speech_buffer = []
                
                # Add pre-speech buffer
                for pre_audio in self._pre_speech_buffer:
                    self._speech_buffer.append(pre_audio)
                
                self._speech_buffer.append(audio_frame)
                
        elif self._state == SpeechState.SPEECH_START:
            self._speech_buffer.append(audio_frame)
            
            time_in_state = timestamp_ms - self._state_start_time_ms
            
            if is_speech:
                self._last_speech_time_ms = timestamp_ms
                
                if time_in_state >= self.min_speech_duration_ms:
                    # Confirmed speech
                    self._state = SpeechState.SPEAKING
                    self._state_start_time_ms = timestamp_ms
                    
                    # Create segment
                    self._current_segment = SpeechSegment(
                        start_time_ms=self._speech_start_time_ms - self.pre_speech_buffer_ms,
                        confidence=smoothed_prob
                    )
                    
                    if self._on_speech_start:
                        self._on_speech_start(self._current_segment)
                        
            else:
                # Check if we should cancel speech start
                silence_duration = timestamp_ms - self._last_speech_time_ms
                if silence_duration > self.min_speech_duration_ms / 2:
                    # False start, return to silence
                    self._state = SpeechState.SILENCE
                    self._state_start_time_ms = timestamp_ms
                    self._speech_buffer = []
                    
        elif self._state == SpeechState.SPEAKING:
            self._speech_buffer.append(audio_frame)
            
            if is_speech:
                self._last_speech_time_ms = timestamp_ms
                
                # Check for max duration
                speech_duration = timestamp_ms - self._speech_start_time_ms
                if speech_duration >= self.max_speech_duration_ms:
                    # Force end speech
                    self._state = SpeechState.SPEECH_END
                    self._state_start_time_ms = timestamp_ms
            else:
                # Check for silence threshold
                if smoothed_prob < self.silence_threshold:
                    silence_duration = timestamp_ms - self._last_speech_time_ms
                    
                    if silence_duration >= self.min_silence_duration_ms:
                        # End of speech
                        self._state = SpeechState.SPEECH_END
                        self._state_start_time_ms = timestamp_ms
                        
        elif self._state == SpeechState.SPEECH_END:
            # Finalize segment
            if self._current_segment:
                self._current_segment.end_time_ms = self._last_speech_time_ms + self.speech_pad_ms
                self._current_segment.audio_data = np.concatenate(self._speech_buffer)
                
                if self._on_speech_end:
                    self._on_speech_end(self._current_segment)
            
            # Reset
            self._speech_buffer = []
            self._current_segment = None
            self._state = SpeechState.SILENCE
            self._state_start_time_ms = timestamp_ms
        
        # Create transition if state changed
        if self._state != previous_state:
            transition = StateTransition(
                from_state=previous_state,
                to_state=self._state,
                timestamp_ms=timestamp_ms,
                segment=self._current_segment
            )
            
            if self._on_state_change:
                self._on_state_change(transition)
            
            logger.debug(
                "State transition",
                from_state=previous_state.name,
                to_state=self._state.name,
                timestamp_ms=timestamp_ms
            )
        
        return transition
    
    def _update_pre_speech_buffer(self, audio_frame: np.ndarray, frame_duration_ms: float):
        """Maintain rolling pre-speech buffer"""
        self._pre_speech_buffer.append(audio_frame)
        
        # Calculate total buffer duration
        total_samples = sum(len(f) for f in self._pre_speech_buffer)
        total_duration_ms = total_samples / self.sample_rate * 1000
        
        # Remove old frames if over limit
        while total_duration_ms > self.pre_speech_buffer_ms and len(self._pre_speech_buffer) > 1:
            removed = self._pre_speech_buffer.popleft()
            total_samples -= len(removed)
            total_duration_ms = total_samples / self.sample_rate * 1000
    
    @property
    def state(self) -> SpeechState:
        return self._state
    
    @property
    def is_speaking(self) -> bool:
        return self._state in (SpeechState.SPEECH_START, SpeechState.SPEAKING)
    
    def reset(self):
        """Reset state machine"""
        self._state = SpeechState.SILENCE
        self._state_start_time_ms = 0.0
        self._current_segment = None
        self._pre_speech_buffer.clear()
        self._speech_buffer = []
        self._probability_history.clear()
        self._last_speech_time_ms = 0.0
        self._speech_start_time_ms = 0.0