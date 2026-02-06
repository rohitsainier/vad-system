# src/voice_chat/conversation.py
"""
Conversation state manager
"""
from dataclasses import dataclass, field
from typing import List, Optional
import time


@dataclass
class Turn:
    """A single conversation turn"""
    user_text: str
    assistant_text: str
    user_audio_duration_ms: float
    assistant_audio_duration_ms: float
    stt_time_ms: float
    llm_time_ms: float
    tts_time_ms: float
    total_time_ms: float
    timestamp: float = field(default_factory=time.time)
    
    @property
    def latency_summary(self) -> str:
        return (
            f"STT: {self.stt_time_ms:.0f}ms | "
            f"LLM: {self.llm_time_ms:.0f}ms | "
            f"TTS: {self.tts_time_ms:.0f}ms | "
            f"Total: {self.total_time_ms:.0f}ms"
        )


class Conversation:
    """Manages conversation state and history"""
    
    def __init__(self, max_turns: int = 100):
        self.max_turns = max_turns
        self.turns: List[Turn] = []
        self.start_time = time.time()
        self._is_active = False
    
    def add_turn(self, turn: Turn):
        """Add a conversation turn"""
        self.turns.append(turn)
        if len(self.turns) > self.max_turns:
            self.turns.pop(0)
    
    @property
    def turn_count(self) -> int:
        return len(self.turns)
    
    @property
    def total_duration_s(self) -> float:
        return time.time() - self.start_time
    
    @property
    def avg_latency_ms(self) -> float:
        if not self.turns:
            return 0
        return sum(t.total_time_ms for t in self.turns) / len(self.turns)
    
    def get_summary(self) -> dict:
        return {
            "turns": self.turn_count,
            "duration_s": self.total_duration_s,
            "avg_latency_ms": self.avg_latency_ms,
            "avg_stt_ms": sum(t.stt_time_ms for t in self.turns) / max(1, len(self.turns)),
            "avg_llm_ms": sum(t.llm_time_ms for t in self.turns) / max(1, len(self.turns)),
            "avg_tts_ms": sum(t.tts_time_ms for t in self.turns) / max(1, len(self.turns)),
        }