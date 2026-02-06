# src/voice_chat/pipeline.py
"""
Voice-to-Voice Chat Pipeline
Orchestrates VAD → STT → LLM → TTS
"""
import numpy as np
import time
import asyncio
from typing import Optional, Callable
from dataclasses import dataclass
import structlog

from .stt_engine import STTEngine, TranscriptionResult
from .llm_engine import LLMEngine, LLMResponse
from .tts_engine import TTSEngine, TTSResult
from .conversation import Conversation, Turn

from ..core.vad_engine import VADEngine
from ..core.state_machine import SpeechSegment
from config.settings import Settings, VADBackend

logger = structlog.get_logger(__name__)


@dataclass
class PipelineConfig:
    """Configuration for voice chat pipeline"""
    # VAD
    vad_backend: str = "silero"
    vad_threshold: float = 0.5
    
    # STT
    whisper_model: str = "base"  # tiny, base, small, medium, large-v3
    whisper_language: Optional[str] = "en"
    
    # LLM
    llm_model: str = "llama3.1:8b"
    llm_temperature: float = 0.7
    llm_max_tokens: int = 256
    system_prompt: Optional[str] = None
    
    # TTS
    tts_engine: str = "auto"  # piper, edge-tts, espeak, auto
    tts_voice: str = "en-US-AriaNeural"
    
    # Pipeline
    sample_rate: int = 16000
    min_speech_ms: int = 500  # Minimum speech to process
    silence_timeout_ms: int = 1000  # Silence before processing


class VoiceChatPipeline:
    """
    Complete voice-to-voice chat pipeline
    
    Flow: Audio → VAD → STT → LLM → TTS → Audio
    """
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        
        # Components
        self._vad_engine: Optional[VADEngine] = None
        self._stt_engine: Optional[STTEngine] = None
        self._llm_engine: Optional[LLMEngine] = None
        self._tts_engine: Optional[TTSEngine] = None
        
        # State
        self._conversation = Conversation()
        self._is_initialized = False
        self._is_processing = False
        
        # Callbacks
        self._on_user_speech: Optional[Callable] = None
        self._on_transcription: Optional[Callable] = None
        self._on_llm_response: Optional[Callable] = None
        self._on_tts_audio: Optional[Callable] = None
        self._on_turn_complete: Optional[Callable] = None
    
    def initialize(self):
        """Initialize all pipeline components"""
        if self._is_initialized:
            return
        
        logger.info("Initializing Voice Chat Pipeline")
        
        # 1. Initialize VAD
        vad_settings = Settings()
        vad_settings.vad.backend = VADBackend(self.config.vad_backend)
        vad_settings.vad.speech_threshold = self.config.vad_threshold
        vad_settings.audio.chunk_duration_ms = 32  # Silero compatible
        
        self._vad_engine = VADEngine(vad_settings.audio, vad_settings.vad)
        self._vad_engine.initialize()
        
        # 2. Initialize STT
        self._stt_engine = STTEngine(
            model_size=self.config.whisper_model,
            language=self.config.whisper_language
        )
        self._stt_engine.initialize()
        
        # 3. Initialize LLM
        self._llm_engine = LLMEngine(
            model=self.config.llm_model,
            system_prompt=self.config.system_prompt,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        self._llm_engine.initialize()
        
        # 4. Initialize TTS
        self._tts_engine = TTSEngine(
            engine=self.config.tts_engine,
            edge_voice=self.config.tts_voice
        )
        self._tts_engine.initialize()
        
        self._is_initialized = True
        logger.info("Voice Chat Pipeline ready!")
    
    def process_speech(self, audio: np.ndarray, sample_rate: int = 16000) -> Optional[TTSResult]:
        """
        Process a speech segment through the full pipeline
        
        Audio → STT → LLM → TTS
        
        Args:
            audio: Speech audio (float32)
            sample_rate: Sample rate
            
        Returns:
            TTSResult with response audio, or None if failed
        """
        if not self._is_initialized:
            self.initialize()
        
        if self._is_processing:
            logger.warning("Pipeline busy, skipping")
            return None
        
        self._is_processing = True
        total_start = time.time()
        
        try:
            audio_duration_ms = len(audio) / sample_rate * 1000
            
            # Skip if too short
            if audio_duration_ms < self.config.min_speech_ms:
                logger.debug("Speech too short, skipping", duration_ms=audio_duration_ms)
                return None
            
            # === Step 1: Speech to Text ===
            logger.info("🎤 Processing speech", duration_ms=f"{audio_duration_ms:.0f}")
            
            stt_start = time.time()
            transcription = self._stt_engine.transcribe(audio, sample_rate)
            stt_time = (time.time() - stt_start) * 1000
            
            if not transcription.text.strip():
                logger.info("Empty transcription, skipping")
                return None
            
            logger.info(f"📝 User: \"{transcription.text}\"")
            
            if self._on_transcription:
                self._on_transcription(transcription)
            
            # === Step 2: LLM Response ===
            llm_start = time.time()
            llm_response = self._llm_engine.chat(transcription.text)
            llm_time = (time.time() - llm_start) * 1000
            
            logger.info(f"🤖 Assistant: \"{llm_response.text}\"")
            
            if self._on_llm_response:
                self._on_llm_response(llm_response)
            
            # === Step 3: Text to Speech ===
            tts_start = time.time()
            tts_result = self._tts_engine.synthesize(llm_response.text)
            tts_time = (time.time() - tts_start) * 1000
            
            total_time = (time.time() - total_start) * 1000
            
            # Record turn
            turn = Turn(
                user_text=transcription.text,
                assistant_text=llm_response.text,
                user_audio_duration_ms=audio_duration_ms,
                assistant_audio_duration_ms=tts_result.duration_ms,
                stt_time_ms=stt_time,
                llm_time_ms=llm_time,
                tts_time_ms=tts_time,
                total_time_ms=total_time
            )
            self._conversation.add_turn(turn)
            
            logger.info(
                "Turn complete",
                latency=turn.latency_summary
            )
            
            if self._on_turn_complete:
                self._on_turn_complete(turn)
            
            if self._on_tts_audio:
                self._on_tts_audio(tts_result)
            
            return tts_result
            
        except Exception as e:
            logger.error("Pipeline error", error=str(e))
            import traceback
            traceback.print_exc()
            return None
            
        finally:
            self._is_processing = False
    
    # Callback registrations
    def on_transcription(self, callback: Callable):
        self._on_transcription = callback
    
    def on_llm_response(self, callback: Callable):
        self._on_llm_response = callback
    
    def on_tts_audio(self, callback: Callable):
        self._on_tts_audio = callback
    
    def on_turn_complete(self, callback: Callable):
        self._on_turn_complete = callback
    
    @property
    def conversation(self) -> Conversation:
        return self._conversation
    
    def clear_history(self):
        """Clear conversation history"""
        self._llm_engine.clear_history()
        self._conversation = Conversation()
    
    def cleanup(self):
        """Cleanup all resources"""
        if self._vad_engine:
            self._vad_engine.cleanup()
        if self._stt_engine:
            self._stt_engine.cleanup()
        if self._llm_engine:
            self._llm_engine.cleanup()
        if self._tts_engine:
            self._tts_engine.cleanup()
        
        self._is_initialized = False
        logger.info("Pipeline cleaned up")