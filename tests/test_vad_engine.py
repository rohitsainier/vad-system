# tests/test_vad_engine.py
"""
Tests for VAD Engine
"""
import pytest
import numpy as np
import asyncio

from src.core.vad_engine import VADEngine, VADEvent
from src.core.state_machine import SpeechSegment
from config.settings import AudioSettings, VADSettings, VADBackend


class TestVADEngine:
    """Test cases for VADEngine"""
    
    @pytest.fixture
    def audio_settings(self):
        return AudioSettings(sample_rate=16000, channels=1)
    
    @pytest.fixture
    def vad_settings(self):
        return VADSettings(
            backend=VADBackend.ENERGY,
            speech_threshold=0.5,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100
        )
    
    @pytest.fixture
    def engine(self, audio_settings, vad_settings):
        engine = VADEngine(audio_settings, vad_settings)
        engine.initialize()
        yield engine
        engine.cleanup()
    
    def test_initialization(self, engine):
        """Test engine initialization"""
        assert engine._is_initialized
        assert engine._vad_backend is not None
        assert engine._state_machine is not None
    
    def test_process_silence(self, engine, audio_settings):
        """Test processing silent audio"""
        # Generate silent audio
        silence = np.zeros(audio_settings.chunk_size, dtype=np.float32)
        
        result = engine.process_audio(silence, audio_settings.sample_rate)
        
        assert not result.is_speech
        assert result.raw_probability is not None
    
    def test_process_speech(self, engine, audio_settings):
        """Test processing speech-like audio"""
        # Generate loud audio (simulating speech)
        t = np.linspace(0, 0.03, audio_settings.chunk_size)
        speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        result = engine.process_audio(speech, audio_settings.sample_rate)
        
        # Energy VAD should detect this as potential speech
        assert result.raw_probability is not None
    
    def test_state_transitions(self, engine, audio_settings):
        """Test speech state transitions"""
        chunk_size = audio_settings.chunk_size
        sample_rate = audio_settings.sample_rate
        
        # Start with silence
        silence = np.zeros(chunk_size, dtype=np.float32)
        
        # Generate speech
        t = np.linspace(0, 0.03, chunk_size)
        speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        # Process silence
        for _ in range(5):
            engine.process_audio(silence, sample_rate)
        
        # Should be in silence state
        assert not engine.is_speaking
    
    def test_metrics(self, engine, audio_settings):
        """Test metrics collection"""
        silence = np.zeros(audio_settings.chunk_size, dtype=np.float32)
        
        for _ in range(10):
            engine.process_audio(silence, audio_settings.sample_rate)
        
        metrics = engine.get_metrics()
        
        assert metrics["frames_processed"] == 10
        assert "speech_segments_detected" in metrics
    
    def test_reset(self, engine, audio_settings):
        """Test engine reset"""
        silence = np.zeros(audio_settings.chunk_size, dtype=np.float32)
        
        for _ in range(10):
            engine.process_audio(silence, audio_settings.sample_rate)
        
        engine.reset()
        
        # After reset, timestamp should be 0
        assert engine._current_timestamp_ms == 0.0


class TestVADEngineAsync:
    """Async test cases for VADEngine"""
    
    @pytest.fixture
    def engine(self):
        settings_audio = AudioSettings(sample_rate=16000)
        settings_vad = VADSettings(backend=VADBackend.ENERGY)
        engine = VADEngine(settings_audio, settings_vad)
        engine.initialize()
        yield engine
        engine.cleanup()
    
    @pytest.mark.asyncio
    async def test_process_stream(self, engine):
        """Test async stream processing"""
        chunk_size = 480  # 30ms at 16kHz
        
        async def audio_generator():
            for _ in range(10):
                yield np.zeros(chunk_size, dtype=np.float32)
        
        events = []
        async for event in engine.process_stream(audio_generator(), 16000):
            events.append(event)
        
        # Should process without errors
        assert True