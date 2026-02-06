# tests/test_backends.py
"""
Tests for VAD Backends
"""
import pytest
import numpy as np

from src.vad_backends.base_vad import VADResult
from src.vad_backends.energy_vad import EnergyVAD
from src.vad_backends.webrtc_vad import WebRTCVAD


class TestEnergyVAD:
    """Test cases for EnergyVAD"""
    
    @pytest.fixture
    def vad(self):
        vad = EnergyVAD(sample_rate=16000, threshold_db=-35.0)
        vad.initialize()
        yield vad
        vad.cleanup()
    
    def test_initialization(self, vad):
        """Test VAD initialization"""
        assert vad._is_initialized
    
    def test_detect_silence(self, vad):
        """Test silence detection"""
        silence = np.zeros(480, dtype=np.float32)
        
        result = vad.process_frame(silence, 0.0)
        
        assert isinstance(result, VADResult)
        assert not result.is_speech
    
    def test_detect_speech(self, vad):
        """Test speech detection"""
        # Generate loud tone
        t = np.linspace(0, 0.03, 480)
        speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        result = vad.process_frame(speech, 0.0)
        
        assert result.is_speech
    
    def test_reset(self, vad):
        """Test VAD reset"""
        speech = np.random.randn(480).astype(np.float32) * 0.5
        
        # Process some frames
        for _ in range(10):
            vad.process_frame(speech, 0.0)
        
        vad.reset()
        
        # Should work after reset
        result = vad.process_frame(speech, 0.0)
        assert isinstance(result, VADResult)


class TestWebRTCVAD:
    """Test cases for WebRTCVAD"""
    
    @pytest.fixture
    def vad(self):
        vad = WebRTCVAD(sample_rate=16000, aggressiveness=3)
        vad.initialize()
        yield vad
        vad.cleanup()
    
    def test_initialization(self, vad):
        """Test VAD initialization"""
        assert vad._is_initialized
    
    def test_supported_frame_sizes(self, vad):
        """Test supported frame sizes"""
        sizes = vad.supported_frame_sizes_ms
        assert 10 in sizes
        assert 20 in sizes
        assert 30 in sizes
    
    def test_detect_silence(self, vad):
        """Test silence detection"""
        silence = np.zeros(480, dtype=np.float32)  # 30ms at 16kHz
        
        result = vad.process_frame(silence, 0.0)
        
        assert isinstance(result, VADResult)
        assert not result.is_speech
    
    def test_invalid_sample_rate(self):
        """Test invalid sample rate"""
        with pytest.raises(ValueError):
            WebRTCVAD(sample_rate=22050)
    
    def test_invalid_aggressiveness(self):
        """Test invalid aggressiveness"""
        with pytest.raises(ValueError):
            WebRTCVAD(sample_rate=16000, aggressiveness=5)