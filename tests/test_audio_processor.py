# tests/test_audio_processor.py
"""
Tests for Audio Processor
"""
import pytest
import numpy as np

from src.core.audio_processor import AudioProcessor, AudioFrame


class TestAudioProcessor:
    """Test cases for AudioProcessor"""
    
    @pytest.fixture
    def processor(self):
        return AudioProcessor(
            target_sample_rate=16000,
            target_channels=1,
            apply_noise_gate=True,
            apply_highpass=True,
            normalize=True
        )
    
    def test_initialization(self, processor):
        """Test processor initialization"""
        assert processor.target_sample_rate == 16000
        assert processor.target_channels == 1
    
    def test_process_float32(self, processor):
        """Test processing float32 audio"""
        audio = np.random.randn(480).astype(np.float32) * 0.1
        
        frame = processor.process(audio, 16000, 1, 0.0)
        
        assert isinstance(frame, AudioFrame)
        assert frame.sample_rate == 16000
        assert frame.is_processed
    
    def test_process_int16(self, processor):
        """Test processing int16 audio"""
        audio = (np.random.randn(480) * 3276).astype(np.int16)
        
        frame = processor.process(audio, 16000, 1, 0.0)
        
        assert frame.is_processed
        assert frame.data.dtype == np.float32
    
    def test_resample(self, processor):
        """Test resampling"""
        # Audio at 48kHz
        audio = np.random.randn(1440).astype(np.float32) * 0.1  # 30ms at 48kHz
        
        frame = processor.process(audio, 48000, 1, 0.0)
        
        # Should be resampled to ~480 samples (30ms at 16kHz)
        assert len(frame.data) == 480 or abs(len(frame.data) - 480) <= 1
    
    def test_stereo_to_mono(self, processor):
        """Test stereo to mono conversion"""
        # Stereo audio (interleaved)
        audio = np.random.randn(960).astype(np.float32) * 0.1
        
        frame = processor.process(audio, 16000, 2, 0.0)
        
        # Should be converted to 480 mono samples
        assert len(frame.data) == 480
    
    def test_rms_db(self, processor):
        """Test RMS dB calculation"""
        # Known amplitude
        audio = np.ones(480, dtype=np.float32) * 0.1
        
        frame = processor.process(audio, 16000, 1, 0.0)
        
        # RMS should be around -20 dB (0.1 amplitude)
        assert -30 < frame.rms_db < -10
    
    def test_noise_gate(self, processor):
        """Test noise gate"""
        # Very quiet audio (below noise gate threshold)
        audio = np.random.randn(480).astype(np.float32) * 0.0001
        
        frame = processor.process(audio, 16000, 1, 0.0)
        
        # Should be attenuated
        assert np.max(np.abs(frame.data)) < 0.01
    
    def test_reset(self, processor):
        """Test processor reset"""
        audio = np.random.randn(480).astype(np.float32) * 0.1
        
        # Process some audio
        processor.process(audio, 16000, 1, 0.0)
        
        # Reset
        processor.reset()
        
        # Should work after reset
        frame = processor.process(audio, 16000, 1, 0.0)
        assert frame.is_processed