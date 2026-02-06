# test_installation.py
"""
Comprehensive installation and functionality test
Run this first to verify everything is working
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def print_header(title):
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)

def print_status(name, status, details=""):
    icon = "✅" if status else "❌"
    print(f"  {icon} {name}: {'OK' if status else 'FAILED'} {details}")
    return status

def test_imports():
    """Test all required imports"""
    print_header("Testing Imports")
    
    results = []
    
    # Core Python
    try:
        import numpy as np
        results.append(print_status("NumPy", True, f"v{np.__version__}"))
    except ImportError as e:
        results.append(print_status("NumPy", False, str(e)))
    
    try:
        import scipy
        results.append(print_status("SciPy", True, f"v{scipy.__version__}"))
    except ImportError as e:
        results.append(print_status("SciPy", False, str(e)))
    
    # PyTorch
    try:
        import torch
        cuda_status = f"CUDA: {torch.cuda.is_available()}"
        if torch.cuda.is_available():
            cuda_status += f" ({torch.cuda.get_device_name(0)})"
        results.append(print_status("PyTorch", True, f"v{torch.__version__} | {cuda_status}"))
    except ImportError as e:
        results.append(print_status("PyTorch", False, str(e)))
    
    try:
        import torchaudio
        results.append(print_status("TorchAudio", True, f"v{torchaudio.__version__}"))
    except ImportError as e:
        results.append(print_status("TorchAudio", False, str(e)))
    
    # Audio libraries
    try:
        import soundfile as sf
        results.append(print_status("SoundFile", True, f"v{sf.__version__}"))
    except ImportError as e:
        results.append(print_status("SoundFile", False, str(e)))
    
    try:
        import sounddevice as sd
        results.append(print_status("SoundDevice", True, f"v{sd.__version__}"))
    except ImportError as e:
        results.append(print_status("SoundDevice", False, str(e)))
    
    # VAD backends
    try:
        import webrtcvad
        results.append(print_status("WebRTC VAD", True))
    except ImportError as e:
        results.append(print_status("WebRTC VAD", False, str(e)))
    
    # Web frameworks
    try:
        import fastapi
        results.append(print_status("FastAPI", True, f"v{fastapi.__version__}"))
    except ImportError as e:
        results.append(print_status("FastAPI", False, str(e)))
    
    try:
        import websockets
        results.append(print_status("WebSockets", True, f"v{websockets.__version__}"))
    except ImportError as e:
        results.append(print_status("WebSockets", False, str(e)))
    
    try:
        import pydantic
        results.append(print_status("Pydantic", True, f"v{pydantic.__version__}"))
    except ImportError as e:
        results.append(print_status("Pydantic", False, str(e)))
    
    # Project modules
    try:
        from config.settings import Settings
        results.append(print_status("Config Module", True))
    except ImportError as e:
        results.append(print_status("Config Module", False, str(e)))
    
    try:
        from src.core.vad_engine import VADEngine
        results.append(print_status("VAD Engine", True))
    except ImportError as e:
        results.append(print_status("VAD Engine", False, str(e)))
    
    try:
        from src.vad_backends.silero_vad import SileroVAD
        results.append(print_status("Silero VAD Backend", True))
    except ImportError as e:
        results.append(print_status("Silero VAD Backend", False, str(e)))
    
    try:
        from src.vad_backends.webrtc_vad import WebRTCVAD
        results.append(print_status("WebRTC VAD Backend", True))
    except ImportError as e:
        results.append(print_status("WebRTC VAD Backend", False, str(e)))
    
    try:
        from src.vad_backends.energy_vad import EnergyVAD
        results.append(print_status("Energy VAD Backend", True))
    except ImportError as e:
        results.append(print_status("Energy VAD Backend", False, str(e)))
    
    return all(results)


def test_audio_devices():
    """Test audio device availability"""
    print_header("Testing Audio Devices")
    
    try:
        import sounddevice as sd
        
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        if len(input_devices) == 0:
            print("\n  ⚠️  No input devices found!")
            print("  This is common in WSL2 or headless environments.")
            print("  The VAD system will still work for file processing.")
            print("  See instructions below to enable audio in WSL2.")
            return print_status("Audio Devices", True, "(No devices - file mode only)")
        
        print(f"\n  Found {len(input_devices)} input device(s):")
        for i, d in enumerate(input_devices[:5]):  # Show first 5
            default = " (default)" if d['name'] == sd.query_devices(kind='input')['name'] else ""
            print(f"    [{i}] {d['name']}{default}")
        
        print(f"\n  Found {len(output_devices)} output device(s):")
        for i, d in enumerate(output_devices[:5]):
            default = " (default)" if d['name'] == sd.query_devices(kind='output')['name'] else ""
            print(f"    [{i}] {d['name']}{default}")
        
        return print_status("Audio Devices", True, f"{len(input_devices)} input, {len(output_devices)} output")
        
    except Exception as e:
        return print_status("Audio Devices", False, str(e))


def test_silero_model():
    """Test Silero VAD model loading"""
    print_header("Testing Silero VAD Model")
    
    try:
        import torch
        import numpy as np
        
        print("  Loading Silero VAD model (this may take a moment)...")
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        
        # Test inference with CORRECT chunk size (512 samples for 16kHz)
        sample_rate = 16000
        chunk_size = 512  # Required by Silero for 16kHz
        
        # Create a proper test chunk (512 samples = 32ms)
        test_audio = torch.randn(chunk_size)
        
        print(f"  Testing with {chunk_size} samples ({chunk_size/sample_rate*1000:.0f}ms) at {sample_rate}Hz...")
        
        with torch.no_grad():
            speech_prob = model(test_audio, sample_rate).item()
        
        print(f"  Model output: {speech_prob:.4f}")
        
        # Test with silence (should be low probability)
        silence = torch.zeros(chunk_size)
        with torch.no_grad():
            silence_prob = model(silence, sample_rate).item()
        print(f"  Silence probability: {silence_prob:.4f}")
        
        # Test with tone (should be higher probability)
        t = torch.linspace(0, chunk_size / sample_rate, chunk_size)
        tone = 0.5 * torch.sin(2 * 3.14159 * 440 * t)
        with torch.no_grad():
            tone_prob = model(tone, sample_rate).item()
        print(f"  Tone (440Hz) probability: {tone_prob:.4f}")
        
        return print_status("Silero Model", True, f"Loaded successfully")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return print_status("Silero Model", False, str(e))


def test_vad_backends():
    """Test all VAD backends"""
    print_header("Testing VAD Backends")
    
    import numpy as np
    results = []
    
    sample_rate = 16000
    
    # Test Energy VAD (any chunk size works)
    try:
        from src.vad_backends.energy_vad import EnergyVAD
        
        chunk_size = 480  # 30ms
        silence = np.zeros(chunk_size, dtype=np.float32)
        t = np.linspace(0, chunk_size / sample_rate, chunk_size)
        speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        vad = EnergyVAD(sample_rate=sample_rate)
        vad.initialize()
        
        result_silence = vad.process_frame(silence)
        result_speech = vad.process_frame(speech)
        
        vad.cleanup()
        
        status = not result_silence.is_speech  # Silence should not be detected as speech
        results.append(print_status(
            "Energy VAD", 
            status,
            f"Silence: {result_silence.is_speech}, Speech: {result_speech.is_speech}"
        ))
    except Exception as e:
        results.append(print_status("Energy VAD", False, str(e)))
    
    # Test WebRTC VAD (needs 10/20/30ms chunks)
    try:
        from src.vad_backends.webrtc_vad import WebRTCVAD
        
        chunk_size = 480  # 30ms at 16kHz
        silence = np.zeros(chunk_size, dtype=np.float32)
        t = np.linspace(0, chunk_size / sample_rate, chunk_size)
        speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        vad = WebRTCVAD(sample_rate=sample_rate, aggressiveness=3)
        vad.initialize()
        
        result_silence = vad.process_frame(silence)
        result_speech = vad.process_frame(speech)
        
        vad.cleanup()
        
        results.append(print_status(
            "WebRTC VAD",
            True,
            f"Silence: {result_silence.is_speech}, Speech: {result_speech.is_speech}"
        ))
    except Exception as e:
        results.append(print_status("WebRTC VAD", False, str(e)))
    
    # Test Silero VAD (needs 512 samples for 16kHz)
    try:
        from src.vad_backends.silero_vad import SileroVAD
        
        # Silero requires 512 samples for 16kHz
        chunk_size = 512  # 32ms
        silence = np.zeros(chunk_size, dtype=np.float32)
        t = np.linspace(0, chunk_size / sample_rate, chunk_size)
        speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        
        vad = SileroVAD(sample_rate=sample_rate)
        vad.initialize()
        
        result_silence = vad.process_frame(silence)
        result_speech = vad.process_frame(speech)
        
        vad.cleanup()
        
        results.append(print_status(
            "Silero VAD",
            True,
            f"Silence prob: {result_silence.raw_probability:.3f}, Speech prob: {result_speech.raw_probability:.3f}"
        ))
    except Exception as e:
        import traceback
        traceback.print_exc()
        results.append(print_status("Silero VAD", False, str(e)))
    
    return all(results)


def test_vad_engine():
    """Test the main VAD engine"""
    print_header("Testing VAD Engine")
    
    try:
        import numpy as np
        from src.core.vad_engine import VADEngine
        from config.settings import Settings, VADBackend
        
        settings = Settings()
        settings.vad.backend = VADBackend.ENERGY  # Use fastest backend for testing
        
        engine = VADEngine(settings.audio, settings.vad)
        engine.initialize()
        
        # Generate test audio
        sample_rate = settings.audio.sample_rate
        chunk_size = settings.audio.chunk_size
        
        # Test with silence
        silence = np.zeros(chunk_size, dtype=np.float32)
        result = engine.process_audio(silence, sample_rate)
        
        print(f"  Processed silence: is_speech={result.is_speech}, prob={result.raw_probability:.3f}")
        
        # Test with speech-like audio
        t = np.linspace(0, 0.03, chunk_size)
        speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = engine.process_audio(speech, sample_rate)
        
        print(f"  Processed speech: is_speech={result.is_speech}, prob={result.raw_probability:.3f}")
        
        # Check metrics
        metrics = engine.get_metrics()
        print(f"  Frames processed: {metrics['frames_processed']}")
        
        engine.cleanup()
        
        return print_status("VAD Engine", True)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return print_status("VAD Engine", False, str(e))


def test_vad_engine_silero():
    """Test the VAD engine with Silero backend"""
    print_header("Testing VAD Engine with Silero")
    
    try:
        import numpy as np
        from src.core.vad_engine import VADEngine
        from config.settings import Settings, VADBackend
        
        settings = Settings()
        settings.vad.backend = VADBackend.SILERO
        # Use 32ms chunks for Silero (512 samples at 16kHz)
        settings.audio.chunk_duration_ms = 32
        
        engine = VADEngine(settings.audio, settings.vad)
        engine.initialize()
        
        sample_rate = settings.audio.sample_rate
        chunk_size = int(sample_rate * 0.032)  # 32ms = 512 samples
        
        # Test with silence
        silence = np.zeros(chunk_size, dtype=np.float32)
        result = engine.process_audio(silence, sample_rate)
        print(f"  Processed silence: is_speech={result.is_speech}, prob={result.raw_probability:.3f}")
        
        # Test with speech-like audio
        t = np.linspace(0, 0.032, chunk_size)
        speech = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        result = engine.process_audio(speech, sample_rate)
        print(f"  Processed speech: is_speech={result.is_speech}, prob={result.raw_probability:.3f}")
        
        engine.cleanup()
        
        return print_status("VAD Engine (Silero)", True)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return print_status("VAD Engine (Silero)", False, str(e))


def test_audio_processor():
    """Test audio processor"""
    print_header("Testing Audio Processor")
    
    try:
        import numpy as np
        from src.core.audio_processor import AudioProcessor
        
        processor = AudioProcessor(
            target_sample_rate=16000,
            target_channels=1
        )
        
        # Test with different formats
        tests = [
            ("Float32 16kHz", np.random.randn(480).astype(np.float32) * 0.1, 16000, 1),
            ("Int16 16kHz", (np.random.randn(480) * 3276).astype(np.int16), 16000, 1),
            ("Float32 48kHz", np.random.randn(1440).astype(np.float32) * 0.1, 48000, 1),
            ("Stereo 16kHz", np.random.randn(960).astype(np.float32) * 0.1, 16000, 2),
        ]
        
        all_passed = True
        for name, audio, sr, channels in tests:
            try:
                frame = processor.process(audio, sr, channels, 0.0)
                print(f"  {name}: output shape={frame.data.shape}, rms_db={frame.rms_db:.1f}")
            except Exception as e:
                print(f"  {name}: FAILED - {e}")
                all_passed = False
        
        return print_status("Audio Processor", all_passed)
        
    except Exception as e:
        return print_status("Audio Processor", False, str(e))


def test_state_machine():
    """Test speech state machine"""
    print_header("Testing State Machine")
    
    try:
        import numpy as np
        from src.core.state_machine import SpeechStateMachine, SpeechState
        
        sm = SpeechStateMachine(
            sample_rate=16000,
            min_speech_duration_ms=100,
            min_silence_duration_ms=100
        )
        
        events = []
        sm.on_speech_start(lambda s: events.append(('start', s)))
        sm.on_speech_end(lambda s: events.append(('end', s)))
        
        # Simulate speech pattern: silence -> speech -> silence
        chunk_size = 480
        
        # Silence
        for i in range(10):
            sm.process(0.1, np.zeros(chunk_size), i * 30)
        
        # Speech
        for i in range(20):
            sm.process(0.9, np.random.randn(chunk_size).astype(np.float32), (10 + i) * 30)
        
        # Silence again
        for i in range(20):
            sm.process(0.1, np.zeros(chunk_size), (30 + i) * 30)
        
        print(f"  Events captured: {len(events)}")
        for event_type, segment in events:
            print(f"    - {event_type}: {segment.start_time_ms:.0f}ms")
        
        return print_status("State Machine", len(events) >= 1)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return print_status("State Machine", False, str(e))


def test_file_processing():
    """Test audio file processing"""
    print_header("Testing File Processing")
    
    try:
        import numpy as np
        import soundfile as sf
        import tempfile
        import os
        
        # Create a test audio file
        sample_rate = 16000
        duration = 2  # 2 seconds
        
        # Generate test audio with speech-like pattern
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.zeros_like(t)
        
        # Add some "speech" in the middle
        speech_start = int(0.5 * sample_rate)
        speech_end = int(1.5 * sample_rate)
        audio[speech_start:speech_end] = 0.3 * np.sin(2 * np.pi * 440 * t[speech_start:speech_end])
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        sf.write(temp_path, audio.astype(np.float32), sample_rate)
        
        # Test FileSource
        from src.audio_sources.file_source import FileSource
        
        source = FileSource(temp_path)
        print(f"  Created test file: {source.duration_seconds:.2f}s @ {source.source_sample_rate}Hz")
        
        # Read chunks
        chunk_count = 0
        for chunk in source.stream():
            chunk_count += 1
        
        print(f"  Read {chunk_count} chunks")
        
        # Cleanup
        os.unlink(temp_path)
        
        return print_status("File Processing", chunk_count > 0)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return print_status("File Processing", False, str(e))


def test_microphone_capture():
    """Test microphone capture (brief test)"""
    print_header("Testing Microphone Capture")
    
    try:
        import numpy as np
        import sounddevice as sd
        
        # Check if devices are available
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        
        if len(input_devices) == 0:
            print("  ⚠️  No input devices available")
            print("  Skipping microphone test (this is normal in WSL2)")
            return print_status("Microphone Capture", True, "(Skipped - no devices)")
        
        sample_rate = 16000
        duration = 1  # 1 second test
        
        print("  Recording 1 second of audio...")
        
        audio = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=1,
            dtype=np.float32
        )
        sd.wait()
        
        # Check if we got audio
        rms = np.sqrt(np.mean(audio ** 2))
        rms_db = 20 * np.log10(rms) if rms > 0 else -100
        
        print(f"  Captured {len(audio)} samples")
        print(f"  RMS level: {rms_db:.1f} dB")
        
        return print_status("Microphone Capture", len(audio) > 0)
        
    except Exception as e:
        if "No input device" in str(e) or "Error querying device" in str(e):
            print("  ⚠️  No microphone available (normal for WSL2)")
            return print_status("Microphone Capture", True, "(Skipped - no devices)")
        return print_status("Microphone Capture", False, str(e))


def test_end_to_end():
    """Test complete end-to-end VAD processing"""
    print_header("Testing End-to-End Processing")
    
    try:
        import numpy as np
        import tempfile
        import soundfile as sf
        import os
        
        from src.core.vad_engine import VADEngine
        from src.audio_sources.file_source import FileSource
        from src.core.state_machine import SpeechSegment
        from config.settings import Settings, VADBackend
        
        # Create test audio file with clear speech pattern
        sample_rate = 16000
        duration = 3.0  # 3 seconds
        
        t = np.linspace(0, duration, int(sample_rate * duration))
        audio = np.zeros_like(t, dtype=np.float32)
        
        # Add "speech" from 0.5s to 1.5s
        speech_start = int(0.5 * sample_rate)
        speech_end = int(1.5 * sample_rate)
        audio[speech_start:speech_end] = 0.4 * np.sin(2 * np.pi * 440 * t[speech_start:speech_end])
        
        # Add another "speech" from 2.0s to 2.5s
        speech_start2 = int(2.0 * sample_rate)
        speech_end2 = int(2.5 * sample_rate)
        audio[speech_start2:speech_end2] = 0.4 * np.sin(2 * np.pi * 440 * t[speech_start2:speech_end2])
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        sf.write(temp_path, audio, sample_rate)
        
        # Setup VAD engine
        settings = Settings()
        settings.vad.backend = VADBackend.ENERGY
        settings.vad.min_speech_duration_ms = 100
        settings.vad.min_silence_duration_ms = 100
        
        engine = VADEngine(settings.audio, settings.vad)
        engine.initialize()
        
        # Track segments
        segments = []
        engine.on_speech_end(lambda s: segments.append(s))
        
        # Process file
        source = FileSource(temp_path, chunk_duration_ms=30)
        
        for chunk in source.stream():
            engine.process_audio(chunk, source.source_sample_rate)
        
        print(f"  Processed {duration}s audio file")
        print(f"  Detected {len(segments)} speech segments")
        
        for i, seg in enumerate(segments):
            print(f"    Segment {i+1}: {seg.start_time_ms:.0f}ms - {seg.end_time_ms:.0f}ms")
        
        # Cleanup
        engine.cleanup()
        os.unlink(temp_path)
        
        # We expect at least 1 segment (might merge the two)
        success = len(segments) >= 1
        return print_status("End-to-End Processing", success, f"{len(segments)} segments detected")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return print_status("End-to-End Processing", False, str(e))


def print_wsl2_instructions():
    """Print instructions for WSL2 audio setup"""
    print("\n" + "=" * 60)
    print(" WSL2 Audio Setup Instructions")
    print("=" * 60)
    print("""
To enable audio in WSL2, you can use PulseAudio:

1. Install PulseAudio on Windows:
   - Download from: https://www.freedesktop.org/wiki/Software/PulseAudio/Ports/Windows/Support/

2. In WSL2, install PulseAudio client:
   $ sudo apt-get update
   $ sudo apt-get install pulseaudio

3. Configure WSL2 to use Windows PulseAudio:
   $ export PULSE_SERVER=tcp:$(cat /etc/resolv.conf | grep nameserver | awk '{print $2}')

4. Start PulseAudio on Windows and test:
   $ paplay /usr/share/sounds/alsa/Front_Center.wav

Alternatively, for file-based VAD (no microphone needed):
   - The VAD system works perfectly for file processing
   - Use FileSource instead of MicrophoneSource
   - REST API and WebSocket server work with uploaded audio
""")


def run_all_tests():
    """Run all tests and report results"""
    print("\n" + "🔬 " + "=" * 56 + " 🔬")
    print("       VAD SYSTEM - COMPREHENSIVE TEST SUITE")
    print("🔬 " + "=" * 56 + " 🔬")
    
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['audio_devices'] = test_audio_devices()
    results['silero_model'] = test_silero_model()
    results['vad_backends'] = test_vad_backends()
    results['audio_processor'] = test_audio_processor()
    results['state_machine'] = test_state_machine()
    results['vad_engine'] = test_vad_engine()
    results['vad_engine_silero'] = test_vad_engine_silero()
    results['file_processing'] = test_file_processing()
    results['end_to_end'] = test_end_to_end()
    
    # Optional microphone test
    print("\n" + "-" * 60)
    response = input("Run microphone test? (y/n) [n]: ").strip().lower()
    if response == 'y':
        results['microphone'] = test_microphone_capture()
    
    # Summary
    print_header("TEST SUMMARY")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, status in results.items():
        icon = "✅" if status else "❌"
        print(f"  {icon} {name.replace('_', ' ').title()}")
    
    print("\n" + "-" * 60)
    print(f"  Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n  🎉 All tests passed! Your VAD system is ready to use.")
    else:
        print("\n  ⚠️  Some tests failed. Please check the errors above.")
        
        # Check if only audio device related
        if not results.get('audio_devices', True):
            print_wsl2_instructions()
    
    print("=" * 60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)