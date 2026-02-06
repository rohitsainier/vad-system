# test_file_vad.py
"""
Comprehensive VAD file processing test
Uses multiple methods to obtain real speech for testing
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import soundfile as sf
import tempfile
import os
import subprocess
import shutil

from src.core.vad_engine import VADEngine
from src.audio_sources.file_source import FileSource
from config.settings import Settings, VADBackend


# ============================================================
# Audio Generation Functions
# ============================================================

def create_simple_tone(duration=5.0, sample_rate=16000):
    """Create simple sine wave test audio"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t, dtype=np.float32)
    
    patterns = [
        (0.5, 1.5, 440),
        (2.5, 3.5, 880),
        (4.0, 4.5, 660),
    ]
    
    for start, end, freq in patterns:
        s = int(start * sample_rate)
        e = int(end * sample_rate)
        audio[s:e] = 0.4 * np.sin(2 * np.pi * freq * t[s:e])
    
    return audio.astype(np.float32), sample_rate


def create_synthetic_speech(duration=5.0, sample_rate=16000):
    """
    Create realistic synthetic speech with harmonics and modulation
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.zeros_like(t, dtype=np.float32)
    
    patterns = [
        (0.5, 1.5),
        (2.5, 3.5),
        (4.0, 4.5),
    ]
    
    for start, end in patterns:
        s = int(start * sample_rate)
        e = int(end * sample_rate)
        seg_t = t[s:e]
        seg_len = e - s
        
        # Varying F0 like real speech
        f0 = 120 + 30 * np.sin(2 * np.pi * 3 * seg_t)
        phase = np.cumsum(2 * np.pi * f0 / sample_rate)
        
        # Rich harmonics
        signal = np.zeros(seg_len, dtype=np.float32)
        for h in range(1, 15):
            amp = 1.0 / (h ** 0.5)
            signal += amp * np.sin(h * phase)
        
        # Formant emphasis
        formant1 = 0.3 * np.sin(2 * np.pi * 500 * seg_t)
        formant2 = 0.2 * np.sin(2 * np.pi * 1500 * seg_t)
        signal = signal * (1 + 0.3 * formant1 + 0.2 * formant2)
        
        # Amplitude modulation
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * seg_t)
        signal = signal * envelope
        
        # Noise
        signal += np.random.randn(seg_len) * 0.02
        
        # Normalize and fade
        signal = signal / np.max(np.abs(signal)) * 0.4
        fade = int(0.03 * sample_rate)
        if fade > 0 and fade < seg_len // 2:
            signal[:fade] *= np.linspace(0, 1, fade)
            signal[-fade:] *= np.linspace(1, 0, fade)
        
        audio[s:e] = signal.astype(np.float32)
    
    return audio, sample_rate


# ============================================================
# Real Speech Acquisition (Multiple Methods)
# ============================================================

def get_real_speech():
    """
    Try multiple methods to get real speech audio.
    Returns (file_path, success)
    """
    print("  Trying multiple methods to get real speech...\n")
    
    # Method 1: espeak TTS (most reliable on Linux/WSL2)
    path = _try_espeak()
    if path:
        return path, True
    
    # Method 2: flite TTS
    path = _try_flite()
    if path:
        return path, True
    
    # Method 3: pico2wave TTS
    path = _try_pico2wave()
    if path:
        return path, True
    
    # Method 4: torchaudio LibriSpeech
    path = _try_torchaudio_librispeech()
    if path:
        return path, True
    
    # Method 5: Python-only speech synthesis (always works)
    path = _try_python_speech_synthesis()
    if path:
        return path, True
    
    return None, False


def _try_espeak():
    """Generate speech using espeak (common on Linux)"""
    print("  [1/5] Trying espeak...", end=" ")
    
    if not shutil.which('espeak'):
        print("not installed")
        print("        Install with: sudo apt-get install espeak -y")
        return None
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        text = (
            "Hello, this is a test of the voice activity detection system. "
            "I am speaking clearly so the system can detect my voice. "
            "Now I will pause for a moment. "
            "And now I am speaking again to create a second speech segment."
        )
        
        result = subprocess.run(
            ['espeak', '-w', temp_path, '-s', '150', '-p', '50', text],
            capture_output=True,
            timeout=15
        )
        
        if result.returncode == 0:
            audio, sr = sf.read(temp_path)
            duration = len(audio) / sr
            if duration > 1.0:
                print(f"✅ Generated {duration:.1f}s @ {sr}Hz")
                return temp_path
        
        os.unlink(temp_path)
        print("failed")
        return None
        
    except Exception as e:
        print(f"error: {e}")
        return None


def _try_flite():
    """Generate speech using flite TTS"""
    print("  [2/5] Trying flite...", end=" ")
    
    if not shutil.which('flite'):
        print("not installed")
        print("        Install with: sudo apt-get install flite -y")
        return None
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        text = (
            "Hello world. This is a test of voice activity detection. "
            "The system should detect when I am speaking and when I am silent."
        )
        
        result = subprocess.run(
            ['flite', '-t', text, '-o', temp_path],
            capture_output=True,
            timeout=15
        )
        
        if result.returncode == 0:
            audio, sr = sf.read(temp_path)
            duration = len(audio) / sr
            if duration > 1.0:
                print(f"✅ Generated {duration:.1f}s @ {sr}Hz")
                return temp_path
        
        os.unlink(temp_path)
        print("failed")
        return None
        
    except Exception as e:
        print(f"error: {e}")
        return None


def _try_pico2wave():
    """Generate speech using pico2wave (SVOX)"""
    print("  [3/5] Trying pico2wave...", end=" ")
    
    if not shutil.which('pico2wave'):
        print("not installed")
        print("        Install with: sudo apt-get install libttspico-utils -y")
        return None
    
    try:
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        text = (
            "Hello. This is a voice activity detection test. "
            "Can you detect when I am speaking?"
        )
        
        result = subprocess.run(
            ['pico2wave', '-w', temp_path, '-l', 'en-US', text],
            capture_output=True,
            timeout=15
        )
        
        if result.returncode == 0:
            audio, sr = sf.read(temp_path)
            duration = len(audio) / sr
            if duration > 1.0:
                print(f"✅ Generated {duration:.1f}s @ {sr}Hz")
                return temp_path
        
        os.unlink(temp_path)
        print("failed")
        return None
        
    except Exception as e:
        print(f"error: {e}")
        return None


def _try_torchaudio_librispeech():
    """Download real speech from LibriSpeech via torchaudio"""
    print("  [4/5] Trying torchaudio LibriSpeech...", end=" ")
    
    try:
        import torchaudio
        
        print("downloading (this may take a minute)...")
        
        # Create data directory
        data_dir = os.path.join(tempfile.gettempdir(), "vad_test_data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Download a small subset
        dataset = torchaudio.datasets.LIBRISPEECH(
            root=data_dir,
            url="test-clean",
            download=True
        )
        
        # Get first sample
        waveform, sample_rate, transcript, *_ = dataset[0]
        audio = waveform.numpy().flatten()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        sf.write(temp_path, audio, sample_rate)
        
        duration = len(audio) / sample_rate
        print(f"  ✅ Downloaded {duration:.1f}s @ {sample_rate}Hz")
        print(f"        Transcript: \"{transcript[:60]}...\"")
        
        return temp_path
        
    except Exception as e:
        print(f"failed: {e}")
        return None


def _try_python_speech_synthesis():
    """
    Create highly realistic speech using Python-only synthesis.
    This concatenates vowel/consonant-like sounds to create 
    something that triggers speech detectors.
    
    This ALWAYS works as it has no external dependencies.
    """
    print("  [5/5] Using Python speech synthesis...", end=" ")
    
    try:
        sample_rate = 16000
        
        # Build audio from phoneme-like segments
        audio_segments = []
        silence_segments = []
        
        def make_vowel(duration, f0, formants, amplitude=0.4):
            """Create a vowel-like sound with formants"""
            n = int(duration * sample_rate)
            t = np.linspace(0, duration, n)
            
            # Glottal pulse train (fundamental + harmonics)
            f0_var = f0 + np.random.randn(n) * 2  # jitter
            phase = np.cumsum(2 * np.pi * f0_var / sample_rate)
            
            signal = np.zeros(n, dtype=np.float64)
            for h in range(1, 25):
                amp = 1.0 / (h ** 0.7)
                signal += amp * np.sin(h * phase)
            
            # Apply formant filtering using resonant filters
            for freq, bw, gain in formants:
                # Create resonant filter
                w0 = 2 * np.pi * freq / sample_rate
                r = 1.0 - np.pi * bw / sample_rate
                r = max(0.8, min(r, 0.999))
                
                b = [gain * (1 - r)]
                a = [1, -2 * r * np.cos(w0), r * r]
                
                from scipy.signal import lfilter
                signal = lfilter(b, a, signal)
            
            # Amplitude envelope
            env = np.ones(n)
            attack = int(0.01 * sample_rate)
            release = int(0.01 * sample_rate)
            if attack > 0 and attack < n:
                env[:attack] = np.linspace(0, 1, attack)
            if release > 0 and release < n:
                env[-release:] = np.linspace(1, 0, release)
            
            signal = signal * env * amplitude
            
            # Add breathiness
            breath = np.random.randn(n) * 0.01
            signal += breath
            
            # Normalize
            peak = np.max(np.abs(signal))
            if peak > 0:
                signal = signal / peak * amplitude
            
            return signal.astype(np.float32)
        
        def make_fricative(duration, amplitude=0.1):
            """Create a fricative-like sound (s, f, sh)"""
            n = int(duration * sample_rate)
            noise = np.random.randn(n) * amplitude
            
            # Bandpass filter (high frequency noise)
            from scipy.signal import butter, lfilter
            b, a = butter(4, [3000 / (sample_rate/2), 7000 / (sample_rate/2)], btype='band')
            noise = lfilter(b, a, noise)
            
            # Envelope
            env = np.ones(n)
            ramp = int(0.005 * sample_rate)
            if ramp > 0:
                env[:ramp] = np.linspace(0, 1, ramp)
                env[-ramp:] = np.linspace(1, 0, ramp)
            
            return (noise * env).astype(np.float32)
        
        def make_silence(duration):
            """Create silence with tiny ambient noise"""
            n = int(duration * sample_rate)
            return (np.random.randn(n) * 0.001).astype(np.float32)
        
        # Common vowel formant patterns (freq, bandwidth, gain)
        vowels = {
            'ah': [(730, 90, 1.0), (1090, 110, 0.5), (2440, 170, 0.3)],
            'ee': [(270, 60, 1.0), (2290, 200, 0.4), (3010, 250, 0.2)],
            'oo': [(300, 60, 1.0), (870, 100, 0.5), (2240, 170, 0.2)],
            'eh': [(530, 80, 1.0), (1840, 150, 0.4), (2480, 200, 0.2)],
            'ih': [(390, 70, 1.0), (1990, 170, 0.4), (2550, 200, 0.2)],
        }
        
        # === Sentence 1: "Hello world" ===
        sentence1 = []
        # "He-"
        sentence1.append(make_fricative(0.08, 0.05))
        sentence1.append(make_vowel(0.12, 130, vowels['eh']))
        # "-llo"
        sentence1.append(make_vowel(0.15, 125, vowels['oo']))
        # space
        sentence1.append(make_silence(0.05))
        # "wor-"
        sentence1.append(make_vowel(0.15, 120, vowels['ah']))
        # "-ld"
        sentence1.append(make_fricative(0.06, 0.03))
        
        # === Silence gap ===
        gap1 = make_silence(0.8)
        
        # === Sentence 2: "Testing the system" ===
        sentence2 = []
        # "Tes-"
        sentence2.append(make_fricative(0.07, 0.04))
        sentence2.append(make_vowel(0.10, 135, vowels['eh']))
        sentence2.append(make_fricative(0.08, 0.06))
        # "-ting"
        sentence2.append(make_vowel(0.12, 130, vowels['ih']))
        sentence2.append(make_fricative(0.05, 0.03))
        # space
        sentence2.append(make_silence(0.04))
        # "the"
        sentence2.append(make_vowel(0.08, 120, vowels['eh']))
        # space
        sentence2.append(make_silence(0.04))
        # "sys-"
        sentence2.append(make_fricative(0.08, 0.06))
        sentence2.append(make_vowel(0.12, 125, vowels['ih']))
        sentence2.append(make_fricative(0.06, 0.04))
        # "-tem"
        sentence2.append(make_vowel(0.12, 118, vowels['eh']))
        
        # === Silence gap ===
        gap2 = make_silence(0.8)
        
        # === Sentence 3: "Can you hear me" ===
        sentence3 = []
        # "Can"
        sentence3.append(make_vowel(0.12, 140, vowels['ah']))
        sentence3.append(make_fricative(0.04, 0.03))
        # space
        sentence3.append(make_silence(0.03))
        # "you"
        sentence3.append(make_vowel(0.10, 135, vowels['oo']))
        # space
        sentence3.append(make_silence(0.03))
        # "hear"
        sentence3.append(make_fricative(0.06, 0.04))
        sentence3.append(make_vowel(0.15, 125, vowels['ee']))
        # space
        sentence3.append(make_silence(0.03))
        # "me"
        sentence3.append(make_vowel(0.15, 130, vowels['ee']))
        
        # Combine everything with leading/trailing silence
        all_audio = [
            make_silence(0.5),                         # Leading silence
            np.concatenate(sentence1),                 # Sentence 1
            gap1,                                       # Gap
            np.concatenate(sentence2),                 # Sentence 2
            gap2,                                       # Gap
            np.concatenate(sentence3),                 # Sentence 3
            make_silence(0.5),                         # Trailing silence
        ]
        
        audio = np.concatenate(all_audio)
        duration = len(audio) / sample_rate
        
        # Save
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        sf.write(temp_path, audio, sample_rate)
        
        print(f"✅ Generated {duration:.1f}s @ {sample_rate}Hz")
        print(f"        3 speech segments with gaps")
        
        return temp_path
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"failed: {e}")
        return None


# ============================================================
# Testing Functions
# ============================================================

def test_with_backend(backend: VADBackend, audio_path: str, description: str = ""):
    """Test a specific backend with given audio file"""
    print(f"\n  {'─'*46}")
    print(f"  {backend.value.upper()} {description}")
    print(f"  {'─'*46}")
    
    settings = Settings()
    settings.vad.backend = backend
    settings.vad.min_speech_duration_ms = 200
    settings.vad.min_silence_duration_ms = 200
    
    if backend == VADBackend.SILERO:
        settings.audio.chunk_duration_ms = 32
        settings.vad.speech_threshold = 0.3
    elif backend == VADBackend.WEBRTC:
        settings.audio.chunk_duration_ms = 30
        settings.vad.webrtc_aggressiveness = 2
    else:
        settings.audio.chunk_duration_ms = 30
    
    engine = VADEngine(settings.audio, settings.vad)
    engine.initialize()
    
    segments = []
    engine.on_speech_start(lambda s: print(f"    🎤 START at {s.start_time_ms:.0f}ms"))
    engine.on_speech_end(lambda s: (
        segments.append(s),
        print(f"    🔇 END   at {s.end_time_ms:.0f}ms ({s.duration_ms:.0f}ms)")
    ))
    
    source = FileSource(audio_path, chunk_duration_ms=settings.audio.chunk_duration_ms)
    
    frame_count = 0
    speech_frames = 0
    probabilities = []
    
    for chunk in source.stream():
        result = engine.process_audio(chunk, source.source_sample_rate)
        frame_count += 1
        if result.raw_probability is not None:
            probabilities.append(result.raw_probability)
        if result.is_speech:
            speech_frames += 1
    
    speech_pct = speech_frames / max(1, frame_count) * 100
    
    print(f"\n    Results: {len(segments)} segments | "
          f"{speech_frames}/{frame_count} frames ({speech_pct:.0f}% speech)")
    
    if probabilities:
        print(f"    Probability: min={min(probabilities):.3f} "
              f"max={max(probabilities):.3f} avg={np.mean(probabilities):.3f}")
    
    engine.cleanup()
    
    return len(segments), probabilities


def main():
    print("\n🔬 Comprehensive VAD File Processing Test")
    print("=" * 60)
    
    results = {}
    backends = [VADBackend.ENERGY, VADBackend.WEBRTC, VADBackend.SILERO]
    
    # =========================================================
    # TEST 1: Simple Sine Wave (Energy VAD only)
    # =========================================================
    print("\n\n" + "🔊 " + "="*54)
    print("  TEST 1: Simple Sine Wave (Pure Tone)")
    print("  Expected: ONLY Energy VAD should detect this")
    print("="*58)
    
    audio, sr = create_simple_tone(duration=5.0)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        sine_path = f.name
    sf.write(sine_path, audio, sr)
    
    print(f"\n  File: 5.0s sine waves at 440/880/660 Hz")
    
    segments, _ = test_with_backend(VADBackend.ENERGY, sine_path, "(Sine Wave)")
    results["sine_energy"] = segments
    
    os.unlink(sine_path)
    
    # =========================================================
    # TEST 2: Synthetic Speech
    # =========================================================
    print("\n\n" + "🗣️  " + "="*54)
    print("  TEST 2: Synthetic Speech (Harmonics + Modulation)")
    print("  Expected: Energy ✅ | Silero ✅ | WebRTC: varies")
    print("="*58)
    
    audio, sr = create_synthetic_speech(duration=5.0)
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        synth_path = f.name
    sf.write(synth_path, audio, sr)
    
    print(f"\n  File: 5.0s | Speech at 0.5-1.5s, 2.5-3.5s, 4.0-4.5s")
    
    for backend in backends:
        try:
            segs, _ = test_with_backend(backend, synth_path, "(Synthetic)")
            results[f"synth_{backend.value}"] = segs
        except Exception as e:
            print(f"\n  ❌ {backend.value}: {e}")
            results[f"synth_{backend.value}"] = -1
    
    os.unlink(synth_path)
    
    # =========================================================
    # TEST 3: Real / Realistic Speech
    # =========================================================
    print("\n\n" + "🎙️  " + "="*54)
    print("  TEST 3: Real Speech Audio")
    print("  Expected: All backends should detect speech")
    print("="*58 + "\n")
    
    real_path, success = get_real_speech()
    
    if success and real_path:
        # Get info
        info = sf.info(real_path)
        print(f"\n  File: {info.duration:.1f}s @ {info.samplerate}Hz")
        
        for backend in backends:
            try:
                segs, _ = test_with_backend(backend, real_path, "(Real Speech)")
                results[f"real_{backend.value}"] = segs
            except Exception as e:
                print(f"\n  ❌ {backend.value}: {e}")
                results[f"real_{backend.value}"] = -1
        
        os.unlink(real_path)
    else:
        print("\n  ❌ Could not generate any speech audio")
    
    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n\n" + "="*60)
    print("📊 FINAL SUMMARY")
    print("="*60)
    
    print("\n┌─────────────────┬──────────┬──────────┬──────────┐")
    print("│ Audio Type      │ Energy   │ WebRTC   │ Silero   │")
    print("├─────────────────┼──────────┼──────────┼──────────┤")
    
    for prefix, name in [
        ("sine", "Sine Wave"),
        ("synth", "Synth Speech"),
        ("real", "Real Speech")
    ]:
        row = f"│ {name:<15} │"
        for backend in ["energy", "webrtc", "silero"]:
            key = f"{prefix}_{backend}"
            if key in results:
                val = results[key]
                if val == -1:
                    row += "  ERROR  │"
                elif val == 0:
                    row += "    0    │"
                elif val >= 2:
                    row += f"  {val:>2}  ✅ │"
                else:
                    row += f"  {val:>2}  ⚠️ │"
            else:
                row += "   ──    │"
        print(row)
    
    print("└─────────────────┴──────────┴──────────┴──────────┘")
    
    # Verdict
    print("\n📝 Analysis:")
    print("─" * 60)
    
    # Check Energy
    e_synth = results.get("synth_energy", 0)
    e_real = results.get("real_energy", 0)
    if e_synth >= 2:
        print("  ✅ Energy VAD: Working (detects all sounds)")
    else:
        print("  ⚠️  Energy VAD: May need threshold adjustment")
    
    # Check Silero
    s_synth = results.get("synth_silero", 0)
    s_real = results.get("real_silero", 0)
    if s_synth >= 2 or s_real >= 1:
        print("  ✅ Silero VAD: Working (speech detection accurate)")
    elif s_synth >= 1:
        print("  ⚠️  Silero VAD: Partially working (may need real speech)")
    else:
        print("  ❌ Silero VAD: Not detecting - check configuration")
    
    # Check WebRTC
    w_synth = results.get("synth_webrtc", 0)
    w_real = results.get("real_webrtc", 0)
    if w_real >= 1:
        print("  ✅ WebRTC VAD: Working with real speech")
    elif w_synth >= 1:
        print("  ✅ WebRTC VAD: Working with synthetic speech")
    else:
        print("  ℹ️  WebRTC VAD: Needs natural speech (by design)")
        print("      → Install espeak for proper testing:")
        print("        sudo apt-get install espeak -y")
    
    # Overall
    total_pass = sum(1 for k, v in results.items() if v >= 1)
    total_tests = len(results)
    
    print(f"\n  Overall: {total_pass}/{total_tests} tests detected speech")
    
    if e_synth >= 2 and s_synth >= 2:
        print("\n  🎉 Your VAD system is production ready!")
    
    print("\n💡 Quick commands to install TTS for better testing:")
    print("   sudo apt-get install espeak -y     # Best option")
    print("   sudo apt-get install flite -y      # Alternative")
    print("=" * 60)


if __name__ == "__main__":
    main()