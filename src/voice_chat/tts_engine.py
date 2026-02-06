"""
TTS Engine - Text-to-Speech with multiple backends
"""
import numpy as np
import subprocess
import tempfile
import asyncio
import wave
import struct
from pathlib import Path
from typing import Optional, Tuple
from dataclasses import dataclass
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TTSResult:
    """Result from TTS synthesis"""
    audio: np.ndarray  # float32 audio
    sample_rate: int
    duration_ms: float
    engine_used: str


class TTSEngine:
    """
    Text-to-Speech engine with multiple backends
    
    Backends:
    - piper: Fast, local neural TTS
    - edge-tts: Microsoft Edge TTS (online)
    - espeak: Basic fallback
    """
    
    PIPER_SAMPLE_RATE = 22050  # Piper default output rate
    
    def __init__(
        self,
        engine: str = "auto",
        piper_model: Optional[str] = None,
        edge_voice: str = "en-US-AriaNeural",
        espeak_voice: str = "en"
    ):
        self.preferred_engine = engine
        self.piper_model = piper_model
        self.edge_voice = edge_voice
        self.espeak_voice = espeak_voice
        
        self._available_engines = []
        self._active_engine = None
        self._is_initialized = False
    
    def initialize(self):
        """Initialize and detect available TTS engines"""
        if self._is_initialized:
            return
        
        # Check available engines
        self._available_engines = []
        
        # Check Piper
        if self._check_piper():
            self._available_engines.append("piper")
            logger.info("Piper TTS available")
        
        # Check edge-tts
        if self._check_edge_tts():
            self._available_engines.append("edge-tts")
            logger.info("Edge TTS available")
        
        # Check espeak
        if self._check_espeak():
            self._available_engines.append("espeak")
            logger.info("eSpeak available")
        
        if not self._available_engines:
            raise RuntimeError("No TTS engines available! Install piper, edge-tts, or espeak.")
        
        # Select engine
        if self.preferred_engine == "auto":
            self._active_engine = self._available_engines[0]
        elif self.preferred_engine in self._available_engines:
            self._active_engine = self.preferred_engine
        else:
            logger.warning(f"{self.preferred_engine} not available, using {self._available_engines[0]}")
            self._active_engine = self._available_engines[0]
        
        logger.info(f"Using TTS engine: {self._active_engine}")
        self._is_initialized = True
    
    def _check_piper(self) -> bool:
        """Check if Piper is available"""
        try:
            result = subprocess.run(
                ["piper", "--help"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False
    
    def _check_edge_tts(self) -> bool:
        """Check if edge-tts is available"""
        try:
            import edge_tts
            return True
        except ImportError:
            return False
    
    def _check_espeak(self) -> bool:
        """Check if espeak is available"""
        try:
            result = subprocess.run(
                ["espeak", "--version"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Try espeak-ng
            try:
                result = subprocess.run(
                    ["espeak-ng", "--version"],
                    capture_output=True,
                    timeout=5
                )
                return result.returncode == 0
            except (FileNotFoundError, subprocess.TimeoutExpired):
                return False
    
    def synthesize(self, text: str) -> TTSResult:
        """
        Synthesize text to speech
        
        Args:
            text: Text to speak
            
        Returns:
            TTSResult with audio
        """
        if not self._is_initialized:
            self.initialize()
        
        if not text.strip():
            # Return silence for empty text
            return TTSResult(
                audio=np.zeros(1600, dtype=np.float32),
                sample_rate=16000,
                duration_ms=100,
                engine_used="none"
            )
        
        # Try active engine, fall back if needed
        engines_to_try = [self._active_engine] + [
            e for e in self._available_engines if e != self._active_engine
        ]
        
        for engine in engines_to_try:
            try:
                if engine == "piper":
                    audio, sr = self._synthesize_piper(text)
                elif engine == "edge-tts":
                    audio, sr = self._synthesize_edge_tts(text)
                elif engine == "espeak":
                    audio, sr = self._synthesize_espeak(text)
                else:
                    continue
                
                duration_ms = len(audio) / sr * 1000
                
                return TTSResult(
                    audio=audio,
                    sample_rate=sr,
                    duration_ms=duration_ms,
                    engine_used=engine
                )
                
            except Exception as e:
                logger.warning(f"{engine} failed: {e}, trying next...")
                continue
        
        raise RuntimeError("All TTS engines failed")
    
    def _synthesize_piper(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using Piper TTS"""
        
        # Piper outputs raw 16-bit PCM audio to stdout
        cmd = ["piper", "--output_raw"]
        
        if self.piper_model:
            cmd.extend(["--model", self.piper_model])
        
        # Run piper with text input
        process = subprocess.run(
            cmd,
            input=text.encode('utf-8'),
            capture_output=True,
            timeout=30
        )
        
        if process.returncode != 0:
            stderr = process.stderr.decode('utf-8', errors='ignore')
            raise RuntimeError(f"Piper failed: {stderr}")
        
        # Raw output is 16-bit signed PCM at 22050 Hz (Piper default)
        raw_audio = process.stdout
        
        if len(raw_audio) < 2:
            raise RuntimeError("Piper returned empty audio")
        
        # Convert raw bytes to numpy array
        # Piper outputs 16-bit signed little-endian PCM
        audio_int16 = np.frombuffer(raw_audio, dtype=np.int16)
        
        # Convert to float32 normalized [-1, 1]
        audio = audio_int16.astype(np.float32) / 32768.0
        
        return audio, self.PIPER_SAMPLE_RATE
    
    def _synthesize_edge_tts(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using Edge TTS"""
        import edge_tts
        import io
        
        # Run async edge-tts
        async def _generate():
            communicate = edge_tts.Communicate(text, self.edge_voice)
            audio_data = b""
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            return audio_data
        
        # Run in event loop
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        mp3_data = loop.run_until_complete(_generate())
        
        # Convert MP3 to WAV using ffmpeg
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as mp3_file:
            mp3_file.write(mp3_data)
            mp3_path = mp3_file.name
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
            wav_path = wav_file.name
        
        try:
            # Convert MP3 to WAV
            subprocess.run([
                "ffmpeg", "-y", "-i", mp3_path,
                "-ar", "22050", "-ac", "1", "-f", "wav",
                wav_path
            ], capture_output=True, timeout=30, check=True)
            
            # Read WAV file
            import soundfile as sf
            audio, sr = sf.read(wav_path)
            audio = audio.astype(np.float32)
            
            return audio, sr
            
        finally:
            # Cleanup temp files
            import os
            try:
                os.unlink(mp3_path)
                os.unlink(wav_path)
            except:
                pass
    
    def _synthesize_espeak(self, text: str) -> Tuple[np.ndarray, int]:
        """Synthesize using espeak/espeak-ng"""
        
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            wav_path = f.name
        
        try:
            # Try espeak-ng first, then espeak
            for cmd in ["espeak-ng", "espeak"]:
                try:
                    result = subprocess.run([
                        cmd, "-v", self.espeak_voice,
                        "-w", wav_path,
                        text
                    ], capture_output=True, timeout=30)
                    
                    if result.returncode == 0:
                        break
                except FileNotFoundError:
                    continue
            else:
                raise RuntimeError("espeak not found")
            
            # Read WAV file
            import soundfile as sf
            audio, sr = sf.read(wav_path)
            audio = audio.astype(np.float32)
            
            return audio, sr
            
        finally:
            import os
            try:
                os.unlink(wav_path)
            except:
                pass
    
    @property
    def available_engines(self):
        return self._available_engines
    
    @property
    def active_engine(self):
        return self._active_engine
    
    def cleanup(self):
        """Cleanup resources"""
        self._is_initialized = False