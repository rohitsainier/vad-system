# src/voice_chat/tts_engine.py
"""
Text-to-Speech Engine
Supports Piper TTS (offline) and edge-tts (online fallback)
"""
import numpy as np
import io
import time
import tempfile
import os
import subprocess
import shutil
from typing import Optional
from dataclasses import dataclass
import soundfile as sf
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TTSResult:
    """Result from text-to-speech"""
    audio: np.ndarray
    sample_rate: int
    duration_ms: float
    processing_time_ms: float
    engine: str


class TTSEngine:
    """
    Text-to-Speech Engine
    
    Priority order:
    1. Piper TTS (fast, offline, good quality)
    2. edge-tts (Microsoft, needs internet, best quality)
    3. espeak (offline, robotic but always works)
    """
    
    def __init__(
        self,
        engine: str = "auto",  # "piper", "edge-tts", "espeak", "auto"
        piper_model: Optional[str] = None,
        edge_voice: str = "en-US-AriaNeural",
        espeak_voice: str = "en",
        output_sample_rate: int = 16000
    ):
        self.preferred_engine = engine
        self.piper_model = piper_model
        self.edge_voice = edge_voice
        self.espeak_voice = espeak_voice
        self.output_sample_rate = output_sample_rate
        
        self._active_engine = None
        self._piper = None
        self._is_initialized = False
    
    def initialize(self):
        """Initialize TTS engine"""
        if self._is_initialized:
            return
        
        if self.preferred_engine == "auto":
            self._active_engine = self._detect_best_engine()
        else:
            self._active_engine = self.preferred_engine
        
        # Initialize specific engine
        if self._active_engine == "piper":
            self._init_piper()
        
        self._is_initialized = True
        logger.info("TTS Engine initialized", engine=self._active_engine)
    
    def _detect_best_engine(self) -> str:
        """Detect the best available TTS engine"""
        
        # Check Piper
        try:
            import piper
            if self.piper_model and os.path.exists(self.piper_model):
                logger.info("Piper TTS available")
                return "piper"
        except ImportError:
            pass
        
        # Check piper CLI
        if shutil.which('piper'):
            return "piper"
        
        # Check edge-tts
        try:
            import edge_tts
            logger.info("edge-tts available")
            return "edge-tts"
        except ImportError:
            pass
        
        # Check espeak
        if shutil.which('espeak'):
            logger.info("espeak available (fallback)")
            return "espeak"
        
        raise RuntimeError(
            "No TTS engine available. Install one:\n"
            "  pip install piper-tts\n"
            "  pip install edge-tts\n"
            "  sudo apt-get install espeak"
        )
    
    def _init_piper(self):
        """Initialize Piper TTS"""
        if self.piper_model:
            try:
                from piper import PiperVoice
                self._piper = PiperVoice.load(self.piper_model)
                logger.info("Piper voice loaded", model=self.piper_model)
            except Exception as e:
                logger.warning("Piper model load failed, using CLI", error=str(e))
    
    def synthesize(self, text: str) -> TTSResult:
        """
        Convert text to speech
        
        Args:
            text: Text to speak
            
        Returns:
            TTSResult with audio data
        """
        if not self._is_initialized:
            self.initialize()
        
        start_time = time.time()
        
        # Clean text for TTS
        text = self._clean_text(text)
        
        if not text.strip():
            return TTSResult(
                audio=np.zeros(1600, dtype=np.float32),
                sample_rate=self.output_sample_rate,
                duration_ms=100,
                processing_time_ms=0,
                engine=self._active_engine
            )
        
        # Route to active engine
        if self._active_engine == "piper":
            audio, sr = self._synthesize_piper(text)
        elif self._active_engine == "edge-tts":
            audio, sr = self._synthesize_edge_tts(text)
        elif self._active_engine == "espeak":
            audio, sr = self._synthesize_espeak(text)
        else:
            raise RuntimeError(f"Unknown TTS engine: {self._active_engine}")
        
        # Resample if needed
        if sr != self.output_sample_rate:
            from scipy.signal import resample
            new_length = int(len(audio) * self.output_sample_rate / sr)
            audio = resample(audio, new_length).astype(np.float32)
            sr = self.output_sample_rate
        
        duration_ms = len(audio) / sr * 1000
        processing_time = (time.time() - start_time) * 1000
        
        result = TTSResult(
            audio=audio,
            sample_rate=sr,
            duration_ms=duration_ms,
            processing_time_ms=processing_time,
            engine=self._active_engine
        )
        
        logger.info(
            "TTS complete",
            engine=self._active_engine,
            text_len=len(text),
            duration_ms=duration_ms,
            processing_ms=processing_time
        )
        
        return result
    
    def _synthesize_piper(self, text: str):
        """Synthesize using Piper TTS"""
        if self._piper:
            # Use Python API
            audio_bytes = io.BytesIO()
            self._piper.synthesize(text, audio_bytes)
            audio_bytes.seek(0)
            audio, sr = sf.read(audio_bytes)
            return audio.astype(np.float32), sr
        
        # Use CLI
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            cmd = f'echo "{text}" | piper --output_file {temp_path}'
            if self.piper_model:
                cmd += f' --model {self.piper_model}'
            
            subprocess.run(cmd, shell=True, capture_output=True, timeout=30)
            audio, sr = sf.read(temp_path)
            return audio.astype(np.float32), sr
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _synthesize_edge_tts(self, text: str):
        """Synthesize using edge-tts (Microsoft)"""
        import asyncio
        import edge_tts
        
        async def _generate():
            communicate = edge_tts.Communicate(text, self.edge_voice)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
                temp_path = f.name
            
            try:
                await communicate.save(temp_path)
                audio, sr = sf.read(temp_path)
                return audio.astype(np.float32), sr
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
        
        # Run async in sync context
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _generate())
                    return future.result(timeout=30)
            else:
                return asyncio.run(_generate())
        except RuntimeError:
            return asyncio.run(_generate())
    
    def _synthesize_espeak(self, text: str):
        """Synthesize using espeak"""
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            subprocess.run(
                ['espeak', '-w', temp_path, '-s', '150', '-v', self.espeak_voice, text],
                capture_output=True,
                timeout=15
            )
            audio, sr = sf.read(temp_path)
            return audio.astype(np.float32), sr
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _clean_text(self, text: str) -> str:
        """Clean text for TTS"""
        import re
        
        # Remove markdown
        text = re.sub(r'\*+', '', text)
        text = re.sub(r'#+\s*', '', text)
        text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        text = re.sub(r'`[^`]*`', '', text)
        
        # Remove special characters
        text = re.sub(r'[{}\[\]|\\<>]', '', text)
        
        # Clean whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def cleanup(self):
        """Cleanup resources"""
        self._piper = None
        self._is_initialized = False