# src/vad_backends/silero_vad.py
"""
Silero VAD implementation - High accuracy neural network based VAD
FIXED: Correct chunk size handling
"""
import torch
import numpy as np
from typing import List, Optional
import structlog
from .base_vad import BaseVAD, VADResult

logger = structlog.get_logger(__name__)


class SileroVAD(BaseVAD):
    """
    Silero VAD - State-of-the-art voice activity detection
    
    IMPORTANT: Silero VAD requires specific chunk sizes:
    - 512 samples (32ms) for 16kHz
    - 256 samples (32ms) for 8kHz
    
    Features:
    - High accuracy
    - Works well with various audio conditions
    - Supports 8kHz and 16kHz
    - Low latency
    """
    
    SUPPORTED_SAMPLE_RATES = [8000, 16000]
    
    # Silero requires exactly these sizes
    CHUNK_SIZES = {
        8000: 256,   # 32ms at 8kHz
        16000: 512,  # 32ms at 16kHz
    }
    
    def __init__(
        self,
        sample_rate: int = 16000,
        threshold: float = 0.5,
        use_onnx: bool = False,
        force_cpu: bool = False
    ):
        super().__init__(sample_rate)
        
        if sample_rate not in self.SUPPORTED_SAMPLE_RATES:
            raise ValueError(f"Sample rate must be one of {self.SUPPORTED_SAMPLE_RATES}")
        
        self.threshold = threshold
        self.use_onnx = use_onnx
        self.force_cpu = force_cpu
        
        # Required chunk size for this sample rate
        self.chunk_size = self.CHUNK_SIZES[sample_rate]
        
        self._model = None
        self._h = None  # Hidden state
        self._c = None  # Cell state
        self._device = None
        
    def initialize(self):
        """Load Silero VAD model"""
        if self._is_initialized:
            return
        
        try:
            # Determine device
            if self.force_cpu:
                self._device = torch.device('cpu')
            else:
                self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Load model
            if self.use_onnx:
                self._model = self._load_onnx_model()
            else:
                self._model, _ = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False,
                    trust_repo=True
                )
                self._model = self._model.to(self._device)
                self._model.eval()
            
            self._is_initialized = True
            self.reset()
            
            logger.info(
                "Silero VAD initialized",
                device=str(self._device),
                sample_rate=self.sample_rate,
                chunk_size=self.chunk_size,
                use_onnx=self.use_onnx
            )
            
        except Exception as e:
            logger.error("Failed to initialize Silero VAD", error=str(e))
            raise
    
    def _load_onnx_model(self):
        """Load ONNX version of the model for faster inference"""
        import onnxruntime as ort
        
        # Download and cache model
        model_path = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=True
        )
        
        providers = ['CPUExecutionProvider']
        if not self.force_cpu:
            providers.insert(0, 'CUDAExecutionProvider')
        
        return ort.InferenceSession(str(model_path), providers=providers)
    
    def process_frame(self, audio_frame: np.ndarray, timestamp_ms: float = 0.0) -> VADResult:
        """
        Process audio frame through Silero VAD
        
        Args:
            audio_frame: Audio samples. Will be chunked to required size (512 for 16kHz)
            timestamp_ms: Timestamp of this frame
            
        Returns:
            VADResult with speech detection info
        """
        if not self._is_initialized:
            self.initialize()
        
        # Ensure correct shape
        if len(audio_frame.shape) > 1:
            audio_frame = audio_frame.flatten()
        
        # Handle chunk size - Silero needs exactly 512 samples for 16kHz
        if len(audio_frame) < self.chunk_size:
            # Pad with zeros if too short
            padded = np.zeros(self.chunk_size, dtype=np.float32)
            padded[:len(audio_frame)] = audio_frame
            audio_frame = padded
        elif len(audio_frame) > self.chunk_size:
            # Process in chunks and return average/max probability
            return self._process_long_audio(audio_frame, timestamp_ms)
        
        duration_ms = len(audio_frame) / self.sample_rate * 1000
        
        if self.use_onnx:
            probability = self._process_onnx(audio_frame)
        else:
            probability = self._process_pytorch(audio_frame)
        
        is_speech = probability >= self.threshold
        
        return VADResult(
            is_speech=is_speech,
            confidence=probability if is_speech else 1.0 - probability,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            raw_probability=probability
        )
    
    def _process_long_audio(self, audio_frame: np.ndarray, timestamp_ms: float) -> VADResult:
        """Process audio longer than chunk_size by chunking and averaging"""
        probabilities = []
        
        # Process in chunks
        for i in range(0, len(audio_frame), self.chunk_size):
            chunk = audio_frame[i:i + self.chunk_size]
            
            if len(chunk) < self.chunk_size:
                # Pad last chunk if needed
                padded = np.zeros(self.chunk_size, dtype=np.float32)
                padded[:len(chunk)] = chunk
                chunk = padded
            
            if self.use_onnx:
                prob = self._process_onnx(chunk)
            else:
                prob = self._process_pytorch(chunk)
            
            probabilities.append(prob)
        
        # Use max probability (if any chunk has speech, consider it speech)
        avg_probability = float(np.mean(probabilities))
        max_probability = float(np.max(probabilities))
        
        # Use max for detection, avg for confidence
        is_speech = max_probability >= self.threshold
        
        duration_ms = len(audio_frame) / self.sample_rate * 1000
        
        return VADResult(
            is_speech=is_speech,
            confidence=max_probability if is_speech else 1.0 - avg_probability,
            timestamp_ms=timestamp_ms,
            duration_ms=duration_ms,
            raw_probability=max_probability
        )
    
    def _process_pytorch(self, audio_frame: np.ndarray) -> float:
        """Process frame using PyTorch model"""
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio_frame).float().to(self._device)
        
        with torch.no_grad():
            # Call model with sample rate
            out = self._model(audio_tensor, self.sample_rate)
        
        return float(out.item())
    
    def _process_onnx(self, audio_frame: np.ndarray) -> float:
        """Process frame using ONNX model"""
        # Prepare inputs
        audio_input = audio_frame.reshape(1, -1).astype(np.float32)
        sr_input = np.array([self.sample_rate], dtype=np.int64)
        
        if self._h is None:
            self._h = np.zeros((2, 1, 64), dtype=np.float32)
            self._c = np.zeros((2, 1, 64), dtype=np.float32)
        
        # Run inference
        ort_inputs = {
            'input': audio_input,
            'sr': sr_input,
            'h': self._h,
            'c': self._c
        }
        
        out, self._h, self._c = self._model.run(None, ort_inputs)
        
        return float(out[0][0])
    
    def reset(self):
        """Reset hidden states"""
        self._h = None
        self._c = None
        # Reset model state if using PyTorch
        if self._model is not None and hasattr(self._model, 'reset_states'):
            self._model.reset_states()
    
    @property
    def supported_frame_sizes_ms(self) -> List[int]:
        """
        Silero VAD works best with 32ms chunks but can handle longer audio
        by processing in chunks internally
        """
        return [32, 64, 96]  # Multiples of 32ms work best
    
    @property
    def required_chunk_size(self) -> int:
        """Get the required chunk size in samples"""
        return self.chunk_size
    
    def cleanup(self):
        """Cleanup resources"""
        self._model = None
        self._h = None
        self._c = None
        self._is_initialized = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()