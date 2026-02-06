# src/utils/metrics.py
"""
Metrics collection for VAD system
"""
import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import threading
import structlog

try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

logger = structlog.get_logger(__name__)


@dataclass
class VADMetrics:
    """Container for VAD metrics"""
    frames_processed: int = 0
    speech_segments: int = 0
    total_speech_duration_ms: float = 0.0
    total_silence_duration_ms: float = 0.0
    average_speech_duration_ms: float = 0.0
    average_confidence: float = 0.0
    processing_time_ms: float = 0.0
    
    # Timing
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "speech_segments": self.speech_segments,
            "total_speech_duration_ms": self.total_speech_duration_ms,
            "total_silence_duration_ms": self.total_silence_duration_ms,
            "average_speech_duration_ms": self.average_speech_duration_ms,
            "average_confidence": self.average_confidence,
            "processing_time_ms": self.processing_time_ms,
            "uptime_seconds": time.time() - self.start_time
        }


class MetricsCollector:
    """
    Metrics collector for VAD system
    
    Features:
    - Internal metrics tracking
    - Prometheus export (optional)
    - Thread-safe operations
    """
    
    def __init__(
        self,
        enable_prometheus: bool = True,
        prometheus_port: int = 9090
    ):
        self.enable_prometheus = enable_prometheus and PROMETHEUS_AVAILABLE
        self.prometheus_port = prometheus_port
        
        self._metrics = VADMetrics()
        self._lock = threading.Lock()
        
        # Per-session metrics
        self._session_metrics: Dict[str, VADMetrics] = defaultdict(VADMetrics)
        
        # Confidence tracking
        self._confidence_sum = 0.0
        self._confidence_count = 0
        
        # Prometheus metrics
        if self.enable_prometheus:
            self._setup_prometheus()
    
    def _setup_prometheus(self):
        """Initialize Prometheus metrics"""
        self._prom_frames = Counter(
            'vad_frames_processed_total',
            'Total number of audio frames processed'
        )
        self._prom_speech_segments = Counter(
            'vad_speech_segments_total',
            'Total number of speech segments detected'
        )
        self._prom_speech_duration = Counter(
            'vad_speech_duration_ms_total',
            'Total speech duration in milliseconds'
        )
        self._prom_processing_time = Histogram(
            'vad_processing_time_ms',
            'Processing time per frame in milliseconds',
            buckets=[0.1, 0.5, 1, 2, 5, 10, 20, 50]
        )
        self._prom_confidence = Gauge(
            'vad_current_confidence',
            'Current speech detection confidence'
        )
        self._prom_is_speech = Gauge(
            'vad_is_speech',
            'Current speech state (1 = speaking, 0 = silence)'
        )
    
    def start_prometheus_server(self):
        """Start Prometheus metrics server"""
        if self.enable_prometheus:
            try:
                start_http_server(self.prometheus_port)
                logger.info("Prometheus server started", port=self.prometheus_port)
            except Exception as e:
                logger.error("Failed to start Prometheus server", error=str(e))
    
    def record_frame(
        self,
        processing_time_ms: float,
        is_speech: bool,
        confidence: float,
        session_id: Optional[str] = None
    ):
        """Record frame processing metrics"""
        with self._lock:
            self._metrics.frames_processed += 1
            self._metrics.processing_time_ms += processing_time_ms
            
            self._confidence_sum += confidence
            self._confidence_count += 1
            self._metrics.average_confidence = self._confidence_sum / self._confidence_count
            
            if session_id:
                self._session_metrics[session_id].frames_processed += 1
        
        if self.enable_prometheus:
            self._prom_frames.inc()
            self._prom_processing_time.observe(processing_time_ms)
            self._prom_confidence.set(confidence)
            self._prom_is_speech.set(1 if is_speech else 0)
    
    def record_speech_segment(
        self,
        duration_ms: float,
        confidence: float,
        session_id: Optional[str] = None
    ):
        """Record speech segment metrics"""
        with self._lock:
            self._metrics.speech_segments += 1
            self._metrics.total_speech_duration_ms += duration_ms
            self._metrics.average_speech_duration_ms = (
                self._metrics.total_speech_duration_ms / self._metrics.speech_segments
            )
            
            if session_id:
                session = self._session_metrics[session_id]
                session.speech_segments += 1
                session.total_speech_duration_ms += duration_ms
        
        if self.enable_prometheus:
            self._prom_speech_segments.inc()
            self._prom_speech_duration.inc(duration_ms)
    
    def record_silence(self, duration_ms: float, session_id: Optional[str] = None):
        """Record silence duration"""
        with self._lock:
            self._metrics.total_silence_duration_ms += duration_ms
            
            if session_id:
                self._session_metrics[session_id].total_silence_duration_ms += duration_ms
    
    def get_metrics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get current metrics"""
        with self._lock:
            if session_id:
                return self._session_metrics[session_id].to_dict()
            return self._metrics.to_dict()
    
    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get metrics for all sessions"""
        with self._lock:
            return {
                session_id: metrics.to_dict()
                for session_id, metrics in self._session_metrics.items()
            }
    
    def reset(self, session_id: Optional[str] = None):
        """Reset metrics"""
        with self._lock:
            if session_id:
                self._session_metrics[session_id] = VADMetrics()
            else:
                self._metrics = VADMetrics()
                self._session_metrics.clear()
                self._confidence_sum = 0.0
                self._confidence_count = 0


# Global metrics instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create global metrics collector"""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector