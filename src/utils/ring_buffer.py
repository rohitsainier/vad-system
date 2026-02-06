# src/utils/ring_buffer.py
"""
Thread-safe ring buffer for audio streaming
"""
import numpy as np
import threading
from typing import Optional, Tuple
from collections import deque


class RingBuffer:
    """
    Thread-safe ring buffer for continuous audio streaming
    """
    
    def __init__(self, capacity_samples: int, dtype=np.float32):
        self.capacity = capacity_samples
        self.dtype = dtype
        self._buffer = np.zeros(capacity_samples, dtype=dtype)
        self._write_pos = 0
        self._read_pos = 0
        self._count = 0
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._not_full = threading.Condition(self._lock)
    
    def write(self, data: np.ndarray, block: bool = True, timeout: float = None) -> int:
        """
        Write data to buffer
        
        Args:
            data: Audio samples to write
            block: Whether to block if buffer is full
            timeout: Maximum time to wait
            
        Returns:
            Number of samples written
        """
        with self._not_full:
            samples_to_write = len(data)
            
            if block:
                while self._count + samples_to_write > self.capacity:
                    if not self._not_full.wait(timeout):
                        return 0
            else:
                if self._count + samples_to_write > self.capacity:
                    # Overwrite oldest data
                    overflow = (self._count + samples_to_write) - self.capacity
                    self._read_pos = (self._read_pos + overflow) % self.capacity
                    self._count -= overflow
            
            # Write data, possibly wrapping around
            if self._write_pos + samples_to_write <= self.capacity:
                self._buffer[self._write_pos:self._write_pos + samples_to_write] = data
            else:
                first_part = self.capacity - self._write_pos
                self._buffer[self._write_pos:] = data[:first_part]
                self._buffer[:samples_to_write - first_part] = data[first_part:]
            
            self._write_pos = (self._write_pos + samples_to_write) % self.capacity
            self._count += samples_to_write
            
            self._not_empty.notify_all()
            
            return samples_to_write
    
    def read(self, num_samples: int, block: bool = True, timeout: float = None) -> Optional[np.ndarray]:
        """
        Read data from buffer
        
        Args:
            num_samples: Number of samples to read
            block: Whether to block if buffer is empty
            timeout: Maximum time to wait
            
        Returns:
            Audio samples or None if timeout
        """
        with self._not_empty:
            if block:
                while self._count < num_samples:
                    if not self._not_empty.wait(timeout):
                        return None
            else:
                if self._count < num_samples:
                    return None
            
            # Read data
            result = np.zeros(num_samples, dtype=self.dtype)
            
            if self._read_pos + num_samples <= self.capacity:
                result[:] = self._buffer[self._read_pos:self._read_pos + num_samples]
            else:
                first_part = self.capacity - self._read_pos
                result[:first_part] = self._buffer[self._read_pos:]
                result[first_part:] = self._buffer[:num_samples - first_part]
            
            self._read_pos = (self._read_pos + num_samples) % self.capacity
            self._count -= num_samples
            
            self._not_full.notify_all()
            
            return result
    
    def peek(self, num_samples: int) -> Optional[np.ndarray]:
        """Peek at data without consuming it"""
        with self._lock:
            if self._count < num_samples:
                return None
            
            result = np.zeros(num_samples, dtype=self.dtype)
            
            if self._read_pos + num_samples <= self.capacity:
                result[:] = self._buffer[self._read_pos:self._read_pos + num_samples]
            else:
                first_part = self.capacity - self._read_pos
                result[:first_part] = self._buffer[self._read_pos:]
                result[first_part:] = self._buffer[:num_samples - first_part]
            
            return result
    
    @property
    def available(self) -> int:
        """Number of samples available to read"""
        with self._lock:
            return self._count
    
    @property
    def free_space(self) -> int:
        """Free space in samples"""
        with self._lock:
            return self.capacity - self._count
    
    def clear(self):
        """Clear the buffer"""
        with self._lock:
            self._read_pos = 0
            self._write_pos = 0
            self._count = 0
            self._not_full.notify_all()


class AudioFrameBuffer:
    """
    Buffer for storing audio frames with timestamps
    """
    
    def __init__(self, max_duration_ms: float, sample_rate: int):
        self.max_samples = int(max_duration_ms / 1000 * sample_rate)
        self.sample_rate = sample_rate
        self._frames = deque()
        self._total_samples = 0
        self._lock = threading.Lock()
    
    def add_frame(self, frame: 'AudioFrame'):
        """Add a frame to the buffer"""
        with self._lock:
            self._frames.append(frame)
            self._total_samples += len(frame.data)
            
            # Remove old frames if over capacity
            while self._total_samples > self.max_samples and len(self._frames) > 1:
                old_frame = self._frames.popleft()
                self._total_samples -= len(old_frame.data)
    
    def get_all(self) -> np.ndarray:
        """Get all buffered audio as a single array"""
        with self._lock:
            if not self._frames:
                return np.array([], dtype=np.float32)
            return np.concatenate([f.data for f in self._frames])
    
    def get_duration_ms(self) -> float:
        """Get total duration of buffered audio"""
        with self._lock:
            return self._total_samples / self.sample_rate * 1000
    
    def clear(self):
        """Clear the buffer"""
        with self._lock:
            self._frames.clear()
            self._total_samples = 0