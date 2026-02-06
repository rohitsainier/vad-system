# src/audio_sources/websocket_source.py
"""
WebSocket audio source for receiving audio over WebSocket
"""
import numpy as np
import asyncio
from typing import Optional, Dict, Any
import websockets
from websockets.client import WebSocketClientProtocol
import json
import structlog

from .base_source import BaseAudioSource
from .stream_source import StreamSource

logger = structlog.get_logger(__name__)


class WebSocketAudioSource(BaseAudioSource):
    """
    WebSocket client audio source
    
    Connects to a WebSocket server and receives audio data
    
    Features:
    - Auto-reconnection
    - Binary and JSON message support
    - Configurable audio format
    - Connection state management
    """
    
    def __init__(
        self,
        uri: str,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_duration_ms: int = 30,
        auto_reconnect: bool = True,
        reconnect_delay: float = 1.0,
        max_reconnect_attempts: int = 10
    ):
        super().__init__(sample_rate, channels, chunk_duration_ms)
        
        self.uri = uri
        self.auto_reconnect = auto_reconnect
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_attempts = max_reconnect_attempts
        
        self._websocket: Optional[WebSocketClientProtocol] = None
        self._stream = StreamSource(sample_rate, channels, chunk_duration_ms)
        self._receive_task: Optional[asyncio.Task] = None
        self._reconnect_count = 0
    
    def open(self) -> None:
        """Open WebSocket connection (starts async task)"""
        if self._is_open:
            return
        
        self._stream.open()
        self._is_open = True
        
        logger.info("WebSocketAudioSource initialized", uri=self.uri)
    
    async def connect(self) -> bool:
        """
        Establish WebSocket connection
        
        Returns:
            True if connected successfully
        """
        try:
            self._websocket = await websockets.connect(
                self.uri,
                ping_interval=30,
                ping_timeout=10
            )
            
            # Send configuration
            await self._websocket.send(json.dumps({
                "type": "configure",
                "sample_rate": self.sample_rate,
                "channels": self.channels
            }))
            
            self._reconnect_count = 0
            logger.info("WebSocket connected", uri=self.uri)
            
            return True
            
        except Exception as e:
            logger.error("WebSocket connection failed", error=str(e))
            return False
    
    async def start_receiving(self):
        """Start receiving audio data"""
        if not self._is_open:
            self.open()
        
        while self._is_open:
            if self._websocket is None or self._websocket.closed:
                if not await self._try_reconnect():
                    break
                continue
            
            try:
                message = await self._websocket.recv()
                
                if isinstance(message, bytes):
                    # Binary audio data
                    self._stream.push_bytes(message)
                elif isinstance(message, str):
                    # JSON message
                    self._handle_json_message(json.loads(message))
                    
            except websockets.exceptions.ConnectionClosed:
                logger.warning("WebSocket connection closed")
                self._websocket = None
                
            except Exception as e:
                logger.error("Error receiving data", error=str(e))
    
    async def _try_reconnect(self) -> bool:
        """Attempt to reconnect"""
        if not self.auto_reconnect:
            return False
        
        if self._reconnect_count >= self.max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return False
        
        self._reconnect_count += 1
        logger.info(
            "Attempting reconnection",
            attempt=self._reconnect_count,
            max_attempts=self.max_reconnect_attempts
        )
        
        await asyncio.sleep(self.reconnect_delay * self._reconnect_count)
        
        return await self.connect()
    
    def _handle_json_message(self, message: Dict[str, Any]):
        """Handle JSON messages from server"""
        msg_type = message.get("type")
        
        if msg_type == "error":
            logger.error("Server error", error=message.get("message"))
        elif msg_type == "configured":
            logger.info("Server configured", data=message.get("data"))
    
    def close(self) -> None:
        """Close WebSocket connection"""
        self._is_open = False
        
        if self._receive_task:
            self._receive_task.cancel()
        
        if self._websocket:
            asyncio.create_task(self._websocket.close())
        
        self._stream.close()
        
        logger.info("WebSocketAudioSource closed")
    
    def read(self, timeout: float = 1.0) -> Optional[np.ndarray]:
        """Read next audio chunk"""
        return self._stream.read(timeout)
    
    async def read_async(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Read next audio chunk (async)"""
        return await self._stream.read_async(timeout)
    
    async def stream_async(self):
        """Stream audio data"""
        # Start receiving in background
        self._receive_task = asyncio.create_task(self.start_receiving())
        
        try:
            async for chunk in self._stream.stream_async():
                yield chunk
        finally:
            self._receive_task.cancel()