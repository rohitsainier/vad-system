# src/server/websocket_server.py
"""
WebSocket server for real-time VAD
"""
import asyncio
import json
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass, asdict
import numpy as np
import websockets
from websockets.server import WebSocketServerProtocol
import structlog

from ..core.vad_engine import VADEngine, VADEvent
from ..core.state_machine import SpeechSegment
from config.settings import Settings

logger = structlog.get_logger(__name__)


@dataclass
class ClientSession:
    """Represents a connected client"""
    websocket: WebSocketServerProtocol
    session_id: str
    vad_engine: VADEngine
    sample_rate: int = 16000
    channels: int = 1


class VADWebSocketServer:
    """
    Production WebSocket server for real-time VAD
    
    Features:
    - Multiple concurrent clients
    - Per-client VAD engines
    - Binary audio data support
    - JSON event responses
    - Graceful shutdown
    """
    
    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8001,
        settings: Optional[Settings] = None
    ):
        self.host = host
        self.port = port
        self.settings = settings or Settings()
        
        self._clients: Dict[str, ClientSession] = {}
        self._server = None
        self._is_running = False
    
    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """Handle a client connection"""
        session_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        
        logger.info("Client connected", session_id=session_id, path=path)
        
        # Create VAD engine for this client
        vad_engine = VADEngine(
            audio_settings=self.settings.audio,
            vad_settings=self.settings.vad
        )
        vad_engine.initialize()
        
        session = ClientSession(
            websocket=websocket,
            session_id=session_id,
            vad_engine=vad_engine,
            sample_rate=self.settings.audio.sample_rate,
            channels=self.settings.audio.channels
        )
        
        self._clients[session_id] = session
        
        # Set up event callbacks
        async def send_event(event_type: str, data: Dict[str, Any]):
            try:
                await websocket.send(json.dumps({
                    "type": event_type,
                    "data": data,
                    "session_id": session_id
                }))
            except Exception as e:
                logger.error("Failed to send event", error=str(e))
        
        def on_speech_start(segment: SpeechSegment):
            asyncio.create_task(send_event("speech_start", {
                "timestamp_ms": segment.start_time_ms,
                "confidence": segment.confidence
            }))
        
        def on_speech_end(segment: SpeechSegment):
            asyncio.create_task(send_event("speech_end", {
                "start_ms": segment.start_time_ms,
                "end_ms": segment.end_time_ms,
                "duration_ms": segment.duration_ms,
                "confidence": segment.confidence
            }))
        
        vad_engine.on_speech_start(on_speech_start)
        vad_engine.on_speech_end(on_speech_end)
        
        try:
            async for message in websocket:
                await self._process_message(session, message)
                
        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected", session_id=session_id)
        except Exception as e:
            logger.error("Client error", session_id=session_id, error=str(e))
        finally:
            # Cleanup
            vad_engine.cleanup()
            del self._clients[session_id]
    
    async def _process_message(self, session: ClientSession, message):
        """Process incoming message"""
        if isinstance(message, bytes):
            # Binary audio data
            audio_data = np.frombuffer(message, dtype=np.int16).astype(np.float32) / 32768.0
            
            result = session.vad_engine.process_audio(
                audio_data,
                session.sample_rate,
                session.channels
            )
            
            # Send probability update
            await session.websocket.send(json.dumps({
                "type": "vad_result",
                "data": {
                    "is_speech": result.is_speech,
                    "probability": result.raw_probability,
                    "confidence": result.confidence,
                    "timestamp_ms": result.timestamp_ms
                }
            }))
            
        elif isinstance(message, str):
            # JSON command
            try:
                cmd = json.loads(message)
                await self._handle_command(session, cmd)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON message", session_id=session.session_id)
    
    async def _handle_command(self, session: ClientSession, cmd: Dict[str, Any]):
        """Handle JSON command"""
        cmd_type = cmd.get("type")
        
        if cmd_type == "configure":
            # Update session configuration
            if "sample_rate" in cmd:
                session.sample_rate = cmd["sample_rate"]
            if "channels" in cmd:
                session.channels = cmd["channels"]
                
            await session.websocket.send(json.dumps({
                "type": "configured",
                "data": {
                    "sample_rate": session.sample_rate,
                    "channels": session.channels
                }
            }))
            
        elif cmd_type == "reset":
            session.vad_engine.reset()
            await session.websocket.send(json.dumps({
                "type": "reset",
                "data": {"status": "ok"}
            }))
            
        elif cmd_type == "get_metrics":
            metrics = session.vad_engine.get_metrics()
            await session.websocket.send(json.dumps({
                "type": "metrics",
                "data": metrics
            }))
    
    async def start(self):
        """Start the WebSocket server"""
        self._is_running = True
        
        self._server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=30,
            ping_timeout=10,
            max_size=10 * 1024 * 1024  # 10MB max message
        )
        
        logger.info(
            "WebSocket server started",
            host=self.host,
            port=self.port
        )
        
        await self._server.wait_closed()
    
    async def stop(self):
        """Stop the server"""
        self._is_running = False
        
        # Close all client connections
        for session in list(self._clients.values()):
            await session.websocket.close()
            session.vad_engine.cleanup()
        
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        
        logger.info("WebSocket server stopped")