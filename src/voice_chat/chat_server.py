# src/voice_chat/chat_server.py
"""
WebSocket server for voice-to-voice chat
With barge-in (interrupt) support
"""
import asyncio
import json
import numpy as np
import os
import time
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import structlog

logger = structlog.get_logger(__name__)

app = FastAPI(title="Voice Chat")

# Serve static files
WEB_DIR = Path(__file__).parent / "web"


@app.get("/")
async def index():
    """Serve the chat UI"""
    html_path = WEB_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text())
    return HTMLResponse("""
    <html><body>
    <h1>Voice Chat</h1>
    <p>Web UI files not found at: """ + str(WEB_DIR) + """</p>
    </body></html>
    """)


@app.get("/health")
async def health():
    return {"status": "healthy", "service": "voice-chat"}


class SessionState:
    """Per-session state tracking for barge-in support"""
    
    def __init__(self):
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_start = None
        self.is_processing = False
        self.is_playing_response = False
        self.barge_in_triggered = False
        self.processing_task: Optional[asyncio.Task] = None
        
        # Barge-in settings
        self.barge_in_speech_frames = 0
        self.barge_in_threshold = 3  # frames of speech to trigger interrupt
        self.silence_timeout = 1.0  # seconds
    
    def reset_speech_buffer(self):
        """Reset speech collection state"""
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_start = None
        self.barge_in_speech_frames = 0
    
    def cancel_processing(self):
        """Cancel any ongoing processing"""
        if self.processing_task and not self.processing_task.done():
            self.processing_task.cancel()
            logger.info("Processing task cancelled due to barge-in")
        self.is_processing = False
        self.is_playing_response = False
        self.barge_in_triggered = True


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """Handle voice chat WebSocket connection"""
    await websocket.accept()
    
    client = websocket.client
    session_id = f"{client.host}:{client.port}" if client else "unknown"
    logger.info("Client connected", session_id=session_id)
    
    # Send immediate acknowledgment
    await websocket.send_json({
        "type": "status",
        "message": "Connected! Initializing pipeline..."
    })
    
    # Initialize pipeline in background (non-blocking)
    pipeline = None
    vad = None
    
    try:
        # Import here to avoid circular imports
        from .pipeline import VoiceChatPipeline, PipelineConfig
        from ..core.vad_engine import VADEngine
        from config.settings import Settings, VADBackend
        
        # Get config from app state or use defaults
        config = getattr(app.state, 'pipeline_config', PipelineConfig())
        
        # Initialize each component with progress updates
        await websocket.send_json({
            "type": "status",
            "message": "🔄 Loading VAD model..."
        })
        
        vad_settings = Settings()
        vad_settings.vad.backend = VADBackend.SILERO
        vad_settings.audio.chunk_duration_ms = 32
        
        # Run blocking init in executor
        loop = asyncio.get_event_loop()
        
        vad = VADEngine(vad_settings.audio, vad_settings.vad)
        await loop.run_in_executor(None, vad.initialize)
        
        await websocket.send_json({
            "type": "status",
            "message": "✅ VAD ready. Loading speech recognition..."
        })
        
        # Initialize pipeline components one by one
        pipeline = VoiceChatPipeline(config)
        
        # Step-by-step initialization
        from .stt_engine import STTEngine
        from .llm_engine import LLMEngine
        from .tts_engine import TTSEngine
        
        # STT
        pipeline._stt_engine = STTEngine(
            model_size=config.whisper_model,
            language=config.whisper_language
        )
        await loop.run_in_executor(None, pipeline._stt_engine.initialize)
        
        await websocket.send_json({
            "type": "status",
            "message": "✅ Whisper ready. Connecting to LLM..."
        })
        
        # LLM
        pipeline._llm_engine = LLMEngine(
            model=config.llm_model,
            system_prompt=config.system_prompt,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens
        )
        await loop.run_in_executor(None, pipeline._llm_engine.initialize)
        
        await websocket.send_json({
            "type": "status",
            "message": "✅ LLM ready. Loading TTS..."
        })
        
        # TTS
        pipeline._tts_engine = TTSEngine(
            engine=config.tts_engine,
            edge_voice=config.tts_voice
        )
        await loop.run_in_executor(None, pipeline._tts_engine.initialize)
        
        # Mark as initialized
        pipeline._vad_engine = vad
        pipeline._is_initialized = True
        
        await websocket.send_json({
            "type": "ready",
            "message": "🎤 All systems ready! Click the microphone to start talking."
        })
        
    except Exception as e:
        error_msg = str(e)
        logger.error("Pipeline initialization failed", error=error_msg)
        
        await websocket.send_json({
            "type": "error",
            "message": f"Initialization failed: {error_msg}"
        })
        
        # Cleanup partial init
        if vad:
            vad.cleanup()
        if pipeline:
            pipeline.cleanup()
        
        await websocket.close()
        return
    
    # === Main message loop with barge-in support ===
    session = SessionState()
    
    try:
        while True:
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=60.0
                )
            except asyncio.TimeoutError:
                try:
                    await websocket.send_json({"type": "ping"})
                except Exception:
                    break
                continue
            
            if "bytes" in message:
                # Binary audio data (int16 PCM from browser)
                audio_bytes = message["bytes"]
                
                if len(audio_bytes) == 0:
                    continue
                
                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_int16.astype(np.float32) / 32768.0
                
                # Run VAD
                try:
                    result = vad.process_audio(audio_float, 16000)
                except Exception as e:
                    logger.error("VAD error", error=str(e))
                    continue
                
                is_speech = bool(result.is_speech)
                probability = float(result.raw_probability or 0)
                
                # Send VAD status to client (client uses this for barge-in too)
                await websocket.send_json({
                    "type": "vad",
                    "is_speech": is_speech,
                    "probability": probability
                })
                
                # === Server-side barge-in detection ===
                if is_speech and session.is_playing_response:
                    session.barge_in_speech_frames += 1
                    
                    if session.barge_in_speech_frames >= session.barge_in_threshold:
                        logger.info("Barge-in detected on server side")
                        session.cancel_processing()
                        session.reset_speech_buffer()
                        
                        await websocket.send_json({
                            "type": "interrupted",
                            "message": "Response interrupted by user"
                        })
                        
                        # Start collecting new speech immediately
                        session.speech_buffer.append(audio_float)
                        session.is_speaking = True
                        continue
                
                if not is_speech and not session.is_playing_response:
                    session.barge_in_speech_frames = 0
                
                # === Normal speech collection ===
                # Skip audio collection while playing response 
                # (unless barge-in is triggered)
                if session.is_processing:
                    continue
                
                if is_speech:
                    session.speech_buffer.append(audio_float)
                    session.is_speaking = True
                    session.silence_start = None
                else:
                    if session.is_speaking:
                        if session.silence_start is None:
                            session.silence_start = time.time()
                        
                        # Keep buffering during short silences
                        session.speech_buffer.append(audio_float)
                        
                        # Check if silence timeout reached
                        if time.time() - session.silence_start >= session.silence_timeout:
                            # Process accumulated speech
                            if session.speech_buffer:
                                full_audio = np.concatenate(session.speech_buffer)
                                duration_ms = len(full_audio) / 16000 * 1000
                                
                                if duration_ms >= 500:  # Minimum 500ms
                                    # Launch processing as a task so we can cancel it
                                    session.is_processing = True
                                    session.barge_in_triggered = False
                                    
                                    session.processing_task = asyncio.create_task(
                                        process_speech_turn(
                                            websocket, pipeline, session,
                                            full_audio, loop
                                        )
                                    )
                            
                            # Reset speech buffer
                            session.reset_speech_buffer()
            
            elif "text" in message:
                # JSON command
                try:
                    cmd = json.loads(message["text"])
                    
                    if cmd.get("type") == "barge_in":
                        # Client detected barge-in
                        logger.info("Client reported barge-in")
                        session.cancel_processing()
                        session.reset_speech_buffer()
                        
                        await websocket.send_json({
                            "type": "interrupted",
                            "message": "Response interrupted"
                        })
                    else:
                        await handle_command(
                            cmd, websocket, pipeline, session, loop
                        )
                except json.JSONDecodeError:
                    pass
    
    except WebSocketDisconnect:
        logger.info("Client disconnected", session_id=session_id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
    finally:
        # Cancel any pending processing
        session.cancel_processing()
        
        if vad:
            vad.cleanup()
        if pipeline:
            pipeline.cleanup()
        logger.info("Session cleaned up", session_id=session_id)


async def process_speech_turn(
    websocket: WebSocket,
    pipeline,
    session: SessionState,
    full_audio: np.ndarray,
    loop
):
    """
    Process a speech turn through the pipeline.
    This runs as an asyncio task so it can be cancelled for barge-in.
    """
    try:
        await websocket.send_json({
            "type": "processing",
            "message": "⏳ Processing your speech..."
        })
        
        # Check if we were interrupted before even starting
        if session.barge_in_triggered:
            logger.info("Skipping processing - barge-in already triggered")
            return
        
        # Process through pipeline (in executor to not block)
        tts_result = await loop.run_in_executor(
            None,
            pipeline.process_speech,
            full_audio,
            16000
        )
        
        # Check again after processing completes
        if session.barge_in_triggered:
            logger.info("Discarding result - barge-in during processing")
            return
        
        if tts_result and pipeline.conversation.turns:
            turn = pipeline.conversation.turns[-1]
            
            # Send transcription
            await websocket.send_json({
                "type": "transcription",
                "text": turn.user_text
            })
            
            # Check for barge-in before sending response
            if session.barge_in_triggered:
                logger.info("Discarding response - barge-in before delivery")
                return
            
            # Send LLM response text
            await websocket.send_json({
                "type": "response",
                "text": turn.assistant_text,
                "latency": turn.latency_summary
            })
            
            # Mark that we're about to play audio
            session.is_playing_response = True
            session.barge_in_speech_frames = 0
            
            # Check for barge-in before sending audio
            if session.barge_in_triggered:
                session.is_playing_response = False
                return
            
            # Send audio response
            response_int16 = (tts_result.audio * 32767).clip(
                -32768, 32767
            ).astype(np.int16)
            await websocket.send_bytes(response_int16.tobytes())
            
            # Estimate playback duration and set a timer to clear playing state
            # (Client also manages this, but this is a safety net)
            playback_duration = len(tts_result.audio) / tts_result.sample_rate
            
            async def clear_playing_state():
                await asyncio.sleep(playback_duration + 0.5)  # small buffer
                if session.is_playing_response and not session.barge_in_triggered:
                    session.is_playing_response = False
                    session.barge_in_speech_frames = 0
            
            asyncio.create_task(clear_playing_state())
            
        else:
            await websocket.send_json({
                "type": "status",
                "message": "Could not process speech. Try again."
            })
    
    except asyncio.CancelledError:
        logger.info("Processing cancelled (barge-in)")
        session.is_playing_response = False
    except Exception as e:
        logger.error("Processing error", error=str(e))
        session.is_playing_response = False
        try:
            await websocket.send_json({
                "type": "error",
                "message": f"Processing error: {str(e)[:100]}"
            })
        except Exception:
            pass
    finally:
        session.is_processing = False


async def handle_command(
    cmd: dict,
    websocket: WebSocket,
    pipeline,
    session: SessionState,
    loop
):
    """Handle JSON commands from client"""
    cmd_type = cmd.get("type")
    
    if cmd_type == "clear":
        session.cancel_processing()
        session.reset_speech_buffer()
        if pipeline:
            pipeline.clear_history()
        await websocket.send_json({
            "type": "cleared",
            "message": "Conversation cleared"
        })
    
    elif cmd_type == "text_chat":
        text = cmd.get("text", "").strip()
        if text and pipeline and pipeline._is_initialized:
            # Stop any playing response
            session.cancel_processing()
            
            await websocket.send_json({
                "type": "processing",
                "message": "⏳ Thinking..."
            })
            
            try:
                # LLM response
                llm_response = await loop.run_in_executor(
                    None, pipeline._llm_engine.chat, text
                )
                
                await websocket.send_json({
                    "type": "response",
                    "text": llm_response.text,
                    "latency": f"LLM: {llm_response.processing_time_ms:.0f}ms"
                })
                
                session.is_playing_response = True
                session.barge_in_speech_frames = 0
                
                # TTS
                tts_result = await loop.run_in_executor(
                    None, pipeline._tts_engine.synthesize, llm_response.text
                )
                
                if tts_result and not session.barge_in_triggered:
                    response_int16 = (tts_result.audio * 32767).clip(
                        -32768, 32767
                    ).astype(np.int16)
                    await websocket.send_bytes(response_int16.tobytes())
                    
                    # Safety timer
                    playback_duration = len(tts_result.audio) / tts_result.sample_rate
                    
                    async def clear_playing_state():
                        await asyncio.sleep(playback_duration + 0.5)
                        if session.is_playing_response and not session.barge_in_triggered:
                            session.is_playing_response = False
                    
                    asyncio.create_task(clear_playing_state())
                    
            except Exception as e:
                session.is_playing_response = False
                await websocket.send_json({
                    "type": "error",
                    "message": f"Error: {str(e)[:100]}"
                })
    
    elif cmd_type == "stats":
        if pipeline:
            summary = pipeline.conversation.get_summary()
            await websocket.send_json({
                "type": "stats",
                "data": summary
            })
    
    elif cmd_type == "pong":
        pass  # Keepalive response, ignore


def start_server(host: str = "0.0.0.0", port: int = 8080, config: dict = None):
    """Start the voice chat server"""
    from .pipeline import PipelineConfig
    
    # Store config in app state
    if config:
        app.state.pipeline_config = PipelineConfig(**config)
    else:
        app.state.pipeline_config = PipelineConfig()
    
    uvicorn.run(app, host=host, port=port, log_level="info")