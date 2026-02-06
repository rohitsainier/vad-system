# src/voice_chat/chat_server.py
"""
WebSocket server for voice-to-voice chat
FIXED: Non-blocking initialization, proper error handling, Ollama auto-detect
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
    
    # === Main message loop ===
    speech_buffer = []
    is_speaking = False
    silence_start = None
    SILENCE_TIMEOUT = 1.0  # seconds of silence before processing
    
    try:
        while True:
            try:
                # Receive with timeout to detect disconnects
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=60.0  # 60 second keepalive
                )
            except asyncio.TimeoutError:
                # Send ping to keep connection alive
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
                
                # Send VAD status
                await websocket.send_json({
                    "type": "vad",
                    "is_speech": bool(result.is_speech),
                    "probability": float(result.raw_probability or 0)
                })
                
                if result.is_speech:
                    speech_buffer.append(audio_float)
                    is_speaking = True
                    silence_start = None
                else:
                    if is_speaking:
                        if silence_start is None:
                            silence_start = time.time()
                        
                        # Keep buffering during short silences
                        speech_buffer.append(audio_float)
                        
                        # Check if silence timeout reached
                        if time.time() - silence_start >= SILENCE_TIMEOUT:
                            # Process accumulated speech
                            if speech_buffer:
                                full_audio = np.concatenate(speech_buffer)
                                duration_ms = len(full_audio) / 16000 * 1000
                                
                                if duration_ms >= 500:  # Minimum 500ms
                                    await websocket.send_json({
                                        "type": "processing",
                                        "message": "⏳ Processing your speech..."
                                    })
                                    
                                    # Process through pipeline (in executor to not block)
                                    try:
                                        tts_result = await loop.run_in_executor(
                                            None,
                                            pipeline.process_speech,
                                            full_audio,
                                            16000
                                        )
                                        
                                        if tts_result and pipeline.conversation.turns:
                                            turn = pipeline.conversation.turns[-1]
                                            
                                            # Send transcription
                                            await websocket.send_json({
                                                "type": "transcription",
                                                "text": turn.user_text
                                            })
                                            
                                            # Send LLM response
                                            await websocket.send_json({
                                                "type": "response",
                                                "text": turn.assistant_text,
                                                "latency": turn.latency_summary
                                            })
                                            
                                            # Send audio response
                                            response_int16 = (tts_result.audio * 32767).clip(
                                                -32768, 32767
                                            ).astype(np.int16)
                                            await websocket.send_bytes(response_int16.tobytes())
                                        else:
                                            await websocket.send_json({
                                                "type": "status",
                                                "message": "Could not process speech. Try again."
                                            })
                                    except Exception as e:
                                        logger.error("Processing error", error=str(e))
                                        await websocket.send_json({
                                            "type": "error",
                                            "message": f"Processing error: {str(e)[:100]}"
                                        })
                            
                            # Reset state
                            speech_buffer = []
                            is_speaking = False
                            silence_start = None
            
            elif "text" in message:
                # JSON command
                try:
                    cmd = json.loads(message["text"])
                    await handle_command(cmd, websocket, pipeline, loop)
                except json.JSONDecodeError:
                    pass
    
    except WebSocketDisconnect:
        logger.info("Client disconnected", session_id=session_id)
    except Exception as e:
        logger.error("WebSocket error", error=str(e))
    finally:
        if vad:
            vad.cleanup()
        if pipeline:
            pipeline.cleanup()
        logger.info("Session cleaned up", session_id=session_id)


async def handle_command(cmd: dict, websocket: WebSocket, pipeline, loop):
    """Handle JSON commands from client"""
    cmd_type = cmd.get("type")
    
    if cmd_type == "clear":
        if pipeline:
            pipeline.clear_history()
        await websocket.send_json({
            "type": "cleared",
            "message": "Conversation cleared"
        })
    
    elif cmd_type == "text_chat":
        text = cmd.get("text", "").strip()
        if text and pipeline and pipeline._is_initialized:
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
                
                # TTS
                tts_result = await loop.run_in_executor(
                    None, pipeline._tts_engine.synthesize, llm_response.text
                )
                
                if tts_result:
                    response_int16 = (tts_result.audio * 32767).clip(
                        -32768, 32767
                    ).astype(np.int16)
                    await websocket.send_bytes(response_int16.tobytes())
                    
            except Exception as e:
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
    
    elif cmd_type == "ping":
        await websocket.send_json({"type": "pong"})


def start_server(host: str = "0.0.0.0", port: int = 8080, config: dict = None):
    """Start the voice chat server"""
    from .pipeline import PipelineConfig
    
    # Store config in app state
    if config:
        app.state.pipeline_config = PipelineConfig(**config)
    else:
        app.state.pipeline_config = PipelineConfig()
    
    uvicorn.run(app, host=host, port=port, log_level="info")