# src/voice_chat/chat_server.py
"""
WebSocket server for voice-to-voice chat
With barge-in (interrupt) support
Uses IndicF5 TTS voice-cloning API
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
        self.speech_buffer = []
        self.is_speaking = False
        self.silence_start = None
        self.barge_in_speech_frames = 0

    def cancel_processing(self):
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

    await websocket.send_json({
        "type": "status",
        "message": "Connected! Initializing pipeline..."
    })

    pipeline = None
    vad = None

    try:
        from .pipeline import VoiceChatPipeline, PipelineConfig
        from ..core.vad_engine import VADEngine
        from config.settings import Settings, VADBackend

        config = getattr(app.state, 'pipeline_config', PipelineConfig())

        # ── VAD ────────────────────────────────────────────────────────
        await websocket.send_json({
            "type": "status",
            "message": "🔄 Loading VAD model..."
        })

        vad_settings = Settings()
        vad_settings.vad.backend = VADBackend.SILERO
        vad_settings.audio.chunk_duration_ms = 32

        loop = asyncio.get_event_loop()

        vad = VADEngine(vad_settings.audio, vad_settings.vad)
        await loop.run_in_executor(None, vad.initialize)

        await websocket.send_json({
            "type": "status",
            "message": "✅ VAD ready. Loading speech recognition..."
        })

        # ── Pipeline shell ─────────────────────────────────────────────
        pipeline = VoiceChatPipeline(config)

        # ── STT ────────────────────────────────────────────────────────
        from .stt_engine import STTEngine

        pipeline._stt_engine = STTEngine(
            model_size=config.whisper_model,
            language=config.whisper_language
        )
        await loop.run_in_executor(None, pipeline._stt_engine.initialize)

        await websocket.send_json({
            "type": "status",
            "message": "✅ Whisper ready. Connecting to LLM..."
        })

        # ── LLM ────────────────────────────────────────────────────────
        from .llm_engine import LLMEngine

        pipeline._llm_engine = LLMEngine(
            model=config.llm_model,
            system_prompt=config.system_prompt,
            temperature=config.llm_temperature,
            max_tokens=config.llm_max_tokens
        )
        await loop.run_in_executor(None, pipeline._llm_engine.initialize)

        await websocket.send_json({
            "type": "status",
            "message": "✅ LLM ready. Connecting to IndicF5 TTS..."
        })

        # ── TTS (IndicF5) ─────────────────────────────────────────────
        from .tts_engine import TTSEngine

        pipeline._tts_engine = TTSEngine(
            indicf5_base_url=config.indicf5_base_url,
            indicf5_reference_voice_key=config.indicf5_reference_voice_key,
            indicf5_sample_rate=config.indicf5_sample_rate,
            indicf5_output_format=getattr(config, 'indicf5_output_format', 'wav'),
            indicf5_timeout=config.indicf5_timeout,
            indicf5_seed=config.indicf5_seed,
            indicf5_normalize=getattr(config, 'indicf5_normalize', True),
        )
        await loop.run_in_executor(None, pipeline._tts_engine.initialize)

        # Report which voice was selected
        active_voice = (
            pipeline._tts_engine._indicf5_reference_voice_key or "(auto)"
        )
        logger.info("IndicF5 TTS initialised", voice=active_voice)

        # ── Done ───────────────────────────────────────────────────────
        pipeline._vad_engine = vad
        pipeline._is_initialized = True

        # Build voice list for the ready message
        try:
            voices = pipeline._tts_engine.list_reference_voices()
            voice_names = list(voices.keys())[:5]  # first 5
            voice_info = f" | Voices: {', '.join(voice_names)}"
            if len(voices) > 5:
                voice_info += f" (+{len(voices) - 5} more)"
        except Exception:
            voice_info = ""

        await websocket.send_json({
            "type": "ready",
            "message": (
                f"🎤 All systems ready! Voice: {active_voice}{voice_info}. "
                "Click the microphone to start talking."
            ),
            "tts_sample_rate": config.indicf5_sample_rate,
        })

    except Exception as e:
        error_msg = str(e)
        logger.error("Pipeline initialization failed", error=error_msg)

        await websocket.send_json({
            "type": "error",
            "message": f"Initialization failed: {error_msg}"
        })

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
                audio_bytes = message["bytes"]
                if len(audio_bytes) == 0:
                    continue

                audio_int16 = np.frombuffer(audio_bytes, dtype=np.int16)
                audio_float = audio_int16.astype(np.float32) / 32768.0

                try:
                    result = vad.process_audio(audio_float, 16000)
                except Exception as e:
                    logger.error("VAD error", error=str(e))
                    continue

                is_speech = bool(result.is_speech)
                probability = float(result.raw_probability or 0)

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

                        session.speech_buffer.append(audio_float)
                        session.is_speaking = True
                        continue

                if not is_speech and not session.is_playing_response:
                    session.barge_in_speech_frames = 0

                # === Normal speech collection ===
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

                        session.speech_buffer.append(audio_float)

                        if time.time() - session.silence_start >= session.silence_timeout:
                            if session.speech_buffer:
                                full_audio = np.concatenate(session.speech_buffer)
                                duration_ms = len(full_audio) / 16000 * 1000

                                if duration_ms >= 500:
                                    session.is_processing = True
                                    session.barge_in_triggered = False

                                    session.processing_task = asyncio.create_task(
                                        process_speech_turn(
                                            websocket, pipeline, session,
                                            full_audio, loop
                                        )
                                    )

                            session.reset_speech_buffer()

            elif "text" in message:
                try:
                    cmd = json.loads(message["text"])

                    if cmd.get("type") == "barge_in":
                        logger.info("Client reported barge-in")
                        session.cancel_processing()
                        session.reset_speech_buffer()

                        await websocket.send_json({
                            "type": "interrupted",
                            "message": "Response interrupted"
                        })
                    elif cmd.get("type") == "set_voice":
                        # Runtime voice switching from the client
                        voice_key = cmd.get("voice_key", "")
                        if voice_key and pipeline and pipeline._tts_engine:
                            pipeline._tts_engine.set_reference_voice(voice_key)
                            await websocket.send_json({
                                "type": "status",
                                "message": f"🔊 Voice changed to: {voice_key}"
                            })
                    elif cmd.get("type") == "list_voices":
                        # Send available voices to client
                        if pipeline and pipeline._tts_engine:
                            voices = pipeline._tts_engine.list_reference_voices(
                                force_refresh=True
                            )
                            voice_list = [
                                {
                                    "key": k,
                                    "author": v.author,
                                    "model": v.model,
                                    "content": v.content,
                                }
                                for k, v in voices.items()
                            ]
                            await websocket.send_json({
                                "type": "voices",
                                "voices": voice_list,
                                "active_voice": (
                                    pipeline._tts_engine._indicf5_reference_voice_key
                                    or "(auto)"
                                ),
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
    """Process a speech turn through the full pipeline (STT → LLM → TTS)."""
    try:
        await websocket.send_json({
            "type": "processing",
            "message": "⏳ Processing your speech..."
        })

        if session.barge_in_triggered:
            logger.info("Skipping processing - barge-in already triggered")
            return

        tts_result = await loop.run_in_executor(
            None,
            pipeline.process_speech,
            full_audio,
            16000
        )

        if session.barge_in_triggered:
            logger.info("Discarding result - barge-in during processing")
            return

        if tts_result and pipeline.conversation.turns:
            turn = pipeline.conversation.turns[-1]

            await websocket.send_json({
                "type": "transcription",
                "text": turn.user_text
            })

            if session.barge_in_triggered:
                logger.info("Discarding response - barge-in before delivery")
                return

            await websocket.send_json({
                "type": "response",
                "text": turn.assistant_text,
                "latency": turn.latency_summary
            })

            session.is_playing_response = True
            session.barge_in_speech_frames = 0

            if session.barge_in_triggered:
                session.is_playing_response = False
                return

            # Send sample-rate header so the client can play back correctly
            await websocket.send_json({
                "type": "audio_meta",
                "sample_rate": tts_result.sample_rate,
                "duration_ms": tts_result.duration_ms,
                "engine": tts_result.engine_used,
                "seed": tts_result.used_seed,
            })

            response_int16 = (tts_result.audio * 32767).clip(
                -32768, 32767
            ).astype(np.int16)
            print(f"DEBUG: tts_result.sample_rate = {tts_result.sample_rate}")
            print(f"DEBUG: audio length = {len(tts_result.audio)}")
            print(f"DEBUG: expected duration = {len(tts_result.audio) / tts_result.sample_rate:.2f}s")
            await websocket.send_bytes(response_int16.tobytes())

            playback_duration = len(tts_result.audio) / tts_result.sample_rate

            async def clear_playing_state():
                await asyncio.sleep(playback_duration + 0.5)
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
            session.cancel_processing()

            await websocket.send_json({
                "type": "processing",
                "message": "⏳ Thinking..."
            })

            try:
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

                # Optional per-call voice override from client
                voice_override = cmd.get("voice_key", None)

                tts_result = await loop.run_in_executor(
                    None,
                    lambda: pipeline._tts_engine.synthesize(
                        llm_response.text,
                        reference_voice_key=voice_override,
                    )
                )

                if tts_result and not session.barge_in_triggered:
                    # Send sample-rate so client can play at correct rate
                    await websocket.send_json({
                        "type": "audio_meta",
                        "sample_rate": tts_result.sample_rate,
                        "duration_ms": tts_result.duration_ms,
                        "engine": tts_result.engine_used,
                        "seed": tts_result.used_seed,
                    })

                    response_int16 = (tts_result.audio * 32767).clip(
                        -32768, 32767
                    ).astype(np.int16)
                    print(f"DEBUG: tts_result.sample_rate = {tts_result.sample_rate}")
                    print(f"DEBUG: audio length = {len(tts_result.audio)}")
                    print(f"DEBUG: expected duration = {len(tts_result.audio) / tts_result.sample_rate:.2f}s")  
                    await websocket.send_bytes(response_int16.tobytes())

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
        pass


def start_server(host: str = "0.0.0.0", port: int = 8080, config: dict = None):
    """Start the voice chat server"""
    from .pipeline import PipelineConfig

    if config:
        app.state.pipeline_config = PipelineConfig(**config)
    else:
        app.state.pipeline_config = PipelineConfig()

    uvicorn.run(app, host=host, port=port, log_level="info")