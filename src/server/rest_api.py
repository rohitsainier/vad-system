# src/server/rest_api.py
"""
REST API for VAD service
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List
import numpy as np
import soundfile as sf
import io
import uuid
import structlog

from ..core.vad_engine import VADEngine
from ..core.state_machine import SpeechSegment
from config.settings import Settings

logger = structlog.get_logger(__name__)

app = FastAPI(
    title="VAD Service API",
    description="Production Voice Activity Detection API",
    version="1.0.0"
)

# Global settings
settings = Settings()

# Store for background jobs
jobs = {}


class VADRequest(BaseModel):
    """Request model for VAD processing"""
    threshold: Optional[float] = 0.5
    min_speech_duration_ms: Optional[int] = 250
    min_silence_duration_ms: Optional[int] = 300
    return_audio: Optional[bool] = False


class SpeechSegmentResponse(BaseModel):
    """Response model for speech segment"""
    start_time_ms: float
    end_time_ms: float
    duration_ms: float
    confidence: float


class VADResponse(BaseModel):
    """Response model for VAD processing"""
    segments: List[SpeechSegmentResponse]
    total_speech_duration_ms: float
    total_audio_duration_ms: float
    speech_ratio: float


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "vad"}


@app.post("/api/v1/vad/process", response_model=VADResponse)
async def process_audio_file(
    file: UploadFile = File(...),
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 300
):
    """
    Process an audio file for voice activity detection
    
    - **file**: Audio file (WAV, FLAC, MP3, etc.)
    - **threshold**: Speech detection threshold (0-1)
    - **min_speech_ms**: Minimum speech duration
    - **min_silence_ms**: Minimum silence duration
    """
    try:
        # Read audio file
        audio_bytes = await file.read()
        audio_buffer = io.BytesIO(audio_bytes)
        
        audio_data, sample_rate = sf.read(audio_buffer, dtype='float32')
        
        # Convert to mono if needed
        if len(audio_data.shape) > 1:
            audio_data = np.mean(audio_data, axis=1)
        
        # Create VAD engine with custom settings
        vad_settings = settings.vad.copy()
        vad_settings.speech_threshold = threshold
        vad_settings.min_speech_duration_ms = min_speech_ms
        vad_settings.min_silence_duration_ms = min_silence_ms
        
        segments: List[SpeechSegment] = []
        
        with VADEngine(settings.audio, vad_settings) as engine:
            # Set up callback
            def on_segment(segment: SpeechSegment):
                segments.append(segment)
            
            engine.on_speech_end(on_segment)
            
            # Process in chunks
            chunk_size = int(sample_rate * 0.03)  # 30ms chunks
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                if len(chunk) < chunk_size:
                    # Pad last chunk
                    chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                
                engine.process_audio(chunk, sample_rate)
        
        # Calculate statistics
        total_audio_ms = len(audio_data) / sample_rate * 1000
        total_speech_ms = sum(s.duration_ms or 0 for s in segments)
        
        return VADResponse(
            segments=[
                SpeechSegmentResponse(
                    start_time_ms=s.start_time_ms,
                    end_time_ms=s.end_time_ms or 0,
                    duration_ms=s.duration_ms or 0,
                    confidence=s.confidence
                )
                for s in segments
            ],
            total_speech_duration_ms=total_speech_ms,
            total_audio_duration_ms=total_audio_ms,
            speech_ratio=total_speech_ms / total_audio_ms if total_audio_ms > 0 else 0
        )
        
    except Exception as e:
        logger.error("Error processing audio", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/vad/process-async")
async def process_audio_async(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """
    Process audio file asynchronously (for large files)
    
    Returns a job ID that can be used to check status
    """
    job_id = str(uuid.uuid4())
    audio_bytes = await file.read()
    
    jobs[job_id] = {"status": "processing", "result": None}
    
    async def process_job():
        try:
            audio_buffer = io.BytesIO(audio_bytes)
            audio_data, sample_rate = sf.read(audio_buffer, dtype='float32')
            
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)
            
            segments = []
            
            with VADEngine(settings.audio, settings.vad) as engine:
                def on_segment(segment: SpeechSegment):
                    segments.append(segment)
                
                engine.on_speech_end(on_segment)
                
                chunk_size = int(sample_rate * 0.03)
                for i in range(0, len(audio_data), chunk_size):
                    chunk = audio_data[i:i + chunk_size]
                    if len(chunk) < chunk_size:
                        chunk = np.pad(chunk, (0, chunk_size - len(chunk)))
                    engine.process_audio(chunk, sample_rate)
            
            jobs[job_id] = {
                "status": "completed",
                "result": {
                    "segments": [
                        {
                            "start_ms": s.start_time_ms,
                            "end_ms": s.end_time_ms,
                            "duration_ms": s.duration_ms
                        }
                        for s in segments
                    ]
                }
            }
        except Exception as e:
            jobs[job_id] = {"status": "failed", "error": str(e)}
    
    background_tasks.add_task(process_job)
    
    return {"job_id": job_id, "status": "processing"}


@app.get("/api/v1/vad/job/{job_id}")
async def get_job_status(job_id: str):
    """Get status of async processing job"""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return jobs[job_id]