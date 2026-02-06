# examples/integration_example.py
"""
Full integration example - Real-time VAD with webhooks
"""
import asyncio
import sys
sys.path.insert(0, '..')

from src.core.vad_engine import VADEngine
from src.audio_sources.microphone_source import MicrophoneSource
from src.handlers.speech_handler import SpeechHandler
from src.handlers.webhook_handler import WebhookHandler
from src.utils.metrics import get_metrics_collector
from src.core.state_machine import SpeechSegment
from config.settings import Settings
from config.logging_config import setup_logging


async def main():
    # Setup logging
    logger = setup_logging(level="INFO", json_format=False)
    logger.info("Starting integration example")
    
    settings = Settings()
    
    # Initialize components
    engine = VADEngine(settings.audio, settings.vad)
    engine.initialize()
    
    mic = MicrophoneSource(
        sample_rate=settings.audio.sample_rate,
        chunk_duration_ms=settings.audio.chunk_duration_ms
    )
    
    speech_handler = SpeechHandler(
        sample_rate=settings.audio.sample_rate,
        output_dir="./speech_segments",
        min_duration_ms=500
    )
    
    webhook_handler = WebhookHandler(
        endpoints=["http://localhost:5000/webhook"],
        max_retries=2
    )
    
    metrics = get_metrics_collector()
    
    # Custom processor
    def process_speech(processed):
        logger.info(
            "Speech processed",
            duration_ms=processed.segment.duration_ms,
            segment_id=processed.metadata.get("segment_id")
        )
    
    speech_handler.add_processor(process_speech)
    
    # Event handlers
    async def on_speech_start(segment: SpeechSegment):
        logger.info("Speech started", timestamp=segment.start_time_ms)
        await webhook_handler.send_speech_start(segment)
    
    async def on_speech_end(segment: SpeechSegment):
        logger.info(
            "Speech ended",
            duration_ms=segment.duration_ms,
            confidence=segment.confidence
        )
        
        # Handle speech
        speech_handler.handle(segment)
        
        # Send webhook
        await webhook_handler.send_speech_end(segment)
        
        # Record metrics
        metrics.record_speech_segment(
            segment.duration_ms or 0,
            segment.confidence
        )
    
    # Register callbacks
    engine.on_speech_start(lambda s: asyncio.create_task(on_speech_start(s)))
    engine.on_speech_end(lambda s: asyncio.create_task(on_speech_end(s)))
    
    print("=" * 60)
    print("Integration Example - Real-time VAD with Webhooks")
    print("=" * 60)
    print("Components initialized:")
    print(f"  - VAD Engine: {settings.vad.backend.value}")
    print(f"  - Sample Rate: {settings.audio.sample_rate} Hz")
    print(f"  - Speech Handler: saving to ./speech_segments")
    print(f"  - Webhook: http://localhost:5000/webhook")
    print()
    print("Speak into your microphone. Press Ctrl+C to stop.")
    print()
    
    try:
        async for audio_chunk in mic.stream():
            import time
            start = time.time()
            
            result = engine.process_audio(
                audio_chunk,
                settings.audio.sample_rate
            )
            
            processing_time = (time.time() - start) * 1000
            
            # Record metrics
            metrics.record_frame(
                processing_time,
                result.is_speech,
                result.confidence
            )
            
            # Visual feedback
            prob = result.raw_probability or 0
            bar = "█" * int(prob * 30)
            indicator = "🟢" if result.is_speech else "⚫"
            print(f"\r{indicator} [{bar:<30}] {prob:.2f}", end="", flush=True)
            
    except KeyboardInterrupt:
        print("\n\nShutting down...")
    
    finally:
        # Flush pending webhooks
        await webhook_handler.flush()
        
        # Print final metrics
        final_metrics = metrics.get_metrics()
        print("\n" + "=" * 60)
        print("Session Metrics:")
        print(f"  Frames processed: {final_metrics['frames_processed']}")
        print(f"  Speech segments: {final_metrics['speech_segments']}")
        print(f"  Total speech: {final_metrics['total_speech_duration_ms']:.0f}ms")
        print(f"  Processing time: {final_metrics['processing_time_ms']:.1f}ms")
        
        webhook_stats = webhook_handler.stats
        print(f"\nWebhook Stats:")
        print(f"  Successful: {webhook_stats['success_count']}")
        print(f"  Failed: {webhook_stats['failure_count']}")
        
        # Cleanup
        engine.cleanup()


if __name__ == "__main__":
    asyncio.run(main())