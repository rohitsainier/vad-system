# examples/realtime_microphone.py
"""
Real-time microphone VAD example
"""
import asyncio
import sys
sys.path.insert(0, '..')

from src.core.vad_engine import VADEngine
from src.audio_sources.microphone_source import MicrophoneSource
from src.core.state_machine import SpeechSegment
from config.settings import Settings


async def main():
    settings = Settings()
    
    # Create VAD engine
    engine = VADEngine(settings.audio, settings.vad)
    engine.initialize()
    
    # Track speech state
    speech_active = False
    
    def on_speech_start(segment: SpeechSegment):
        nonlocal speech_active
        speech_active = True
        print(f"\n🎤 Speech started at {segment.start_time_ms:.0f}ms")
    
    def on_speech_end(segment: SpeechSegment):
        nonlocal speech_active
        speech_active = False
        print(f"🔇 Speech ended - Duration: {segment.duration_ms:.0f}ms")
    
    engine.on_speech_start(on_speech_start)
    engine.on_speech_end(on_speech_end)
    
    # Create microphone source
    mic = MicrophoneSource(
        sample_rate=settings.audio.sample_rate,
        chunk_duration_ms=settings.audio.chunk_duration_ms
    )
    
    print("=" * 50)
    print("Real-time Voice Activity Detection")
    print("=" * 50)
    print("Speak into your microphone...")
    print("Press Ctrl+C to stop\n")
    
    try:
        async for audio_chunk in mic.stream():
            result = engine.process_audio(
                audio_chunk,
                settings.audio.sample_rate
            )
            
            # Visual indicator
            indicator = "🟢" if result.is_speech else "⚫"
            prob_bar = "█" * int(result.raw_probability * 20) if result.raw_probability else ""
            print(f"\r{indicator} [{prob_bar:<20}] {result.raw_probability:.2f}" if result.raw_probability else "", end="")
            
    except KeyboardInterrupt:
        print("\n\nStopping...")
    finally:
        engine.cleanup()
        
    # Print summary
    metrics = engine.get_metrics()
    print("\n" + "=" * 50)
    print("Session Summary")
    print("=" * 50)
    print(f"Frames processed: {metrics['frames_processed']}")
    print(f"Speech segments: {metrics['speech_segments_detected']}")
    print(f"Total speech: {metrics['total_speech_duration_ms']:.0f}ms")


if __name__ == "__main__":
    asyncio.run(main())