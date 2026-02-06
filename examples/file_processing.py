# examples/file_processing.py
"""
File processing example - Process audio files for VAD
"""
import sys
sys.path.insert(0, '..')

from pathlib import Path
from src.core.vad_engine import VADEngine
from src.audio_sources.file_source import FileSource
from src.handlers.speech_handler import SpeechHandler
from src.core.state_machine import SpeechSegment
from config.settings import Settings


def process_audio_file(file_path: str, output_dir: str = None):
    """
    Process an audio file and extract speech segments
    
    Args:
        file_path: Path to audio file
        output_dir: Optional directory to save speech segments
    """
    settings = Settings()
    
    print(f"Processing: {file_path}")
    print("=" * 50)
    
    # Load audio file
    source = FileSource(file_path)
    print(f"Duration: {source.duration_seconds:.2f} seconds")
    print(f"Sample rate: {source.source_sample_rate} Hz")
    print(f"Channels: {source.channels}")
    print()
    
    # Create VAD engine
    engine = VADEngine(settings.audio, settings.vad)
    engine.initialize()
    
    # Create speech handler
    handler = SpeechHandler(
        sample_rate=settings.audio.sample_rate,
        output_dir=output_dir
    )
    
    # Track segments
    segments = []
    
    def on_speech_end(segment: SpeechSegment):
        segments.append(segment)
        processed = handler.handle(segment)
        if processed:
            print(f"  Segment {len(segments)}: "
                  f"{segment.start_time_ms:.0f}ms - {segment.end_time_ms:.0f}ms "
                  f"({segment.duration_ms:.0f}ms)")
    
    engine.on_speech_end(on_speech_end)
    
    # Process file
    print("Detected speech segments:")
    
    for chunk in source.stream():
        engine.process_audio(chunk, source.source_sample_rate)
    
    # Print summary
    print()
    print("=" * 50)
    print("Summary:")
    print(f"  Total segments: {len(segments)}")
    
    if segments:
        total_speech = sum(s.duration_ms or 0 for s in segments)
        total_audio = source.duration_seconds * 1000
        print(f"  Total speech: {total_speech:.0f}ms")
        print(f"  Speech ratio: {total_speech / total_audio * 100:.1f}%")
    
    engine.cleanup()
    
    return segments


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Process audio file for VAD")
    parser.add_argument("file", help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output directory for segments")
    
    args = parser.parse_args()
    
    process_audio_file(args.file, args.output)