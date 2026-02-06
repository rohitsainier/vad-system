# test_voice_pipeline.py
"""
Test voice chat pipeline with audio files (no browser needed)
"""
import sys
sys.path.insert(0, '.')

import numpy as np
import soundfile as sf
import tempfile
import subprocess
import os

from src.voice_chat.pipeline import VoiceChatPipeline, PipelineConfig


def generate_speech(text: str) -> str:
    """Generate speech using espeak"""
    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
        path = f.name
    subprocess.run(
        ['espeak', '-w', path, '-s', '140', text],
        capture_output=True, timeout=10
    )
    return path


def main():
    print("=" * 60)
    print("  🎤 Voice Pipeline Test (File-based)")
    print("=" * 60)
    
    # Configure pipeline
    config = PipelineConfig(
        whisper_model="base",      # Use 'small' or 'medium' for better accuracy
        llm_model="translategemma",   # Change to your model
        tts_engine="espeak",       # Use espeak since we're in WSL2
        whisper_language="en"
    )
    
    print(f"\n  Whisper model: {config.whisper_model}")
    print(f"  LLM model: {config.llm_model}")
    print(f"  TTS engine: {config.tts_engine}")
    
    # Initialize pipeline
    print("\n⏳ Initializing pipeline (first run downloads models)...\n")
    pipeline = VoiceChatPipeline(config)
    pipeline.initialize()
    print("✅ Pipeline ready!\n")
    
    # Test conversations
    test_inputs = [
        "Hello, how are you today?",
        "What is the capital of France?",
        "Tell me a short joke.",
        "What can you help me with?",
    ]
    
    for i, text in enumerate(test_inputs):
        print(f"\n{'='*60}")
        print(f"  Turn {i+1}")
        print(f"{'='*60}")
        
        # Generate speech input using espeak
        print(f"\n  📢 Generating speech: \"{text}\"")
        audio_path = generate_speech(text)
        
        # Load audio
        audio, sr = sf.read(audio_path)
        audio = audio.astype(np.float32)
        os.unlink(audio_path)
        
        print(f"  🎤 Audio: {len(audio)/sr:.1f}s @ {sr}Hz")
        
        # Process through pipeline
        print(f"  ⏳ Processing...\n")
        
        result = pipeline.process_speech(audio, sr)
        
        if result:
            turn = pipeline.conversation.turns[-1]
            
            print(f"  📝 You said:     \"{turn.user_text}\"")
            print(f"  🤖 Assistant:    \"{turn.assistant_text}\"")
            print(f"  ⚡ Latency:      {turn.latency_summary}")
            print(f"  🔊 Response:     {result.duration_ms:.0f}ms audio")
            
            # Save response audio
            output_path = f"test_audio/response_{i+1}.wav"
            os.makedirs("test_audio", exist_ok=True)
            sf.write(output_path, result.audio, result.sample_rate)
            print(f"  💾 Saved:        {output_path}")
        else:
            print(f"  ❌ No response generated")
    
    # Summary
    summary = pipeline.conversation.get_summary()
    
    print(f"\n\n{'='*60}")
    print(f"  📊 Session Summary")
    print(f"{'='*60}")
    print(f"  Turns:           {summary['turns']}")
    print(f"  Avg latency:     {summary['avg_latency_ms']:.0f}ms")
    print(f"  Avg STT time:    {summary['avg_stt_ms']:.0f}ms")
    print(f"  Avg LLM time:    {summary['avg_llm_ms']:.0f}ms")
    print(f"  Avg TTS time:    {summary['avg_tts_ms']:.0f}ms")
    print(f"{'='*60}\n")
    
    pipeline.cleanup()


if __name__ == "__main__":
    main()