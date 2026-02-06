# run_voice_chat.py
"""
Start the Voice Chat Server
"""
import sys
sys.path.insert(0, '.')

import argparse
import os


def main():
    parser = argparse.ArgumentParser(description='Voice Chat with Local LLM')
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--llm', default='translategemma', help='Ollama model name')
    parser.add_argument('--whisper', default='base', 
                       choices=['tiny', 'base', 'small', 'medium', 'large-v3'],
                       help='Whisper model size')
    parser.add_argument('--tts', default='auto', 
                       choices=['auto', 'espeak', 'edge-tts', 'piper'],
                       help='TTS engine')
    parser.add_argument('--ollama-host', default=None,
                       help='Ollama host URL (auto-detect if not set)')
    
    args = parser.parse_args()
    
    # Set Ollama host if provided
    if args.ollama_host:
        os.environ['OLLAMA_HOST'] = args.ollama_host
    
    # Quick pre-flight checks
    print("=" * 60)
    print("  🎤 Voice Chat with Local LLM")
    print("=" * 60)
    
    # Check Ollama
    print("\n  Checking Ollama...", end=" ")
    try:
        from src.voice_chat.llm_engine import LLMEngine
        host = LLMEngine._detect_ollama_host()
        print(f"✅ Found at {host}")
    except Exception as e:
        print(f"⚠️  {e}")
        print("  Make sure Ollama is running: ollama serve")
    
    # Check espeak
    import shutil
    print("  Checking TTS...", end=" ")
    if args.tts == 'espeak' or args.tts == 'auto':
        if shutil.which('espeak'):
            print("✅ espeak available")
        else:
            print("⚠️  espeak not found (install: sudo apt-get install espeak)")
    
    # Check web files
    from pathlib import Path
    web_dir = Path("src/voice_chat/web")
    print(f"  Checking web UI...", end=" ")
    if (web_dir / "index.html").exists():
        print("✅ Found")
    else:
        print(f"⚠️  Not found at {web_dir}")
    
    print(f"\n  Configuration:")
    print(f"    LLM Model:    {args.llm}")
    print(f"    Whisper:      {args.whisper}")
    print(f"    TTS Engine:   {args.tts}")
    print(f"    Server:       http://localhost:{args.port}")
    print(f"\n  📱 Open in your browser: http://localhost:{args.port}")
    print("=" * 60 + "\n")
    
    # Start server
    from src.voice_chat.chat_server import start_server
    
    config = {
        'llm_model': args.llm,
        'whisper_model': args.whisper,
        'tts_engine': args.tts,
    }
    
    start_server(host=args.host, port=args.port, config=config)


if __name__ == "__main__":
    main()