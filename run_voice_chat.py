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
    
    # Server
    parser.add_argument('--host', default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    
    # LLM
    parser.add_argument('--llm', default='translategemma', help='Ollama model name')
    parser.add_argument('--ollama-host', default=None,
                        help='Ollama host URL (auto-detect if not set)')
    
    # Whisper
    parser.add_argument('--whisper', default='base',
                        choices=['tiny', 'base', 'small', 'medium', 'large-v3'],
                        help='Whisper model size')
    
    # IndicF5 TTS
    parser.add_argument('--indicf5-url', default='http://localhost:8000',
                        help='IndicF5 TTS API base URL')
    parser.add_argument('--indicf5-voice', default='',
                        help='Default reference voice key (auto-picks first if empty)')
    parser.add_argument('--indicf5-sample-rate', type=int, default=24000,
                        help='Output audio sample rate')
    parser.add_argument('--indicf5-seed', type=int, default=-1,
                        help='Random seed for reproducible generation (-1 = random)')
    parser.add_argument('--indicf5-timeout', type=int, default=120,
                        help='API request timeout in seconds')
    
    args = parser.parse_args()
    
    # Set Ollama host if provided
    if args.ollama_host:
        os.environ['OLLAMA_HOST'] = args.ollama_host
    
    # ── Pre-flight checks ──────────────────────────────────────────────
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
    
    # Check IndicF5 TTS server
    print(f"  Checking IndicF5 TTS at {args.indicf5_url}...", end=" ")
    try:
        from src.voice_chat.tts_engine import IndicF5Client
        client = IndicF5Client(
            base_url=args.indicf5_url,
            timeout=10,
        )
        if client.health_check():
            print("✅ Server healthy")
            
            # List available voices
            voices = client.list_reference_voices()
            if voices:
                print(f"  Available voices ({len(voices)}):")
                for key, v in voices.items():
                    marker = " 👈 (selected)" if key == args.indicf5_voice else ""
                    print(f"    • {key} — author: {v.author}, model: {v.model}{marker}")
                
                if args.indicf5_voice and args.indicf5_voice not in voices:
                    print(f"  ⚠️  Voice '{args.indicf5_voice}' not found, will auto-pick")
            else:
                print("  ⚠️  No reference voices uploaded yet")
                print("     Upload one via the API or engine.upload_reference_voice()")
        else:
            print("⚠️  Server returned unhealthy status")
        client.close()
    except Exception as e:
        print(f"⚠️  Not reachable — {e}")
        print(f"  Make sure the IndicF5 server is running at {args.indicf5_url}")
    
    # Check web files
    from pathlib import Path
    web_dir = Path("src/voice_chat/web")
    print(f"  Checking web UI...", end=" ")
    if (web_dir / "index.html").exists():
        print("✅ Found")
    else:
        print(f"⚠️  Not found at {web_dir}")
    
    # ── Configuration summary ──────────────────────────────────────────
    print(f"\n  Configuration:")
    print(f"    LLM Model:       {args.llm}")
    print(f"    Whisper:         {args.whisper}")
    print(f"    TTS Engine:      IndicF5")
    print(f"    IndicF5 URL:     {args.indicf5_url}")
    print(f"    IndicF5 Voice:   {args.indicf5_voice or '(auto)'}")
    print(f"    Sample Rate:     {args.indicf5_sample_rate}")
    print(f"    Seed:            {args.indicf5_seed}")
    print(f"    Server:          http://localhost:{args.port}")
    print(f"\n  📱 Open in your browser: http://localhost:{args.port}")
    print("=" * 60 + "\n")
    
    # ── Start server ───────────────────────────────────────────────────
    from src.voice_chat.chat_server import start_server
    
    config = {
        'llm_model': args.llm,
        'whisper_model': args.whisper,
        'tts_engine': 'indicf5',
        'indicf5_base_url': args.indicf5_url,
        'indicf5_reference_voice_key': args.indicf5_voice,
        'indicf5_sample_rate': args.indicf5_sample_rate,
        'indicf5_seed': args.indicf5_seed,
        'indicf5_timeout': args.indicf5_timeout,
    }
    
    start_server(host=args.host, port=args.port, config=config)


if __name__ == "__main__":
    main()