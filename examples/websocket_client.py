# examples/websocket_client.py
"""
WebSocket client example for VAD service
"""
import asyncio
import websockets
import json
import numpy as np
import sounddevice as sd


async def main():
    uri = "ws://localhost:8001"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to VAD server")
        
        # Configure
        await websocket.send(json.dumps({
            "type": "configure",
            "sample_rate": 16000,
            "channels": 1
        }))
        
        response = await websocket.recv()
        print(f"Configuration response: {response}")
        
        # Start audio capture
        sample_rate = 16000
        chunk_size = int(sample_rate * 0.03)  # 30ms
        
        audio_queue = asyncio.Queue()
        
        def audio_callback(indata, frames, time, status):
            if status:
                print(f"Audio status: {status}")
            audio_queue.put_nowait(indata.copy())
        
        stream = sd.InputStream(
            samplerate=sample_rate,
            channels=1,
            dtype=np.int16,
            blocksize=chunk_size,
            callback=audio_callback
        )
        
        async def send_audio():
            while True:
                audio = await audio_queue.get()
                await websocket.send(audio.tobytes())
        
        async def receive_events():
            async for message in websocket:
                event = json.loads(message)
                
                if event["type"] == "speech_start":
                    print(f"\n🎤 Speech started at {event['data']['timestamp_ms']:.0f}ms")
                elif event["type"] == "speech_end":
                    print(f"🔇 Speech ended - Duration: {event['data']['duration_ms']:.0f}ms")
                elif event["type"] == "vad_result":
                    prob = event["data"]["probability"]
                    indicator = "🟢" if event["data"]["is_speech"] else "⚫"
                    print(f"\r{indicator} Probability: {prob:.2f}", end="")
        
        print("\nListening... Press Ctrl+C to stop\n")
        
        with stream:
            try:
                await asyncio.gather(
                    send_audio(),
                    receive_events()
                )
            except KeyboardInterrupt:
                print("\nDisconnecting...")


if __name__ == "__main__":
    asyncio.run(main())