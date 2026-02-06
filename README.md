# Production VAD System

A production-ready Voice Activity Detection (VAD) system with multiple backends, real-time streaming support, and comprehensive APIs.

## Features

- **Multiple VAD Backends**
  - Silero VAD (Neural network, high accuracy)
  - WebRTC VAD (Fast, lightweight)
  - Energy-based VAD (Simple, minimal CPU)
  - Hybrid VAD (Combines all backends)

- **Real-time Processing**
  - Microphone input
  - WebSocket streaming
  - Low-latency detection

- **Production Ready**
  - REST API
  - WebSocket server
  - Docker support
  - Prometheus metrics
  - Structured logging

## Installation

### Option 1: Conda Environment (Recommended)


```bash
# Clone repository
git clone https://github.com/rohitsainier/vad-system.git
cd vad-system

conda create -y -n vad python=3.10
conda activate vad
pip install -r requirements.txt

#Verify Installation
python test_installation.py

#Test
python test_file_vad.py

