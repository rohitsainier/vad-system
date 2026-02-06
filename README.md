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

#### Using environment.yml (Recommended)

```bash
# Clone repository
git clone https://github.com/example/vad-system.git
cd vad-system

# Create conda environment from file
conda env create -f environment.yml

# Activate environment
conda activate vad-system

# Verify installation
python -c "from src.core.vad_engine import VADEngine; print('Installation successful!')"