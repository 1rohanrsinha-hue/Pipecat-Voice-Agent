# 🤖 Pipecat Voice Agent

A real-time conversational AI voice agent built with **Pipecat**, running entirely on a **RunPod RTX 3090 GPU**. No paid APIs — everything runs locally.

**Target latency: < 800ms end-to-end**

---

## 🏗️ Architecture

```
Browser (mic) → LiveKit WebRTC → RunPod RTX 3090
                                        ↓
                              FastWhisper STT (~140ms)
                                        ↓
                              Ollama Llama 3.2 3B (~350ms)
                                        ↓
                              Kokoro TTS on GPU (~300ms)
                                        ↓
                              Back to browser (~600ms total)
```

### Stack
| Component | Tool | Purpose |
|---|---|---|
| **Framework** | Pipecat 1.2.1 | Pipeline orchestration |
| **Transport** | LiveKit WebRTC | Browser to Server audio |
| **STT** | FastWhisper distil-large-v3 | Speech to text |
| **VAD** | Silero VAD | Voice activity detection |
| **LLM** | Ollama Llama 3.2 3B | Language model |
| **TTS** | Kokoro ONNX (GPU) | Text to speech |
| **GPU** | RunPod RTX 3090 24GB | Inference |

---

## Latency Results

| Component | Latency |
|---|---|
| FastWhisper STT | ~130-250ms |
| Ollama LLM | ~250-600ms |
| Kokoro TTS (GPU) | ~180-800ms |
| **Total** | **~500-800ms** |

---

## Setup

### 1. RunPod Instance
- GPU: **RTX 3090** (24GB VRAM)
- Template: RunPod PyTorch 2.1 (CUDA 12.1)
- Disk: 50GB

### 2. Install Dependencies
```bash
apt-get update && apt-get install -y ffmpeg portaudio19-dev libsndfile1 curl zstd
pip install faster-whisper kokoro-onnx soundfile aiohttp numpy torch
pip install "pipecat-ai[livekit,silero]"
pip install livekit livekit-api
pip install onnxruntime-gpu nvidia-cudnn-cu12
```

### 3. Install Ollama + Model
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &
ollama pull llama3.2:3b
```

### 4. Download Kokoro Models
```bash
mkdir -p models/kokoro
wget -O models/kokoro/kokoro-v0_19.onnx \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/kokoro-v0_19.onnx
wget -O models/kokoro/voices.bin \
  https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files/voices.bin
```

### 5. LiveKit Credentials
Sign up free at livekit.io and get your LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET.

---

## Running

```bash
export ONNX_PROVIDER=CUDAExecutionProvider
export LD_LIBRARY_PATH=/usr/local/lib/python3.11/dist-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export LIVEKIT_URL="wss://your-project.livekit.cloud"
export LIVEKIT_API_KEY="your_api_key"
export LIVEKIT_API_SECRET="your_api_secret"

ollama serve &
python3 main.py
```

The terminal prints a shareable link. Send it to anyone — they open it in Chrome, allow microphone, and start talking to the AI.

---

## How It Works

### Pipecat Pipeline
Each component is a Pipecat FrameProcessor. Data flows as typed frames:

```
transport.input()          <- LiveKit mic audio
FastWhisperSTTProcessor    <- AudioRawFrame -> TranscriptionFrame
OllamaLLMProcessor         <- TranscriptionFrame -> TextFrame
KokoroTTSProcessor         <- TextFrame -> TTSAudioRawFrame
LatencyLogger              <- logs ms per stage
transport.output()         <- audio back to browser
```

### VAD
Silero VAD runs inside the STT processor. Only collects audio when speech is detected. When silence exceeds 800ms it transcribes. This prevents hallucinations from background noise and ensures clean turn-taking.

### GPU Acceleration
- FastWhisper: CUDA float16
- Ollama: all 29 layers on GPU
- Kokoro: CUDAExecutionProvider via ONNX_PROVIDER env variable

---

## Files

| File | Purpose |
|---|---|
| `main.py` | Main Pipecat pipeline |
| `server.py` | HTTP server for index.html and LiveKit tokens |
| `index.html` | Browser frontend with echo cancellation |

---

## Author

**Rohan Sinha**
https://github.com/1rohanrsinha-hue
