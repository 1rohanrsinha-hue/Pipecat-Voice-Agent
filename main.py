import asyncio, time, json, numpy as np, aiohttp, os, torch
from faster_whisper import WhisperModel
from kokoro_onnx import Kokoro
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.frames.frames import UserAudioRawFrame, TextFrame, TranscriptionFrame, TTSAudioRawFrame
from pipecat.processors.frame_processor import FrameProcessor
from pipecat.transports.livekit.transport import LiveKitTransport, LiveKitParams
from livekit import api
import soundfile as sf
import io

LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "")
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
KOKORO_MODEL = "models/kokoro/kokoro-v0_19.onnx"
KOKORO_VOICES = "models/kokoro/voices.bin"
KOKORO_VOICE = "af_bella"
ROOM_NAME = "voice-agent"
SYSTEM_PROMPT = "You are a helpful concise voice assistant. Keep responses to 1-3 sentences. No bullet points or markdown. Just natural conversational speech."

SILENCE_THRESHOLD = 0.1
SILENCE_DURATION = 0.8
MIN_SPEECH_DURATION = 0.3


class FastWhisperSTTProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._bot_speaking = False
        self._audio_buffer = []
        self._silence_frames = 0
        self._is_speaking = False
        self._sample_rate = 16000

        print("Loading FastWhisper...")
        self.whisper = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
        print("FastWhisper ready")

        print("Loading Silero VAD...")
        self.vad_model, _ = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            trust_repo=True
        )
        self.vad_model.eval()
        print("Silero VAD ready")

    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        chunk = audio_chunk[:512] if len(audio_chunk) >= 512 else np.pad(audio_chunk, (0, 512 - len(audio_chunk)))
        tensor = torch.FloatTensor(chunk)
        with torch.no_grad():
            confidence = self.vad_model(tensor, self._sample_rate).item()
        return confidence > SILENCE_THRESHOLD

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)

        if isinstance(frame, UserAudioRawFrame):
            if self._bot_speaking:
                return
            audio_np = np.frombuffer(frame.audio, dtype=np.int16).astype(np.float32) / 32768.0
            speech_detected = self._is_speech(audio_np)

            if speech_detected:
                if not self._is_speaking:
                    self._is_speaking = True
                    self._audio_buffer = []
                    self._silence_frames = 0
                    print("  🎤 Speaking...")
                self._audio_buffer.append(audio_np)
                self._silence_frames = 0
            else:
                if self._is_speaking:
                    self._silence_frames += 1
                    self._audio_buffer.append(audio_np)
                    silence_secs = self._silence_frames * (len(audio_np) / self._sample_rate)
                    if silence_secs >= SILENCE_DURATION:
                        self._is_speaking = False
                        await self._transcribe()
            return

        await self.push_frame(frame, direction)

    async def _transcribe(self):
        if not self._audio_buffer:
            return
        audio_np = np.concatenate(self._audio_buffer)
        self._audio_buffer = []
        if len(audio_np) / self._sample_rate < MIN_SPEECH_DURATION:
            return
        t0 = time.time()
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None, lambda: self.whisper.transcribe(audio_np, language="en", vad_filter=False)
        )
        text = " ".join(s.text for s in segments).strip()
        if text:
            print(f"  STT ({int((time.time()-t0)*1000)}ms): {text}")
            await self.push_frame(TranscriptionFrame(text=text, user_id="user", timestamp=str(time.time())))


class OllamaLLMProcessor(FrameProcessor):
    def __init__(self):
        super().__init__()
        self.history = [{"role": "system", "content": SYSTEM_PROMPT}]

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame):
            t0 = time.time()
            self.history.append({"role": "user", "content": frame.text})
            reply = await self._call_ollama()
            self.history.append({"role": "assistant", "content": reply})
            print(f"  LLM ({int((time.time()-t0)*1000)}ms): {reply[:60]}")
            await self.push_frame(TextFrame(text=reply))
        else:
            await self.push_frame(frame, direction)

    async def _call_ollama(self):
        result = ""
        async with aiohttp.ClientSession() as session:
            async with session.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL, "messages": self.history,
                "stream": True, "options": {"temperature": 0.7, "num_predict": 80}
            }) as resp:
                async for line in resp.content:
                    line = line.decode().strip()
                    if not line: continue
                    try:
                        chunk = json.loads(line)
                        result += chunk.get("message", {}).get("content", "")
                    except: pass
        return result.strip()


class KokoroTTSProcessor(FrameProcessor):
    def __init__(self, stt_processor):
        super().__init__()
        self.stt = stt_processor
        print("Loading Kokoro...")
        self.kokoro = Kokoro(KOKORO_MODEL, KOKORO_VOICES)
        print("Kokoro ready")

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TextFrame):
            t0 = time.time()
            self.stt._bot_speaking = True
            loop = asyncio.get_event_loop()
            # Split into sentences for faster first audio
            sentences = [s.strip() for s in frame.text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            for sentence in sentences:
                samples, sr = await loop.run_in_executor(
                    None, lambda s=sentence: self.kokoro.create(s, voice=KOKORO_VOICE, speed=1.1, lang="en-us")
                )
                audio_bytes = (samples * 32768).astype(np.int16).tobytes()
                await self.push_frame(TTSAudioRawFrame(audio=audio_bytes, sample_rate=sr, num_channels=1))
            print(f"  TTS ({int((time.time()-t0)*1000)}ms)")
            await asyncio.sleep(0.3)
            self.stt._bot_speaking = False
        else:
            await self.push_frame(frame, direction)


class LatencyLogger(FrameProcessor):
    def __init__(self):
        super().__init__()
        self._start = None

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, TranscriptionFrame): self._start = time.time()
        if isinstance(frame, TTSAudioRawFrame) and self._start:
            ms = int((time.time() - self._start) * 1000)
            print(f"  {'✅ OK' if ms < 800 else '⚠️  SLOW'} Total latency: {ms}ms\n")
            self._start = None
        await self.push_frame(frame, direction)


def generate_token(room: str, identity: str) -> str:
    grants = api.VideoGrants(room_join=True, room=room, can_publish=True, can_subscribe=True)
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity(identity).with_grants(grants)
    return token.to_jwt()


async def main():
    print("\n" + "="*55)
    print("  Pipecat Voice Agent — LiveKit WebRTC")
    print("="*55)

    bot_token = generate_token(ROOM_NAME, "ai-bot")
    user_token = generate_token(ROOM_NAME, "user")

    join_url = f"https://meet.livekit.io/custom?liveKitUrl={LIVEKIT_URL}&token={user_token}"
    print(f"\n{'='*55}")
    print(f"  SHARE THIS LINK WITH YOUR INTERVIEWER:")
    print(f"  {join_url}")
    print(f"{'='*55}\n")

    transport = LiveKitTransport(
        url=LIVEKIT_URL, token=bot_token, room_name=ROOM_NAME,
        params=LiveKitParams(audio_in_enabled=True, audio_out_enabled=True)
    )

    stt = FastWhisperSTTProcessor()
    pipeline = Pipeline([
        transport.input(), stt, OllamaLLMProcessor(),
        KokoroTTSProcessor(stt), LatencyLogger(), transport.output(),
    ])

    runner = PipelineRunner()
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=False))

    @transport.event_handler("on_participant_joined")
    async def on_joined(transport, participant):
        print(f"  👋 Participant joined — start talking!\n")

    @transport.event_handler("on_participant_left")
    async def on_left(transport, participant, reason):
        print(f"  👋 Participant left.")
        await task.cancel()

    await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())