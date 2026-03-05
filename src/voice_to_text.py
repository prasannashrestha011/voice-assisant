import sounddevice as sd
import numpy as np
import whisper
import tempfile
import scipy.io.wavfile as wav
import torch
import os

# ── Silero VAD loader (cached after first call) ───────────────────────────────
_vad_model = None
_get_speech_ts = None

def _load_vad():
    global _vad_model, _get_speech_ts
    if _vad_model is None:
        print("Loading Silero VAD model...")
        _vad_model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _get_speech_ts = utils[0]   # get_speech_timestamps
        print("Silero VAD ready.")
    return _vad_model, _get_speech_ts


class VoiceTranscriber:
    def __init__(
        self,
        model_size: str = "small",
        sample_rate: int = 16000,
        # VAD knobs
        vad_threshold: float = 0.5,       # confidence to call a chunk "speech"
        min_speech_ms: int = 300,          # ignore bursts shorter than this
        min_silence_ms: int = 400,         # gap needed to split segments
        speech_pad_ms: int = 100,          # padding kept around each speech chunk
        min_audio_ratio: float = 0.10,     # skip transcription if <10 % is speech
    ):
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        self.sample_rate = sample_rate
        self.vad_threshold = vad_threshold
        self.min_speech_ms = min_speech_ms
        self.min_silence_ms = min_silence_ms
        self.speech_pad_ms = speech_pad_ms
        self.min_audio_ratio = min_audio_ratio
        _load_vad()
        print("VoiceTranscriber ready.")

    # ── low-level helpers ─────────────────────────────────────────────────────

    def record(self, duration: int = 5) -> np.ndarray:
        print(f"\n🎙️  Recording {duration}s — speak now!")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
        )
        sd.wait()
        print("⏹️  Recording done.")
        return audio.flatten()

    def has_speech(self, audio: np.ndarray) -> bool:
        """Return True only if Silero VAD detects meaningful speech."""
        vad_model, get_speech_ts = _load_vad()
        tensor = torch.from_numpy(audio)

        timestamps = get_speech_ts(
            tensor,
            vad_model,
            sampling_rate=self.sample_rate,
            threshold=self.vad_threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

        if not timestamps:
            return False

        # Sum total speech samples
        speech_samples = sum(t["end"] - t["start"] for t in timestamps)
        ratio = speech_samples / len(audio)
        print(f"🔊 Speech ratio: {ratio:.1%}  (segments: {len(timestamps)})")
        return ratio >= self.min_audio_ratio

    def extract_speech_only(self, audio: np.ndarray) -> np.ndarray:
        """Concatenate only the speech segments (strips silence)."""
        vad_model, get_speech_ts = _load_vad()
        tensor = torch.from_numpy(audio)

        timestamps = get_speech_ts(
            tensor,
            vad_model,
            sampling_rate=self.sample_rate,
            threshold=self.vad_threshold,
            min_speech_duration_ms=self.min_speech_ms,
            min_silence_duration_ms=self.min_silence_ms,
            speech_pad_ms=self.speech_pad_ms,
        )

        if not timestamps:
            return np.array([], dtype=np.float32)

        chunks = [audio[t["start"]: t["end"]] for t in timestamps]
        return np.concatenate(chunks)

    def transcribe(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_path = f.name
            wav.write(tmp_path, self.sample_rate, (audio * 32767).astype(np.int16))
        result = self.model.transcribe(tmp_path)
        os.unlink(tmp_path)
        return result["text"].strip()


    def listen(self, duration: int = 5) -> str | None:
        """
        Record, run VAD, transcribe only if speech detected.
        Returns transcribed text, or None if no speech found.
        """
        audio = self.record(duration=duration)

        if not self.has_speech(audio):
            print("🔇 No speech detected — skipping transcription.")
            return None

        print("🔄 Transcribing...")
        speech_audio = self.extract_speech_only(audio)
        text = self.transcribe(speech_audio)
        print(f"🗣️  You said: \"{text}\"")
        return text

    def listen_loop(self, duration: int = 5, stop_phrase: str = "stop"):
        """
        Keep listening until stop_phrase is spoken.
        Silent recordings are skipped automatically.
        """
        print(f"🔁 Listening loop started. Say '{stop_phrase}' to quit.\n")
        while True:
            text = self.listen(duration=duration)

            if text is None:
                continue                       # silent — try again

            yield text

            if stop_phrase.lower() in text.lower():
                print("🛑 Stop phrase detected. Exiting.")
                break

