import numpy as np
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import urllib.request
from pathlib import Path
from kokoro_onnx import Kokoro


# ── Model file URLs (GitHub releases) ────────────────────────────────────────
_MODEL_URL  = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
_VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"

_DEFAULT_MODEL_PATH  = Path(__file__).parent / "kokoro-v1.0.onnx"
_DEFAULT_VOICES_PATH = Path(__file__).parent / "voices-v1.0.bin"


def _download_if_missing(url: str, dest: Path):
    if dest.exists():
        return
    print(f"📥 Downloading {dest.name} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"✅ Saved to {dest}")


class TextToSpeech:
    """
    Local Text-to-Speech using Kokoro ONNX.

    Install:
        pip install kokoro-onnx soundfile sounddevice

    Model files (~310 MB total) are auto-downloaded on first run and cached
    next to this script. You can also pass explicit paths via model_path /
    voices_path if you keep them elsewhere.
    """

    # Available voices — English (en) and others
    VOICES = {
        # American English
        "af_heart":   "American Female - Heart (warm)",
        "af_bella":   "American Female - Bella (soft)",
        "af_sarah":   "American Female - Sarah (clear)",
        "af_nicole":  "American Female - Nicole (professional)",
        "am_adam":    "American Male   - Adam (deep)",
        "am_michael": "American Male   - Michael (natural)",
        # British English
        "bf_emma":    "British Female  - Emma (elegant)",
        "bf_isabella":"British Female  - Isabella (warm)",
        "bm_george":  "British Male    - George (authoritative)",
        "bm_lewis":   "British Male    - Lewis (casual)",
    }

    def __init__(
        self,
        voice: str = "af_heart",
        speed: float = 1.0,
        lang: str = "en-us",
        sample_rate: int = 24000,
        model_path: str | Path | None = None,
        voices_path: str | Path | None = None,
    ):
        """
        Args:
            voice:        Voice ID from VOICES dict above.
            speed:        Speech speed multiplier (0.5 – 2.0).
            lang:         Language code e.g. 'en-us', 'en-gb'.
            sample_rate:  Output sample rate (Kokoro native = 24000 Hz).
            model_path:   Path to kokoro-v1.0.onnx  (auto-downloaded if None).
            voices_path:  Path to voices-v1.0.bin   (auto-downloaded if None).
        """
        if voice not in self.VOICES:
            raise ValueError(
                f"Unknown voice '{voice}'. Choose from:\n"
                + "\n".join(f"  {k}: {v}" for k, v in self.VOICES.items())
            )

        # Resolve / download model files
        model_path  = Path(model_path)  if model_path  else _DEFAULT_MODEL_PATH
        voices_path = Path(voices_path) if voices_path else _DEFAULT_VOICES_PATH
        _download_if_missing(_MODEL_URL,  model_path)
        _download_if_missing(_VOICES_URL, voices_path)

        print(f"Loading Kokoro TTS  (voice={voice}, speed={speed})...")
        self.kokoro = Kokoro(str(model_path), str(voices_path))
        self.voice = voice
        self.speed = speed
        self.lang = lang
        self.sample_rate = sample_rate
        print("Kokoro TTS ready.")

    # ── core synthesis ────────────────────────────────────────────────────────

    def synthesize(self, text: str) -> np.ndarray:
        """
        Convert text → audio samples (float32, mono).

        Returns:
            np.ndarray of shape (N,) with values in [-1, 1].
        """
        if not text or not text.strip():
            raise ValueError("Input text is empty.")

        samples, sample_rate = self.kokoro.create(
            text,
            voice=self.voice,
            speed=self.speed,
            lang=self.lang,
        )
        # Kokoro returns (samples, sr); store actual sr in case it differs
        self._last_sr = sample_rate
        return samples

    # ── playback ──────────────────────────────────────────────────────────────

    def speak(self, text: str, blocking: bool = True) -> np.ndarray:
        """
        Synthesize and play audio through the default output device.

        Args:
            text:     Text to speak.
            blocking: If True, wait until playback finishes before returning.

        Returns:
            The raw audio samples (useful for saving later).
        """
        print(f"🔊 Speaking: \"{text[:80]}{'...' if len(text) > 80 else ''}\"")
        samples = self.synthesize(text)
        sr = getattr(self, "_last_sr", self.sample_rate)
        sd.play(samples, samplerate=sr)
        if blocking:
            sd.wait()
        return samples

    def stop(self):
        """Stop any currently playing audio."""
        sd.stop()

    # ── file I/O ──────────────────────────────────────────────────────────────

    def save(self, text: str, path: str) -> str:
        """
        Synthesize text and save to a WAV file.

        Args:
            text: Text to synthesize.
            path: Output file path (should end in .wav).

        Returns:
            Absolute path to the saved file.
        """
        samples = self.synthesize(text)
        sr = getattr(self, "_last_sr", self.sample_rate)
        sf.write(path, samples, sr)
        print(f"💾 Saved to: {path}")
        return os.path.abspath(path)

    def speak_and_save(self, text: str, path: str, blocking: bool = True) -> str:
        """Speak aloud and simultaneously save to file."""
        samples = self.synthesize(text)
        sr = getattr(self, "_last_sr", self.sample_rate)
        sf.write(path, samples, sr)
        print(f"🔊 Speaking & saving to: {path}")
        sd.play(samples, samplerate=sr)
        if blocking:
            sd.wait()
        return os.path.abspath(path)

    # ── convenience ───────────────────────────────────────────────────────────

    def change_voice(self, voice: str):
        """Hot-swap the voice without reloading the model."""
        if voice not in self.VOICES:
            raise ValueError(f"Unknown voice '{voice}'.")
        self.voice = voice
        print(f"🎙️  Voice changed to: {voice} — {self.VOICES[voice]}")

    def change_speed(self, speed: float):
        """Adjust speech speed (0.5 = half, 2.0 = double)."""
        if not (0.3 <= speed <= 2.5):
            raise ValueError("Speed must be between 0.3 and 2.5.")
        self.speed = speed
        print(f"⏩ Speed set to {speed}x")

    @classmethod
    def list_voices(cls):
        """Print all available voices."""
        print("\nAvailable Kokoro voices:")
        for voice_id, desc in cls.VOICES.items():
            print(f"  {voice_id:<14} {desc}")
        print()


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # List all voices
    TextToSpeech.list_voices()

    # Init with a warm American female voice
    tts = TextToSpeech(voice="af_heart", speed=1.0)

    # Basic speak
    tts.speak("Hello! I am Kokoro, your local text to speech engine.")

    # Save to file
    tts.save("This audio was saved to a file.", "output.wav")

    # Change voice mid-session
    tts.change_voice("bm_george")
    tts.speak("And now I sound like a British gentleman.")

    # Speed demo
    tts.change_speed(1.3)
    tts.speak("I can also speak faster if you need me to.")