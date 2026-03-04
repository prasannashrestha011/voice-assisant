import sounddevice as sd
import numpy as np
import whisper
import tempfile
import scipy.io.wavfile as wav


class VoiceTranscriber:
    def __init__(self, model_size: str = "small", sample_rate: int = 16000):
        print(f"Loading Whisper {model_size} model...")
        self.model = whisper.load_model(model_size)
        self.sample_rate = sample_rate
        print(f"Model ready.")

    def record(self, duration: int = 5) -> np.ndarray:
        print(f"Recording for {duration} seconds... speak now!")
        audio = sd.rec(
            int(duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32"
        )
        sd.wait()
        print("Recording done.")
        return audio.flatten()

    def transcribe(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wav.write(f.name, self.sample_rate, (audio * 32767).astype(np.int16))
            result = self.model.transcribe(f.name)
        return result["text"].strip()

    def listen(self, duration: int = 5) -> str:
        audio = self.record(duration=duration)
        print("Transcribing...")
        text = self.transcribe(audio)
        print(f"You said: {text}")
        return text

    def listen_loop(self, duration: int = 5, stop_phrase: str = "stop"):
        """Keep listening until stop_phrase is spoken."""
        print(f"Listening loop started. Say '{stop_phrase}' to quit.")
        while True:
            text = self.listen(duration=duration)
            yield text
            if stop_phrase.lower() in text.lower():
                print("Stop phrase detected. Exiting.")
                break


if __name__ == "__main__":
    transcriber = VoiceTranscriber(model_size="small")

    # Single shot
    text = transcriber.listen(duration=20)

    # Loop mode
    for text in transcriber.listen_loop(duration=5):
        print(f"Got: {text}")