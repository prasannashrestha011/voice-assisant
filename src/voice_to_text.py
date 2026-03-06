import sounddevice as sd
import numpy as np
import whisper
import tempfile
import scipy.io.wavfile as wav
import scipy.signal as signal
import torch
import os
import queue
import sys
from .audio.echo_cancel import _find_pulse_device,_setup_echo_cancel,_teardown_echo_cancel
from .ui.console_status import STATUS_IDLE,STATUS_PROCESSING,STATUS_RECORDING,_print_result,_print_status

_vad_model = None

def _load_vad():
    global _vad_model
    if _vad_model is None:
        print("Loading Silero VAD model...")
        _vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        print("Silero VAD ready.")
    return _vad_model






class VoiceTranscriber:
    def __init__(
        self,
        model_size: str = "medium",
        vad_threshold: float = 0.5,
        silence_timeout_s: float = 1.5,
        pre_roll_chunks: int = 5,
        max_record_s: float = 30.0,
    ):
        print(f"Loading Whisper {model_size} model…")
        self.model = whisper.load_model(model_size)

        self.sample_rate       = 16000
        self.vad_threshold     = vad_threshold # count as speech if chunk probs is higher than threshold
        self.silence_timeout_s = silence_timeout_s # how long the silence duration triggers the stop recording 
        self.pre_roll_chunks   = pre_roll_chunks # 
        self.max_record_s      = max_record_s
        self.chunk_samples     = 512 # number of audio sample contain in one chunk
        self.native_rate       = 48000 # actual microphone hardware sample rate
        self.native_chunk      = int(self.chunk_samples * self.native_rate / self.sample_rate)

        _load_vad()
        print("VoiceTranscriber ready.\n")

    def _resample(self, chunk: np.ndarray) -> np.ndarray:
        if self.native_rate == self.sample_rate:
            return chunk
        return signal.resample_poly(
            chunk, up=self.sample_rate, down=self.native_rate
        ).astype(np.float32)

    def _is_speech(self, chunk: np.ndarray) -> bool:
        vad  = _load_vad()
        prob = vad(torch.from_numpy(chunk.copy()), self.sample_rate).item()
        return prob >= self.vad_threshold

    def _transcribe(self, audio: np.ndarray) -> str:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        wav.write(tmp, self.sample_rate, (audio * 32767).astype(np.int16))
        result = self.model.transcribe(tmp, language="en", task="transcribe")
        os.unlink(tmp)
        return result["text"].strip()

    def run_forever(self, stop_phrase: str = "stop recording"):
        chunk_s     = self.chunk_samples / self.sample_rate
        max_silence = int(self.silence_timeout_s / chunk_s)
        max_chunks  = int(self.max_record_s      / chunk_s)

        audio_q: queue.Queue[np.ndarray] = queue.Queue()
        original_source = _setup_echo_cancel()

        # Dynamically find device — works across machines
        device_index, _ = _find_pulse_device()
        print(f"Using device index {device_index}: {sd.query_devices(device_index)['name']}")
        print(f"🔁 Running continuously. Say \"{stop_phrase}\" to quit.\n")
        _print_status(STATUS_IDLE)

        stream = sd.InputStream(
            device=device_index,
            samplerate=self.native_rate,
            channels=1,
            dtype="float32",
            blocksize=self.native_chunk,
            latency="low",
            callback=lambda indata, f, t, s: (
                audio_q.put(self._resample(indata[:, 0].copy()))
            ),
        )

        try:
            stream.start()

            while True:
                pre_roll  = []
                in_speech = False

                while not in_speech:
                    chunk = audio_q.get()
                    pre_roll.append(chunk)
                    if len(pre_roll) > self.pre_roll_chunks:
                        pre_roll.pop(0)
                    if self._is_speech(chunk):
                        in_speech    = True
                        recording    = pre_roll.copy()
                        total_chunks = len(recording)
                        silence_cnt  = 0
                        _print_status(STATUS_RECORDING)

                while True:
                    chunk = audio_q.get()
                    recording.append(chunk)
                    total_chunks += 1
                    if self._is_speech(chunk):
                        silence_cnt = 0
                    else:
                        silence_cnt += 1
                    if silence_cnt >= max_silence or total_chunks >= max_chunks:
                        break

                _print_status(STATUS_PROCESSING)
                audio = np.concatenate(recording)
                max_val = np.max(np.abs(audio))
                if max_val > 0:
                    audio = audio / max_val * 0.95

                text = self._transcribe(audio)
                _print_result(text)
                _print_status(STATUS_IDLE)

                if stop_phrase.lower() in text.lower():
                    sys.stdout.write("🛑 Stop phrase detected. Goodbye!\n")
                    sys.stdout.flush()
                    break

        except KeyboardInterrupt:
            sys.stdout.write("\n⚠️  Interrupted.\n")

        finally:
            stream.stop()   # stop accepting new audio
            stream.close()  # release the mic hardware
            _teardown_echo_cancel(original_source)
            sys.stdout.write("🎤 Microphone released.\n")
            sys.stdout.flush()


if __name__ == "__main__":
    t = VoiceTranscriber(model_size="medium")
    t.run_forever(stop_phrase="stop recording")