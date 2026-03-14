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
import threading
from typing import Callable, Optional

import atexit
import signal as sys_signal

from pathlib import Path

from .logger.logger import _logger
from .audio.echo_cancel import _find_pulse_device,_setup_echo_cancel,_teardown_echo_cancel
from .ui.console_status import STATUS_IDLE,STATUS_PROCESSING,STATUS_RECORDING,_print_result,_print_status

_vad_model = None

def _load_vad():
    global _vad_model
    if _vad_model is None:
        _logger.debug("Loading Silero VAD model...")
        _vad_model, _ = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            trust_repo=True,
        )
        _logger.debug("Silero VAD ready.")
    return _vad_model






class VoiceTranscriber:
    def __init__(
        self,
        tts=None,
        on_interrupt: Optional[Callable[[], None]] = None,
        model_size: str = "small",
        vad_threshold: float = 0.5,
        silence_timeout_s: float = 1.5,
        pre_roll_chunks: int = 8,
        interrupt_speech_chunks: int = 3,
        max_record_s: float = 30.0,
    ):
        _logger.debug(f"Loading Whisper {model_size} model…")
        self.model = whisper.load_model(model_size)
        self.tts=tts
        self.on_interrupt = on_interrupt

        self.sample_rate       = 16000
        self.vad_threshold     = vad_threshold # count as speech if chunk probs is higher than threshold
        self.silence_timeout_s = silence_timeout_s # how long the silence duration triggers the stop recording 
        self.pre_roll_chunks   = pre_roll_chunks # 
        self.interrupt_speech_chunks = interrupt_speech_chunks
        self.max_record_s      = max_record_s # maximum recording seconds
        self.chunk_samples     = 512 # number of audio sample contain in one chunk
        self.native_rate       = 48000 # actual microphone hardware sample rate per sec
        self.native_chunk      = int(self.chunk_samples * self.native_rate / self.sample_rate) # no. of sample each chunks holds
        self._original_source = _setup_echo_cancel()
        device_index, _       = _find_pulse_device()
        self._audio_q: queue.Queue[np.ndarray] = queue.Queue()
        self._stream=None
        self._is_running=True

        self._register_signal_handlers()
        atexit.register(self.shutdown)

        self._stream = sd.InputStream(
            device=device_index,
            samplerate=self.native_rate,
            channels=1,
            dtype="float32",
            blocksize=self.native_chunk,
            latency="low",
            callback=lambda indata, f, t, s: (
                self._audio_q.put(self._resample(indata[:, 0].copy()))
            ),
        )
        _load_vad()
        self._stream.start()
        _logger.debug("VoiceTranscriber ready.\n")

    def _resample(self, chunk: np.ndarray) -> np.ndarray:
        """converts the device sample rate to sample rate required by whisper and vad model"""
        if self.native_rate == self.sample_rate:
            return chunk
        return signal.resample_poly(
            chunk, up=self.sample_rate, down=self.native_rate
        ).astype(np.float32)

    def _is_speech(self, chunk: np.ndarray) -> bool:
        """checks if the deteched sound is a interrupt or actual speech"""
        vad  = _load_vad()
        prob = vad(torch.from_numpy(chunk.copy()), self.sample_rate).item()
        return prob >= self.vad_threshold

    def _transcribe(self, audio: np.ndarray) -> str:
        """converts text to speech"""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp = f.name
        wav.write(tmp, self.sample_rate, (audio * 32767).astype(np.int16))
        result = self.model.transcribe(tmp, language="en", task="transcribe")
        os.unlink(tmp)
        return result["text"].strip()
    


    def listen(self) -> str:
        _logger.debug("[listen] started")

        chunk_s     = self.chunk_samples / self.sample_rate
        max_silence = int(self.silence_timeout_s / chunk_s)
        max_chunks  = int(self.max_record_s / chunk_s)

        _logger.debug("[listen] chunk_s:", chunk_s)
        _logger.debug("[listen] max_silence:", max_silence)
        _logger.debug("[listen] max_chunks:", max_chunks)

        # Phase 0 — drain stale audio from TTS playback
        _logger.debug("[listen] phase 0 draining audio queue")
        drained = 0
        try:
            while True:
                self._audio_q.get_nowait()
                drained += 1
        except queue.Empty:
            pass
        if drained > 0:
            _logger.debug(f"[listen] drained {drained} stale audio chunks")

        recording = []
        pre_roll = []
        interrupted_tts = False
        interrupted_notified = False
        speech_run = 0
        barge_in_run = 0
        # Faster barge-in interrupt than “start recording” threshold.
        barge_in_chunks = min(2, self.interrupt_speech_chunks)

        _logger.debug(40*"=")
        # Phase 1 — wait for speech
        _logger.debug("[listen] phase 1 waiting for speech")
        _print_status(STATUS_IDLE)

        while True:
            chunk = self._audio_q.get()

            if self._is_speech(chunk):

                pre_roll.append(chunk)
                speech_run += 1
                barge_in_run += 1

                # If TTS is currently active, interrupt it quickly on sustained speech.
                if (
                    self.tts is not None
                    and self.tts.play_evnt.is_set()
                    and barge_in_run >= barge_in_chunks
                ):
                    if not interrupted_notified and self.on_interrupt is not None:
                        try:
                            self.on_interrupt()
                        except Exception as e:
                            _logger.error("[listen] interrupt callback failed:", e)
                        interrupted_notified = True

                    if not interrupted_tts:
                        _logger.debug("[listen] interrupting TTS")
                        try:
                            self.tts.interrupt(wait=False)
                        except Exception as e:
                            _logger.error("[listen] TTS interrupt failed:", e)
                        interrupted_tts = True

                if speech_run < self.interrupt_speech_chunks:
                    if len(pre_roll) > self.pre_roll_chunks:
                        pre_roll.pop(0)
                    continue

                break

            speech_run = 0
            barge_in_run = 0
            pre_roll.append(chunk)

            if len(pre_roll) > self.pre_roll_chunks:
                pre_roll.pop(0)

        # Phase 2 — recording
        _logger.debug(40*"-")
        _logger.debug("[listen] phase 2 recording started")
        _print_status(STATUS_RECORDING)

        recording = pre_roll.copy()
        total_chunks = len(recording)
        silence_cnt = 0

        while True:
            chunk = self._audio_q.get()
            recording.append(chunk)
            total_chunks += 1

            if self._is_speech(chunk):
                silence_cnt = 0
            else:
                silence_cnt += 1


            if silence_cnt >= max_silence:
                _logger.debug("[listen] stopping recording due to silence")
                break

            if total_chunks >= max_chunks:
                _logger.debug("[listen] stopping recording due to max length")
                break

        # Phase 3 — transcription
        _logger.debug(40*"-")
        _logger.debug("[listen] phase 3 transcription")
        _print_status(STATUS_PROCESSING)

        audio = np.concatenate(recording)

        max_val = np.max(np.abs(audio))
        _logger.debug("[listen] max audio value:", max_val)

        if max_val > 0:
            audio = audio / max_val * 0.95
            _logger.debug("[listen] audio normalized")

        result = self._transcribe(audio)

        _logger.debug("[listen] transcription result:", result)
        _print_result(result)

        return result


    """methods assoicated with graceful shutdown of mic and restoration of default mic source """
    def _register_signal_handlers(self):
        """Register handlers for graceful shutdown on SIGINT (Ctrl+C) and SIGTERM."""
        if threading.current_thread() is not threading.main_thread():
            _logger.debug("[Signal] skipping handler registration (not main thread)")
            return

        def _handle_signal(signum, frame):
            sig_name = "SIGINT (Ctrl+C)" if signum == sys_signal.SIGINT else "SIGTERM"
            _logger.debug(f"\n[Signal] {sig_name} received — shutting down mic gracefully...")
            self.shutdown()
            sys.exit(0)

        sys_signal.signal(sys_signal.SIGINT,  _handle_signal)
        sys_signal.signal(sys_signal.SIGTERM, _handle_signal)

    def shutdown(self):
        """Gracefully stop the audio stream and tear down echo cancellation."""
        if not self._is_running:
            return  # Prevent double-shutdown

        self._is_running = False
        _logger.debug("\n[Shutdown] Stopping audio stream...")

        try:
            if self._stream is not None:
                if self._stream.active:
                    self._stream.stop()
                self._stream.close()
                self._stream = None
                _logger.debug("[Shutdown] Audio stream closed.")
        except Exception as e:
            _logger.debug(f"[Shutdown] Warning: error closing stream: {e}")

        try:
            _teardown_echo_cancel(self._original_source)
            _logger.debug("[Shutdown] Echo cancellation restored.")
        except Exception as e:
            _logger.debug(f"[Shutdown] Warning: error restoring echo cancel: {e}")

        # Drain the audio queue to unblock any waiting .get() calls
        try:
            while not self._audio_q.empty():
                self._audio_q.get_nowait()
        except queue.Empty:
            pass

        print("[Shutdown] Done. Goodbye!")

if __name__=="__main__":
    v=VoiceTranscriber(model_size="small")
    text=v.listen()

                
                