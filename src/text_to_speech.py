import os
import queue
import threading
import time
import soundfile as sf
import numpy as np
from pathlib import Path
from kokoro_onnx import Kokoro
import sounddevice as sd

from .logger.logger import _logger
from .audio.audio_visualizer import visualize


def _clean_text(text: str) -> str:
    return " ".join((text or "").strip().split())


_SHUTDOWN=object()
class _PlayJob:
    def __init__(self,sample:np.ndarray,sr:int):
       self.sample=sample
       self.sr=sr
       self.cancel=threading.Event()
       self.done_evnt=threading.Event()

class TextToSpeech:
    def __init__(self):
        self.voice="af_bella"
        self.speed=1.0
        self.lang="en-us"
        base_path = Path(__file__).parent
        model_path  = base_path / "kokoro-v1.0.onnx"
        voices_path = base_path / "voices-v1.0.bin"
        self.kokoro=Kokoro(str(model_path),str(voices_path))

        #queue
        self.audio_queue:queue.Queue=queue.Queue()
        self.current_job:_PlayJob | None=None
        #events
        self.play_evnt=threading.Event()

        # Used to prevent in-flight synthesis from enqueueing after an interrupt.
        self._epoch_lock = threading.Lock()
        self._epoch = 0
        #lock
        self.current_lock=threading.Lock()
        self.exc_lock=threading.Lock()
        self.worker_thread=threading.Thread(target=self.audio_worker,daemon=True)
        self.worker_thread.start()
        
        

    def audio_worker(self):
        """Background loop to consume and play queued audio jobs."""
        while True:

            _logger.debug("\033[93m[audio_worker] started\033[0m\n")
            job = None
            try:
                job = self.audio_queue.get()
                _logger.debug("\n[audio_worker] got job:", job)

                if job is _SHUTDOWN:
                    _logger.debug("[audio_worker] received shutdown signal")
                    return

                with self.current_lock:
                    self.current_job = job
                    _logger.debug("[audio_worker] set current_job =", self.current_job)

                self.play_evnt.set()

                try:
                    self._play_with_visualizer(job)

                except Exception as e:
                    _logger.error(f"[audio_worker] playback error: {e}")

                finally:
                    job.done_evnt.set()
                    with self.current_lock:
                        self.current_job = None

                    if self.audio_queue.empty():
                        self.play_evnt.clear()

            except Exception as e:
                # Prevent silent worker death
                _logger.error(f"[audio_worker] loop error (continuing): {e}")
                time.sleep(0.05)
            finally:
                # queue.Queue.task_done is optional here, but safe if used with join later
                if job is not None and job is not _SHUTDOWN:
                    try:
                        self.audio_queue.task_done()
                    except ValueError:
                        pass

    def _play_with_visualizer(self, job: _PlayJob) -> None:
        sample = job.sample
        sr = job.sr

        if sample.ndim == 1:
            channels = 1
        else:
            channels = sample.shape[1]

        idx = 0

        def callback(outdata, frames, time_info, status):
            nonlocal idx

            if job.cancel.is_set():
                outdata.fill(0)
                raise sd.CallbackStop()

            end = idx + frames
            chunk = sample[idx:end]

            if channels == 1:
                chunk = chunk.reshape(-1, 1)

            if chunk.shape[0] < frames:
                outdata[: chunk.shape[0]] = chunk
                outdata[chunk.shape[0] :] = 0
                visualize(outdata[: chunk.shape[0]], frames, time_info, status)
                idx = end
                raise sd.CallbackStop()

            outdata[:] = chunk
            visualize(outdata, frames, time_info, status)
            idx = end

        with sd.OutputStream(
            samplerate=sr,
            channels=channels,
            dtype=sample.dtype,
            callback=callback,
        ):
            while not job.cancel.is_set() and idx < len(sample):
                sd.sleep(20)
         
    
    def interrupt(self, wait: bool = True) -> None:
        _logger.debug("[interrupt] called, wait =", wait)

        # Bump epoch first so any in-flight synthesis can be discarded.
        with self._epoch_lock:
            self._epoch += 1

        with self.current_lock:
            current = self.current_job
            _logger.debug("[interrupt] current_job =", current)

        if current is None:
            self.flush()
            self.play_evnt.clear()
            print("[interrupt] no current job")
            return

        _logger.debug("[interrupt] signaling cancel")
        current.cancel.set()

        if wait:
            print("[interrupt] waiting for worker to finish current job")
            current.done_evnt.wait(timeout=1.0)

        # Also drop anything already queued (important for streaming sentence-by-sentence)
        self.flush()

        _logger.debug("[interrupt] finished")


    def flush(self) -> None:
        """Drop any queued audio jobs that haven't started yet."""
        drained = 0
        while True:
            try:
                job = self.audio_queue.get_nowait()
            except queue.Empty:
                break

            if job is _SHUTDOWN:
                # Preserve shutdown sentinel for the worker.
                self.audio_queue.put(_SHUTDOWN)
                break

            if isinstance(job, _PlayJob):
                job.cancel.set()
                job.done_evnt.set()

            drained += 1
            try:
                self.audio_queue.task_done()
            except ValueError:
                pass

        with self.current_lock:
            no_current = self.current_job is None

        if drained and no_current and self.audio_queue.empty():
            self.play_evnt.clear()

        if drained:
            _logger.debug(f"[flush] dropped {drained} queued audio job(s)")



    def synthesize(self,text)->tuple[np.ndarray,int]:
        sample,sr=self.kokoro.create(text,self.voice,self.speed,self.lang)
        return sample,sr
    def speak(self, sample, sr):
        job = _PlayJob(sample, sr)
        self.audio_queue.put(job)

        # Safety timeout so caller doesn't block forever if backend hangs
        done = job.done_evnt.wait(timeout=30.0)
        if not done:
            _logger.debug("[speak] timeout waiting for playback completion")
        return job


    def enqueue_text(self, text: str) -> _PlayJob | None:
        """Synthesize and enqueue audio without waiting for playback."""
        text = _clean_text(text)
        if not text:
            return None

        with self._epoch_lock:
            epoch = self._epoch

        sample, sr = self.synthesize(text)

        # If an interrupt happened while we were synthesizing, drop this job.
        with self._epoch_lock:
            if self._epoch != epoch:
                return None

        job = _PlayJob(sample, sr)
        self.audio_queue.put(job)

        # signal: TTS is active or has pending audio
        self.play_evnt.set()
        return job
        
        
    def exc(self, text):
        with self.exc_lock:
            text = _clean_text(text)
            if not text:
                return

            with self._epoch_lock:
                epoch = self._epoch

            sample, sr = self.synthesize(text)

            with self._epoch_lock:
                if self._epoch != epoch:
                    return

            with self.current_lock:
                has_current_job = self.current_job is not None

            if has_current_job:
                self.interrupt(wait=True)

            self.speak(sample, sr)
            print("finished audio")

    def shutdown(self) -> None:
        self.interrupt(wait=False)
        self.audio_queue.put(_SHUTDOWN)
        if self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)

if __name__=="__main__":
    t1=TextToSpeech()
    t1.exc("Helloworldhowreouafterwinningsuchelection")
    t1.exc("Banana Banana  BananaBanana")