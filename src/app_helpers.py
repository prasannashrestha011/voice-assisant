import queue
import re
import shutil
import subprocess
import time

from .logger.logger import _logger
from .text_to_speech import TextToSpeech


def split_sentences(buffer: str) -> tuple[list[str], str]:
	# Split on end-of-sentence punctuation; keep trailing partial.
	parts = re.split(r"(?<=[.!?])\s+", buffer)
	if len(parts) <= 1:
		return [], buffer
	return parts[:-1], parts[-1]


def clean_for_tts(text: str) -> str:
	return " ".join((text or "").strip().split())


def stop_ollama() -> None:
	if shutil.which("ollama") is None:
		return
	try:
		subprocess.run(["ollama", "stop", "qwen2.5:1.5b"], check=False)
	except Exception as e:
		_logger.error("[shutdown] ollama stop failed:", e)


def drain_queue(q: queue.Queue[str]) -> None:
	try:
		while True:
			q.get_nowait()
	except queue.Empty:
		pass


def throttle_tts(tts: TextToSpeech, max_backlog: int = 2) -> None:
	while tts.audio_queue.qsize() >= max_backlog:
		time.sleep(0.02)
