import queue
import threading

from dotenv import load_dotenv

from .logger.logger import _logger
from .llm import LLM
from .app_helpers import clean_for_tts
from .app_helpers import drain_queue
from .app_helpers import split_sentences
from .app_helpers import stop_ollama
from .app_helpers import throttle_tts
from .text_to_speech import TextToSpeech
from .voice_to_text import VoiceTranscriber

load_dotenv()



def _listener_loop(
	stop_ev: threading.Event,
	input_q: queue.Queue[str],
	tts: TextToSpeech,
	on_interrupt: callable,
) -> None:
	vts = VoiceTranscriber(tts=tts, on_interrupt=on_interrupt)
	while not stop_ev.is_set():
		try:
			text = vts.listen()
		except Exception as e:
			_logger.error("[listener] error:", e)
			continue

		if not text:
			continue

		drain_queue(input_q)
		input_q.put(text)


def _make_on_token(
	llm: LLM,
	tts: TextToSpeech,
	buffer_ref: dict[str, str],
) -> callable:
	def on_token(token: str) -> None:
		if llm.is_cancelled():
			return

		buffer_ref["buffer"] += token
		sentences, remainder = split_sentences(buffer_ref["buffer"])
		buffer_ref["buffer"] = remainder

		for sentence in sentences:
			if llm.is_cancelled():
				return
			clean = clean_for_tts(sentence)
			if not clean:
				continue
			throttle_tts(tts)
			tts.enqueue_text(clean)

	return on_token


def _flush_leftover(llm: LLM, tts: TextToSpeech, buffer: str) -> None:
	if llm.is_cancelled():
		return
	leftover = clean_for_tts(buffer)
	if leftover:
		throttle_tts(tts)
		tts.enqueue_text(leftover)
###

def main() -> None:
	tts = TextToSpeech()
	llm = LLM()
	input_q: queue.Queue[str] = queue.Queue(maxsize=1)
	stop_ev = threading.Event()

	def on_interrupt() -> None:
		llm.cancel()
		tts.interrupt(wait=False)

	listener_thread = threading.Thread(
		target=_listener_loop,
		args=(stop_ev, input_q, tts, on_interrupt),
		daemon=True,
	)
	listener_thread.start()

	try:
		while True:
			prompt = input_q.get()
			if not prompt:
				continue

			# Stream tokens and speak sentence-by-sentence.
			buffer_ref = {"buffer": ""}
			on_token = _make_on_token(llm, tts, buffer_ref)
			llm.generate(prompt, stream=True, on_token=on_token)
			_flush_leftover(llm, tts, buffer_ref["buffer"])
	except KeyboardInterrupt:
		pass
	finally:
		stop_ev.set()
		tts.shutdown()
		stop_ollama()


if __name__ == "__main__":
	main()
