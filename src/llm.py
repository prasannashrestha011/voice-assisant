
from typing import Callable, Optional
import threading

from langchain_ollama import ChatOllama
from dotenv import load_dotenv

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────────────────────
class LLM:
    def __init__(self):
        self.llm = ChatOllama(
            model="qwen2.5:1.5b",
            temperature=0.3,
            think=False,
            device="cuda",
        )
        self._cancel_ev = threading.Event()

    def cancel(self) -> None:
        self._cancel_ev.set()

    def reset_cancel(self) -> None:
        self._cancel_ev.clear()

    def is_cancelled(self) -> bool:
        return self._cancel_ev.is_set()

    def generate(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        stream: bool = True,
    ) -> str:
        self.reset_cancel()

        if not stream:
            result = self.llm.invoke(prompt)
            return result.content

        parts: list[str] = []
        for chunk in self.llm.stream(prompt):
            if self._cancel_ev.is_set():
                break

            text = chunk.content or ""
            if not text:
                continue

            parts.append(text)
            if on_token is not None:
                on_token(text)

        return "".join(parts)