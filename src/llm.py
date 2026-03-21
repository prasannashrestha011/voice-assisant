from datetime import datetime
from typing import Callable, Optional
import threading
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
load_dotenv()
MAX_HISTORY = 20
# ── LLM Setup ─────────────────────────────────────────────────────────────────
class LLM:
    def __init__(self, tools: dict | None = None):
        self.llm = ChatOllama(
            # model="qwen2.5:1.5b",
            model="qwen3.5:2b",
            temperature=0.3,
            think=False,
            device="cuda",
        )
        self._cancel_ev = threading.Event()
        self.tools = tools or {}
        self._tool_schemas = [schema for schema, _ in self.tools.values()]
        self._history: list[dict] = []

    def cancel(self) -> None:
        self._cancel_ev.set()

    def reset_cancel(self) -> None:
        self._cancel_ev.clear()

    def is_cancelled(self) -> bool:
        return self._cancel_ev.is_set()

    def clear_memory(self) -> None:
        self._history = []

    def _run_tool(self, tool_name: str, tool_args: dict) -> str:
        if tool_name not in self.tools:
            return f"Unknown tool: {tool_name}"
        _, fn = self.tools[tool_name]
        try:
            return str(fn(tool_args))
        except Exception as e:
            return f"Tool error: {e}"

    def generate(
        self,
        prompt: str,
        on_token: Optional[Callable[[str], None]] = None,
        stream: bool = True,
        max_tool_rounds: int = 5,
    ) -> str:
        self.reset_cancel()
        self._history.append({"role": "user", "content": prompt})
        if len(self._history) > MAX_HISTORY:
            self._history = self._history[-MAX_HISTORY:]
        messages = [
            {
                "role": "system",
                "content": (
            f"You are Dexter AI, a concise voice assistant developed by Prasanna Shrestha. "
            f"Today's date is {datetime.now().strftime('%A, %B %d, %Y')}. "

            # Tool usage — critical
            "You have access to real-time tools. NEVER say you lack real-time information. "
            "ALWAYS use the web_search tool for: current events, news, sports scores, weather, "
            "prices, standings, results, rankings, or anything time-sensitive. "
            "If the user asks about anything happening in the world, search first, then answer. "
            "Do NOT answer from memory for time-sensitive topics — use web_search. "

            # Identity
            "Do not disclose the underlying model name, it is confidential. "

            # Voice response style
            "You are a voice assistant — keep all responses short and spoken-friendly. "
            "When a tool returns large data, summarize in 1-2 short sentences. "
            "Never read out raw data, URLs, or long lists. "
            "State simple values directly and briefly."
                )
            },
            *self._history
        ]
        llm = self.llm.bind_tools(self._tool_schemas) if self._tool_schemas else self.llm
        for _ in range(max_tool_rounds):
            if self._cancel_ev.is_set():
                break
            response = llm.invoke(messages)
            tool_calls = getattr(response, "tool_calls", None)
            if tool_calls:
                messages.append(response)
                self._history.append(response)
                for tc in tool_calls:
                    tool_name = tc["name"]
                    tool_args = tc["args"]
                    result = self._run_tool(tool_name, tool_args)
                    tool_msg = {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": result,
                    }
                    messages.append(tool_msg)
                    self._history.append(tool_msg)
                continue

            # ── final response: real streaming ────────────────────────────
            final_parts: list[str] = []
            if stream and on_token:
                for chunk in llm.stream(messages):  # ← real stream, not fake char loop
                    if self._cancel_ev.is_set():
                        break
                    text = chunk.content or ""
                    if not text:
                        continue
                    final_parts.append(text)
                    on_token(text)
            else:
                final_parts.append(response.content or "")

            final_text = "".join(final_parts)
            self._history.append({"role": "assistant", "content": final_text})
            return final_text

        return ""