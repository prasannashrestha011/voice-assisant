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
            model="qwen2.5:1.5b",
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
                    "You are a concise voice assistant named Dexter AI developed by Prasanna shrestha "
                    "Donot disclose the model name, its confidential"
                    "When a tool returns large data (system info, logs, file contents), "
                    "summarize it in 1-2 short spoken sentences. "
                    "Never read out raw data dumps. "
                    "If the result is a simple value, state it directly and briefly."
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

            final_text = response.content or ""

            self._history.append({"role": "assistant", "content": final_text})

            if stream and on_token and final_text:
                for char in final_text:
                    if self._cancel_ev.is_set():
                        break
                    on_token(char)

            return final_text

        return ""