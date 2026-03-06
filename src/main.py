from typing import TypedDict, Optional
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from .voice_to_text import VoiceTranscriber
from .text_to_speech import TextToSpeech

load_dotenv()

# ── LLM Setup ─────────────────────────────────────────────────────────────────
llm = ChatOllama(
    model="qwen3.5:2b",
    temperature=0.3,
    think=False,
    device="cuda"
)

# ── Global instances (loaded once) ────────────────────────────────────────────
_transcriber: Optional[VoiceTranscriber] = None
_tts: Optional[TextToSpeech] = None

# ── Config ────────────────────────────────────────────────────────────────────
RECORD_DURATION = 7
STOP_PHRASE     = "stop listening"
TTS_VOICE       = "af_heart"
TTS_SPEED       = 1.0


# ── LangGraph State ───────────────────────────────────────────────────────────
class PipelineState(TypedDict):
    user_input:           str
    llm_response:         str
    should_stop:          bool
    has_input:            bool          # False when VAD detected silence
    conversation_history: list[dict]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def voice_input_node(state: PipelineState) -> PipelineState:
    """Record audio → VAD filter → Whisper transcription."""
    text = _transcriber.listen(duration=RECORD_DURATION)

    if text is None:
        # Silence detected by VAD — skip LLM, loop back
        return {**state, "user_input": "", "has_input": False, "should_stop": False}

    should_stop = STOP_PHRASE.lower() in text.lower()
    return {**state, "user_input": text, "has_input": True, "should_stop": should_stop}


def _clean_for_speech(text: str) -> str:
    """Strip markdown symbols that sound bad when spoken."""
    import re
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)   # **bold** / *italic*
    text = re.sub(r"`{1,3}[^`]*`{1,3}", "", text)           # `code`
    text = re.sub(r"#{1,6}\s*", "", text)                    # ## headers
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)   # [link](url)
    text = re.sub(r"[-•]\s+", "", text)                      # bullet points
    text = re.sub(r"\s{2,}", " ", text)                      # extra spaces
    return text.strip()


def _sentence_splitter(buffer: str) -> tuple[list[str], str]:
    """
    Split buffer into complete sentences and return
    (ready_sentences, leftover_fragment).
    Splits on . ! ? followed by whitespace or end-of-string.
    """
    import re
    parts = re.split(r"(?<=[.!?])\s+", buffer)
    if len(parts) <= 1:
        return [], buffer           # no complete sentence yet
    return parts[:-1], parts[-1]   # last part may be incomplete


def llm_response_node(state: PipelineState) -> PipelineState:
    """Stream LLM tokens → split into sentences → speak each sentence immediately."""
    history = state.get("conversation_history", [])

    messages = [
        ("system", (
            "You are a helpful voice assistant. "
            "Respond in plain spoken sentences only — "
            "no bullet points, no markdown, no special characters. "
            "Keep answers concise and natural-sounding."
        ))
    ]
    for turn in history:
        messages.append(("human",     turn["user"]))
        messages.append(("assistant", turn["assistant"]))
    messages.append(("human", state["user_input"]))

    print("\n🤖 Assistant: ", end="", flush=True)

    full_answer = ""
    buffer      = ""

    # ── Stream tokens from LLM ────────────────────────────────────────────────
    for chunk in llm.stream(messages):
        token = chunk.content
        if not token:
            continue

        print(token, end="", flush=True)   # live console output
        full_answer += token
        buffer      += token

        # Try to extract complete sentences from the buffer
        sentences, buffer = _sentence_splitter(buffer)
        for sentence in sentences:
            clean = _clean_for_speech(sentence)
            if clean:
                _tts.speak(clean, blocking=True)   # speak immediately, wait before next

    # ── Speak any remaining fragment ─────────────────────────────────────────
    leftover = _clean_for_speech(buffer)
    if leftover:
        _tts.speak(leftover, blocking=True)

    print("\n")   # newline after streamed output

    updated_history = history + [{"user": state["user_input"], "assistant": full_answer.strip()}]
    return {**state, "llm_response": full_answer.strip(), "conversation_history": updated_history}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_listen(state: PipelineState) -> str:
    """
    After voice_input_node:
      - silence  → loop back to listen (skip LLM)
      - stop phrase spoken → end
      - valid speech → send to LLM
    """
    if state["should_stop"]:
        return "end"
    if not state["has_input"]:
        return "listen"       # silent clip, try again
    return "respond"


def route_after_respond(state: PipelineState) -> str:
    return "end" if state["should_stop"] else "listen"


# ── Graph ─────────────────────────────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    graph = StateGraph(PipelineState)

    graph.add_node("listen",  voice_input_node)
    graph.add_node("respond", llm_response_node)

    graph.add_edge(START, "listen")

    graph.add_conditional_edges(
        "listen",
        route_after_listen,
        {"listen": "listen", "respond": "respond", "end": END},
    )
    graph.add_conditional_edges(
        "respond",
        route_after_respond,
        {"listen": "listen", "end": END},
    )

    return graph.compile()


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    _transcriber = VoiceTranscriber(model_size="medium")
    _tts         = TextToSpeech(voice=TTS_VOICE, speed=TTS_SPEED)

    pipeline = build_pipeline()

    print("\n" + "=" * 55)
    print("  🎙️  Voice ➜ LLM ➜ 🔊 TTS  Pipeline")
    print("  LangGraph + Silero VAD + Whisper + Ollama + Kokoro")
    print("=" * 55)
    print(f"  Say \"{STOP_PHRASE}\" at any time to quit.\n")

    # Greet the user
    _tts.speak("Hello! I'm ready. How can I help you?")

    initial_state: PipelineState = {
        "user_input":           "",
        "llm_response":         "",
        "should_stop":          False,
        "has_input":            False,
        "conversation_history": [],
    }

    final_state = pipeline.invoke(initial_state)

    print("\n── Conversation ended ──")
    print(f"Total turns: {len(final_state['conversation_history'])}")