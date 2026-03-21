import requests
import json

OLLAMA_URL = "http://127.0.0.1:11434/api/chat"


def refine_query(raw_query: str, model: str = "llama3.2") -> str:
    """
    Uses a fast LLM call to rewrite the user's query into
    an optimized web search query before hitting DuckDuckGo.
    """
    prompt = f"""You are a search query optimizer. Your job is to rewrite a user's question into the best possible web search query.

Rules:
- Always include the current year (2026) for anything time-sensitive (sports, news, events, prices)
- Be specific: include team names, competition names, tournament stages
- Remove conversational words like "who", "what is", "can you tell me"
- Keep it under 10 words
- Return ONLY the refined query, nothing else, no explanation, no quotes

Examples:
User: "who are the winners of round of 16 in champions league this season"
Refined: UEFA Champions League 2026 round of 16 results

User: "what is the latest iphone"
Refined: latest iPhone model 2026

User: "how is the weather in kathmandu"
Refined: Kathmandu weather today

User: "who is the richest person right now"
Refined: richest person in the world 2026

Now refine this query:
User: "{raw_query}"
Refined:"""

    try:
        response = requests.post(OLLAMA_URL, json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "options": {"temperature": 0, "num_predict": 30}  # deterministic + fast
        }, timeout=8)

        data = response.json()
        refined = data["message"]["content"].strip()

        # Sanitize — strip quotes if LLM wrapped it
        refined = refined.strip('"').strip("'").strip()

        print(f"[refine] '{raw_query}' → '{refined}'")
        return refined

    except Exception as e:
        print(f"[refine] Failed, using raw query: {e}")
        return raw_query  # fallback to original