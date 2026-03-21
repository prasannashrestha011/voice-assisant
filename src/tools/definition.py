from src.tools.terminal import neofetch_tool
from src.preferences.services import services
from .wrapper.wrapper import make_tool
from .web_scraper import web_search_tool
from datetime import datetime

TOOLS = {
    "neo_fetch": (
        make_tool(
            "neo_fetch",
            "fetch system specification in terminal using neofetch",
            {}
        ),
        lambda args: neofetch_tool()
    ),
    "change_voice": (
        make_tool(
            "change_voice",
            "Change the TTS voice. Available: af_bella, af_sarah, af_nicole, af_sky, am_adam, am_michael, bf_emma, bf_isabella, bm_george, bm_lewis"+
  "Change the TTS voice. Users can say British/American male/female, or use specific codes.",
            {"voice": {"type": "string", "description": "Voice code e.g. af_bella"}}
        ),
        lambda args: services.set_pref("voice", args["voice"])
    ),
    "change_speed": (
        make_tool(
            "change_speed",
            "Change the TTS speaking speed. 1.0 is normal, 0.5 is slow, 2.0 is fast",
            {"speed": {"type": "number", "description": "Speed multiplier e.g. 1.2"}}
        ),
        lambda args: services.set_pref("speed", float(args["speed"]))
    ),
    "web_search": (
        make_tool(
            "web_search",
            f"Search the web and read full content from top results. The current date is {datetime.now().strftime('%Y-%m-%d')}. "
            "Use for current events, news, sports, weather, prices, or any real-time information. summarized the found result, donot just reside it. "
            "Just provide a search query — no URL needed.",
            {
                "query": {
                    "type": "string",
                    "description": f"What to search for e.g. 'Champions League quarterfinals {datetime.now().year}'"
                }
            }
        ),
        lambda args: web_search_tool(args["query"])
    )
}