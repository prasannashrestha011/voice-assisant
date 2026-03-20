from src.tools.terminal import neofetch_tool

from .wrapper.wrapper import make_tool
TOOLS = {
    "neo_fetch":(
        make_tool(
            "neo_fetch",
            "fetch system specification in terminal using neofetch",
            {}

        ),
        lambda args: neofetch_tool()
    )
}
