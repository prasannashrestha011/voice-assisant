import subprocess
from langchain_core.tools import tool


def neofetch_tool() -> str:
    try:
        # Open a new terminal window and run neofetch visually
        subprocess.Popen([
            "gnome-terminal", "--",
            "bash", "-c",
            "neofetch; echo; echo 'Press Enter to close...'; read"
        ])
        return "Opened neofetch in a new terminal window"  # ← short string for LLM to speak
    except FileNotFoundError:
        return "Error: gnome-terminal is not installed"
    except Exception as e:
        return f"Error: {e}"