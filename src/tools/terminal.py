import subprocess
from langchain_core.tools import tool


@tool
def neofetch_tool()->str:
    """
    Runs 'neofetch' in the assistant terminal and returns the output as text.
    """

    try:
        result = subprocess.run(["neofetch", "--stdout"], capture_output=True, text=True)
        return result.stdout  # must return output as string
    except FileNotFoundError:
        print("Neofetch not installed")