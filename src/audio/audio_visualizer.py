import sys
import numpy as np

_STATE = {"ema": 0.0, "peak": 0.0}

def visualize(indata, frames, time, status, width: int = 60) -> None:
    if indata.size == 0:
        return

    level = float(np.abs(indata).max())

    alpha = 0.2
    _STATE["ema"] = alpha * level + (1.0 - alpha) * _STATE["ema"]

    if level > _STATE["peak"]:
        _STATE["peak"] = level
    else:
        _STATE["peak"] *= 0.98

    bar_len = max(0, min(width, int(_STATE["ema"] * width)))
    peak_pos = max(0, min(width - 1, int(_STATE["peak"] * width)))

    bar = ["-"] * width
    for i in range(bar_len):
        bar[i] = "#"
    if peak_pos < width:
        bar[peak_pos] = "|"

    # \033[2K clears entire line, \r goes back to start
    sys.stdout.write(f"\033[2K\r[{''.join(bar)}]")
    sys.stdout.flush()