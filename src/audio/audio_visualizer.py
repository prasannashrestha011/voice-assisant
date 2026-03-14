import sys

import numpy as np

_STATE = {"ema": 0.0, "peak": 0.0}

def visualize(indata, frames, time, status, width: int = 60) -> None:
    """Render a smoothed ASCII level meter with peak hold."""
    if indata.size == 0:
        return

    level = float(np.abs(indata).max())

    # Exponential smoothing
    alpha = 0.2
    _STATE["ema"] = alpha * level + (1.0 - alpha) * _STATE["ema"]

    # Peak hold with decay
    if level > _STATE["peak"]:
        _STATE["peak"] = level
    else:
        _STATE["peak"] *= 0.98

    bar_len = max(0, min(width, int(_STATE["ema"] * width)))
    peak_pos = max(0, min(width - 1, int(_STATE["peak"] * width)))

    # Color thresholds via ANSI codes
    ema = _STATE["ema"]
    if ema < 0.6:
        color = "\033[32m"   # green
    elif ema < 0.85:
        color = "\033[33m"   # yellow
    else:
        color = "\033[31m"   # red
    reset = "\033[0m"

    # Build bar
    bar = list(" " * width)
    for i in range(bar_len):
        bar[i] = "█"
    bar[peak_pos] = "▌"

    # Color the filled portion
    colored = color + "".join(bar[:bar_len]) + reset
    peak_char = ("\033[91m" if ema >= 0.85 else "\033[33m") + bar[peak_pos] + reset
    tail = "".join(bar[bar_len + 1:]) if bar_len < peak_pos else ""

    if bar_len < peak_pos:
        meter = colored + " " * (peak_pos - bar_len - 1) + peak_char + tail
    elif bar_len == peak_pos:
        meter = colored[:-len(reset)] + peak_char[peak_char.index("▌"):]  
        meter = color + "".join(bar[:bar_len]) + peak_char + reset
    else:
        meter = colored + tail

    # Clip warning
    clip = " \033[31mCLIP!\033[0m" if level > 0.95 else "      "

    sys.stdout.write(f"\r[{meter}] {_STATE['ema']:.2f} pk:{_STATE['peak']:.2f}{clip}")
    sys.stdout.flush()