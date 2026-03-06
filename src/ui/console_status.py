import sys

STATUS_IDLE       = "⬜ IDLE       │ Waiting for speech…"
STATUS_RECORDING  = "🔴 RECORDING  │ Speech detected — capturing…"
STATUS_PROCESSING = "🔄 PROCESSING │ Transcribing…"

def _print_status(status: str):
    sys.stdout.write(f"\r\033[K{status}")
    sys.stdout.flush()

def _print_result(text: str):
    sys.stdout.write(f"\n🗣️  {text}\n")
    sys.stdout.flush()