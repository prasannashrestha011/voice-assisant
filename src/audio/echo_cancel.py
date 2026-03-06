"""
We temporarily disable default audio source to use virtual microphone

"""
import sounddevice as sd
import subprocess


_echo_module_id = None  # track loaded module
ECHO_CANCEL_SOURCE_NAME = "echo-cancel-source"
ECHO_CANCEL_SINK_NAME   = "echo-cancel-sink"

def _get_default_source() -> str:
    """Dynamically get the current default source before we change it."""
    result = subprocess.run(["pactl", "info"], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if "Default Source:" in line:
            return line.split(": ", 1)[1].strip()
    return ""

def _find_pulse_device() -> tuple[int, int]:
    """Dynamically find pulse/pipewire device index and sample rate."""
    for name in ("pulse", "pipewire"):
        for i, d in enumerate(sd.query_devices()):
            if d['name'].lower() == name and d['max_input_channels'] > 0:
                return i, int(d['default_samplerate'])
    raise RuntimeError("No suitable audio input device found (pulse/pipewire)")

def _setup_echo_cancel() -> str:
    """Load echo-cancel module and set as default. Returns original source name."""
    global _echo_module_id

    original_source = _get_default_source()

    # Unload any stale instances
    subprocess.run(["pactl", "unload-module", "module-echo-cancel"],
                   capture_output=True)

    result = subprocess.run([
        "pactl", "load-module", "module-echo-cancel",
        "aec_method=webrtc",
        "aec_args=analog_gain_control=0 digital_gain_control=1 noise_suppression=0 high_pass_filter=0",
        f"source_name={ECHO_CANCEL_SOURCE_NAME}",
        f"sink_name={ECHO_CANCEL_SINK_NAME}",
    ], capture_output=True, text=True, check=True)

    _echo_module_id = result.stdout.strip()
    print(f"Loaded echo-cancel module (id={_echo_module_id})")

    subprocess.run(["pactl", "set-default-source", ECHO_CANCEL_SOURCE_NAME], check=True)
    print(f"Set default source to '{ECHO_CANCEL_SOURCE_NAME}'")

    return original_source

def _teardown_echo_cancel(original_source: str):
    """Restore original source and unload echo-cancel module."""
    global _echo_module_id

    if original_source:
        subprocess.run(["pactl", "set-default-source", original_source],
                       capture_output=True)
        print(f"\nRestored default source to '{original_source}'")

    if _echo_module_id:
        subprocess.run(["pactl", "unload-module", _echo_module_id],
                       capture_output=True)
        print(f"Unloaded echo-cancel module (id={_echo_module_id})")
        _echo_module_id = None