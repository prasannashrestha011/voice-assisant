import json
from dataclasses import dataclass, asdict
from pathlib import Path
from src.logger.logger import _logger

BASE_DIR = Path(__file__).resolve().parents[2]
PREFS_PATH = BASE_DIR / "data" / "user_preferences.json"

@dataclass
class UserPreferences:
    voice: str = "af_sarah"
    speed: float = 1.0
    lang: str = "en-us"

    def save(self) -> None:
        with open(PREFS_PATH, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls) -> "UserPreferences":
        if not PREFS_PATH.exists():
            prefs = cls()
            prefs.save()  # create default file on first run
            return prefs
        try:
            with open(PREFS_PATH) as f:
                data = json.load(f)
                _logger.info(data)
            return cls(**data)
        except Exception:
            return cls()  # fallback to defaults if file is corrupt