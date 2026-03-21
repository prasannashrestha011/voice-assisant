from __future__ import annotations
from typing import TYPE_CHECKING
from src.preferences.preference import UserPreferences

if TYPE_CHECKING:
    from src.text_to_speech import TextToSpeech

"""responsible for manupulating user preferences"""
class Services:
    tts: "TextToSpeech | None" = None
    prefs: UserPreferences = UserPreferences.load()

    @classmethod
    def set_pref(cls, key: str, value) -> str:
        setattr(cls.prefs, key, value)
        cls.prefs.save()
        return f"{key} changed to {value}"

services = Services()