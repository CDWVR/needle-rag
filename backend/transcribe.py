"""Voice input: short recordings are transcribed by Whisper on OpenRouter under a hard daily spend cap.

Cost controls, outermost first:
  1. A separate OpenRouter key (OPENROUTER_STT_API_KEY) whose credit limit is set in the OpenRouter dashboard.
  2. A daily USD cap kept here (persisted on the data volume, UTC day). Each call reserves a worst-case
     estimate before it is sent, then the real cost reported by OpenRouter replaces the estimate.
  3. A maximum clip length (enforced in the browser, and by a byte limit here) and per-client rate limits.
"""

import base64
import json
import logging
import os
import threading
import time
from typing import Optional

import httpx

from config import env_float, env_int, env_str
from llm import record_cost
from paths import data_path
from pipeline_logic import InfraError, NeedleError

log = logging.getLogger("needle.transcribe")

STT_URL = "https://openrouter.ai/api/v1/audio/transcriptions"
STT_MODEL = env_str("STT_MODEL", "openai/whisper-large-v3-turbo")
STT_MAX_SECONDS = env_int("STT_MAX_SECONDS", 15)
STT_DAILY_USD = env_float("STT_DAILY_USD", 0.01)
# Worst-case price per second used to reserve budget before a call. OpenRouter's example response
# shows about 0.000055 USD/s; this is deliberately higher so the cap is never overshot by a price rise.
STT_RESERVE_PER_SECOND = env_float("STT_RESERVE_PER_SECOND", 0.0001)
# 15 s of browser Opus is about 60 KB; this leaves room for other codecs but bounds a crafted upload.
STT_MAX_BYTES = env_int("STT_MAX_BYTES", 200_000)
STT_TIMEOUT_SECONDS = 30


def stt_key() -> str:
    """The speech key, falling back to the main OpenRouter key. Empty means voice input is off."""
    return (os.getenv("OPENROUTER_STT_API_KEY", "").strip() or os.getenv("OPENROUTER_API_KEY", "").strip())


def enabled() -> bool:
    return bool(stt_key()) and STT_DAILY_USD > 0


def sniff_format(data: bytes) -> Optional[str]:
    """Identify the container from its leading bytes; the client's content type is not trusted."""
    if data[:4] == b"\x1a\x45\xdf\xa3":
        return "webm"
    if data[:4] == b"OggS":
        return "ogg"
    if data[4:8] == b"ftyp":
        return "m4a"
    if data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        return "wav"
    if data[:3] == b"ID3" or data[:2] in (b"\xff\xfb", b"\xff\xf3", b"\xff\xf2"):
        return "mp3"
    return None


class DailyBudget:
    """USD spent on speech today (UTC), persisted so a restart cannot reset the cap."""

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()

    @staticmethod
    def _today() -> str:
        return time.strftime("%Y-%m-%d", time.gmtime())

    def _load(self) -> float:
        try:
            with open(self.path, encoding="utf-8") as handle:
                state = json.load(handle)
            return float(state["usd"]) if state.get("day") == self._today() else 0.0
        except (OSError, ValueError, KeyError, TypeError):
            return 0.0

    def _store(self, usd: float) -> None:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        temporary = self.path + ".tmp"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump({"day": self._today(), "usd": round(usd, 6)}, handle)
        os.replace(temporary, self.path)

    def reserve(self, amount: float) -> bool:
        with self._lock:
            spent = self._load()
            if spent + amount > STT_DAILY_USD:
                return False
            self._store(spent + amount)
            return True

    def settle(self, reserved: float, actual: float) -> None:
        with self._lock:
            self._store(max(0.0, self._load() - reserved + actual))

    def spent(self) -> float:
        with self._lock:
            return self._load()


budget = DailyBudget(data_path("voice", "spend.json"))


def reserve_for(seconds: float) -> float:
    clip = min(max(float(seconds or 0), 1.0), float(STT_MAX_SECONDS))
    return round(clip * STT_RESERVE_PER_SECOND, 6)


def transcribe(audio: bytes, claimed_seconds: float, language: str = "") -> str:
    """Return the transcript. Raises NeedleError with a user-facing message on any refusal."""
    if not enabled():
        raise NeedleError("Voice input is not configured on this server.")
    if not audio:
        raise NeedleError("No audio was received.")
    if len(audio) > STT_MAX_BYTES:
        raise NeedleError(f"That recording is too long. Keep it under {STT_MAX_SECONDS} seconds.")
    audio_format = sniff_format(audio)
    if not audio_format:
        raise NeedleError("That audio format is not supported.")
    reserved = reserve_for(claimed_seconds)
    if not budget.reserve(reserved):
        raise NeedleError("Voice input has reached today's limit. Type your question instead.")
    actual = reserved
    try:
        body = {
            "model": STT_MODEL,
            "input_audio": {"data": base64.b64encode(audio).decode("ascii"), "format": audio_format},
        }
        if language and language.isalpha() and len(language) in (2, 3):
            body["language"] = language.lower()
        try:
            response = httpx.post(
                STT_URL,
                headers={"Authorization": f"Bearer {stt_key()}", "Content-Type": "application/json"},
                json=body,
                timeout=STT_TIMEOUT_SECONDS,
            )
        except httpx.HTTPError as exc:
            log.warning("speech request failed: %s", type(exc).__name__)
            actual = 0.0  # a request that never completed is not billed
            raise InfraError("Voice input could not reach the speech service.", status_code=None, kind="network") from exc
        if response.status_code >= 500:
            actual = 0.0  # OpenRouter does not bill failed generations
            raise InfraError("The speech service is unavailable.", status_code=response.status_code, kind="upstream")
        if response.status_code != 200:
            actual = 0.0
            try:
                reason = str((response.json().get("error") or {}).get("message") or "")[:300]
            except (ValueError, AttributeError):
                reason = ""
            log.error("speech request refused: HTTP %s %s", response.status_code, reason)
            if response.status_code in (401, 402, 403):
                # A key problem is the owner's to fix (limit too low, key revoked); say so, not "bad recording".
                raise InfraError("Voice input is unavailable right now.", status_code=response.status_code, kind="key")
            raise NeedleError("The speech service refused that recording.")
        payload = response.json()
        usage = payload.get("usage") or {}
        try:
            billed = float(usage.get("cost"))
        except (TypeError, ValueError):
            billed = None
        if billed is None:
            billed = float(usage.get("seconds") or claimed_seconds or 0) * STT_RESERVE_PER_SECOND
        actual = max(0.0, billed)
        text = " ".join(str(payload.get("text") or "").split())[:1000]
        return text if any(char.isalnum() for char in text) else ""  # silence comes back as "." or "..."
    finally:
        budget.settle(reserved, actual)
        record_cost("stt", actual)
