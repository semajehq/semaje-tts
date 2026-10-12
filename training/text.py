"""Text normalization for character-level ("text" phoneme) Piper voices.

Kikuyu and Dholuo have no espeak-ng voice, but both orthographies are close to
phonemic, so the voice is trained on characters. Piper's text mode splits the
NFD form into code points, so "ĩ" becomes "i" + U+0303 (both in Piper's
default symbol table). Everything here is pure and gate-tested.
"""
from __future__ import annotations

import re
import unicodedata

from piper.phoneme_ids import DEFAULT_PHONEME_ID_MAP

_REPLACE = {
    "’": "'", "‘": "'", "ʼ": "'", "`": "'", "“": '"', "”": '"', "–": "-", "—": "-", "_": "-",
    "…": "...", " ": " ",
}
# Characters the model never needs to pronounce.
_DROP = set('"#%&*+/<=>@[\\]^{|}~°')


def normalize(text: str) -> str:
    """Lowercase, unify quotes/dashes, drop markup symbols, collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = "".join(_REPLACE.get(c, c) for c in text).lower()
    text = "".join(" " if c in _DROP else c for c in text)
    text = re.sub(r"\s*\n\s*", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def symbols(text: str) -> list[str]:
    """Exactly what Piper's text mode feeds the model."""
    return list(unicodedata.normalize("NFD", text))


def unknown_symbols(text: str) -> set[str]:
    return {s for s in symbols(text) if s not in DEFAULT_PHONEME_ID_MAP}


def reject_reason(text: str) -> str | None:
    """Why an utterance is unusable for training, or None if it is fine.

    Digits are rejected rather than verbalized: number words differ per
    language and the model would otherwise learn to read "3" as a sound.
    """
    if not text:
        return "empty"
    if any(c.isdigit() for c in text):
        return "digits"
    if unknown_symbols(text):
        return "unknown-symbols"
    return None
