"""Character error rate, matching semaje-stt's eval normalization (kept local so
this repo has no code dependency on semaje-stt, only on its HTTP contract)."""
import re
import unicodedata


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in text)
    return re.sub(r"\s+", " ", text).strip()


def _distance(a: list, b: list) -> int:
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (a[i - 1] != b[j - 1]))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    r, h = list(normalize(ref).replace(" ", "")), list(normalize(hyp).replace(" ", ""))
    return _distance(r, h) / max(1, len(r))
