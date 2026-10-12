"""Build a Piper training set from Google WAXAL's TTS split for one language.

    python -m training.waxal --lang kik --out work/kik [--limit 40] [--stt-url http://localhost:8011]

WAXAL (CC-BY-4.0 / CC-BY-SA-4.0) ships MP3 + transcripts for 8 speakers per
language. Not every row is usable. Measured on the Kikuyu and Dholuo test
splits: some audio is far longer than its text (a 167 s clip for a 330-char
line), some text far longer than its audio, and one Kikuyu speaker describes
pictures in her own words instead of reading the script. Rows are filtered by:

  1. alignment: 4-20 chars/second of speech and <= 20 s (deterministic)
  2. text: no digits, every symbol in Piper's table (deterministic)
  3. optional ASR agreement: a running semaje-stt must transcribe the clip
     within --max-cer of its text (catches paraphrased, non-verbatim reads)

Output: wavs/<id>.wav (22,050 Hz mono), metadata.csv (id|speaker|text),
heldout.csv (test-split lines for evaluation), report.json (what was dropped, why).
"""
from __future__ import annotations

import argparse
import collections
import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import uuid
from pathlib import Path

from training.text import normalize, reject_reason

URL = "https://huggingface.co/datasets/google/WaxalNLP/resolve/main/data/TTS/{lang}/{lang}-{split}-00000.parquet"
CACHE = Path(os.getenv("WAXAL_CACHE", str(Path.home() / ".cache" / "semaje" / "waxal")))
MIN_CPS, MAX_CPS, MAX_SECONDS, MIN_SECONDS = 4.0, 20.0, 20.0, 1.0
SAMPLE_RATE = 22050
# Who recorded each language's WAXAL TTS data (from the dataset card), for attribution.
PROVIDERS = {"kik": "Loud and Clear", "luo": "Loud and Clear", "swa": "Loud and Clear / Makerere University",
             "lug": "Loud and Clear / Makerere University", "ibo": "Media Trust", "hau": "Media Trust",
             "yor": "Media Trust", "wol": "AIMS Senegal", "ful": "Media Trust / AIMS Senegal",
             "aka": "University of Ghana", "ewe": "University of Ghana", "ach": "Makerere University",
             "nyn": "Makerere University"}


def fetch(lang: str, split: str) -> Path:
    path = CACHE / f"{lang}-{split}.parquet"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(lang=lang, split=split), tmp)
        tmp.replace(path)
    return path


def alignment_reason(chars: int, seconds: float) -> str | None:
    if seconds < MIN_SECONDS:
        return "too-short"
    if seconds > MAX_SECONDS:
        return "too-long"
    rate = chars / seconds
    if rate < MIN_CPS:
        return "audio-longer-than-text"
    if rate > MAX_CPS:
        return "text-longer-than-audio"
    return None


def decode(mp3: bytes, dst: Path) -> float:
    """MP3 → 22.05 kHz mono 16-bit WAV; returns duration in seconds."""
    with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
        f.write(mp3)
        f.flush()
        subprocess.run(["ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", f.name, "-ac", "1",
                        "-ar", str(SAMPLE_RATE), "-c:a", "pcm_s16le", str(dst)], check=True)
    import wave
    with wave.open(str(dst)) as w:
        return w.getnframes() / w.getframerate()


def asr_cer(stt_url: str, wav: Path, lang: str, text: str) -> float:
    from semaje_tts.evals.metrics import cer
    boundary = uuid.uuid4().hex
    body = io.BytesIO()
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"language\"\r\n\r\n{lang}\r\n".encode())
    body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
               f"Content-Type: audio/wav\r\n\r\n".encode() + wav.read_bytes() + f"\r\n--{boundary}--\r\n".encode())
    req = urllib.request.Request(f"{stt_url}/transcribe", data=body.getvalue(), method="POST",
                                 headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urllib.request.urlopen(req, timeout=600) as res:
        return cer(text, json.loads(res.read()).get("text", ""))


def build(lang: str, out: Path, limit: int = 0, stt_url: str = "", max_cer: float = 0.6,
          splits=("train", "validation")) -> dict:
    import pyarrow.parquet as pq

    wavs = out / "wavs"
    wavs.mkdir(parents=True, exist_ok=True)
    kept, dropped = [], collections.Counter()
    per_speaker = collections.defaultdict(lambda: collections.Counter())
    for split in splits:
        for row in pq.read_table(fetch(lang, split)).to_pylist():
            if limit and len(kept) >= limit:
                break
            text = normalize(row["text"])
            speaker = row["speaker_id"] or "speaker"
            reason = reject_reason(text)
            wav = wavs / f"{row['id']}.wav"
            if reason is None:
                seconds = decode(row["audio"]["bytes"], wav)
                reason = alignment_reason(len(text), seconds)
                if reason is None and stt_url and asr_cer(stt_url, wav, lang, text) > max_cer:
                    reason = "asr-disagrees"
            if reason:
                dropped[reason] += 1
                per_speaker[speaker][reason] += 1
                wav.unlink(missing_ok=True)
                continue
            kept.append((row["id"], speaker, text))
            per_speaker[speaker]["kept"] += 1
    with open(out / "metadata.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\").writerows(kept)
    # Held-out evaluation lines: the test split's text, never trained on.
    heldout = [(r["id"], normalize(r["text"])) for r in pq.read_table(fetch(lang, "test"), columns=["id", "text"]).to_pylist()]
    heldout = [(i, t) for i, t in heldout if reject_reason(t) is None][:50]
    with open(out / "heldout.csv", "w", encoding="utf-8", newline="") as f:
        csv.writer(f, delimiter="|", quoting=csv.QUOTE_NONE, escapechar="\\").writerows(heldout)
    speakers = sorted({s for _, s, _ in kept})
    report = {
        "lang": lang, "kept": len(kept), "dropped": dict(dropped), "speakers": speakers,
        "perSpeaker": {k: dict(v) for k, v in sorted(per_speaker.items())},
        "source": "google/WaxalNLP", "provider": PROVIDERS.get(lang, "see dataset card"),
        "license": "CC-BY-SA-4.0" if lang not in ("aka", "ewe") else "CC-BY-4.0",
        "filters": {"charsPerSecond": [MIN_CPS, MAX_CPS], "maxSeconds": MAX_SECONDS,
                    "asrMaxCer": max_cer if stt_url else None},
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lang", required=True, help="WAXAL code: kik, luo, ibo, hau, yor, wol, lug, swa, aka, ewe, ...")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="keep at most N utterances (smoke tests)")
    ap.add_argument("--stt-url", default=os.getenv("STT_URL", ""), help="enable ASR-agreement filtering")
    ap.add_argument("--max-cer", type=float, default=0.6)
    ap.add_argument("--splits", default="train,validation", help="WAXAL splits to train on (smoke: validation)")
    args = ap.parse_args(argv)
    report = build(args.lang, Path(args.out), args.limit, args.stt_url.rstrip("/"), args.max_cer,
                   splits=tuple(s.strip() for s in args.splits.split(",") if s.strip()))
    print(json.dumps({k: report[k] for k in ("lang", "kept", "dropped", "speakers")}, ensure_ascii=False))
    return 0 if report["kept"] else 1


if __name__ == "__main__":
    sys.exit(main())
