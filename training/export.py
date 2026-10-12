"""Turn a trained checkpoint into a semaje custom voice.

    python -m training.export --run runs/kik --data work/kik --out voices/semaje-kik-waxal

Writes <id>.onnx + <id>.onnx.json (Piper format) + <id>.semaje.json with the
language, speakers, license, and attribution. Copy the three files into
TTS_CUSTOM_VOICE_DIR (or the tts container's /voices/custom) and semaje-tts
serves the voice with no code change.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from training.train import BASE_LICENSE

LANGUAGE_NAMES = {"kik": "Kikuyu", "luo": "Dholuo", "ibo": "Igbo", "hau": "Hausa", "yor": "Yoruba", "wol": "Wolof",
                  "lug": "Luganda", "swa": "Swahili", "aka": "Akan", "ewe": "Ewe", "ful": "Fula", "ach": "Acholi",
                  "nyn": "Nyankole"}


def pick_checkpoint(run: Path, explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    # Prefer the best-by-MOS checkpoint, then best-by-mel, then last.
    for pattern in ("epoch=*-val_mos=*.ckpt", "epoch=*-val_mel=*.ckpt", "last.ckpt"):
        found = list(run.glob(f"lightning_logs/*/checkpoints/{pattern}"))
        if found:
            if "val_mos" in pattern:
                return max(found, key=lambda p: float(p.stem.split("val_mos=")[1]))
            if "val_mel" in pattern:
                return min(found, key=lambda p: float(p.stem.split("val_mel=")[1]))
            return max(found, key=lambda p: p.stat().st_mtime)
    raise FileNotFoundError(f"no checkpoint under {run}")


def semaje_metadata(voice_id: str, report: dict, checkpoint: Path, default_speaker: int) -> dict:
    lang = report["lang"]
    return {
        "language": lang,
        "name": f"{LANGUAGE_NAMES.get(lang, lang)} (WAXAL, {len(report['speakers'])} speakers)",
        "quality": "medium",
        "speakers": report["speakers"],
        "defaultSpeaker": default_speaker,
        "license": report["license"],
        "commercialUse": "yes",  # CC-BY(-SA): commercial use allowed with attribution (and share-alike for SA)
        "attribution": (f"Trained by semaje on Google WAXAL ({report['provider']}), {report['license']}; "
                        f"warm-started from {BASE_LICENSE}."),
        "training": {"kept": report["kept"], "dropped": report["dropped"], "checkpoint": checkpoint.name},
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint")
    ap.add_argument("--default-speaker", type=int, default=0)
    args = ap.parse_args(argv)
    run, data, out = Path(args.run), Path(args.data), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    voice_id = out.name
    ckpt = pick_checkpoint(run, args.checkpoint)
    subprocess.run([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(ckpt),
                    "--output-file", str(out / f"{voice_id}.onnx")], check=True)
    shutil.copy(run / "config.json", out / f"{voice_id}.onnx.json")
    report = json.loads((data / "report.json").read_text(encoding="utf-8"))
    meta = semaje_metadata(voice_id, report, ckpt, args.default_speaker)
    (out / f"{voice_id}.semaje.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"voice ready: {out} (from {ckpt.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
