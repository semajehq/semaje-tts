"""Fine-tune a multi-speaker Piper voice on a training set built by training.waxal.

    python -m training.train --data work/kik --run runs/kik [--max-epochs 2000] [--accelerator gpu]

Warm-starts from Piper's LibriTTS-R medium checkpoint (904 speakers, CC-BY-4.0
data) with a non-strict load: every matching-shape layer (text encoder, flow,
decoder, speaker conditioning) transfers and the speaker table is re-learned
for the WAXAL speakers. Symbols are characters (phoneme_type=text). Resumes
automatically from runs/<lang>/lightning_logs/**/last.ckpt when present.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/libritts_r/medium/best.ckpt"
BASE_LICENSE = "Piper libritts_r medium (MIT code; LibriTTS-R data CC-BY-4.0)"
CACHE = Path(os.getenv("PIPER_BASE_CACHE", str(Path.home() / ".cache" / "semaje" / "piper-base")))


def base_checkpoint() -> Path:
    path = CACHE / "libritts_r-medium-best.ckpt"
    if not path.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".part")
        urllib.request.urlretrieve(BASE_URL, tmp)
        tmp.replace(path)
    return path


def mos_available() -> bool:
    """Piper's UTMOS predictor needs torchaudio, which piper-tts[train] does not install."""
    import importlib.util
    return importlib.util.find_spec("torchaudio") is not None


def latest_checkpoint(run: Path) -> Path | None:
    found = sorted(run.glob("lightning_logs/*/checkpoints/last.ckpt"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def command(data: Path, run: Path, max_epochs: int, batch_size: int, accelerator: str,
            devices: str, precision: str, max_steps: int, resume: Path | None) -> list[str]:
    speakers = json.loads((data / "report.json").read_text(encoding="utf-8"))["speakers"]
    use_mos = mos_available()
    args = [
        sys.executable, "-m", "training.cli", "fit",
        "--model.mos_metric", "utmos" if use_mos else "none",
        "--data.voice_name", run.name,
        "--data.csv_path", str(data / "metadata.csv"),
        "--data.audio_dir", str(data / "wavs"),
        "--data.cache_dir", str(run / "cache"),
        "--data.config_path", str(run / "config.json"),
        "--data.espeak_voice", "en-us",          # unused with phoneme_type=text, but required
        "--data.phoneme_type", "text",
        "--data.batch_size", str(batch_size),
        "--data.num_workers", str(min(8, os.cpu_count() or 1)),
        "--model.sample_rate", "22050",
        "--model.num_speakers", str(len(speakers)),
        "--trainer.accelerator", accelerator,
        "--trainer.devices", devices,
        "--trainer.precision", precision,
        "--trainer.max_epochs", str(max_epochs),
        "--trainer.default_root_dir", str(run),
        "--trainer.check_val_every_n_epoch", "1",
    ]
    if max_steps > 0:
        args += ["--trainer.max_steps", str(max_steps)]
    if resume is not None:
        args += ["--ckpt_path", str(resume)]
    else:
        args += ["--model.warmstart_ckpt", str(base_checkpoint())]
    return args


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--max-epochs", type=int, default=2000)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke tests)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--accelerator", default="auto", help="gpu | cpu | mps | auto")
    ap.add_argument("--devices", default="auto")
    ap.add_argument("--precision", default="32-true", help="32-true (safe) | 16-mixed (faster on GPUs)")
    args = ap.parse_args(argv)
    run = Path(args.run)
    run.mkdir(parents=True, exist_ok=True)
    resume = latest_checkpoint(run)
    cmd = command(Path(args.data), run, args.max_epochs, args.batch_size, args.accelerator,
                  args.devices, args.precision, args.max_steps, resume)
    (run / "train-command.txt").write_text(" ".join(cmd) + "\n", encoding="utf-8")
    print(("resuming from " + str(resume)) if resume else "warm-starting from LibriTTS-R", flush=True)
    env = {**os.environ, "SEMAJE_TRAIN_MOS": "1" if mos_available() else "0"}
    return subprocess.call(cmd, env=env)


if __name__ == "__main__":
    sys.exit(main())
