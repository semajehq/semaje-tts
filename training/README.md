# Training voices for languages nobody ships

Build commercially usable Piper voices (CC-BY-SA-4.0) from Google's WAXAL
TTS recordings: Kikuyu, Dholuo, and every other WAXAL language (Igbo, Hausa,
Yoruba, Wolof, Luganda, Swahili, Akan, Ewe, Fula, Acholi, Nyankole).

Credit where it's due: C-elo Labs showed that WAXAL plus African Next Voices
gives good Kikuyu voices (their fine-tuned MMS-TTS model is non-commercial).
We use the same openly licensed WAXAL recordings with a different recipe so the
result can be used commercially: a Piper (VITS) model warm-started from Piper's
LibriTTS-R checkpoint (CC-BY-4.0 data), with characters as symbols because
espeak-ng has no Kikuyu or Dholuo phonemizer.

## On a GPU box (recommended)

```bash
docker build -f training/Dockerfile -t semaje-tts-train .
docker run --gpus all -v $PWD/work:/app/work semaje-tts-train make voice LANG=kik ACCEL=gpu EPOCHS=2000
docker run --gpus all -v $PWD/work:/app/work semaje-tts-train make voice LANG=luo ACCEL=gpu EPOCHS=2000
```

Or natively: `make train-setup`, then `make voice LANG=kik ACCEL=gpu`.

| Step | What happens | Output |
|---|---|---|
| `voice-data` | downloads WAXAL `<lang>_tts`, drops misaligned / digit / unknown-symbol rows (and, with `STT_URL`, rows semaje-stt can't match), resamples to 22.05 kHz | `work/<lang>/` + `report.json` (what was dropped and why, per speaker) |
| `voice-train` | warm-starts from LibriTTS-R, trains a multi-speaker model (one speaker per WAXAL voice); resumes from `last.ckpt` if interrupted | `work/run-<lang>/` (TensorBoard logs, checkpoints) |
| `voice-export` | picks the best checkpoint (by predicted MOS, then mel loss) and writes the voice bundle | `work/voices/semaje-<lang>-waxal/` |

Disk: about 10 GB per language (WAXAL audio ~0.8 GB, base checkpoint ~0.9 GB,
up to 5 training checkpoints at ~0.9 GB each).

Expect roughly 1,000–1,400 usable clips per language after filtering. On one
A10G/A100, 2,000 epochs takes a few hours; listen to checkpoints along the
way (`tensorboard --logdir work/run-kik`) and stop when it sounds clean.

**Data quality matters more than steps.** WAXAL's Kikuyu set includes a
speaker who describes pictures in her own words instead of reading the script,
plus clips whose audio is several times longer than the text. The length filter
catches the second problem; pass `STT_URL=http://<semaje-stt>:8011` to
`make voice-data` to catch the first (rows whose transcription disagrees with
the script by more than 60% CER are dropped).

## Install the voice

```bash
cp work/voices/semaje-kik-waxal/* /path/to/voices/custom/        # TTS_CUSTOM_VOICE_DIR
# Docker: docker compose cp work/voices/semaje-kik-waxal/. tts:/voices/custom/
```

semaje-tts picks it up on the next request: commercially licensed custom voices
outrank the non-commercial MMS fallback. Choose a speaker with
`{"speaker": N}` (names are in the `.semaje.json`); the default is
`--default-speaker` at export.

## Measure it

```bash
make eval LANGS=kik,luo     # TTS → STT round trip, CER against held-out WAXAL test lines
```

## License of the result

WAXAL's Kikuyu and Dholuo recordings are CC-BY-SA-4.0 (Loud and Clear for
Google), and the base checkpoint's data is CC-BY-4.0, so voices are released
under **CC-BY-SA-4.0**: commercial use is allowed, with attribution and
share-alike. The `.semaje.json` carries the attribution string; semaje-tts
returns it in the `x-semaje-license` header of every response.
