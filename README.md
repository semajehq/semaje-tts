# semaje-tts

Open-source text-to-speech: Piper by default, MMS-TTS for 1,100 languages when
an operator opts into its non-commercial license, custom voices for everything
else. Contract: [contract.md](contract.md). Part of
[semaje](https://github.com/semajehq/semaje); runs standalone too.

```bash
docker run -p 8012:8012 -v semaje-voices:/voices ghcr.io/semajehq/semaje-tts:1.0.0
curl -X POST localhost:8012/synthesize -H 'content-type: application/json' \
  -d '{"text":"Habari ya asubuhi","language":"sw","format":"mp3"}' -o habari.mp3
```

Native:

```bash
make setup                 # Piper + two default voices (SEMAJE_MMS=1 make setup for MMS-TTS)
make run                   # :8012
make test                  # gate tests
make eval                  # TTS → STT round trip (needs semaje-stt running)
```

The voice catalog and language resolution come from
[semaje-languages](https://github.com/semajehq/semaje-languages).

## Measured round trip (TTS → STT CER, 2026-10-12)

| language | voice | CER |
|---|---|---|
| Swahili | piper `sw_CD-lanfrica-medium` | 5.7% |
| English | piper `en_GB-cori-high` | 7.1% |
| Nepali | piper `ne_NP-chitwan-medium` | 9.1% |
| French | piper `fr_FR-mls-medium` | 17.0% (digit normalization) |
| Kikuyu | mms `kik` (non-commercial) | 4.5% |
| Yoruba | mms `yor` (non-commercial) | 7.1% |
| Amharic | mms `amh` (non-commercial) | 18.2% |

Warm Piper synthesis: ~0.1 s for 4 s of speech on an M4 CPU.

## Training new voices

`training/` builds commercially usable (CC-BY-SA-4.0) multi-speaker Piper voices
from Google WAXAL for Kikuyu, Dholuo, Igbo, Hausa, Yoruba, Wolof, Luganda,
Swahili, Akan, Ewe, Fula, Acholi, and Nyankole. See
[training/README.md](training/README.md): one command on a GPU box per language.

## Coverage gap, stated plainly

No prebuilt open voice exists today for Igbo, Zulu, Xhosa, Wolof, or Luo
(neither Piper nor MMS-TTS ships them). For Igbo, Wolof, and Luo (Dholuo) plus
Kikuyu, `training/` builds one from WAXAL. Zulu and Xhosa are not in WAXAL yet:
any Piper-format voice dropped in `TTS_CUSTOM_VOICE_DIR` is picked up with no
code change.
