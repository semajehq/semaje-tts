"""Gate tests: the voice-training pipeline's pure parts (no torch, no downloads)."""
import json
from pathlib import Path

from training.export import pick_checkpoint, semaje_metadata
from training.text import normalize, reject_reason, symbols, unknown_symbols
from training.train import command
from training.waxal import alignment_reason


def test_normalize_kikuyu():
    assert normalize("Mũndũ o wothe ena kihoto. \nGũtirĩ mũndũ!") == "mũndũ o wothe ena kihoto. gũtirĩ mũndũ!"
    assert normalize("bũrũri_inĩ") == "bũrũri-inĩ"            # WAXAL uses _ where Kikuyu uses -
    assert normalize("“Nĩ ’wega”  #tag") == "nĩ 'wega tag"   # double quotes dropped, ’ → '


def test_tilde_vowels_decompose_into_known_symbols():
    assert symbols("ĩ") == ["i", "̃"]
    assert unknown_symbols(normalize("Gĩkũyũ na Dholuo")) == set()


def test_reject_reasons():
    assert reject_reason("") == "empty"
    assert reject_reason("thaa 3:45") == "digits"
    assert reject_reason("bei ni 5€".replace("5", "")) == "unknown-symbols"
    assert reject_reason(normalize("Ũhoro waku?")) is None


def test_alignment_filter():
    assert alignment_reason(120, 10.0) is None
    assert alignment_reason(330, 167.0) == "too-long"
    assert alignment_reason(30, 10.0) == "audio-longer-than-text"
    assert alignment_reason(98, 1.7) == "text-longer-than-audio"
    assert alignment_reason(5, 0.5) == "too-short"


def _data(tmp_path, speakers):
    (tmp_path / "report.json").write_text(json.dumps({"speakers": speakers}))
    return tmp_path


def test_train_command_warmstarts_then_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr("training.train.base_checkpoint", lambda: Path("/cache/base.ckpt"))
    monkeypatch.setattr("training.train.mos_available", lambda: False)
    data = _data(tmp_path, ["Female 1", "Male 1", "Male 2"])
    first = command(data, tmp_path / "run", 100, 16, "gpu", "1", "16-mixed", -1, None)
    assert first[first.index("--model.num_speakers") + 1] == "3"
    assert first[first.index("--data.phoneme_type") + 1] == "text"
    assert first[first.index("--model.warmstart_ckpt") + 1] == "/cache/base.ckpt"
    assert "--ckpt_path" not in first and "--trainer.max_steps" not in first
    assert first[first.index("--model.mos_metric") + 1] == "none"
    assert first[2:4] == ["training.cli", "fit"]   # our CLI: no crash-prone val_mos default
    resumed = command(data, tmp_path / "run", 100, 16, "gpu", "1", "16-mixed", 50, Path("last.ckpt"))
    assert resumed[resumed.index("--ckpt_path") + 1] == "last.ckpt"
    assert "--model.warmstart_ckpt" not in resumed
    assert resumed[resumed.index("--trainer.max_steps") + 1] == "50"


def test_pick_checkpoint_prefers_best_mos(tmp_path):
    ck = tmp_path / "lightning_logs" / "version_0" / "checkpoints"
    ck.mkdir(parents=True)
    for name in ["last.ckpt", "epoch=10-val_mel=0.5000.ckpt", "epoch=12-val_mos=3.10.ckpt", "epoch=20-val_mos=3.40.ckpt"]:
        (ck / name).write_bytes(b"")
    assert pick_checkpoint(tmp_path, None).name == "epoch=20-val_mos=3.40.ckpt"
    (ck / "epoch=12-val_mos=3.10.ckpt").unlink(); (ck / "epoch=20-val_mos=3.40.ckpt").unlink()
    assert pick_checkpoint(tmp_path, None).name == "epoch=10-val_mel=0.5000.ckpt"


def test_semaje_metadata_records_license_and_attribution():
    report = {"lang": "kik", "speakers": ["Female 2", "Male 1"], "license": "CC-BY-SA-4.0",
              "provider": "Loud and Clear", "kept": 900, "dropped": {"digits": 40}}
    meta = semaje_metadata("semaje-kik-waxal", report, Path("epoch=20-val_mos=3.4.ckpt"), 1)
    assert meta["language"] == "kik" and meta["defaultSpeaker"] == 1 and meta["commercialUse"] == "yes"
    assert "Google WAXAL (Loud and Clear)" in meta["attribution"] and "LibriTTS-R" in meta["attribution"]
    assert meta["name"] == "Kikuyu (WAXAL, 2 speakers)"
