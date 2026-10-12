# semaje-tts — common tasks. The venv lives outside the repo (see setup.sh).
SEMAJE_HOME ?= $(HOME)/.local/share/semaje
PY ?= $(SEMAJE_HOME)/venv-tts/bin/python

.PHONY: setup run test eval image
setup:   ## venv + default Piper voices (SEMAJE_MMS=1 for MMS-TTS)
	./setup.sh
run:     ## serve on :8012
	./run.sh
test:    ## gate tests (needs ffmpeg)
	$(PY) -m pytest -q
eval:    ## TTS → STT round-trip intelligibility (this service + semaje-stt running)
	$(PY) -u -m semaje_tts.evals.eval_roundtrip --langs $(or $(LANGS),en,sw,fr,ne)
image:
	docker build -t semaje-tts .

# ── Voice training (see training/README.md) ───────────────────────────
TRAIN_PY ?= $(SEMAJE_HOME)/venv-train/bin/python
LANG ?= kik
WORK ?= work
ACCEL ?= auto
EPOCHS ?= 2000
.PHONY: train-setup voice-data voice-train voice-export voice
train-setup:  ## training venv (torch, lightning, piper[train]) + alignment extension
	training/setup.sh
voice-data:   ## WAXAL → filtered Piper dataset in $(WORK)/$(LANG)  (STT_URL=... adds ASR filtering)
	$(TRAIN_PY) -m training.waxal --lang $(LANG) --out $(WORK)/$(LANG) $(if $(STT_URL),--stt-url $(STT_URL))
voice-train:  ## fine-tune (resumes automatically); ACCEL=gpu on a CUDA box
	$(TRAIN_PY) -m training.train --data $(WORK)/$(LANG) --run $(WORK)/run-$(LANG) --accelerator $(ACCEL) --max-epochs $(EPOCHS)
voice-export: ## best checkpoint → semaje custom voice in $(WORK)/voices/semaje-$(LANG)-waxal
	$(TRAIN_PY) -m training.export --run $(WORK)/run-$(LANG) --data $(WORK)/$(LANG) --out $(WORK)/voices/semaje-$(LANG)-waxal
voice: voice-data voice-train voice-export
