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
