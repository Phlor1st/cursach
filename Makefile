# GNU Make (Git Bash / WSL). На чистом PowerShell используйте: python run_pipeline.py <команда>
PYTHON ?= python

.PHONY: process synthetic train copy-models

process:
	$(PYTHON) run_pipeline.py process

process-no-origin:
	set RAMAN_USE_ORIGIN=0 && $(PYTHON) run_pipeline.py process --no-origin

synthetic:
	$(PYTHON) run_pipeline.py synthetic

train:
	$(PYTHON) run_pipeline.py train

copy-models:
	$(PYTHON) run_pipeline.py copy-models
