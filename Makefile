# Define shell
SHELL := /bin/bash

# Python path
export PYTHONPATH := $(shell pwd)

# Define variables
MANUSCRIPT := figures_and_tables

# Default target
all: prepdata estimation manuscript
all_clean: clean prepdata estimation manuscript

# ADD to create virtual environment later
act_venv:
	pyenv activate env-notice
	pip install -r requirements.txt

clean:
	rm -rf ./data/*
	rm -rf ./data/raw/*
	rm -rf ./output/*
	rm -f ./scripts/quants/*
	python ./scripts/00_init.py
	python ./scripts/01_clean_data.py

prepdata:
	python ./scripts/02_sample_and_ipw.py
	python ./scripts/03_summary_stats.py
	python ./scripts/04_reg_tables.py
	python ./scripts/05_hazard_plots.py

estimation:
	python ./scripts/e01_estimation.py

manuscript:
	cd ./draft; \
	pdflatex $(MANUSCRIPT).tex; \
	bibtex $(MANUSCRIPT); \
	pdflatex $(MANUSCRIPT).tex; \
	pdflatex $(MANUSCRIPT).tex; \
	rm -f $(MANUSCRIPT).aux $(MANUSCRIPT).bbl $(MANUSCRIPT).blg $(MANUSCRIPT).log $(MANUSCRIPT).out $(MANUSCRIPT).toc