# Define shell
SHELL := /bin/bash

# Python path
export PYTHONPATH := $(shell pwd)

# Manuscript name
MANUSCRIPT := figures_and_tables

# Scripts for main program
SCRIPTS := $(shell ls ./scripts/*.py)
SCRIPTS := $(filter-out ./scripts/00_init.py, $(SCRIPTS))
SCRIPTS := $(filter-out ./scripts/01_clean_data.py, $(SCRIPTS))

# Files to clean before running the main program (optional)
CLEANFILES += $(shell find ./scripts/quants/ -type f \! -name "sim*" ! -name "ext*")
CLEANFILES += $(shell ls ./output/*)
CLEANFILES += $(shell ls ./data/*)

# Default target
all: act-venv mainprog manuscript
all_clean: act-venv clean all

act-venv:
	@eval "$(pyenv virtualenv-init -)"
	@eval "$(activate env-notice)"
	@pip install -r requirements.txt

clean:
	for files in $(CLEANFILES); do \
		rm -f $$files; \
	done
	python ./scripts/00_init.py
	echo "Data cleaning..."
	python ./scripts/01_clean_data.py
	echo "Data cleaning done."

mainprog: 
	for script in $(SCRIPTS); do \
		echo $$script; \
		python $$script; \
	done

manuscript:
	cd ./draft; \
	pdflatex $(MANUSCRIPT).tex; \
	bibtex $(MANUSCRIPT); \
	pdflatex $(MANUSCRIPT).tex; \
	pdflatex $(MANUSCRIPT).tex; \
	rm -f $(MANUSCRIPT).aux $(MANUSCRIPT).bbl $(MANUSCRIPT).blg $(MANUSCRIPT).log $(MANUSCRIPT).out $(MANUSCRIPT).toc