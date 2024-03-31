PYTHON := $(shell command -v python3 || command -v python)
PIP_OR_BREW := $(shell command -v pip || command -v brew)

.PHONY: virtual_env

all: virtual_env

virtual_env:
	[ -d myenv ] || @$(PYTHON) -m venv myenv
	. myenv/bin/activate 
	@$(PIP_OR_BREW) install -r requirements.txt

run_command:
	@$(PYTHON) -c "x = 1; print(x)"

