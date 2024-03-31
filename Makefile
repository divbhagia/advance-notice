.PHONY: virtual_env

all: virtual_env

virtual_env:
	[ -d myenv ] || python -m venv myenv
	. myenv/bin/activate 
	pip install -r requirements.txt

run_command:
	. myenv/bin/activate
	python -c "x = 1; print(x)"

