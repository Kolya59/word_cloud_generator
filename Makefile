.PHONY: install clean

install:
	python3 -m venv venv
	source ./venv/bin/activate 
	pip install -r requirements.txt

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
