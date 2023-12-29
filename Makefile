.PHONY: install clean

install:
	python3 -m venv venv
	chmod +x venv/bin/activate
	source ./venv/bin/activate 
	pip install -r requirements.txt
	python setup.py

clean:
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -exec rm -r {} +
