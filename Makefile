TEST_PATH=./
PYTHON3_PATH=${HOME}/miniconda3/bin/python 

all: test git

venv: venv/bin/activate

venv/bin/activate: requirements.txt
	@- test -d venv || virtualenv -p ${PYTHON3_PATH} venv
	@- . venv/bin/activate; pip install -Ur requirements.txt
	@- touch venv/bin/activate

test: venv
	@echo "Running unit tests."
	@echo ""
	@- . venv/bin/activate; nosetests -s --with-coverage ${TEST_PATH}; deactivate
	@ rm -rf venv 
	@echo ""

clean:
	@echo "Cleaning *.pyc, *.DS_Store, and other junk files..."
	@- find . -name '*.pyc' -exec rm -f {} +
	@- find . -name '*.pyo' -exec rm -f {} +
	@- find . -name '__pycache__' -exec rm -rf {} +
	@echo ""

git: clean
	@echo "Syncing with repository"
	@echo ""
	@- git add --all .
	@- git commit -am "Autocommit"
	@- git push origin master