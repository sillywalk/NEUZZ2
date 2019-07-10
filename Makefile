ROOT_PATH=./
PYTHON3_PATH=${HOME}/miniconda3/bin/python 

all: test git

venv: environment.yml
	@- test -d venv || conda env create --prefix=$(ROOT_PATH)/venv -f environment.yml
	@- conda activate $(ROOT_PATH)/venv; conda env update --prefix=$(ROOT_PATH)/venv --file environment.yml

test: venv
	@echo "Running unit tests."
	@echo ""
	@- conda activate $(ROOT_PATH)/venv; nosetests -s --with-coverage ${ROOT_PATH}; conda deactivate
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