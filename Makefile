.PHONY: help
help:
	@echo "Project Makefile"

.PHONY: install-deps
install-deps:
	poetry install

.PHONY: install-pytorch
install_pytorch:
	poetry run pip uninstall torch --yes
	poetry run pip install torch  --extra-index-url https://download.pytorch.org/whl/cu116

.PHONY: format
format:
	@echo "--> Format the python code"
	poetry run autoflake --in-place -r ./  --ignore-init-module-imports --remove-all-unused-imports --remove-unused-variables
	poetry run black --line-length 100 .
	poetry run isort . --profile black

.PHONY: setup-wandb
setup-wandb:
	poetry run wandb login

.PHONY: trim-notebooks
trim_notebooks:
	find . -path ./.venv -prune -o -name "*.ipynb" -exec poetry run jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace {} \;

.PHONY: build-datasets
build-datasets:
	poetry run python scripts/create_cv_folds.py data/raw_datasets/wtbdata_245days.csv data/train_val_test 1 --n_test_days=80 --n_val_days=0 --gap=0
	poetry run python scripts/create_cv_folds.py data/raw_datasets/wtbdata_245days.csv data/cv_folds 5 --n_test_days=80 --n_val_days=0 --gap=0

.PHONY: score
score:
	@read -p "Enter the name of the folder containing the model inside the archive/ folder: " folder; \
	read -p "Do you want to save the results? [true/false]: " save; \
	poetry run python scoring/evaluation.py --model_path ./archive/$$folder --save_results $$save

.PHONY: train
train:
	@read -p "Enter the name of the folder containing the model inside the archive/ folder: " folder; \
	read -p "Enter the name of the WANDB project that will store the training results: " wandb_project; \
	poetry run python ./archive/$$folder/training.py /code/data/train_val_test/train/ /code/data/train_val_test/test/ $$wandb_project

.PHONY: train-cv
train-cv:
	@read -p "Enter the name of the folder containing the model inside the archive/ folder: " folder; \
	read -p "Enter the name of the WANDB project that will store the training results: " wandb_project; \
	read -p "Enter the name of this run model: " prefix_name_run; \
	poetry run python scripts/metatrainer.py ./archive/$$folder/training.py /code/data/cv_folds/ $$wandb_project $$prefix_name_run


.PHONY: build-containers
build-containers:
	docker-compose -f ./docker/docker-compose.yaml up --build

.PHONY: run-containers
run-containers:
	docker-compose -f ./docker/docker-compose.yaml up

.PHONY: remove-containers
remove-containers:
	docker-compose -f ./docker/docker-compose.yaml down --rmi 'all'