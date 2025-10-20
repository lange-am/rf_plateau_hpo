@echo off
python -m venv .venv
call .venv\Scripts\activate
python -m pip install -U pip
pip install -e .[dev]
python -m ipykernel install --user --name rf-plateau-hpo --display-name "rf-plateau-hpo"
jupyter lab
