#!/usr/bin/env bash
set -e
python -m venv .venv
. .venv/bin/activate
python -m pip install -U pip
pip install -e ".[dev]"
python -m ipykernel install --user --name rf-plateau-hpo --display-name "rf-plateau-hpo"
exec jupyter lab
