#!/bin/bash

REQUIREMENTS_FILE=$1

python3 -m venv venv

source ./venv/bin/activate

pip install --upgrade pip
pip install -r $REQUIREMENTS_FILE

deactivate
