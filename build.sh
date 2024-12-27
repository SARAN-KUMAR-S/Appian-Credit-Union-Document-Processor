#!/usr/bin/env bash
set -e

apt-get update -y && apt-get install -y poppler-utils tesseract-ocr
pip install -r requirements.txt

python -m spacy download en_core_web_sm
