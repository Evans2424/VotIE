# VotIE: Voting Information Extraction from Municipal Meeting Minutes

Official repository for the paper "Voting Information Extraction (VotIE): 
Structured Information Extraction from Municipal Meeting Minutes"

[![License](https://img.shields.io/badge/License-MIT-blue.svg)]()
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)]()

## Quick Start
```bash
# Clone repository
git clone https://github.com/yourname/votie.git
cd votie

# Install dependencies
pip install -r requirements.txt

# Download data (if hosted separately)
# python scripts/download_data.py

# Train model
python scripts/train.py --config configs/deberta_crf.yaml

# Evaluate
python scripts/evaluate.py --model models/deberta-crf-best --data data/votie_conll/test.conll