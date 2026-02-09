# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VotIE (Voting Information Extraction) is a research framework for extracting structured voting information from Portuguese municipal meeting minutes, under review for ACL 2026. It implements two approaches:

1. **Discriminative sequence labeling** with BIO tagging (CRF, BiLSTM, BERTimbau, DeBERTa, XLM-RoBERTa)
2. **LLM-based extraction** using Google Gemini with zero/few-shot prompting

The system extracts 8 entity types: VOTER-FAVOR, VOTER-AGAINST, VOTER-ABSTENTION, VOTER-ABSENT, VOTING, SUBJECT, COUNTING-UNANIMITY, COUNTING-MAJORITY.

Best model: XLM-RoBERTa + CRF (93.2% macro F1). Full dataset available at https://rdm.inesctec.pt/dataset/cs-2025-007.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Quick start with sample data (no full dataset needed)
python scripts/quick_start.py
python scripts/evaluate_sample_data.py --sample-file data/sample_data.jsonl

# Train a model
python scripts/train.py --config configs/xlmr_crf.yaml
python scripts/train.py --config configs/deberta_crf.yaml --experiment-name my_experiment

# Full pipeline (train → predict → evaluate)
python scripts/run_pipeline.py --config configs/deberta_crf.yaml --name my_experiment
python scripts/run_pipeline.py --config configs/deberta_crf.yaml --name my_experiment --predict-only
python scripts/run_pipeline.py --config configs/deberta_crf.yaml --name my_experiment --evaluate-only

# Predict and evaluate separately
python scripts/predict.py models/deberta_crf/my_experiment/best_model data/citilink_spans/test.jsonl predictions/output.jsonl cuda
python scripts/evaluate.py predictions/output.jsonl evaluation/results.json

# Cross-municipality (LOMO) experiments
python scripts/run_municipality_experiments.py

# LLM extraction (requires GOOGLE_API_KEY)
python scripts/llm_extraction/extract_gemini_spans.py --strategy zero_shot
python scripts/llm_extraction/extract_gemini_spans.py --strategy few_shot
python scripts/llm_extraction/evaluate_spans.py results/llm_extraction/gemini_few_standard.jsonl

# Validate configs
python scripts/validate_experiments.py --smoke-test
```

## Architecture

### Data Flow

Input JSONL (span or BIO format) → `src/data/dataset.py` loads and converts to BIO tags via `src/data/span_to_bio.py` → tokenization with subword alignment → model training/inference → BIO validation → entity-level evaluation with seqeval.

### Key Architectural Patterns

- **Unified training**: `scripts/train.py` auto-detects model type from YAML config. All configs live in `configs/`.
- **Base model pattern**: `src/models/base.py` (`BaseVotIEModel`) provides shared functionality (weighted loss, BIO validation, bias initialization). Each model subclass (`xlmr_models.py`, `deberta_models.py`, `bertimbau_models.py`, `crf.py`, `bilstm_crf.py`) implements architecture-specific logic.
- **Two trainers**: `src/trainer.py` handles transformer models; `src/bilstm_trainer.py` handles BiLSTM separately.
- **Windowing**: Documents exceeding 512 tokens are split into overlapping windows (50-token overlap) and reassembled after inference.
- **Class imbalance**: O-tag weight reduced to 0.01 in loss computation to handle severe class imbalance.
- **LLM module**: `src/llm_extraction/` is self-contained with its own base class (`shared/base_extractor.py`), Gemini-specific implementation (`gemini/`), prompt strategies (`gemini/prompts.py`), and few-shot selection (`few_shot_selector.py`).

### Data Formats

- **Span format** (`data/citilink_spans/`): `{"id": ..., "text": ..., "spans": [{"text": ..., "type": ..., "start": ..., "end": ...}]}`
- **BIO format** (`data/citilink_bio/`): `{"id": ..., "text": ..., "tokens": [...], "labels": ["B-VOTER-FAVOR", "I-VOTER-FAVOR", ...]}`
- Sample data (30 examples) included at `data/sample_data.jsonl`; full dataset must be downloaded from RDM.

### Configuration

YAML configs in `configs/` control model architecture, training hyperparameters, data paths, and windowing. Municipality-specific LOMO configs are in `configs/municipality_experiments/`. Key settings: model name, architecture (linear/crf), learning rate, batch size, epochs, patience, seed (42), device.

## Key Dependencies

PyTorch, Hugging Face Transformers, pytorch-crf, seqeval, sklearn-crfsuite, google-generativeai, pydantic, PyYAML.
