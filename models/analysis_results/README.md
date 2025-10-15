# VotIE Model Results Analysis

This directory contains comprehensive analysis of all model experiments for the VotIE project.

## Generated Files

### 1. Span-Level Results
- **`span_level_overall.csv`**: Overall span-level metrics (precision, recall, F1) for all models
- **`span_level_per_entity.csv`**: Detailed metrics broken down by entity type
- **`per_entity_f1_pivot.csv`**: Pivot table showing F1 scores across all models and entity types
- **`per_entity_precision_pivot.csv`**: Pivot table for precision metrics
- **`per_entity_recall_pivot.csv`**: Pivot table for recall metrics

### 2. Event-Level Results
- **`event_level_metrics.csv`**: Event-level performance including:
  - Event detection accuracy
  - Complete event accuracy
  - Subject extraction F1
  - Participant-vote extraction F1

### 3. LaTeX Tables (Ready for Paper)
- **`table_span_results.tex`**: Span-level performance table
  - Values shown as percentages
  - **Best values in each column are highlighted in bold**
  
- **`table_event_results.tex`**: Event-level performance table
  - Values shown as percentages
  - **Best values in each column are highlighted in bold**

### 4. Error Analysis
- **`deberta_crf_error_analysis.json`**: Detailed error analysis for DeBERTa-CRF model
  - Subject errors (missing, boundary, incorrect)
  - Participant-vote errors (missing, extra, position wrong)
  - Counting errors
  - Voting expression errors
  - Event detection errors
  - Multiple simultaneous errors

- **`deberta_crf_error_analysis_table.tex`**: LaTeX table of error distribution

## Key Findings

### Span-Level Performance (Best Model: DeBERTa-CRF)
- **Best Overall F1**: DeBERTa-CRF (93.0%)
- **Best Precision**: DeBERTa-CRF (91.1%)
- **Best Recall**: DeBERTa-Linear (95.6%)

### Event-Level Performance (Best Model: DeBERTa-CRF)
- **Best Complete Event Accuracy**: DeBERTa-CRF (83.3%)
- **Best Subject F1**: DeBERTa-CRF (87.6%)
- **Best Participant-Vote F1**: XLM-R-CRF (98.1%)
- **Best Event Detection**: BERT-Linear and XLM-R-Linear (95.4%)

### Best Models by Entity Type
- **COUNTING-MAJORITY**: CRF (96.3%)
- **COUNTING-UNANIMITY**: DeBERTa-CRF (97.2%)
- **SUBJECT**: DeBERTa-CRF (84.3%)
- **VOTER-ABSENT**: BERT-CRF (95.5%)
- **VOTER-ABSTENTION**: DeBERTa-CRF (93.9%)
- **VOTER-AGAINST**: DeBERTa-Linear (97.6%)
- **VOTER-FAVOR**: XLM-R-CRF (95.3%)
- **VOTING**: XLM-R-CRF (96.8%)

## Usage in Paper

The generated LaTeX tables can be directly included in your paper:

```latex
% Span-level results
\input{tables/table_span_results.tex}

% Event-level results
\input{tables/table_event_results.tex}

% Error analysis (if generated)
\input{tables/deberta_crf_error_analysis_table.tex}
```

## Regenerating Results

To regenerate all analysis files:

```bash
cd /home/jevans/VotIE/models
python3 analyze_results.py
```

This will:
1. Load all `evaluation_results.json` files from model directories
2. Generate CSV files with detailed metrics
3. Create LaTeX tables with proper formatting
4. Run error analysis on DeBERTa-CRF model

## Notes

- All tables use percentage formatting for better readability
- Best values in each column are **highlighted in bold**
- The script automatically identifies the best-performing model for each metric
- Error analysis requires the EventConstructor utility to convert BIO predictions to event format
