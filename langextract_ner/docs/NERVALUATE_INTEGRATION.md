# Enhanced NER Evaluation with Nervaluate

The LangExtract system now supports **enhanced evaluation metrics** using the `nervaluate` library for more comprehensive Named Entity Recognition assessment.

## 🚀 Installation

```bash
pip install nervaluate
```

## 📊 What Nervaluate Adds

### **4 Evaluation Schemes**

| Scheme | Description | Use Case |
|--------|-------------|----------|
| **Strict** | Exact boundary + exact entity type | Gold standard evaluation |
| **Exact** | Exact boundary, any type | Boundary detection accuracy |
| **Partial** | Overlapping boundaries, any type | Near-miss detection |
| **Type** | Any overlap + correct type | Entity type classification |

### **5 Error Types Tracked**

- **COR** (Correct): Perfect matches
- **INC** (Incorrect): Wrong entity type but correct boundary
- **PAR** (Partial): Partial boundary overlap
- **MIS** (Missing): False negatives (missed entities)
- **SPU** (Spurious): False positives (hallucinated entities)

## 🎯 Benefits for Municipal Voting NER

### **1. Boundary Flexibility**
Municipal text often has ambiguous entity boundaries:
```
"o Executivo Municipal de Lisboa deliberou"
vs
"o Executivo Municipal deliberou"
```
- **Exact scheme**: Catches boundary accuracy regardless of type
- **Partial scheme**: Credits partial matches

### **2. Type Classification Analysis**
Distinguishes between boundary errors vs classification errors:
```
Predicted: "deliberou" → SUBJECT
Truth:     "deliberou" → VOTING
```
- **INC error**: Correct boundary, wrong type
- Helps identify if model has boundary detection vs classification issues

### **3. Entity Detection vs Recognition**
```
"por unanimidade aprovou a proposta"
```
- **Type scheme**: Credits finding "por unanimidade" even with wrong boundaries
- **Partial scheme**: Credits overlapping detection

## 📈 Report Integration

The enhanced evaluation provides **comparative analysis**:

### **Executive Summary**
```markdown
## Performance Summary

### Overall Metrics (Seqeval IOB2)
| Metric | Score |
|--------|-------|
| Precision | 0.875 |
| Recall | 0.823 |
| F1-Score | 0.848 |

### Enhanced Metrics (Nervaluate)
| Evaluation Scheme | Precision | Recall | F1-Score |
|-------------------|-----------|--------|----------|
| **Strict** | 0.875 | 0.823 | 0.848 |
| **Exact** | 0.921 | 0.867 | 0.893 |
| **Partial** | 0.945 | 0.891 | 0.917 |
| **Type** | 0.892 | 0.845 | 0.868 |
```

### **Key Insights**
- **Strict vs Exact**: Shows entity type classification accuracy
- **Exact vs Partial**: Shows boundary precision importance
- **Partial vs Type**: Shows overlap detection capability

## 🔧 Usage Examples

### **Automatic Integration**
The system automatically uses nervaluate if available:

```python
from evaluation import VotingEntityEvaluator

evaluator = VotingEntityEvaluator(data_loader)

# Automatically includes both seqeval and nervaluate metrics
report = evaluator.evaluate_extraction_results(predictions, "pt", municipalities)

# Access enhanced metrics
if report.nervaluate_metrics:
    strict_f1 = report.nervaluate_metrics['strict']['overall']['f1']
    partial_f1 = report.nervaluate_metrics['partial']['overall']['f1']

    print(f"Strict F1: {strict_f1:.3f}")
    print(f"Partial F1: {partial_f1:.3f}")
    print(f"Boundary tolerance: {partial_f1 - strict_f1:.3f}")
```

### **Error Analysis**
```python
# Analyze error patterns by scheme
strict_errors = report.nervaluate_metrics['strict']
for entity_type, metrics in strict_errors.items():
    if entity_type != 'overall':
        cor = metrics.get('correct', 0)
        inc = metrics.get('incorrect', 0)
        par = metrics.get('partial', 0)
        mis = metrics.get('missing', 0)
        spu = metrics.get('spurious', 0)

        print(f"{entity_type}:")
        print(f"  Correct: {cor}, Type errors: {inc}")
        print(f"  Partial: {par}, Missing: {mis}, Spurious: {spu}")
```

## 📋 Publication Benefits

### **Research Papers**
- **Comparative evaluation** against multiple schemes
- **Detailed error anaaccuracylysis** for different entity types
- **Boundary sensitivity analysis** (strict vs partial)
- **Classification ** (exact vs type)

### **Model Development**
- **Diagnose specific issues**: boundary detection vs type classification
- **Optimize for use case**: strict accuracy vs coverage
- **Track improvement areas**: which error types decrease over iterations

### **Reproducibility**
- **Standard evaluation framework** used across NER research
- **Multiple metrics** for comprehensive assessment
- **Detailed error breakdown** for transparency

## ⚡ Performance Impact

- **Minimal overhead**: ~10-20% additional evaluation time
- **Optional integration**: Falls back to seqeval if nervaluate unavailable
- **Comprehensive output**: 4x more detailed metrics for publication

## 🔍 Example Output

```json
{
  "strict": {
    "overall": {"precision": 0.875, "recall": 0.823, "f1": 0.848},
    "VOTER-FAVOR": {"precision": 0.920, "recall": 0.865, "f1": 0.892}
  },
  "exact": {
    "overall": {"precision": 0.921, "recall": 0.867, "f1": 0.893}
  },
  "partial": {
    "overall": {"precision": 0.945, "recall": 0.891, "f1": 0.917}
  },
  "type": {
    "overall": {"precision": 0.892, "recall": 0.845, "f1": 0.868}
  }
}
```

This comprehensive evaluation framework provides **publication-ready metrics** that meet the highest standards for NER research and practical deployment assessment.