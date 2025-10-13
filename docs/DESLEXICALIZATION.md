# Deslexicalization Guide

## Overview

Deslexicalization is a data preprocessing technique that replaces specific named entities in training data with generic tokens or synthetic alternatives. This helps models generalize better by preventing overfitting to specific individuals, municipalities, or locations mentioned in the training data.

## Why Use Deslexicalization?

Municipal meeting minutes often contain:
- **Specific person names** (voters, officials) that may not appear in new documents
- **Municipality-specific terms** (city names, local institutions) that vary across regions
- **Location references** (streets, buildings) unique to each municipality

By deslexicalizing these entities, models learn to recognize the *patterns* around entities rather than memorizing specific names, leading to better cross-municipality and cross-domain generalization.

## How It Works

The deslexicalization system replaces three types of entities:

### 1. VOTER Entities (Person Names)
- **Strategy**: Uses intelligent classification to distinguish actual person names from institutional references
- **Replacement Options**:
  - `use_fake_names: true` - Replaces with realistic fake names using Faker library
  - `use_fake_names: false` - Replaces with generic `[PERSON]` token
- **Examples**:
  - "Dr. João Silva" → "Dr. António Costa" (fake name)
  - "Maria Santos" → "Beatriz Oliveira" (fake name)
  - "Senhor Vereador José" → `[PERSON]` (generic token)

### 2. Municipality Terms
- **Strategy**: Identifies municipality names and related terms
- **Replacement**: `@MUNICIPALITY` placeholder
- **Examples**:
  - "Câmara Municipal do Porto" → "Câmara Municipal @MUNICIPALITY"
  - "Lisboa" → "@MUNICIPALITY"
  - "Município de Coimbra" → "Município de @MUNICIPALITY"

### 3. Location References
- **Strategy**: Detects location-specific terms
- **Replacement**: `@LOCAL` placeholder
- **Examples**:
  - "Praça do Município" → "@LOCAL"
  - "Rua Principal" → "@LOCAL"
  - "Edifício dos Paços do Concelho" → "@LOCAL"

## Configuration

Add a `deslexicalization` section to your experiment config file:

```yaml
deslexicalization:
  # Master switch - set to true to enable
  enabled: true

  # VOTER entity replacement strategy
  use_fake_names: true           # true = fake names, false = [PERSON] token
  faker_locale: 'pt_PT'          # Locale for fake name generation

  # Entity type controls
  replace_municipalities: true   # Replace municipality terms with @MUNICIPALITY
  replace_locations: true        # Replace location terms with @LOCAL

  # Logging
  log_file: "logs/deslexicalization.jsonl"  # Log file path (auto-named per experiment)
```

### Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enabled` | boolean | `false` | Master switch to enable/disable deslexicalization |
| `use_fake_names` | boolean | `true` | If true, use Faker to generate realistic fake names. If false, use `[PERSON]` token |
| `faker_locale` | string | `'pt_PT'` | Locale for Faker library (e.g., `'pt_PT'`, `'en_US'`, `'es_ES'`) |
| `replace_municipalities` | boolean | `true` | Replace municipality-specific terms with `@MUNICIPALITY` |
| `replace_locations` | boolean | `true` | Replace location references with `@LOCAL` |
| `log_file` | string | `'logs/deslexicalization.jsonl'` | Base path for log files (actual file includes experiment name) |

## Usage Examples

### Example 1: Basic Deslexicalization

**Config**: `configs/main_experiment/bert_crf_deslexicalized.yaml`

```bash
python scripts/train.py \
  --config configs/main_experiment/bert_crf_deslexicalized.yaml \
  --experiment-name delex_experiment
```

This will:
- Replace VOTER names with Portuguese fake names
- Replace municipality terms with `@MUNICIPALITY`
- Replace locations with `@LOCAL`
- Log all operations to `logs/delex_experiment_deslexicalization.jsonl`

### Example 2: Generic Tokens Only

Create a custom config with:

```yaml
deslexicalization:
  enabled: true
  use_fake_names: false  # Use [PERSON] instead of fake names
  replace_municipalities: true
  replace_locations: true
```

### Example 3: Selective Deslexicalization

Only replace person names, keep municipalities and locations:

```yaml
deslexicalization:
  enabled: true
  use_fake_names: true
  replace_municipalities: false  # Keep original municipality names
  replace_locations: false       # Keep original location names
```

### Example 4: Cross-Lingual Training

For English training data:

```yaml
deslexicalization:
  enabled: true
  use_fake_names: true
  faker_locale: 'en_US'  # English fake names
  replace_municipalities: true
  replace_locations: true
```

## Understanding the Logs

Each deslexicalization operation is logged to a JSONL file for debugging and analysis.

**Log Location**: `logs/{experiment_name}_deslexicalization.jsonl`

### Log Entry Format

```json
{
  "type": "deslexicalization",
  "timestamp": "2025-10-13T10:30:45.123456",
  "example_id": "doc_001",
  "original_tokens": ["Dr.", "João", "Silva", "votou", "a", "favor"],
  "original_text": "Dr. João Silva votou a favor",
  "replacements": [
    {
      "entity_type": "VOTER",
      "original": "Dr. João Silva",
      "replacement": "fake_name",
      "start_idx": 0,
      "end_idx": 3
    }
  ],
  "modified_tokens": ["Dr.", "António", "Costa", "votou", "a", "favor"],
  "modified_text": "Dr. António Costa votou a favor"
}
```

### Log Summary Entry

At the end of processing, a summary is logged:

```json
{
  "type": "summary",
  "timestamp": "2025-10-13T10:35:00.000000",
  "examples_modified": 1234,
  "examples_unmodified": 456,
  "voter_replacements": 2345,
  "municipality_replacements": 567,
  "location_replacements": 123
}
```

## Implementation Details

### Intelligent Entity Classification

The system uses sophisticated pattern matching to avoid false positives:

**Protected Patterns** (NOT deslexicalized):
- Institutional references: "A Câmara", "O Presidente"
- Political parties: "do BE", "pelo PS", "da CDS"
- Role titles alone: "Vereador", "Secretário"
- Generic references: "A eleita", "O presente"

**Deslexicalized Patterns**:
- Person names with titles: "Dr. João Silva"
- Multi-word proper names: "Maria dos Santos"
- Role + name combinations: "Presidente João Costa"

### Integration with Training Pipeline

Deslexicalization is applied:
- ✅ **Training data**: Fully deslexicalized
- ❌ **Validation data**: Original (untouched)
- ❌ **Test data**: Original (untouched)

This ensures fair evaluation while allowing the model to learn generalized patterns.

### Performance Impact

- **Speed**: Minimal overhead (~5-10% additional loading time)
- **Memory**: No significant increase
- **Disk Space**: Log files are small (typically <10MB for large datasets)

## Troubleshooting

### Issue: Faker not installed

**Error**: `Faker not available, falling back to token replacement`

**Solution**: Install Faker:
```bash
pip install faker
```

### Issue: Log file permissions

**Error**: `Permission denied: logs/deslexicalization.jsonl`

**Solution**: Create logs directory:
```bash
mkdir -p logs
chmod 755 logs
```

### Issue: Too many/too few replacements

**Solution**: Check the log file to see what's being replaced:
```bash
# View summary
tail -1 logs/{experiment_name}_deslexicalization.jsonl | jq .

# View first few operations
head -10 logs/{experiment_name}_deslexicalization.jsonl | jq .
```

Adjust the configuration flags (`replace_municipalities`, `replace_locations`) as needed.

## Best Practices

1. **Always review logs**: Check the deslexicalization log after first run to ensure expected behavior

2. **Use fake names for realistic data**: `use_fake_names: true` maintains better linguistic patterns than generic tokens

3. **Match faker locale to data**: Use `faker_locale: 'pt_PT'` for Portuguese, `'en_US'` for English, etc.

4. **Version control configs**: Keep separate config files for deslexicalized vs. non-deslexicalized experiments

5. **Compare results**: Train identical models with/without deslexicalization to measure impact

6. **Cross-municipality experiments**: Deslexicalization is especially valuable for leave-one-municipality-out experiments

## Research Applications

Deslexicalization is particularly useful for:

- **Cross-municipality generalization**: Testing if models work across different cities
- **Domain adaptation**: Adapting models trained on one region to another
- **Privacy preservation**: Removing specific names from training data
- **Data augmentation**: Creating more diverse training examples
- **Fairness testing**: Ensuring models don't memorize specific individuals

## Citation

If you use this deslexicalization system in your research, please cite:

```bibtex
@software{votie_deslexicalization,
  title={VotIE Deslexicalization System},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/votie}
}
```

## See Also

- Example config: `configs/main_experiment/bert_crf_deslexicalized.yaml`
- Implementation: `src/data/deslexicalization.py`
- Person name detection: `deslexicalization_utils.py`
- Training script: `scripts/train.py`
