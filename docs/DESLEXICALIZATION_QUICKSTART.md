# Deslexicalization Quick Start

## What Was Added

A complete deslexicalization system has been integrated into your NER training pipeline. This system can:

1. **Replace VOTER names** with fake names (using Faker) or generic tokens
2. **Replace municipality names** with `@MUNICIPALITY` placeholder
3. **Replace location references** with `@LOCAL` placeholder
4. **Log all operations** to JSONL files for debugging and analysis

## Files Created/Modified

### New Files
- `src/data/deslexicalization.py` - Main deslexicalization module
- `configs/main_experiment/bert_crf_deslexicalized.yaml` - Example config with deslexicalization
- `docs/DESLEXICALIZATION.md` - Comprehensive documentation
- `logs/{experiment_name}_deslexicalization.jsonl` - Auto-generated log files

### Modified Files
- `src/data/dataset.py` - Updated `load_jsonl_file()` and `load_ner_dataset_from_jsonl()` to support deslexicalization
- `scripts/train.py` - Updated `train_transformer()` to pass deslexicalization config
- `configs/main_experiment/bert_crf.yaml` - Added deslexicalization section (disabled by default)

## Quick Usage

### 1. Train with Deslexicalization (Default Settings)

```bash
python scripts/train.py \
  --config configs/main_experiment/bert_crf_deslexicalized.yaml \
  --experiment-name my_delex_experiment
```

This will:
- Replace VOTER names with Portuguese fake names
- Replace municipalities with @MUNICIPALITY
- Replace locations with @LOCAL
- Log to `logs/my_delex_experiment_deslexicalization.jsonl`

### 2. Train without Deslexicalization

```bash
python scripts/train.py \
  --config configs/main_experiment/bert_crf.yaml \
  --experiment-name my_normal_experiment
```

The default configs have deslexicalization disabled.

### 3. Custom Deslexicalization

Create your own config file or modify an existing one:

```yaml
deslexicalization:
  enabled: true
  use_fake_names: true              # or false for [PERSON] tokens
  faker_locale: 'pt_PT'             # or 'en_US', 'es_ES', etc.
  replace_municipalities: true      # or false
  replace_locations: true           # or false
  log_file: "logs/deslexicalization.jsonl"
```

## Configuration Reference

| Option | Default | Description |
|--------|---------|-------------|
| `enabled` | `false` | Enable/disable deslexicalization |
| `use_fake_names` | `true` | Use Faker for realistic names vs `[PERSON]` token |
| `faker_locale` | `'pt_PT'` | Locale for fake name generation |
| `replace_municipalities` | `true` | Replace municipality terms |
| `replace_locations` | `true` | Replace location terms |
| `log_file` | `'logs/...'` | Base path for log files |

## Checking Results

### View Deslexicalization Statistics

```bash
# View summary (last line of log)
tail -1 logs/my_experiment_deslexicalization.jsonl | python -m json.tool

# Count replacements
grep "type.*deslexicalization" logs/my_experiment_deslexicalization.jsonl | wc -l

# View first 5 operations
head -6 logs/my_experiment_deslexicalization.jsonl | python -m json.tool
```

### Example Log Entry

```json
{
  "type": "deslexicalization",
  "timestamp": "2025-10-13T10:30:45",
  "example_id": "segment_001",
  "replacements": [
    {
      "entity_type": "VOTER",
      "original": "Dr. João Silva",
      "replacement": "fake_name",
      "start_idx": 0,
      "end_idx": 3
    },
    {
      "entity_type": "MUNICIPALITY",
      "count": 2,
      "replacement": "@MUNICIPALITY"
    }
  ],
  "original_text": "Dr. João Silva votou a favor na Câmara do Porto",
  "modified_text": "Dr. António Costa votou a favor na Câmara @MUNICIPALITY"
}
```

## Important Notes

1. **Deslexicalization applies only to TRAINING data** - validation and test sets remain unchanged
2. **Logs are created per experiment** - each experiment gets its own log file
3. **Faker must be installed** - If not installed, system falls back to generic tokens
   ```bash
   pip install faker
   ```
4. **Check logs after first run** - Review the log file to ensure expected behavior

## Troubleshooting

**Issue**: Module not found error
```bash
# Make sure you're running from project root
cd /Users/josepedroevans/Desktop/VotIE
python scripts/train.py --config ...
```

**Issue**: Faker not available
```bash
pip install faker
```

**Issue**: No replacements happening
- Check that `deslexicalization.enabled: true` in config
- Check that your data has VOTER entities (VOTANTE labels)
- Review log file to see what's being detected

## Next Steps

- Read full documentation: `docs/DESLEXICALIZATION.md`
- Try the example config: `configs/main_experiment/bert_crf_deslexicalized.yaml`
- Compare results with/without deslexicalization
- Use for cross-municipality experiments
- Customize replacement strategies for your use case

## Example Workflow

```bash
# 1. Train baseline model (no deslexicalization)
python scripts/train.py \
  --config configs/main_experiment/bert_crf.yaml \
  --experiment-name baseline

# 2. Train with deslexicalization
python scripts/train.py \
  --config configs/main_experiment/bert_crf_deslexicalized.yaml \
  --experiment-name deslexicalized

# 3. Check deslexicalization log
cat logs/deslexicalized_deslexicalization.jsonl | python -m json.tool | less

# 4. Compare results
python scripts/evaluate.py \
  --model models/bert_crf/baseline/best_model \
  --test-data data/votie_bio/test.jsonl

python scripts/evaluate.py \
  --model models/bert_crf/deslexicalized/best_model \
  --test-data data/votie_bio/test.jsonl
```

## Support

For issues or questions:
1. Check `docs/DESLEXICALIZATION.md` for detailed documentation
2. Review example configs in `configs/main_experiment/`
3. Inspect log files in `logs/` directory
4. Check source code in `src/data/deslexicalization.py`
