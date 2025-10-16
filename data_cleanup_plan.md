# Data Module Cleanup - Visual Plan

## Current Structure (BEFORE)

```
src/data/
│
├── __init__.py (6 lines)
│   └── Exports: dataset.*, windowing.*
│
├── preprocessing.py (0 lines) ❌ EMPTY - DELETE
│
├── postprocessing.py (96 lines) ✅ KEEP AS-IS
│   ├── postprocess_model_output()          [USED: bert_models, xlmr_models, deberta_models]
│   └── postprocess_labels_to_original()    [USED: models/__init__.py]
│
├── validation.py (130 lines) ⚠️ DUPLICATE - DELETE AFTER CONSOLIDATION
│   ├── validate_bio_sequence()             [USED: models/base.py - CAN REPLACE]
│   │   └── 🔴 DUPLICATE of dataset.py version
│   └── validate_bio_batch()                [USED: NOBODY - ORPHAN]
│
├── windowing.py (610 lines) ❌ UNUSED CLASS - DELETE
│   └── WindowingManager                    [USED: NOBODY - 560+ lines of dead code]
│       ├── needs_windowing()
│       ├── create_windows()
│       ├── merge_windowed_predictions()
│       └── [many internal methods]
│
└── dataset.py (1054 lines) ✅ KEEP - CORE MODULE
    ├── Data Classes ✅
    │   ├── VoteExample
    │   ├── VoteFeatures
    │   └── NERDataset
    │
    ├── Loading Functions ✅
    │   ├── load_jsonl_file()                    [USED: train.py, predict.py, evaluate.py]
    │   ├── load_ner_dataset()                   [USED: Training pipeline]
    │   ├── load_ner_dataset_with_dynamic_windowing()
    │   └── load_ner_dataset_from_jsonl()        [USED: train.py - MAIN LOADER]
    │
    ├── Tokenization ✅
    │   ├── align_tokens_with_subwords()         [USED: predict.py, training]
    │   ├── clean_problematic_characters()       [USED: align_tokens_with_subwords]
    │   └── is_problematic_token()               [USED: validation]
    │
    ├── Windowing ✅
    │   └── create_dynamic_windows()             [USED: predict.py, training, legacy]
    │       └── 🔴 ALTERNATIVE to WindowingManager (this one is USED)
    │
    ├── Validation ⚠️
    │   ├── validate_bio_sequence()              [USED: bilstm_trainer, crf, dataset]
    │   │   └── 🔴 DUPLICATE of validation.py version (KEEP THIS ONE)
    │   └── validate_example_data()              [USED: load_jsonl_file]
    │
    ├── Conversion ✅
    │   ├── jsonl_to_examples()
    │   ├── jsonl_to_examples_with_dynamic_windowing()
    │   └── convert_examples_to_features()       [USED: All loaders]
    │
    └── Utilities ✅
        ├── create_label_mapping()               [USED: All loaders]
        ├── calculate_paper_class_weights()
        └── get_class_weights()
```

---

## Proposed Structure (AFTER)

```
src/data/
│
├── __init__.py (5 lines) ✅ CLEANED
│   └── Exports: dataset.*, postprocessing.*
│
├── postprocessing.py (96 lines) ✅ NO CHANGE
│   ├── postprocess_model_output()
│   └── postprocess_labels_to_original()
│
└── dataset.py (1054 lines) ✅ KEEP + ADD validate_bio_batch if needed
    ├── [All existing functions remain]
    └── validate_bio_batch() ← ADD if needed (from old validation.py)
```

---

## File Deletion Plan

### 1. Delete `preprocessing.py` ❌
**Reason**: Empty file (0 lines)
**Impact**: None - never imported
**Action**: 
```bash
rm src/data/preprocessing.py
```

### 2. Delete `windowing.py` ❌
**Reason**: 610 lines of unused code
**Impact**: None - `WindowingManager` never instantiated
**Action**:
```bash
rm src/data/windowing.py
```

### 3. Delete `validation.py` ⚠️
**Reason**: Duplicate of dataset.py functions
**Impact**: ONE import needs updating
**Action**:
```bash
# First update the import
# Then delete the file
rm src/data/validation.py
```

---

## Import Updates Required

### File: `src/models/base.py` (line 94)

**BEFORE**:
```python
from src.data.validation import validate_bio_sequence
```

**AFTER**:
```python
from src.data.dataset import validate_bio_sequence
```

### File: `src/data/__init__.py`

**BEFORE**:
```python
"""Data loading and preprocessing utilities."""

from .dataset import *
from .windowing import *
```

**AFTER**:
```python
"""Data loading and preprocessing utilities."""

from .dataset import *
from .postprocessing import *
```

---

## Duplication Details

### Duplicate 1: `validate_bio_sequence()`

```
┌─────────────────────────────────────────────────────────────┐
│ Function: validate_bio_sequence()                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Location 1: src/data/dataset.py (line 112)                 │
│   Used by:                                                  │
│     ✓ src/data/dataset.py (validate_example_data)         │
│     ✓ src/bilstm_trainer.py (2 locations)                 │
│     ✓ src/models/crf.py                                    │
│   Status: ✅ KEEP THIS ONE (more widely used)             │
│                                                             │
│ Location 2: src/data/validation.py (line 20)               │
│   Used by:                                                  │
│     ✓ src/models/base.py (1 location)                     │
│   Status: ❌ DELETE (less integrated)                      │
│                                                             │
│ Differences:                                                │
│   • Parameter naming: tag_sequence vs labels               │
│   • Logging verbosity: slightly different                  │
│   • Logic: IDENTICAL                                        │
│                                                             │
│ Resolution:                                                 │
│   1. Update models/base.py import                          │
│   2. Delete validation.py                                  │
└─────────────────────────────────────────────────────────────┘
```

### Duplicate 2: Windowing Implementations

```
┌─────────────────────────────────────────────────────────────┐
│ Functionality: Sequence Windowing                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Implementation 1: create_dynamic_windows() (functional)     │
│   Location: src/data/dataset.py (line 614)                 │
│   Lines: ~140                                               │
│   Used by:                                                  │
│     ✓ scripts/predict.py                                   │
│     ✓ src/data/dataset.py (training pipeline)             │
│     ✓ helper scripts/predict_legacy.py                    │
│   Status: ✅ KEEP (actively used)                          │
│                                                             │
│ Implementation 2: WindowingManager (OOP)                    │
│   Location: src/data/windowing.py (line 26)                │
│   Lines: ~560                                               │
│   Used by:                                                  │
│     ❌ NOBODY - never instantiated                         │
│   Status: ❌ DELETE (dead code)                            │
│                                                             │
│ Differences:                                                │
│   • Approach: Functional vs OOP                            │
│   • Complexity: Simple vs sophisticated                    │
│   • Integration: Integrated vs isolated                    │
│                                                             │
│ Resolution:                                                 │
│   1. Keep create_dynamic_windows() in dataset.py          │
│   2. Delete entire windowing.py file                       │
└─────────────────────────────────────────────────────────────┘
```

---

## Cleanup Commands

Execute in order:

```bash
# 1. Update import in base.py
sed -i '' 's/from src.data.validation/from src.data.dataset/' src/models/base.py

# 2. Update __init__.py
cat > src/data/__init__.py << 'EOF'
"""Data loading and preprocessing utilities."""

from .dataset import *
from .postprocessing import *
EOF

# 3. Delete dead files
rm src/data/preprocessing.py
rm src/data/windowing.py
rm src/data/validation.py

# 4. Verify no broken imports
python -c "from src.data import *; print('✓ Imports OK')"
python -c "from src.models.base import *; print('✓ Base model OK')"
python -c "from scripts.train import *; print('✓ Training OK')"
```

---

## Verification Tests

After cleanup, run:

```bash
# Test 1: Import verification
python -c "
from src.data.dataset import (
    load_jsonl_file,
    align_tokens_with_subwords,
    create_dynamic_windows,
    validate_bio_sequence,
    load_ner_dataset_from_jsonl
)
print('✓ All critical functions importable')
"

# Test 2: Postprocessing still works
python -c "
from src.data.postprocessing import (
    postprocess_model_output,
    postprocess_labels_to_original
)
print('✓ Postprocessing functions importable')
"

# Test 3: No references to deleted modules
if grep -r "from.*validation import" src/ scripts/ --exclude-dir=__pycache__; then
    echo "❌ Found remaining validation imports"
else
    echo "✓ No validation imports found"
fi

if grep -r "from.*windowing import" src/ scripts/ --exclude-dir=__pycache__; then
    echo "❌ Found remaining windowing imports"
else
    echo "✓ No windowing imports found"
fi

if grep -r "from.*preprocessing import" src/ scripts/ --exclude-dir=__pycache__; then
    echo "❌ Found remaining preprocessing imports"
else
    echo "✓ No preprocessing imports found"
fi

# Test 4: Training pipeline
python scripts/train.py --config configs/main_experiment/bert_linear.yaml --dry-run

# Test 5: Prediction pipeline
python scripts/predict.py --help
```

---

## Metrics

### Lines of Code Reduction

| File | Before | After | Change |
|------|--------|-------|--------|
| `preprocessing.py` | 0 | ❌ DELETED | -0 |
| `windowing.py` | 610 | ❌ DELETED | -610 |
| `validation.py` | 130 | ❌ DELETED | -130 |
| `postprocessing.py` | 96 | 96 | 0 |
| `dataset.py` | 1054 | 1054 | 0 |
| `__init__.py` | 6 | 5 | -1 |
| **TOTAL** | **1,896** | **1,155** | **-741 (39%)** |

### Complexity Reduction

- **Files**: 6 → 3 (-50%)
- **Empty files**: 1 → 0 (-100%)
- **Duplicate functions**: 2 → 0 (-100%)
- **Unused classes**: 1 → 0 (-100%)
- **Dead code**: ~750 lines → 0 (-100%)

---

## Risk Assessment

### ✅ LOW RISK (Safe to execute)

1. **Deleting `preprocessing.py`**
   - Empty file
   - No imports found
   - Zero impact

2. **Deleting `windowing.py`**
   - Class never instantiated
   - No imports found
   - Zero impact on functionality

3. **Consolidating validation**
   - One import to update (models/base.py)
   - Simple find/replace
   - Identical functionality

### ⚠️ MEDIUM RISK (Test thoroughly)

1. **Updating `__init__.py`**
   - Public API change
   - Test all imports after
   - Should be safe (windowing not used)

### ❌ HIGH RISK (Not recommended now)

1. **Refactoring dataset.py**
   - Major structural change
   - Many imports to update
   - Better as future work

---

## Timeline

**Estimated time**: 2-3 hours

1. **Preparation** (30 min)
   - Backup current state
   - Create git branch
   - Review changes

2. **Execution** (30 min)
   - Update imports
   - Delete files
   - Update __init__.py

3. **Testing** (60 min)
   - Run verification tests
   - Test training pipeline
   - Test prediction pipeline
   - Check all imports

4. **Documentation** (30 min)
   - Update README if needed
   - Document changes
   - Commit with clear message

---

## Git Workflow

```bash
# 1. Create cleanup branch
git checkout -b cleanup/data-module
git commit -am "Checkpoint before data module cleanup"

# 2. Execute cleanup
# ... (run cleanup commands)

# 3. Commit changes
git add .
git commit -m "refactor: Clean up src/data module

- Remove empty preprocessing.py (0 lines)
- Remove unused WindowingManager class (610 lines)
- Consolidate duplicate validate_bio_sequence()
- Update imports in models/base.py
- Clean up __init__.py exports

Removes 741 lines (39%) of dead/duplicate code with zero functionality loss.
"

# 4. Test
python scripts/train.py --config configs/main_experiment/bert_linear.yaml --dry-run

# 5. Merge if tests pass
git checkout main
git merge cleanup/data-module
git push
```

---

## Success Criteria

- [ ] All files deleted successfully
- [ ] No broken imports
- [ ] Training pipeline runs
- [ ] Prediction pipeline runs
- [ ] All tests pass
- [ ] No references to deleted modules
- [ ] Code quality improved (less duplication)
- [ ] Repository ready for publication

---

*Cleanup Plan Generated: 2025-10-16*
