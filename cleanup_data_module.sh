#!/bin/bash
#
# Cleanup script for src/data module
# 
# This script safely removes dead code, consolidates duplicates,
# and updates imports for the VotIE data module.
#
# USAGE: bash cleanup_data_module.sh
#
# Before running:
# 1. Review DATA_FUNCTIONS_REVIEW.md
# 2. Backup your work or create a git branch
# 3. Read through this script
#
# After running:
# 1. Run the verification tests
# 2. Test training and prediction pipelines
# 3. Commit changes if successful

set -e  # Exit on error

echo "========================================="
echo "VotIE Data Module Cleanup"
echo "========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check we're in the right directory
if [ ! -f "DATA_FUNCTIONS_REVIEW.md" ]; then
    echo -e "${RED}Error: Run this script from the VotIE root directory${NC}"
    exit 1
fi

echo "Step 1: Creating backup..."
BACKUP_DIR="backup_data_module_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"
cp -r src/data "$BACKUP_DIR/"
echo -e "${GREEN}✓ Backup created in $BACKUP_DIR${NC}"
echo ""

echo "Step 2: Updating import in src/models/base.py..."
if grep -q "from src.data.validation import validate_bio_sequence" src/models/base.py; then
    sed -i '' 's/from src.data.validation import validate_bio_sequence/from src.data.dataset import validate_bio_sequence/' src/models/base.py
    echo -e "${GREEN}✓ Import updated in base.py${NC}"
else
    echo -e "${YELLOW}⚠ Import already updated or not found${NC}"
fi
echo ""

echo "Step 3: Updating src/data/__init__.py..."
cat > src/data/__init__.py << 'EOF'
"""Data loading and preprocessing utilities."""

from .dataset import *
from .postprocessing import *
EOF
echo -e "${GREEN}✓ __init__.py updated${NC}"
echo ""

echo "Step 4: Removing dead files..."

# Remove preprocessing.py (empty)
if [ -f "src/data/preprocessing.py" ]; then
    rm src/data/preprocessing.py
    echo -e "${GREEN}✓ Removed preprocessing.py (empty file)${NC}"
else
    echo -e "${YELLOW}⚠ preprocessing.py already removed${NC}"
fi

# Remove windowing.py (unused WindowingManager)
if [ -f "src/data/windowing.py" ]; then
    rm src/data/windowing.py
    echo -e "${GREEN}✓ Removed windowing.py (610 lines of unused code)${NC}"
else
    echo -e "${YELLOW}⚠ windowing.py already removed${NC}"
fi

# Remove validation.py (duplicate)
if [ -f "src/data/validation.py" ]; then
    rm src/data/validation.py
    echo -e "${GREEN}✓ Removed validation.py (duplicate functions)${NC}"
else
    echo -e "${YELLOW}⚠ validation.py already removed${NC}"
fi

echo ""
echo "========================================="
echo "Cleanup Complete!"
echo "========================================="
echo ""
echo "Files removed:"
echo "  - preprocessing.py (0 lines - empty)"
echo "  - windowing.py (610 lines - unused)"
echo "  - validation.py (130 lines - duplicate)"
echo "  Total: 741 lines (39% reduction)"
echo ""

echo "Running verification tests..."
echo ""

# Test 1: Import verification
echo -n "Test 1: Importing data module... "
if python3 -c "from src.data import *" 2>/dev/null; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

echo -n "Test 2: Importing dataset functions... "
if python3 -c "from src.data.dataset import load_jsonl_file, align_tokens_with_subwords, create_dynamic_windows, validate_bio_sequence" 2>/dev/null; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

echo -n "Test 3: Importing postprocessing... "
if python3 -c "from src.data.postprocessing import postprocess_model_output, postprocess_labels_to_original" 2>/dev/null; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

echo -n "Test 4: Importing models.base... "
if python3 -c "from src.models.base import *" 2>/dev/null; then
    echo -e "${GREEN}✓ PASS${NC}"
else
    echo -e "${RED}✗ FAIL${NC}"
    exit 1
fi

# Test 5: Check for broken references
echo -n "Test 5: Checking for broken validation imports... "
if grep -r "from.*data.validation import" src/ scripts/ 2>/dev/null | grep -v "__pycache__" | grep -q .; then
    echo -e "${RED}✗ FAIL - Found remaining validation imports${NC}"
    grep -r "from.*data.validation import" src/ scripts/ 2>/dev/null | grep -v "__pycache__"
    exit 1
else
    echo -e "${GREEN}✓ PASS${NC}"
fi

echo -n "Test 6: Checking for broken windowing imports... "
if grep -r "from.*data.windowing import" src/ scripts/ 2>/dev/null | grep -v "__pycache__" | grep -q .; then
    echo -e "${RED}✗ FAIL - Found remaining windowing imports${NC}"
    grep -r "from.*data.windowing import" src/ scripts/ 2>/dev/null | grep -v "__pycache__"
    exit 1
else
    echo -e "${GREEN}✓ PASS${NC}"
fi

echo -n "Test 7: Checking for broken preprocessing imports... "
if grep -r "from.*data.preprocessing import" src/ scripts/ 2>/dev/null | grep -v "__pycache__" | grep -q .; then
    echo -e "${RED}✗ FAIL - Found remaining preprocessing imports${NC}"
    grep -r "from.*data.preprocessing import" src/ scripts/ 2>/dev/null | grep -v "__pycache__"
    exit 1
else
    echo -e "${GREEN}✓ PASS${NC}"
fi

echo ""
echo "========================================="
echo -e "${GREEN}All Tests Passed!${NC}"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Test training: python scripts/train.py --help"
echo "2. Test prediction: python scripts/predict.py --help"
echo "3. Run full pipeline test if desired"
echo "4. Review changes: git diff"
echo "5. Commit: git commit -am 'refactor: Clean up src/data module'"
echo ""
echo "Backup available at: $BACKUP_DIR"
echo ""
