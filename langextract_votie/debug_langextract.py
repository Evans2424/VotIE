#!/usr/bin/env python
"""
Debug script to test LangExtract extraction with minimal setup
"""

import os
import sys
from pathlib import Path
import logging

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Test imports
logger.info("Testing imports...")
try:
    import langextract as lx
    logger.info(f"✓ LangExtract imported")
except Exception as e:
    logger.error(f"✗ Failed to import langextract: {e}")
    sys.exit(1)

# Test API key
logger.info("Checking API key...")

#Load dotenv
from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv('LANGEXTRACT_API_KEY') or os.getenv('GOOGLE_API_KEY')
if not api_key:
    logger.error("✗ No API key found")
    sys.exit(1)
logger.info(f"✓ API key found: {api_key[:10]}...")

# Test simple extraction
logger.info("Testing simple extraction...")
test_text = "O Presidente votou a favor da proposta."

simple_prompt = "Extract voting entities: VOTER-FAVOR, SUBJECT"

simple_examples = [
    lx.data.ExampleData(
        text="Maria votou a favor.",
        extractions=[
            lx.data.Extraction(
                extraction_class="VOTER-FAVOR",
                extraction_text="Maria"
            )
        ]
    )
]

logger.info("Calling lx.extract with minimal config...")
logger.info(f"  Text: {test_text}")
logger.info(f"  Model: gemini-2.5-flash")
logger.info(f"  Examples: {len(simple_examples)}")

try:
    import signal
    
    def timeout_handler(signum, frame):
        raise TimeoutError("Extract call timed out after 30 seconds")
    
    # Set 30 second timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(30)
    
    logger.info("Starting extraction (30s timeout)...")
    result = lx.extract(
        text_or_documents=test_text,
        prompt_description=simple_prompt,
        examples=simple_examples,
        model_id="gemini-2.5-flash",
        api_key=api_key
    )
    
    signal.alarm(0)  # Cancel timeout
    
    logger.info(f"✓ Extraction succeeded!")
    logger.info(f"  Extractions: {len(result.extractions)}")
    for ext in result.extractions:
        logger.info(f"    - {ext.extraction_class}: '{ext.extraction_text}'")
    
except TimeoutError as e:
    logger.error(f"✗ TIMEOUT: {e}")
    logger.error("LangExtract is hanging - this indicates an API or network issue")
    sys.exit(1)
except Exception as e:
    logger.error(f"✗ Extraction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

logger.info("=" * 80)
logger.info("SUCCESS - LangExtract is working correctly")
logger.info("=" * 80)
