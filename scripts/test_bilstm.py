#!/usr/bin/env python3
"""
Test BiLSTM model loading and prediction
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.predict import load_model, predict_example

def test_bilstm_model():
    """Test BiLSTM model loading and prediction."""
    
    model_path = "models/bilstm_fasttext/bilstm_final"
    test_tokens = ["O", "presidente", "votou", "a", "favor", "."]
    
    try:
        print("Loading BiLSTM model...")
        model, tokenizer, id_to_label, config = load_model(model_path, device='cpu')
        
        print(f"Model type: {config.get('model_type', 'unknown')}")
        print(f"Number of labels: {len(id_to_label)}")
        print(f"Labels: {list(id_to_label.values())}")
        
        # Test prediction
        print("\nTesting prediction...")
        test_example = {
            'id': 'test_001',
            'tokens': test_tokens
        }
        
        result = predict_example(model, tokenizer, id_to_label, test_example, device='cpu')
        
        print(f"Input tokens: {result['tokens']}")
        print(f"Predictions: {result['pred_labels']}")
        
        print("\n✅ BiLSTM model test completed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ BiLSTM model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_bilstm_model()