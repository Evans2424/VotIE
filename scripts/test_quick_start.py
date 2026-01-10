"""
Test script for the quick_start.py example using the local model.
"""

import sys
import os

# Add the model directory to path to use local model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'XLM-RoBERTa-CRF-VotIE'))

from transformers import AutoTokenizer, AutoModel

print("Loading model and tokenizer from local directory...")
model_path = "../XLM-RoBERTa-CRF-VotIE"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(model_path, trust_remote_code=True)

print("Model loaded successfully!\n")

# Example text from sample data
text = """A CÂMARA EM FACE DO AUTO DE VISTORIA ELABORADO PELA COMISSÃO DE VISTORIAS, DELIBEROU, POR TODOS OS MEMBROS PRESENTES, QUE O REFERIDO PRÉDIO E AS SUAS FRAÇÕES AUTÓNOMAS, AS QUAIS SÃO DISTINTAS E ISOLADAS ENTRE SI E COM SAÍDA PRÓPRIA PARA A VIA PÚBLICA, REÚNEM OS REQUISITOS LEGAIS PARA NELE SER INSTITUÍDO O REGIME DE PROPRIEDADE HORIZONTAL."""

print("Testing with example text...")
print(f"Text: '{text[:100]}...'\n")

# Tokenize
inputs = tokenizer(text, return_tensors="pt")

# Get predictions
print("Getting predictions...")
predictions = model.decode(**inputs, tokenizer=tokenizer, text=text)

# Print results
print(f"\n{'Word':<30} {'Label':<30}")
print("-" * 60)
entity_count = 0
for pred in predictions:
    if pred['label'] != 'O':  # Only show entities
        print(f"{pred['word']:<30} {pred['label']:<30}")
        entity_count += 1

print(f"\nFound {entity_count} entity tokens")

# Test with offsets
print("\n" + "="*60)
print("Testing with character offsets...")
print("="*60 + "\n")
predictions_with_offsets = model.decode(**inputs, tokenizer=tokenizer, text=text, return_offsets=True)
for pred in predictions_with_offsets:
    if pred['label'] != 'O':  # Only show entities
        print(f"{pred['word']:<20} {pred['label']:<25} (chars {pred['start']}-{pred['end']})")

print("\n✓ Test completed successfully!")
