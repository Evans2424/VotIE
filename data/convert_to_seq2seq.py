import json
import logging
from pathlib import Path
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def convert_spans_to_seq2seq(input_file: str, output_file: str):
    """
    Converts BIO/Span JSONL data into T5 Seq2Seq format.
    
    Input:
      {"text": "Mr. Silva voted yes", "spans": [{"start": 0, "end": 9, "label": "VOTER"}]}
    
    Output:
      {
        "input_text": "extract votes: Mr. Silva voted yes", 
        "target_text": "[ Mr. Silva | VOTER ] voted yes"
      }
    """
    input_path = Path(input_file)
    output_path = Path(output_file)
    
    with open(input_path, 'r', encoding='utf-8') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
        
        count = 0
        errors = 0
        
        for line in tqdm(f_in, desc=f"Converting {input_path.name}"):
            try:
                data = json.loads(line)
                original_text = data['text']
                spans = data.get('spans', [])
                
                # 1. Sort spans by start position to ensure correct reconstruction
                spans.sort(key=lambda x: x['start'])
                
                # 2. Reconstruct the text with tags
                target_text = ""
                last_idx = 0
                
                for span in spans:
                    start = span['start']
                    end = span['end']
                    label = span['label']
                    
                    # Sanity check: overlapping spans (basic check)
                    if start < last_idx:
                        # Skip nested/overlapping entity for simplicity
                        continue
                        
                    # Append text BEFORE the entity
                    target_text += original_text[last_idx:start]
                    
                    # Append the entity with TANL formatting: [ Entity Text | Label ]
                    entity_text = original_text[start:end]
                    target_text += f"[ {entity_text} | {label} ]"
                    
                    last_idx = end
                
                # Append remaining text after the last entity
                target_text += original_text[last_idx:]
                
                # 3. Create the T5 example
                # We add a prefix "extract votes:" to guide the model
                t5_example = {
                    "input_text": f"extract votes: {original_text}",
                    "target_text": target_text,
                    "metadata": {
                        "id": data.get('id'),
                        "doc_id": data.get('document_id')
                    }
                }
                
                f_out.write(json.dumps(t5_example, ensure_ascii=False) + '\n')
                count += 1
                
            except Exception as e:
                errors += 1
                continue

    logger.info(f"✅ Converted {count} examples. (Skipped {errors} errors). Saved to {output_path}")

if __name__ == "__main__":
    datasets = [
        ("data/votie_bio/train.jsonl", "data/t5_seq2seq/train.json"),
        ("data/votie_bio/dev.jsonl", "data/t5_seq2seq/dev.json"),
        ("data/votie_bio/test.jsonl", "data/t5_seq2seq/test.json")
    ]
    
    # Create directory if not exists
    Path("data/t5_seq2seq").mkdir(parents=True, exist_ok=True)
    
    for in_file, out_file in datasets:
        if Path(in_file).exists():
            convert_spans_to_seq2seq(in_file, out_file)
        else:
            logger.warning(f"⚠️ Input file not found: {in_file}")