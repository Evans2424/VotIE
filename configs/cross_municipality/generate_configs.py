#!/usr/bin/env python3
"""
Generate cross-municipality experiment configs for all models.
"""

from pathlib import Path

MUNICIPALITIES = {
    "alandroal": {"display": "Alandroal", "id": "M01"},
    "campomaior": {"display": "Campomaior", "id": "M02"},
    "covilha": {"display": "Covilhã", "id": "M03"},
    "fundao": {"display": "Fundão", "id": "M04"},
    "guimaraes": {"display": "Guimarães", "id": "M05"},
    "porto": {"display": "Porto", "id": "M06"}
}

MODELS = {
    "bert_crf": {
        "model_name": "neuralmind/bert-base-portuguese-cased",
        "architecture": "crf"
    },
    "deberta_crf": {
        "model_name": "microsoft/mdeberta-v3-base",
        "architecture": "crf"
    }
}

CONFIG_TEMPLATE = """# Cross-Municipality Experiment: Test on {muni_display}
# Train on 5 municipalities, test on {muni_display} (held-out)
# LOMO (Leave-One-Municipality-Out) split

model:
  name: "{model_key}"
  model_name: "{model_name}"
  architecture: "{architecture}"
  dropout: 0.1
  num_labels: null  # will be set automatically

data:
  data_dir: "data/votie_lomo/{muni_id}_{muni_key}"
  train_file: "train.jsonl"
  dev_file: "dev.jsonl"
  test_file: "test.jsonl"

training:
  batch_size: 16
  epochs: 10
  learning_rate: 2e-5
  warmup_proportion: 0.1
  weight_decay: 0.01
  patience: 3
  device: 0  # CUDA device ID
  apply_bio_validation: true

optimization:
  optimizer: "adamw"
  scheduler: "linear_with_warmup"
  grad_clip: 1.0

augmentation:
  max_length: 512
  padding: "max_length"
  truncation: true

windowing:
  enable_windowing: true
  overlap_tokens: 50
"""

def generate_configs():
    """Generate config files for all cross-municipality experiments."""

    base_dir = Path(__file__).parent

    print("Generating cross-municipality configs...")
    print()

    for model_key, model_info in MODELS.items():
        model_dir = base_dir / model_key
        model_dir.mkdir(exist_ok=True)

        print(f"Model: {model_key}")

        for muni_key, muni_info in MUNICIPALITIES.items():
            config_content = CONFIG_TEMPLATE.format(
                muni_key=muni_key,
                muni_display=muni_info['display'],
                muni_id=muni_info['id'],
                model_key=model_key,
                model_name=model_info['model_name'],
                architecture=model_info['architecture']
            )

            config_file = model_dir / f"test_{muni_key}.yaml"
            with open(config_file, 'w') as f:
                f.write(config_content)

            print(f"  ✓ {config_file.name}")

        print()

    print("✅ All configs generated!")
    print()
    print("Usage examples:")
    print("  python train.py bert_crf --config configs/cross_municipality/bert_crf/test_campomaior.yaml")
    print("  python train.py deberta_crf --config configs/cross_municipality/deberta_crf/test_alandroal.yaml")

if __name__ == "__main__":
    generate_configs()
