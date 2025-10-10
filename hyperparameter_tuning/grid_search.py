#!/usr/bin/env python3
"""
Simple Grid Search for BERT-CRF and DeBERTa-CRF baselines.

Performs sequential grid search over learning rates and batch sizes,
saving results to JSON without saving model checkpoints.

Usage:
    nohup python grid_search.py > grid_search.log 2>&1 &
"""

import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from itertools import product

# Add root to path
ROOT_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(ROOT_DIR))

# Configure logging for nohup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('grid_search_detailed.log')
    ]
)
logger = logging.getLogger(__name__)

from baselines.transformers.trainer import NERTrainer
from baselines.transformers.model import BertCRFNER, DebertaCRFNER, XLMRCRFNER
from baselines.transformers.config_loader_transformer import load_config, extract_data_params
from core.data.dataset import load_ner_dataset_with_dynamic_windowing

# Grid search parameters
LEARNING_RATES = [1e-5, 2e-5, 3e-5, 5e-5]
BATCH_SIZES = [16, 32]
EPOCHS = [5, 7, 10]
MODELS = ['xlmr_crf']

# Fixed parameters
MAX_LENGTH = 512
OVERLAP_TOKENS = 50
WARMUP_PROPORTION = 0.1
MAX_GRAD_NORM = 1.0
PATIENCE = 3
DEVICE = 2  # GPU device


def train_and_evaluate(model_name, learning_rate, batch_size, num_epochs):
    """Train model with given hyperparameters and return metrics."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Training {model_name} - LR: {learning_rate}, BS: {batch_size}, Epochs: {num_epochs}")
    logger.info(f"{'='*80}")

    try:
        # Load config to get data paths
        config = load_config(baseline_name=model_name)
        data_params = extract_data_params(config)

        # Load data
        base_dir = ROOT_DIR.parent
        data_dir = base_dir / data_params['data_dir']

        model_config = config['model']
        hf_model_name = model_config['model_name']

        logger.info(f"Loading datasets from {data_dir}")
        train_dataset, dev_dataset, test_dataset, label_to_id, id_to_label, tokenizer = \
            load_ner_dataset_with_dynamic_windowing(
                data_dir=data_dir,
                train_files=data_params['train_files'],
                dev_file=data_params['dev_file'],
                test_file=data_params['test_file'],
                tokenizer_name=hf_model_name,
                max_length=MAX_LENGTH,
                overlap_tokens=OVERLAP_TOKENS,
                enable_windowing=True
            )

        logger.info(f"Train: {len(train_dataset)}, Dev: {len(dev_dataset)}, Test: {len(test_dataset)}")

        # Create model
        if model_name == 'bert_crf':
            model = BertCRFNER(model_name=hf_model_name, num_labels=len(label_to_id))
        elif model_name == 'deberta_crf':
            model = DebertaCRFNER(model_name=hf_model_name, num_labels=len(label_to_id))
        elif model_name == 'xlmr_crf':
            model = XLMRCRFNER(model_name=hf_model_name, num_labels=len(label_to_id))
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Create output dir (will not save models, just for trainer compatibility)
        output_dir = Path(f"grid_search_temp/{model_name}_lr{learning_rate}_bs{batch_size}_ep{num_epochs}")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Create trainer
        trainer = NERTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            output_dir=output_dir,
            id_to_label=id_to_label,
            num_train_epochs=num_epochs,
            train_batch_size=batch_size,
            eval_batch_size=batch_size * 2,
            learning_rate=learning_rate,
            classifier_learning_rate=1e-3,
            warmup_proportion=WARMUP_PROPORTION,
            max_grad_norm=MAX_GRAD_NORM,
            patience=PATIENCE,
            tokenizer=tokenizer,
            base_model=hf_model_name,
            device=DEVICE,
            apply_bio_validation=True
        )

        # Train
        logger.info("Starting training...")
        start_time = datetime.now()
        results = trainer.train()
        training_time = (datetime.now() - start_time).total_seconds()

        # Test evaluation
        logger.info("Evaluating on test set...")
        test_dataloader = trainer._get_dataloader(test_dataset, batch_size * 2, shuffle=False)
        test_results = trainer.final_evaluate(test_dataloader, eval_set_name="test")

        # Collect results
        # The trainer returns different key names in its results dict depending on
        # implementation (e.g. 'best_eval_score' instead of 'best_val_f1'). Be
        # defensive here and try multiple keys; also derive the best epoch from
        # training_history if the trainer did not return a explicit best_epoch.
        best_dev_f1 = results.get('best_eval_score', results.get('best_val_f1', 0.0))

        # Try to get best epoch directly, otherwise infer it from training_history
        best_dev_epoch = results.get('best_epoch')
        if best_dev_epoch is None:
            training_history = results.get('training_history', {}) or {}
            f1_list = training_history.get('entity_f1_strict') if isinstance(training_history, dict) else None
            if f1_list:
                try:
                    best_idx = max(range(len(f1_list)), key=lambda i: f1_list[i])
                    best_dev_epoch = best_idx + 1  # make it 1-based
                except Exception:
                    best_dev_epoch = -1
            else:
                best_dev_epoch = -1

        # Normalize test metric keys (trainer uses entity_* names)
        test_f1 = test_results.get('entity_f1_strict', test_results.get('f1', 0.0))
        test_precision = test_results.get('entity_precision', test_results.get('precision', 0.0))
        test_recall = test_results.get('entity_recall', test_results.get('recall', 0.0))

        result_dict = {
            'model': model_name,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'training_time_seconds': training_time,
            'best_dev_f1': float(best_dev_f1),
            'best_dev_epoch': int(best_dev_epoch) if isinstance(best_dev_epoch, (int, float)) else -1,
            'test_f1': float(test_f1),
            'test_precision': float(test_precision),
            'test_recall': float(test_recall),
            'timestamp': datetime.now().isoformat()
        }

        logger.info(f"✅ Dev F1: {result_dict['best_dev_f1']:.4f}")
        logger.info(f"✅ Test F1: {result_dict['test_f1']:.4f}")

        # Clean up temp directory to save space
        import shutil
        if output_dir.exists():
            shutil.rmtree(output_dir)

        return result_dict

    except Exception as e:
        logger.error(f"❌ Error training {model_name} with LR={learning_rate}, BS={batch_size}, Epochs={num_epochs}: {e}")
        import traceback
        traceback.print_exc()
        return {
            'model': model_name,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }


def is_experiment_completed(all_results, model_name, learning_rate, batch_size, num_epochs):
    """Check if an experiment with these exact parameters has been completed successfully."""
    for result in all_results:
        if (result.get('model') == model_name and
            result.get('learning_rate') == learning_rate and
            result.get('batch_size') == batch_size and
            result.get('num_epochs') == num_epochs and
            'error' not in result and
            result.get('test_f1') is not None):  # Ensure it completed successfully
            return True
    return False


def main():
    """Run grid search."""
    logger.info("="*80)
    logger.info("Starting Grid Search for BERT-CRF and DeBERTa-CRF")
    logger.info("="*80)
    logger.info(f"Learning Rates: {LEARNING_RATES}")
    logger.info(f"Batch Sizes: {BATCH_SIZES}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Models: {MODELS}")
    logger.info("="*80)

    # Results file
    results_file = Path("grid_search_results.json")
    all_results = []

    # Load existing results if any
    if results_file.exists():
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        logger.info(f"Loaded {len(all_results)} existing results")

    # Calculate total experiments and identify pending ones
    all_experiments = []
    pending_experiments = []
    
    for model_name in MODELS:
        for lr, bs, ep in product(LEARNING_RATES, BATCH_SIZES, EPOCHS):
            experiment = (model_name, lr, bs, ep)
            all_experiments.append(experiment)
            
            if not is_experiment_completed(all_results, model_name, lr, bs, ep):
                pending_experiments.append(experiment)

    total_experiments = len(all_experiments)
    completed_experiments = total_experiments - len(pending_experiments)
    
    logger.info(f"Total experiments: {total_experiments}")
    logger.info(f"Completed experiments: {completed_experiments}")
    logger.info(f"Pending experiments: {len(pending_experiments)}")
    
    if len(pending_experiments) == 0:
        logger.info("All experiments already completed!")
        return

    # Grid search - only run pending experiments
    for i, (model_name, lr, bs, ep) in enumerate(pending_experiments, 1):
        logger.info(f"\n[{i}/{len(pending_experiments)}] Running experiment...")
        logger.info(f"Model: {model_name}, LR: {lr}, BS: {bs}, Epochs: {ep}")

        # Train and evaluate
        result = train_and_evaluate(model_name, lr, bs, ep)
        all_results.append(result)

        # Save results incrementally
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)

        logger.info(f"Results saved to {results_file}")
        logger.info(f"Progress: {i}/{len(pending_experiments)} pending experiments completed")

    # Final summary
    logger.info("\n" + "="*80)
    logger.info("Grid Search Complete!")
    logger.info("="*80)

    # Count successful vs failed experiments
    successful_results = [r for r in all_results if 'error' not in r and r.get('test_f1') is not None]
    failed_results = [r for r in all_results if 'error' in r or r.get('test_f1') is None]
    
    logger.info(f"Total experiments: {len(all_results)}")
    logger.info(f"Successful: {len(successful_results)}")
    logger.info(f"Failed: {len(failed_results)}")

    # Print best results per model
    for model_name in MODELS:
        model_results = [r for r in all_results
                        if r.get('model') == model_name
                        and 'error' not in r
                        and r.get('test_f1') is not None]

        if model_results:
            best = max(model_results, key=lambda x: x.get('test_f1', 0))
            logger.info(f"\n{model_name.upper()} - Best Test F1: {best['test_f1']:.4f}")
            logger.info(f"  Parameters: LR={best['learning_rate']}, BS={best['batch_size']}, Epochs={best['num_epochs']}")
            logger.info(f"  Dev F1: {best['best_dev_f1']:.4f}")
            logger.info(f"  Training time: {best.get('training_time_seconds', 0):.1f}s")
        else:
            logger.info(f"\n{model_name.upper()} - No successful experiments found")

    logger.info(f"\nFull results saved to: {results_file}")


if __name__ == "__main__":
    main()
