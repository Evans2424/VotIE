#!/usr/bin/env python3
"""
VotIE Dataset Generation Pipeline - Master Orchestrator

This script runs the complete pipeline to generate all publication-ready dataset formats:
1. Convert Citilink to Event format (with main splits)
2. Convert Events to Spans format
3. Convert Events to CoNLL format
4. Create LOMO splits (all formats)
5. Generate comprehensive documentation

Run this script to generate the complete VotIE dataset.
"""

import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_script(script_path: Path, description: str) -> bool:
    """
    Run a Python script and return success status.

    Args:
        script_path: Path to script
        description: Human-readable description

    Returns:
        True if script succeeded
    """
    logger.info(f"\n{'='*80}")
    logger.info(f"Running: {description}")
    logger.info(f"Script: {script_path}")
    logger.info(f"{'='*80}\n")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            check=True,
            capture_output=False,
            text=True
        )
        logger.info(f"\n✓ {description} completed successfully\n")
        return True

    except subprocess.CalledProcessError as e:
        logger.error(f"\n✗ {description} failed with return code {e.returncode}\n")
        return False

    except Exception as e:
        logger.error(f"\n✗ {description} failed with error: {e}\n")
        return False


def check_prerequisites() -> bool:
    """Check that all prerequisites are met."""
    logger.info("Checking prerequisites...")

    # Check that citilink directory exists
    citilink_dir = Path("private/citilink")
    if not citilink_dir.exists():
        logger.error(f"Citilink directory not found: {citilink_dir}")
        logger.error("Please ensure citilink/ directory exists with Portuguese municipality files")
        return False

    # Check for municipality files
    municipalities = ["alandroal", "campomaior", "covilha", "fundao", "guimaraes", "porto"]
    missing_files = []

    for muni in municipalities:
        file_path = citilink_dir / f"{muni}_citilink_pt.json"
        if not file_path.exists():
            missing_files.append(file_path)

    if missing_files:
        logger.error("Missing municipality files:")
        for f in missing_files:
            logger.error(f"  {f}")
        return False

    logger.info("✓ All prerequisites met")
    return True


def generate_summary_report(output_dir: Path):
    """Generate a final summary report of all generated datasets."""
    logger.info("\n" + "="*80)
    logger.info("GENERATING SUMMARY REPORT")
    logger.info("="*80)

    report_lines = []
    report_lines.append("# VotIE Dataset Generation Summary")
    report_lines.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("\n## Generated Datasets\n")

    # Check main splits
    report_lines.append("### Main Splits (Stratified)")
    for format_name in ["votie_events", "votie_spans", "votie_conll"]:
        format_dir = output_dir / format_name
        if format_dir.exists():
            report_lines.append(f"- **{format_name}/**")
            for split in ["train", "dev", "test"]:
                if format_name == "votie_conll":
                    split_file = format_dir / f"{split}.conll"
                else:
                    split_file = format_dir / f"{split}.jsonl"

                if split_file.exists():
                    size_mb = split_file.stat().st_size / (1024 * 1024)
                    report_lines.append(f"  - {split_file.name} ({size_mb:.2f} MB)")

    # Check LOMO splits
    lomo_dir = output_dir / "votie_lomo"
    if lomo_dir.exists():
        report_lines.append("\n### LOMO Splits (Leave-One-Municipality-Out)")
        municipalities = sorted([d.name for d in lomo_dir.iterdir() if d.is_dir()])
        report_lines.append(f"- {len(municipalities)} municipality splits:")
        for muni in municipalities:
            report_lines.append(f"  - **{muni}/** (events, spans, conll)")

    # Save report
    report_file = output_dir / "GENERATION_SUMMARY.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))

    logger.info(f"Summary report saved to: {report_file}")

    # Print to console
    print("\n" + "="*80)
    print("DATASET GENERATION SUMMARY")
    print("="*80)
    for line in report_lines[2:]:  # Skip title
        print(line)
    print("="*80 + "\n")


def main():
    """Main orchestration pipeline."""
    start_time = datetime.now()

    print("\n" + "="*80)
    print("VotIE DATASET GENERATION PIPELINE")
    print("="*80)
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")

    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed. Aborting.")
        return 1

    scripts_dir = Path("data/scripts")
    output_dir = Path("data")

    # Define pipeline steps
    steps = [
        (scripts_dir / "convert_citilink_to_events.py",
         "Step 1/5: Convert Citilink to Event format with main splits"),

        (scripts_dir / "convert_events_to_spans.py",
         "Step 2/5: Convert Event format to Spans format"),

        (scripts_dir / "convert_events_to_conll.py",
         "Step 3/5: Convert Event format to CoNLL format"),

        (scripts_dir / "create_lomo_splits.py",
         "Step 4/5: Create LOMO splits (all formats)"),
    ]

    # Run pipeline
    all_succeeded = True
    for script_path, description in steps:
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            all_succeeded = False
            break

        if not run_script(script_path, description):
            logger.error(f"Pipeline failed at: {description}")
            all_succeeded = False
            break

    # Generate final documentation and summary
    if all_succeeded:
        logger.info("\n" + "="*80)
        logger.info("Step 5/5: Generating final documentation")
        logger.info("="*80 + "\n")

        generate_summary_report(output_dir)

    # Final status
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"Status: {'SUCCESS' if all_succeeded else 'FAILED'}")
    print(f"Duration: {duration}")
    print(f"Started:  {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")

    if all_succeeded:
        print("\nGenerated datasets:")
        print("  - data/votie_events/     (Event-based format with main splits)")
        print("  - data/votie_spans/      (Span-based format)")
        print("  - data/votie_conll/      (CoNLL sequence labeling format)")
        print("  - data/votie_lomo/       (LOMO splits for cross-domain evaluation)")
        print("\nNext steps:")
        print("  1. Review data/GENERATION_SUMMARY.md")
        print("  2. Check data/conversion.log for detailed logs")
        print("  3. Review validation reports in each format directory")
        print("  4. Generate README.md files (run manually if needed)")
    else:
        print("\nPipeline failed. Check logs above for details.")
        print("Review data/conversion.log for full error details.")

    print("="*80 + "\n")

    return 0 if all_succeeded else 1


if __name__ == "__main__":
    sys.exit(main())
