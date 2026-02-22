#!/usr/bin/env python3
"""Main pipeline orchestrator for the Passos Magicos ML pipeline.

Runs all 6 pipeline steps in sequence. Each step is a standalone Python
script that can also be executed independently.

Usage:
    python run_pipeline.py                   # Run full pipeline
    python run_pipeline.py --dry-run         # Print steps without executing
    python run_pipeline.py --start-from 3   # Resume from step 3 (03_eda.py)
"""

import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

STEPS: list[tuple[str, str]] = [
    ("01 - Load & Merge CSVs", "src/pipeline/01_load.py"),
    ("02 - Validate Target Column", "src/pipeline/02_validate.py"),
    ("03 - Exploratory Data Analysis", "src/pipeline/03_eda.py"),
    ("04 - Preprocessing & Split", "src/pipeline/04_preprocess.py"),
    ("05 - Train & Select Model", "src/pipeline/05_train.py"),
    ("06 - Evaluate on Test Set", "src/pipeline/06_evaluate.py"),
]


def parse_args() -> tuple[bool, int]:
    """Parse CLI flags from sys.argv.

    Returns:
        Tuple of (dry_run: bool, start_from: int) where start_from is 1-indexed.
    """
    dry_run = "--dry-run" in sys.argv
    start_from = 1

    for i, arg in enumerate(sys.argv):
        if arg == "--start-from" and i + 1 < len(sys.argv):
            try:
                start_from = int(sys.argv[i + 1])
            except ValueError:
                print(f"  [WARN] Invalid --start-from value '{sys.argv[i + 1]}' -- defaulting to 1")

    return dry_run, start_from


def run_step(name: str, script: str, dry_run: bool = False) -> bool:
    """Execute a single pipeline step as a subprocess.

    Args:
        name: Human-readable step name for logging.
        script: Path to the Python script relative to project root.
        dry_run: If True, print the command without executing.

    Returns:
        True if the step succeeded or dry_run is active, False otherwise.
    """
    script_path = PROJECT_ROOT / script

    if not script_path.exists():
        print(f"  [ERROR] Script not found: {script_path}")
        return False

    if dry_run:
        print(f"  [DRY-RUN] Would execute: python {script_path}")
        return True

    print(f"  Executing: python {script_path.name}")
    start = time.time()

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False,
    )

    elapsed = time.time() - start
    if result.returncode == 0:
        print(f"  [OK] Completed in {elapsed:.1f}s")
        return True

    print(f"  [FAIL] Exit code {result.returncode} after {elapsed:.1f}s")
    return False


def main() -> None:
    """Run all pipeline steps in sequence with optional skip and dry-run support."""
    dry_run, start_from = parse_args()

    print("=" * 60)
    print("  Passos Magicos ML Pipeline")
    print("=" * 60)

    mode_label = "DRY-RUN" if dry_run else "FULL EXECUTION"
    print(f"  Mode: {mode_label}")
    if start_from > 1:
        print(f"  Starting from step: {start_from}")
    print()

    total_start = time.time()
    failed = False

    for step_num, (name, script) in enumerate(STEPS, start=1):
        if step_num < start_from:
            print(f"--- Step {step_num}/{len(STEPS)}: {name} [SKIPPED] ---")
            continue

        print(f"\n--- Step {step_num}/{len(STEPS)}: {name} ---")
        success = run_step(name, script, dry_run=dry_run)

        if not success:
            print(f"\n[ABORT] Pipeline failed at step {step_num}: {name}")
            failed = True
            break

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 60)

    if failed:
        print(f"  Pipeline FAILED after {total_elapsed:.1f}s")
        sys.exit(1)
    else:
        status = "DRY-RUN complete" if dry_run else "Pipeline complete"
        print(f"  {status} in {total_elapsed:.1f}s")
        if not dry_run:
            print("  Outputs:")
            print("    - model/best_model.pkl")
            print("    - metrics/report.json")
            print("    - metrics/eda_summary.json")
            print("    - metrics/confusion_matrix.png")
            print("    - metrics/feature_importance.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
