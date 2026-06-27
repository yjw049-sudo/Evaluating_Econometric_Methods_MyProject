"""Run the 1953-1993 data-processing pipeline in order."""

import subprocess
import sys
from pathlib import Path


WORK_DIR = Path(__file__).resolve().parent

SCRIPTS = [
    "merge_processed_with_lipad.py",
    "merge_speeches_parlinfo.py",
    "run_kmeans.py",
]


def run_script(script_name):
    """Run one Python script and stop if it fails."""
    script_path = WORK_DIR / script_name

    if not script_path.exists():
        raise FileNotFoundError(f"Script was not found: {script_path}")

    print(f"\n{'=' * 70}")
    print(f"Running: {script_name}")
    print(f"{'=' * 70}")

    subprocess.run(
        [sys.executable, str(script_path)],
        cwd=WORK_DIR,
        check=True,
    )

    print(f"Completed: {script_name}")


def main():
    """Run all pipeline scripts sequentially."""
    for script_name in SCRIPTS:
        run_script(script_name)

    print("\nAll pipeline scripts completed successfully.")


if __name__ == "__main__":
    main()
