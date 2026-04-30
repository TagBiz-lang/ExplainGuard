#!/usr/bin/env python3
"""
ExplainGuard Clean - Main Entry Point

Quick setup and execution for ExplainGuard evaluation.
Provides interactive menu for common tasks.
"""

import sys
import subprocess
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from config import Config
from src.utils.helpers import setup_logging

logger = setup_logging(__name__)


def print_banner():
    print("\n" + "=" * 80)
    print("  EXPLAINGUARD CLEAN - OPTIMIZED WORKFLOW")
    print("=" * 80 + "\n")


def print_menu():
    print("Select an operation:\n")
    print("  1) Setup environment (initialize directories & link datasets)")
    print("  2) Evaluate single dataset")
    print("  3) Evaluate multiple datasets (auto-discover)")
    print("  4) View recent reports")
    print("  5) Exit\n")


def setup_environment():
    """Run setup script"""
    logger.info("Running setup...")
    result = subprocess.run(
        ["python", str(project_root / "scripts" / "setup.py")],
        cwd=str(project_root),
    )
    return result.returncode == 0


def evaluate_single():
    """Run single dataset evaluation"""
    dataset_path = input("Enter dataset path (or press Enter for auto-detect): ").strip()
    dataset_name = input("Enter dataset name (optional): ").strip() or None
    device = input("Device (cuda/cpu, default: cuda): ").strip() or "cuda"

    cmd = ["python", str(project_root / "scripts" / "evaluate.py")]
    if dataset_path:
        cmd.append(dataset_path)
    if dataset_name:
        cmd.extend(["--name", dataset_name])
    cmd.extend(["--device", device])

    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode == 0


def evaluate_multi():
    """Run multi-dataset evaluation"""
    dataset_dir = input("Dataset directory (press Enter for auto-detect): ").strip()
    device = input("Device (cuda/cpu, default: cuda): ").strip() or "cuda"

    cmd = ["python", str(project_root / "scripts" / "evaluate_multi_dataset.py")]
    if dataset_dir:
        cmd.extend(["--dataset_dir", dataset_dir])
    cmd.extend(["--device", device])

    result = subprocess.run(cmd, cwd=str(project_root))
    return result.returncode == 0


def view_reports():
    """List available reports"""
    print("\nAvailable reports:\n")

    reports_dir = Config.REPORTS_DIR
    if not reports_dir.exists():
        print("  No reports generated yet.")
        return

    reports = list(reports_dir.glob("*.json"))
    if not reports:
        print("  No reports found.")
        return

    for idx, report in enumerate(sorted(reports), 1):
        size_kb = report.stat().st_size / 1024
        print(f"  {idx}) {report.name} ({size_kb:.1f} KB)")

    print("\nUse a JSON viewer to inspect reports, e.g.:")
    print(f"  cat {reports_dir}/evaluation_*.json | python -m json.tool\n")


def main():
    print_banner()

    while True:
        print_menu()
        choice = input("Enter choice (1-5): ").strip()

        print()

        if choice == "1":
            if setup_environment():
                logger.info("✓ Setup completed successfully\n")
            else:
                logger.error("✗ Setup failed\n")

        elif choice == "2":
            if evaluate_single():
                logger.info("✓ Evaluation completed successfully\n")
            else:
                logger.error("✗ Evaluation failed\n")

        elif choice == "3":
            if evaluate_multi():
                logger.info("✓ Multi-dataset evaluation completed successfully\n")
            else:
                logger.error("✗ Multi-dataset evaluation failed\n")

        elif choice == "4":
            view_reports()

        elif choice == "5":
            logger.info("Exiting...")
            sys.exit(0)

        else:
            logger.error("Invalid choice. Please try again.\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        sys.exit(1)
