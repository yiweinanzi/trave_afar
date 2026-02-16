#!/usr/bin/env python
"""
Test runner script for GoAfar project.

Usage:
    python run_tests.py                    # Run all tests
    python run_tests.py --unit             # Run only unit tests
    python run_tests.py --integration       # Run only integration tests
    python run_tests.py --coverage         # Run with coverage report
    python run_tests.py --module embedding  # Run specific module tests
    python run_tests.py --verbose         # Verbose output
"""
import argparse
import subprocess
import sys
from pathlib import Path


def run_pytest(args):
    """Run pytest with the given arguments."""
    cmd = ["python", "-m", "pytest"]

    # Add pytest.ini config
    cmd.extend(["-c", "pytest.ini"])

    # Add verbosity
    if args.verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")

    # Add coverage
    if args.coverage:
        cmd.extend([
            "--cov=src/embedding",
            "--cov=src/routing",
            "--cov=src/llm4rec",
            "--cov=src/service",
            "--cov=src/ranking",
            "--cov-report=term-missing",
            "--cov-report=html:outputs/coverage/html",
        ])

    # Filter by test type
    if args.unit:
        cmd.append("-m")
        cmd.append("unit")
    elif args.integration:
        cmd.append("-m")
        cmd.append("integration")
    elif args.slow:
        cmd.append("-m")
        cmd.append("slow")

    # Filter by module
    if args.module:
        module_path = f"tests/{args.module}"
        cmd.append(module_path)

    # Specific test file
    if args.test:
        cmd.append(args.test)

    # Additional args
    if args.remaining:
        cmd.extend(args.remaining)

    # Run tests
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    return result.returncode


def main():
    parser = argparse.ArgumentParser(
        description="Run GoAfar project tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--unit",
        action="store_true",
        help="Run only unit tests",
    )
    parser.add_argument(
        "--integration",
        action="store_true",
        help="Run only integration tests",
    )
    parser.add_argument(
        "--slow",
        action="store_true",
        help="Run only slow tests",
    )
    parser.add_argument(
        "--coverage",
        "-c",
        action="store_true",
        help="Generate coverage report",
    )
    parser.add_argument(
        "--module",
        "-m",
        choices=["embedding", "routing", "llm4rec", "service", "ranking"],
        help="Run tests for specific module",
    )
    parser.add_argument(
        "--test",
        "-t",
        help="Run specific test file",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )
    parser.add_argument(
        "remaining",
        nargs=argparse.REMAINDER,
        help="Additional arguments to pass to pytest",
    )

    args = parser.parse_args()

    # Create outputs directory
    (Path(__file__).parent / "outputs" / "coverage").mkdir(parents=True, exist_ok=True)

    # Run tests
    exit_code = run_pytest(args)

    # Print summary
    if exit_code == 0:
        print("\n" + "="*60)
        print("All tests passed!")
        print("="*60)
    else:
        print("\n" + "="*60)
        print("Some tests failed.")
        print("="*60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
