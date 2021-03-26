"""Console script for mabrl."""
import sys

from mabrl.example_experiment import main as main_experiments


def main():
    """Console script for mabrl."""
    main_experiments()
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
