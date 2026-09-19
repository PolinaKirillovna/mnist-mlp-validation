"""Enable ``python -m mnist_validation``."""

from __future__ import annotations

import sys

from mnist_validation.cli import main

if __name__ == "__main__":
    sys.exit(main())
