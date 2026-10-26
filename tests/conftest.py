"""Global test configuration and environment defaults."""

import os

# Default browser to headless in test suites unless explicitly overridden
os.environ.setdefault("AG_BROWSER_HEADLESS", "1")
