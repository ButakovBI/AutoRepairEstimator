import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ['REPO_ROOT']).resolve()

BOOTSTRAP_NAME = 'package_bootstrap'
BUILD_DIR = 'build'
DEFAULT_VERSIONS_PATH = PROJECT_ROOT / BUILD_DIR / BOOTSTRAP_NAME / 'versions.json'
DEFAULT_WHEEL_DIR = PROJECT_ROOT / BUILD_DIR / 'whl'
