import json
import logging
import os
from pathlib import Path

from package_bootstrap.constants import BOOTSTRAP_NAME, DEFAULT_VERSIONS_PATH, PROJECT_ROOT

_logger = logging.getLogger(__name__)


class PackageManager:
    PYPROJECT_FILE = 'pyproject.toml'
    SETUP_FILE = 'setup.py'
    _versions: dict[str, str] | None = None

    @classmethod
    def discover_local_packages(cls, root_dir: Path | None = PROJECT_ROOT) -> dict[str, Path]:
        result = {}
        for root, _, files in os.walk(root_dir):
            root_path = Path(root)
            if cls.PYPROJECT_FILE in files and cls.SETUP_FILE in files:
                result[root_path.name] = root_path
        _logger.info(f"[{BOOTSTRAP_NAME}] Discovered packages: {result}")
        return result

    @classmethod
    def get_version(cls, name: str, versions_file: Path | None = DEFAULT_VERSIONS_PATH):
        cls._load_versions(versions_file)
        return cls._versions.get(name)

    @classmethod
    def get_versioned_packages(cls, names: list[str], versions_file: Path | None = DEFAULT_VERSIONS_PATH) -> list[str]:
        cls._load_versions(versions_file)
        res = []
        for name in names:
            if name in cls._versions:
                res.append(f"{name}=={cls._versions[name]}")
            else:
                res.append(name)
        return res

    @classmethod
    def _load_versions(cls, versions_filepath: Path) -> None:
        if cls._versions is not None:
            return
        with versions_filepath.open(encoding='utf-8') as f:
            raw = json.load(f)
        cls._versions = {item['name']: item['version'] for item in raw}
