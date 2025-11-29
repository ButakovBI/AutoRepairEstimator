import logging
import os
import shutil
import subprocess
from pathlib import Path

from package_bootstrap.constants import BOOTSTRAP_NAME
from package_bootstrap.package_manager import PackageManager


class Bootstrap:

    def __init__(self, wheel_dir: Path, root_dir: Path):
        self.wheel_dir = wheel_dir
        self.root_dir = root_dir
        self._clean_wheel_dir()

    def build_wheels(self, packages: list[str]) -> None:
        packages = set(packages)
        found_packages = PackageManager.discover_local_packages(self.root_dir)
        package_paths_to_clean = []

        def build_and_collect(name: str):
            package_path = self._build_package(name, found_packages)
            package_paths_to_clean.append(package_path)

        if BOOTSTRAP_NAME in packages:
            build_and_collect(BOOTSTRAP_NAME)
            packages.remove(BOOTSTRAP_NAME)

        for name in packages:
            build_and_collect(name)

        if package_paths_to_clean:
            self._cleanup_package_artifacts(package_paths_to_clean)

    def _build_package(self, name: str, found_packages: dict[str, Path]) -> Path:
        if name not in found_packages:
            raise RuntimeError(f"[{BOOTSTRAP_NAME}] Package not found: {name}")

        package_path = found_packages[name]
        self._logger.info(f"Building package: {name}")

        try:
            subprocess.run([
                "pip3", "wheel",
                "--wheel-dir", str(self.wheel_dir),
                "--find-links", str(self.wheel_dir),
                str(package_path)
            ], check=True)
            self._logger.info(f"Built wheel for {name}")
        except Exception as e:
            self._logger.error(f"Failed to build wheel for {name}: {e}")
            raise RuntimeError(f"[{BOOTSTRAP_NAME}] Build failed for {name}")
        return package_path

    def _clean_wheel_dir(self) -> None:
        if self.wheel_dir.exists():
            for f in self.wheel_dir.glob("*"):
                f.unlink()
        self.wheel_dir.mkdir(parents=True, exist_ok=True)

    def _cleanup_package_artifacts(self, package_paths: list[Path]):
        self._logger.info('Cleaning up build artifacts...')
        for package_path in package_paths:
            build_dir = package_path / 'build'
            if build_dir.is_dir():
                shutil.rmtree(build_dir)

            egg_dir = Path(f'{package_path}/source/{os.path.basename(package_path)}.egg-info')
            if egg_dir.is_dir():
                shutil.rmtree(egg_dir)

    @property
    def _logger(self) -> logging.Logger:
        return logging.getLogger(__name__)
