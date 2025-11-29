import argparse

from package_bootstrap.constants import DEFAULT_WHEEL_DIR, PROJECT_ROOT
from package_bootstrap.package_manager import PackageManager
from package_bootstrap.bootstrap import Bootstrap


def main():
    parser = argparse.ArgumentParser(description="Build selected local packages into wheels")
    parser.add_argument(
        "--packages",
        nargs="+",
        help="List of package names to build space-separated",
    )
    args = parser.parse_args()
    wheel_dir = PROJECT_ROOT / DEFAULT_WHEEL_DIR

    if args.packages:
        packages_to_build = args.packages
    else:
        packages_to_build = PackageManager.discover_local_packages(PROJECT_ROOT)

    bootstrap = Bootstrap(wheel_dir, PROJECT_ROOT)
    bootstrap.build_wheels(packages_to_build)
