import setuptools

from package_bootstrap.package_manager import PackageManager

PACKAGE_NAME = 'misis_orchestrator'
VERSION = '2.1.1'
AUTHOR = 'ButakovBI'

REQUIRES = [
    "aiokafka",
    "asyncpg",
    "pydantic",
    "pydantic_settings",
    "sqlalchemy",
]

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    description='Misis Orchestrator',
    packages=setuptools.find_packages(where='source'),
    package_dir={'': 'source'},
    install_requires=PackageManager.get_versioned_packages(REQUIRES),
)
