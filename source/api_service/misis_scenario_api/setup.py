import setuptools

from package_bootstrap.package_manager import PackageManager

PACKAGE_NAME = 'misis_scenario_api'
VERSION = '2.2.3'
AUTHOR = 'ButakovBI'

REQUIRES = [
    "aioboto3",
    "aiokafka",
    "asyncpg",
    "fastapi",
    "httpx",
    "pydantic",
    "pydantic_settings",
    "python-multipart",
    "sqlalchemy",
    "uvicorn[standard]"
]

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    description='Misis Scenario API',
    packages=setuptools.find_packages(where='source'),
    package_dir={'': 'source'},
    install_requires=PackageManager.get_versioned_packages(REQUIRES),
)
