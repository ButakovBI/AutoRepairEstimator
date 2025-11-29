import setuptools

PACKAGE_NAME = 'deploy_manager'
VERSION = '1.0.1'
AUTHOR = 'ButakovBI'


setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    description='Docker and docker-compose deploy manager',
    packages=setuptools.find_packages(where='source'),
    package_dir={'': 'source'},
    entry_points={'console_scripts': [f'run-deploy={PACKAGE_NAME}.main:main']},
)
