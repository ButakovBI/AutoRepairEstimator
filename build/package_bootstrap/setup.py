import setuptools

PACKAGE_NAME = 'package_bootstrap'
VERSION = '1.0.0'
AUTHOR = 'ButakovBI'

setuptools.setup(
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    description='Package Bootstrap',
    packages=setuptools.find_packages(where='source'),
    package_dir={'': 'source'},
    entry_points={'console_scripts': [f'package-bootstrap={PACKAGE_NAME}.main:main']},
)
