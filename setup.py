"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

# Always prefer setuptools over distutils
from setuptools import setup, find_packages
# To use a consistent encoding
# from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

# Get the version string from the VERSION file
with open(path.join(here, 'VERSION'), 'r') as f:
    version = f.readline().strip()

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='burstcalc',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version=version,
    description='Burst Searching tools',
    long_description=long_description,
    author='Qi Feng',
    author_email='',
    install_requires=[
        'matplotlib',
        'numpy',
        'tables',
        'pandas',
        'scipy',
    ],
    entry_points={
        'console_scripts': [
            'run_burst=burstcalc:main',
            'run1burst=burstcalc:run1'
        ],
    },
)
