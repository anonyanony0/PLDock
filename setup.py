import sys
import time
from setuptools import setup, find_packages
setup(
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'atom3-py3',
        'pandas',
        'biopandas',
        'tqdm',
        'easy-parallel-py3',
        'numpy',
        'biopython',
        'loguru',
        'deepchem'
    ],
    python_requires='>=3.7,<3.10')