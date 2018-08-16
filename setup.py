from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name='BBS Gaussian Processes',
    version=getenv("VERSION", "LOCAL"),
    description='Fitting Gaussian Process models to Breeding Bird Survey data.',
    packages=find_packages()
)
