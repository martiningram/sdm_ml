from os import getenv
from setuptools import setup
from setuptools import find_packages


setup(
    name='Machine learning for (Joint) Species Distribution Modelling',
    version=getenv("VERSION", "LOCAL"),
    description='Fitting machine learning models to (joint) species models.',
    packages=find_packages()
)
