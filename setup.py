from distutils.core import setup
from setuptools import find_packages

setup(
    name='intrinsic-rl',
    version='0.1dev',
    packages=find_packages(),
    long_description=open('README.md').read()
)
