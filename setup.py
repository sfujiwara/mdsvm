# coding: utf-8

from setuptools import setup, find_packages
from mdsvm import __license__, __author__, __version__

setup(
    name='mdsvm',
    description='SVM implementation with minimal dependency to other software',
    version=__version__,
    license=__license__,
    author=__author__,
    author_email='shuhei.fujiwara@gmail.com',
    url='https://github.com/sfujiwara/mdsvm',
    packages=find_packages(),
    # install_requires=['numpy'],
)
