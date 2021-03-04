#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.rst") as fp:
    readme = fp.read()

with open("HISTORY.rst") as fp:
    history = fp.read()

setup(
    name="pyGEEMs",
    version="0.2.0",
    packages=find_packages(exclude=[".*tests.*"]),
    test_suite="tests",
    author="Albert Kottke",
    author_email="albert.kottke@gmail.com",
    description="Geotechnical earthquake engineering models implemented in Python.",
    license="MIT",
    long_description=readme + "\n\n" + history,
    long_description_content_type="text/markdown",
    install_requires=[
        "numpy",
        "numba",
        "setuptools",
        "scipy",
    ],
    url="http://github.com/arkottke/pygeems",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Environment :: Console",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering",
        "Intended Audience :: Science/Research",
    ],
)
