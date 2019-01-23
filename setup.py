#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as fp:
    readme = fp.read()

setup(
    name='pygeems',
    version='0.1.0',
    packages=find_packages(exclude=['.*tests.*']),
    test_suite='tests',
    author='Albert Kottke',
    author_email='albert.kottke@gmail.com',
    description='Geotechnical earthquake engineering models implemented in Python.',
    license='MIT',
    long_description=readme,
    long_description_content_type='text/markdown',
    url='http://github.com/arkottke/pyrvt',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering',
        'Intended Audience :: Science/Research',
    ], )
