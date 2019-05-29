#!/usr/bin/env python

"""
Copyright (c) 2019, EPFL/Blue Brain Project

This file is part of BlueBrain SNAP library <https://github.com/BlueBrain/snap>

This library is free software; you can redistribute it and/or modify it under
the terms of the GNU Lesser General Public License version 3.0 as published
by the Free Software Foundation.

This library is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
details.

You should have received a copy of the GNU Lesser General Public License
along with this library; if not, write to the Free Software Foundation, Inc.,
51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

from setuptools import setup, find_packages


setup(
    name='bluesnap',
    install_requires=[
        'cached_property>=1.0',
        'functools32;python_version<"3.2"',
        'h5py>=2.2',
        'libsonata>=0.0.2',
        'neurom>=1.3',
        'numpy>=1.8',
        'pandas>=0.17',
        'six>=1.0',
    ],
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=[
        'setuptools_scm',
    ],
    author="BlueBrain Project, EPFL",
    author_email="bbp-ou-nse@groupes.epfl.ch",
    description="(bluesnap)",
    long_description="(bluesnap)",
    url='https://github.com/BlueBrain/snap',
    keywords=[
        'SONATA',
        'BlueBrainProject'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.6',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering',
        'Topic :: Utilities',
        'License :: OSI Approved :: GNU Lesser General Public '
        'License v3 (LGPLv3)',
    ]
)
