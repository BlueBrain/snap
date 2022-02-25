#!/usr/bin/env python

# Copyright (c) 2019, EPFL/Blue Brain Project

# This file is part of BlueBrain SNAP library <https://github.com/BlueBrain/snap>

# This library is free software; you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License version 3.0 as published
# by the Free Software Foundation.

# This library is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public License for more
# details.

# You should have received a copy of the GNU Lesser General Public License
# along with this library; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

from setuptools import find_packages, setup

with open("README.rst") as f:
    README = f.read()

setup(
    name="bluepysnap",
    python_requires=">=3.7",
    install_requires=[
        "cached_property>=1.0",
        "h5py>=3.0.1,<4.0.0",
        "libsonata>=0.1.6,<1.0.0",
        "morphio>=3.0.0,<4.0.0",
        "morph-tool>=2.4.3,<3.0.0",
        "numpy>=1.8,<2.0.0",
        "pandas>=1.0.0,<2.0.0",
        "click>=7.0",
        "more-itertools>=8.2.0",
    ],
    extras_require={
        "docs": ["sphinx", "sphinx-bluebrain-theme"],
        "plots": ["matplotlib>=3.0.0,<4.0.0"],
    },
    packages=find_packages(),
    use_scm_version=True,
    setup_requires=[
        "setuptools_scm",
    ],
    entry_points="""
        [console_scripts]
        bluepysnap=bluepysnap.cli:cli
    """,
    author="Blue Brain Project, EPFL",
    description="Simulation and Neural network Analysis Productivity layer",
    long_description=README,
    long_description_content_type="text/x-rst",
    license="LGPLv3",
    url="https://github.com/BlueBrain/snap",
    keywords=["SONATA", "BlueBrainProject"],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering",
        "Topic :: Utilities",
        "License :: OSI Approved :: GNU Lesser General Public License v3 (LGPLv3)",
    ],
)
