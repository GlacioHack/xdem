"""This file now only serves for backward-compatibility for routines explicitly calling python setup.py"""
from setuptools import find_packages, setup

setup(
    name="xdem",
    use_scm_version=True,  # Enable versioning with setuptools_scm
    setup_requires=["setuptools_scm"],  # Ensure setuptools_scm is used to determine the version
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "xdem = xdem.xdem_cli:main",
        ],
    },
)
