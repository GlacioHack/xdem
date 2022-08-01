import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.install import install

FULLVERSION = "0.0.6"
VERSION = FULLVERSION

with open(os.path.join(os.path.dirname(__file__), "README.md")) as infile:
    LONG_DESCRIPTION = infile.read()

setup(
    name="xdem",
    version=FULLVERSION,
    description="Set of tools to manipulate Digital Elevation Models (DEMs) ",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/GlacioHack/xdem",
    author="The GlacioHack Team",
    author_email="this-is-not-an-email@a.com",  # This is needed for PyPi unfortunately.
    license="BSD-3",
    packages=["xdem"],
    install_requires=[
        "numpy",
        "scipy",
        "rasterio",
        "geopandas",
        "pyproj",
        "tqdm",
        "scikit-gstat",
        "scikit-image",
        "geoutils",
    ],
    extras_require={
        "rioxarray": ["rioxarray"],
        "richdem": ["richdem"],
        "opencv": ["opencv"],
        "pytransform3d": ["pytransform3d"],
    },
    python_requires=">=3.7",
    scripts=[],
    zip_safe=False,
)

write_version = True


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(os.path.dirname(__file__), "xdem", "version.py")

    a = open(filename, "w")
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()


if write_version:
    write_version_py()
