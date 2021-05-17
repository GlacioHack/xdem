import os
import subprocess
import sys

from setuptools import setup
from setuptools.command.install import install

FULLVERSION = '0.0.2-2'
VERSION = FULLVERSION

class PostInstall(install):
    """
    Post install script to install github dependencies.

    The reason that this is needed is that PyPi has no functionality to parse github dependencies,
    so this is a workaround until they fix it.

    Inspired by: https://github.com/BaderLab/saber/issues/35#issuecomment-467827175
    """
    pkgs = "git+https://github.com/GlacioHack/GeoUtils.git"


    def run(self):
        """Install / reinstall each required GitHub package."""
        install.run(self)

        print(subprocess.getoutput(f"{sys.executable} -m pip install --force-reinstall {self.pkgs}"))



with open(os.path.join(os.path.dirname(__file__), "README.md")) as infile:
    LONG_DESCRIPTION = infile.read()

setup(name='xdem',
      version=FULLVERSION,
      description='Set of tools to manipulate Digital Elevation Models (DEMs) ',
      long_description=LONG_DESCRIPTION,
      long_description_content_type="text/markdown",
      url='https://github.com/GlacioHack/xdem',
      author='The GlacioHack Team',
      author_email="john@doe.com",  # This is needed for PyPi unfortunately.
      license='BSD-3',
      packages=['xdem'],
      install_requires=['numpy', 'scipy', 'rasterio', 'geopandas',
                        'pyproj', 'tqdm', 'scikit-gstat', 'scikit-image'],
      extras_require={'rioxarray': ['rioxarray'], 'richdem': ['richdem'], 'pdal': [
          'pdal'], 'opencv': ['opencv'], "pytransform3d": ["pytransform3d"]},
      scripts=[],
      cmdclass={"install": PostInstall},
      zip_safe=False)

write_version = True


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(os.path.dirname(__file__), 'xdem',
                             'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()


if write_version:
    write_version_py()
