from setuptools import setup
from os import path

FULLVERSION = '0.0.1'
VERSION = FULLVERSION

setup(name='demutils',
      version=FULLVERSION,
      description='',
      url='',
      author='The GlacioHack Team',
      license='BSD-3',
      packages=['DemUtils'],
      install_requires=['numpy', 'scipy', 'rasterio', 'geopandas', 'pyproj'],
      extras_require={'rioxarray': ['rioxarray'], 'richdem': ['richdem'], 'pdal':['pdal']},
      scripts=[],
      zip_safe=False)

write_version = True


def write_version_py(filename=None):
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = path.join(path.dirname(__file__), 'DemUtils',
                             'version.py')

    a = open(filename, 'w')
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()


if write_version:
    write_version_py()
