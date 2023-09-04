from __future__ import annotations

import os

from setuptools import setup

setup()

write_version = True


def write_version_py(filename: str | None = None) -> None:
    cnt = """\
version = '%s'
short_version = '%s'
"""
    if not filename:
        filename = os.path.join(os.path.dirname(__file__), "xdem", "__version__.py")

    a = open(filename, "w")
    try:
        a.write(cnt % (FULLVERSION, VERSION))
    finally:
        a.close()


if write_version:
    write_version_py()
