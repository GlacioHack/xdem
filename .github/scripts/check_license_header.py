"""Check source license headers, allowing different copyright years for xDEM developers and CNES."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HEADER_FILE = Path(__file__).with_name("license-header.txt")
COPYRIGHT_PATTERN = re.compile(
    r"\A(?:# Copyright \(c\) [0-9]{4} " r"(?:xDEM developers|Centre National d'Etudes Spatiales \(CNES\))\n)+"
)


def check_license_header(path: Path, license_body: str) -> str | None:
    """Check copyright lines and license text without changing the source file."""

    # Normalize line endings so Windows and Unix files use the same template
    try:
        content = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        return f"Cannot read file: {error}"

    # First, we allow xDEM developers and CNES with any year
    copyright_lines = COPYRIGHT_PATTERN.match(content)
    if copyright_lines is None:
        return "Expected copyright lines with four-digit years for xDEM developers and/or CNES"
    # Then we check the complete license text
    if not content[copyright_lines.end() :].startswith(license_body):
        return f"License text must match {HEADER_FILE.name} after the copyright lines"
    return None


def main(argv: list[str]) -> int:
    """Report invalid license headers and return a nonzero exit status if any fail."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="Python source files to check")
    args = parser.parse_args(argv)

    license_body = HEADER_FILE.read_text(encoding="utf-8").split("\n", maxsplit=1)[1]
    failed = False
    for path in args.files:
        error = check_license_header(path, license_body)
        if error is not None:
            print(f"{path}: {error}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
