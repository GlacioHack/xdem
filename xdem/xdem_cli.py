import argparse
from argparse import ArgumentParser

import argcomplete

import xdem


def get_parser() -> ArgumentParser:
    """
    ArgumentParser for xdem

    :return: parser
    """
    parser = argparse.ArgumentParser(
        description="Compare Digital Elevation Models",
        fromfile_prefix_chars="@",
    )

    parser.add_argument(
        "reference_dem",
        help="path to a reference dem",
    )

    parser.add_argument(
        "dem_to_be_aligned",
        help="path to a second dem",
    )

    parser.add_argument(
        "--loglevel",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"),
        help="Logger level (default: INFO. Should be one of (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )

    parser.add_argument(
        "--version",
        "-v",
        action="version",
        version=f"%(prog)s {xdem.__version__}",
    )

    return parser


def main() -> None:
    """
    Call xDEM's main
    """
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    try:
        xdem.run(args.reference_dem, args.dem_to_be_aligned, args.loglevel)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
