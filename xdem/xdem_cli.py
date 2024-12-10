# Copyright (c) 2024 xDEM developers
#
# This file is part of the xDEM project:
# https://github.com/glaciohack/xdem
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
#
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" CLI configuration for xDEM"""
import argparse
import logging
from argparse import ArgumentParser

import argcomplete

import xdem


def get_parser() -> ArgumentParser:
    """
    ArgumentParser for xdem

    :return: parser
    """
    parser = argparse.ArgumentParser(prog="xdem", description="xDEM command-line interface")

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

    subparsers = parser.add_subparsers(title="Subcommands", dest="command")

    # Subcommand for coregistration
    coregister_parser = subparsers.add_parser("coregister", help="Coregister two DEMs")
    coregister_parser.add_argument("reference_dem", help="path to a reference dem")
    coregister_parser.add_argument("dem_to_be_aligned", help="path to a second dem")

    return parser


def main() -> None:
    """
    Call xDEM's main
    """
    parser = get_parser()
    argcomplete.autocomplete(parser)
    args = parser.parse_args()

    # Show help if no subcommand is provided
    if not args.command:
        parser.print_help()
        return

    # Set the logging configuration
    logging.basicConfig(level=args.loglevel)

    # Handle coregister subcommand
    if args.command == "coregister":
        try:
            xdem.coregister(args.reference_dem, args.dem_to_be_aligned)
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
