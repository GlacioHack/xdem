# Copyright (c) 2025 Centre National d'Etudes Spatiales (CNES).
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


import argparse
import ctypes.util
import logging
import sys

from xdem._misc import import_optional
from xdem.workflows import Accuracy, Topo
from xdem.workflows.schemas import COMPLETE_CONFIG_ACCURACY, COMPLETE_CONFIG_TOPO

try:
    lib_gobject_name = ctypes.util.find_library("gobject-2.0")
    lib_pango_name = ctypes.util.find_library("pango-1.0")
    if lib_gobject_name and lib_pango_name:
        from weasyprint import HTML

        _has_libgobject = True
    else:
        _has_libgobject = False
    _has_weasyprint = _has_libgobject
except ImportError:
    _has_weasyprint = False


def main() -> None:
    """
    Main function for the CLI
    """

    yaml = import_optional("yaml", package_name="pyyaml")

    parser = argparse.ArgumentParser(prog="xdem", description="CLI tool to run xDEM workflows", add_help=False)
    parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )

    subparsers = parser.add_subparsers(
        dest="command",
        help="Available workflows as subcommand (see xdem [workflow] -h"
        " for more information on the specific workflow)",
    )

    # Subcommand: info
    topo_parser = subparsers.add_parser(
        "topo",
        help="Run topography workflow",
        description="Run the topography workflow using a YAML configuration file.",
        epilog="examples:\n"
        "  xdem topo --config config.yaml --output myoutputfolder\n"
        "  xdem topo --template-config --output template_config.yaml",
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    topo_parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit"
    )
    topo_group = topo_parser.add_mutually_exclusive_group(required=True)
    topo_group.add_argument(
        "--config",
        help="Path to YAML configuration file",
    )
    topo_parser.add_argument(
        "--output",
        help="(Optional) Path to output folder with --config (overrides configuration file)"
        "or file with --template-config",
    )
    topo_group.add_argument("--template-config", action="store_true", help="Show template of YAML configuration file")

    # Subcommand: accuracy
    diff_parser = subparsers.add_parser(
        "accuracy",
        help="Run accuracy workflow",
        description="Run the accuracy assessment workflow for elevation data using a YAML configuration file.",
        epilog="examples:\n"
        "  xdem accuracy --config config.yaml --output myoutputfolder\n"
        "  xdem accuracy --template-config --output template_config.yaml",
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    diff_parser.add_argument(
        "-h", "--help", action="help", default=argparse.SUPPRESS, help="Show this help message and exit"
    )
    diff_group = diff_parser.add_mutually_exclusive_group(required=True)
    diff_group.add_argument("--config", help="Path to YAML configuration file")
    diff_group.add_argument("--template-config", action="store_true", help="Show template of YAML configuration file")
    diff_parser.add_argument(
        "--output",
        help="(Optional) Path to output folder with --config (overrides configuration file)"
        "or file with --template-config",
    )
    args = parser.parse_args(args=None if sys.argv[1:] else ["--help"])

    # Instance logger
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    # fontTools creates noisy logs
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools").propagate = False

    if args.command == "topo":
        if args.template_config:
            if args.output:
                with open(args.output, "w") as yaml_file:
                    yaml.dump(COMPLETE_CONFIG_TOPO, yaml_file, sort_keys=False, allow_unicode=True)
                logging.info("Default config saved in" + args.output)
            else:
                yaml_string = yaml.dump(COMPLETE_CONFIG_TOPO, sort_keys=False, allow_unicode=True)
                logging.info("\n" + yaml_string)
        elif args.config:
            logger.info("Running topo workflow")
            workflow = Topo(args.config, args.output)
            workflow.run()

    elif args.command == "accuracy":
        if args.template_config:
            if args.output:
                with open(args.output, "w") as yaml_file:
                    yaml.dump(COMPLETE_CONFIG_ACCURACY, yaml_file, sort_keys=False, allow_unicode=True)
                logging.info("Default config saved in" + args.output)
            else:
                yaml_string = yaml.dump(COMPLETE_CONFIG_ACCURACY, sort_keys=False, allow_unicode=True)
                logging.info("\n" + yaml_string)
        elif args.config:
            logger.info("Running accuracy workflow")
            workflow = Accuracy(args.config, args.output)  # type: ignore
            workflow.run()

    else:
        raise ValueError(f"{args.command} doesn't exist, valid command are 'accuracy', 'topo'")

    if args.config:
        if not _has_weasyprint:
            msg = "Optional dependency 'weasyprint' required. " "Install it directly or through: pip install xdem[opt]."
            raise ImportError(msg)

        logger.info("Generating HTML and PDF report")
        HTML(workflow.outputs_folder / "report.html").write_pdf(workflow.outputs_folder / "report.pdf")

    logger.info("End of execution")


if __name__ == "__main__":
    main()
