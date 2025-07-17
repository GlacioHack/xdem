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
import logging

from weasyprint import HTML

from xdem.workflows import Compare, TopoSummary, Uncertainty


def main() -> None:
    """
    Main function for the CLI
    """

    parser = argparse.ArgumentParser(prog="xdem", description="CLI tool to process DEM workflows")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available workflows as subcommand")

    # Subcommand: info
    qualify_parser = subparsers.add_parser(
        "topo-summary",
        help="Run DEM qualification workflow",
        description="Run a DEM information workflow using a YAML configuration file.",
        epilog="Example: xdem info config.yaml",
    )
    qualify_parser.add_argument(
        "config",
        help="Path to YAML configuration file",
    )

    # Subcommand: compare
    coreg_parser = subparsers.add_parser(
        "compare",
        help="Run DEM comparison workflow",
        description="Run a DEM comparison workflow using a YAML configuration file.",
        epilog="Example: xdem compare config.yaml",
    )
    coreg_parser.add_argument("config", help="Path to YAML configuration file")

    # Subcommand: uncertainty
    coreg_parser = subparsers.add_parser(
        "uncertainty",
        help="Run DEM uncertainty workflow",
        description="Run a DEM uncertainty workflow using a YAML configuration file.",
        epilog="Example: xdem uncertainty config.yaml",
    )
    coreg_parser.add_argument("config", help="Path to YAML configuration file")

    args = parser.parse_args()

    # Instance logger
    log_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logging.basicConfig(level=log_level, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    # fontTools creates noisy logs
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    logging.getLogger("fontTools").propagate = False

    if args.command == "topo-summary":
        logger.info("Running DEM information workflow")
        workflow = TopoSummary(args.config)
        workflow.run()

    elif args.command == "compare":
        logger.info("Running DEM comparison workflow")
        workflow = Compare(args.config)  # type: ignore
        workflow.run()

    elif args.command == "uncertainty":
        logger.info("Running DEM comparison workflow")
        workflow = Uncertainty(args.config)  # type: ignore
        workflow.run()

    else:
        raise ValueError(f"{args.command} doesn't exist, valid command are 'compare', 'info' or 'uncertainty")

    logger.info("Generate report")
    HTML(workflow.outputs_folder / "report.html").write_pdf(workflow.outputs_folder / "report.pdf")

    logger.info("End of execution")


if __name__ == "__main__":
    main()
