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

"""
VolChange class from workflows.
"""
from typing import Any

from xdem.workflows.schemas import VOLCHANGE_SCHEMA
from xdem.workflows.workflows import Workflows


class VolChange(Workflows):
    """
    VolChange class from workflows.
    """

    def __init__(self, config_dem: str | dict[str, Any]) -> None:
        """
        Initialize VolChange class
        :param config_dem: Path to a user configuration file
        """
        self.schema = VOLCHANGE_SCHEMA

        super().__init__(config_dem)

    def run(self) -> None:
        """
        Run function for the coregistration workflow
        :return: None
        """

        print("WIP : Hello world ! :)")

    def create_html(self, list_dict: list[tuple[str, dict[str, Any]]]) -> None:
        """
        Create HTML page from png files and table
        :param list_dict: list containing tuples of title and various dictionaries
        :return: None
        """
