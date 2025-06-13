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

"""Small functions for testing, examples, and other miscellaneous uses."""
from __future__ import annotations

import copy
import functools
import sys
import warnings
from typing import Any, Callable

from packaging.version import Version

try:
    import yaml  # type: ignore

    _has_yaml = True
except ImportError:
    _has_yaml = False

import xdem


def deprecate(removal_version: Version = None, details: str = None) -> Callable[[Any], Any]:
    """
    Trigger a DeprecationWarning for the decorated function.

    :param func: The function to be deprecated.
    :param removal_version: Optional. The version at which this will be removed.
                            If this version is reached, a ValueError is raised.
    :param details: Optional. A description for why the function was deprecated.

    :triggers DeprecationWarning: For any call to the function.

    :raises ValueError: If 'removal_version' was given and the current version is equal or higher.

    :returns: The decorator to decorate the function.
    """

    def deprecator_func(func: Callable[[Any], Any]) -> Callable[[Any], Any]:
        @functools.wraps(func)  # type: ignore
        def new_func(*args: Any, **kwargs: Any) -> Any:

            # Get current base version (without dev changes)
            current_version = Version(Version(xdem.__version__).base_version)

            # True if it should warn, False if it should raise an error
            should_warn = removal_version is None or removal_version > current_version

            # Add text depending on the given arguments and 'should_warn'.
            text = (
                f"Call to deprecated function '{func.__name__}'."
                if should_warn
                else f"Deprecated function '{func.__name__}' was removed in {removal_version}."
            )

            # Add the details explanation if it was given, and make sure the sentence is ended.
            if details is not None:
                details_frm = details.strip()
                if details_frm[0].islower():
                    details_frm = details_frm[0].upper() + details_frm[1:]

                text += " " + details_frm

                if not any(text.endswith(c) for c in ".!?"):
                    text += "."

            if should_warn and removal_version is not None:
                text += f" This functionality will be removed in version {removal_version}."
            elif not should_warn:
                text += f" Current version: {xdem.__version__}."

            if should_warn:
                warnings.warn(text, category=DeprecationWarning, stacklevel=2)
            else:
                raise ValueError(text)

            return func(*args, **kwargs)

        return new_func

    return deprecator_func


def copy_doc(
    module_to_copy: object,
    remove_dem_res_params: bool = False,
) -> Callable:  # type: ignore
    """
    A decorator to copy docstring from a function to another one while replacing the docstring.
    Used for copying xdem.terrain documentation to xdem.DEM.

    :param module_to_copy: Name of module to copy the function from
    :param remove_dem_res_params: To remove the parameters dem: and resolution: in the terrain docstring,
        as they are useless for the DEM class.
    """

    def decorator(decorated: Callable) -> Callable:  # type: ignore
        # Get name of decorated object
        # If object is a property, get name through fget
        try:
            decorated_name = decorated.fget.__name__
        # Otherwise, directly with the name attribute
        except AttributeError:
            decorated_name = decorated.__name__

        # Get parent doc
        other_doc = getattr(module_to_copy, decorated_name).__doc__

        # Replace argument description of dem and resolution (not used in the DEM class, only in terrain)
        if remove_dem_res_params:

            # Get Python version (spaces of docstring are not handled the same way after Python 3.13)
            pyv = Version(".".join(str(getattr(sys.version_info, v)) for v in ["major", "minor", "micro"]))

            # Find and remove them if they exist
            if ":param dem:" in other_doc:
                if pyv >= Version("3.13"):
                    dem_section = "\n:param dem:" + other_doc.split("\n:param dem:")[1].split("\n")[0]
                else:
                    dem_section = "\n    :param dem:" + other_doc.split("\n    :param dem:")[1].split("\n")[0]

                other_doc = other_doc.replace(dem_section, "")
            if ":param resolution:" in other_doc:
                if pyv >= Version("3.13"):
                    resolution_section = (
                        "\n:param resolution:" + other_doc.split("\n:param resolution:")[1].split("\n")[0]
                    )
                else:
                    resolution_section = (
                        "\n    :param resolution:" + other_doc.split("\n    :param resolution:")[1].split("\n")[0]
                    )

                other_doc = other_doc.replace(resolution_section, "")

        # Remove docstring examples
        other_doc = other_doc.split(":examples:")[0]

        decorated.__doc__ = other_doc

        return decorated

    return decorator


def diff_environment_yml(
    fn_env: str | dict[str, Any], fn_devenv: str | dict[str, Any], print_dep: str = "both", input_dict: bool = False
) -> None:
    """
    Compute the difference between environment.yml and dev-environment.yml for setup of continuous integration,
    while checking that all the dependencies listed in environment.yml are also in dev-environment.yml
    :param fn_env: Filename path to environment.yml
    :param fn_devenv: Filename path to dev-environment.yml
    :param print_dep: Whether to print conda differences "conda", pip differences "pip" or both.
    :param input_dict: Whether to consider the input as a dict (for testing purposes).
    """

    if not _has_yaml:
        raise ValueError("Test dependency needed. Install 'pyyaml'.")

    if not input_dict:
        # Load the yml as dictionaries
        yaml_env = yaml.safe_load(open(fn_env))  # type: ignore
        yaml_devenv = yaml.safe_load(open(fn_devenv))  # type: ignore
    else:
        # We need a copy as we'll pop things out and don't want to affect input
        # dict.copy() is shallow and does not work with embedded list in dicts (as is the case here)
        yaml_env = copy.deepcopy(fn_env)
        yaml_devenv = copy.deepcopy(fn_devenv)

    # Extract the dependencies values
    conda_dep_env = yaml_env["dependencies"]
    conda_dep_devenv = yaml_devenv["dependencies"]

    # Check if there is any pip dependency, if yes pop it from the end of the list
    if isinstance(conda_dep_devenv[-1], dict):
        pip_dep_devenv = conda_dep_devenv.pop()["pip"]

        # Remove the package's self install for devs via pip, if it exists
        if "-e ./" in pip_dep_devenv:
            pip_dep_devenv.remove("-e ./")

        # Check if there is a pip dependency in the normal env as well, if yes pop it also
        if isinstance(conda_dep_env[-1], dict):
            pip_dep_env = conda_dep_env.pop()["pip"]

            # The diff below computes the dependencies that are in env but not in dev-env
            # It should be empty, otherwise we raise an error
            diff_pip_check = list(set(pip_dep_env) - set(pip_dep_devenv))
            if len(diff_pip_check) != 0:
                raise ValueError(
                    "The following pip dependencies are listed in env but not dev-env: " + ",".join(diff_pip_check)
                )

            # The diff below computes the dependencies that are in dev-env but not in env, to add during CI
            diff_pip_dep = list(set(pip_dep_devenv) - set(pip_dep_env))

        # If there is no pip dependency in env, all the ones of dev-env need to be added during CI
        else:
            diff_pip_dep = pip_dep_devenv

    # If there is no pip dependency, we ignore this step
    else:
        diff_pip_dep = []

    # If the diff is empty for pip, return a string "None" to read easily in bash
    if len(diff_pip_dep) == 0:
        diff_pip_dep = ["None"]

    # We do the same for the conda dependency, first a sanity check that everything that is in env is also in dev-ev
    diff_conda_check = list(set(conda_dep_env) - set(conda_dep_devenv))
    if len(diff_conda_check) != 0:
        raise ValueError("The following dependencies are listed in env but not dev-env: " + ",".join(diff_conda_check))

    # Then the difference to add during CI
    diff_conda_dep = list(set(conda_dep_devenv) - set(conda_dep_env))

    # Join the lists
    joined_list_conda_dep = " ".join(diff_conda_dep)
    joined_list_pip_dep = " ".join(diff_pip_dep)

    # Print to be captured in bash
    if print_dep == "both":
        print(joined_list_conda_dep)
        print(joined_list_pip_dep)
    elif print_dep == "conda":
        print(joined_list_conda_dep)
    elif print_dep == "pip":
        print(joined_list_pip_dep)
    else:
        raise ValueError('The argument "print_dep" can only be "conda", "pip" or "both".')
