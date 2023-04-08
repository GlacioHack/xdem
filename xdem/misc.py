"""Small functions for testing, examples, and other miscellaneous uses."""
from __future__ import annotations

import copy
import functools
import warnings
from typing import Any, Callable

from packaging.version import Version

try:
    import yaml  # type: ignore

    _has_yaml = True
except ImportError:
    _has_yaml = False

try:
    import cv2

    _has_cv2 = True
except ImportError:
    _has_cv2 = False

import numpy as np

import xdem.version
from xdem._typing import NDArrayf


def generate_random_field(shape: tuple[int, int], corr_size: int) -> NDArrayf:
    """
    Generate a semi-random gaussian field (to simulate a DEM or DEM error)

    :param shape: The output shape of the field.
    :param corr_size: The correlation size of the field.

    :examples:
        >>> np.random.seed(1)
        >>> generate_random_field((4, 5), corr_size=2).round(2)
        array([[0.47, 0.5 , 0.56, 0.63, 0.65],
               [0.49, 0.51, 0.56, 0.62, 0.64],
               [0.56, 0.56, 0.57, 0.59, 0.59],
               [0.57, 0.57, 0.57, 0.58, 0.58]])

    :returns: A numpy array of semi-random values from 0 to 1
    """

    if not _has_cv2:
        raise ValueError("Optional dependency needed. Install 'opencv'")

    field = cv2.resize(
        cv2.GaussianBlur(
            np.repeat(
                np.repeat(
                    np.random.randint(0, 255, (shape[0] // corr_size, shape[1] // corr_size), dtype="uint8"),
                    corr_size,
                    axis=0,
                ),
                corr_size,
                axis=1,
            ),
            ksize=(2 * corr_size + 1, 2 * corr_size + 1),
            sigmaX=corr_size,
        )
        / 255,
        dsize=(shape[1], shape[0]),
    )
    return field


def deprecate(removal_version: str = None, details: str = None) -> Callable[[Any], Any]:
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
            # True if it should warn, False if it should raise an error
            should_warn = removal_version is None or Version(removal_version) > Version(xdem.version.version)

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
                text += f" Current version: {xdem.version.version}."

            if should_warn:
                warnings.warn(text, category=DeprecationWarning, stacklevel=2)
            else:
                raise ValueError(text)

            return func(*args, **kwargs)

        return new_func

    return deprecator_func


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
        raise ValueError("Test dependency needed. Install 'pyyaml'")

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
