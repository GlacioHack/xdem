"""Test the xdem.misc functions."""

from __future__ import annotations

import os
import re

import pytest
import yaml  # type: ignore
from packaging.version import Version

import xdem
import xdem.misc


class TestMisc:
    def test_environment_files(self) -> None:
        """Check that environment yml files are properly written: all dependencies of env are also in dev-env"""

        fn_env = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "environment.yml"))
        fn_devenv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dev-environment.yml"))

        # Load the yml as dictionaries
        yaml_env = yaml.safe_load(open(fn_env))
        yaml_devenv = yaml.safe_load(open(fn_devenv))

        # Extract the dependencies values
        conda_dep_env = yaml_env["dependencies"]
        conda_dep_devenv = yaml_devenv["dependencies"]

        # Check if there is any pip dependency, if yes pop it from the end of the list
        if isinstance(conda_dep_devenv[-1], dict):
            pip_dep_devenv = conda_dep_devenv.pop()

            # Check if there is a pip dependency in the normal env as well, if yes pop it also
            if isinstance(conda_dep_env[-1], dict):
                pip_dep_env = conda_dep_env.pop()

                # The diff below computes the dependencies that are in env but not in dev-env
                # It should be empty, otherwise we raise an error
                diff_pip_check = list(set(pip_dep_env) - set(pip_dep_devenv))
                assert len(diff_pip_check) == 0

        # We do the same for the conda dependency, first a sanity check that everything that is in env is also in dev-ev
        diff_conda_check = list(set(conda_dep_env) - set(conda_dep_devenv))
        assert len(diff_conda_check) == 0

    @pytest.mark.parametrize("deprecation_increment", [-1, 0, 1, None])  # type: ignore
    @pytest.mark.parametrize("details", [None, "It was completely useless!", "dunnowhy"])  # type: ignore
    def test_deprecate(self, deprecation_increment: int | None, details: str | None) -> None:
        """
        Test the deprecation warnings/errors.

        If the removal_version is larger than the current, it should warn.
        If the removal_version is smaller or equal, it should raise an error.

        :param deprecation_increment: The version number relative to the current version.
        :param details: An optional explanation for the description.
        """

        current_version = Version(Version(xdem.__version__).base_version)

        # Set the removal version to be the current version plus the increment (e.g. 0.0.5 + 1 -> 0.0.6)
        if deprecation_increment is not None:
            # If the micro version is already 0 and it is a decrement, decrement minor version instead
            if current_version.micro == 0 and deprecation_increment == -1:
                removal_version_tuple = (current_version.major, current_version.minor + deprecation_increment, 0)
            # Otherwise, increment micro version
            else:
                removal_version_tuple = (
                    current_version.major,
                    current_version.minor,
                    current_version.micro + deprecation_increment,
                )

            # Convert to version
            removal_version = Version(".".join([str(v) for v in removal_version_tuple]))
        else:
            removal_version = None

        # Define a function with no use that is marked as deprecated.
        @xdem.misc.deprecate(removal_version, details=details)  # type: ignore
        def useless_func() -> int:
            return 1

        # Example of why Version needs to be used below
        assert not "0.0.10" > "0.0.8"
        assert Version("0.0.10") > Version("0.0.8")

        # If True, a warning is expected. If False, a ValueError is expected.
        should_warn = removal_version is None or removal_version > current_version

        # Add the expected text depending on the parametrization.
        text = (
            "Call to deprecated function 'useless_func'."
            if should_warn
            else f"Deprecated function 'useless_func' was removed in {removal_version}."
        )

        if details is not None:
            text += " " + details.strip().capitalize()

            if not any(text.endswith(c) for c in ".!?"):
                text += "."

        if should_warn and removal_version is not None:
            text += f" This functionality will be removed in version {removal_version}."
        elif not should_warn:
            text += f" Current version: {current_version}."

        # Expect either a warning or an exception with the right text.
        if should_warn:
            with pytest.warns(DeprecationWarning, match="^" + text + "$"):
                useless_func()
        else:
            with pytest.raises(ValueError, match=re.escape(text)):
                useless_func()

    def test_diff_environment_yml(self, capsys) -> None:  # type: ignore

        # Test with synthetic environment
        env = {"dependencies": ["python==3.9", "numpy", "pandas"]}
        devenv = {"dependencies": ["python==3.9", "numpy", "pandas", "otherdep"]}

        # This should print the difference between the two
        xdem.misc.diff_environment_yml(env, devenv, input_dict=True, print_dep="conda")

        # Capture the stdout and check it is indeed the right diff
        captured = capsys.readouterr().out
        assert captured == "otherdep\n"

        # This should print the difference including pip
        xdem.misc.diff_environment_yml(env, devenv, input_dict=True, print_dep="both")

        captured = capsys.readouterr().out
        assert captured == "otherdep\nNone\n"

        env2 = {"dependencies": ["python==3.9", "numpy", "pandas"]}
        devenv2 = {"dependencies": ["python==3.9", "numpy", "pandas", "otherdep", {"pip": ["geoutils", "-e ./"]}]}

        # The diff function should not account for -e ./ that is the local install for developers
        xdem.misc.diff_environment_yml(env2, devenv2, input_dict=True, print_dep="both")
        captured = capsys.readouterr().out

        assert captured == "otherdep\ngeoutils\n"

        # This should print only pip
        xdem.misc.diff_environment_yml(env2, devenv2, input_dict=True, print_dep="pip")
        captured = capsys.readouterr().out

        assert captured == "geoutils\n"

        # This should raise an error because print_dep is not well defined
        with pytest.raises(ValueError, match='The argument "print_dep" can only be "conda", "pip" or "both".'):
            xdem.misc.diff_environment_yml(env2, devenv2, input_dict=True, print_dep="lol")

        # When the dependencies are not defined in dev-env but in env, it should raise an error
        # For normal dependencies
        env3 = {"dependencies": ["python==3.9", "numpy", "pandas", "lol"]}
        devenv3 = {"dependencies": ["python==3.9", "numpy", "pandas", "otherdep", {"pip": ["geoutils"]}]}
        with pytest.raises(ValueError, match="The following dependencies are listed in env but not dev-env: lol"):
            xdem.misc.diff_environment_yml(env3, devenv3, input_dict=True, print_dep="pip")

        # For pip dependencies
        env4 = {"dependencies": ["python==3.9", "numpy", "pandas", {"pip": ["lol"]}]}
        devenv4 = {"dependencies": ["python==3.9", "numpy", "pandas", "otherdep", {"pip": ["geoutils"]}]}
        with pytest.raises(ValueError, match="The following pip dependencies are listed in env but not dev-env: lol"):
            xdem.misc.diff_environment_yml(env4, devenv4, input_dict=True, print_dep="pip")
