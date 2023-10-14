from __future__ import annotations

import argparse

import yaml  # type: ignore


def environment_yml_nopy(fn_env: str, py_version: str, add_deps: list[str] = None) -> None:
    """
    Generate temporary environment-py3.XX.yml files forcing python versions for setup of continuous integration.

    :param fn_env: Filename path to environment.yml
    :param py_version: Python version to force.
    :param add_deps: Additional dependencies to solve for directly (for instance graphviz fails with mamba update).
    """

    # Load the yml as dictionary
    yaml_env = yaml.safe_load(open(fn_env))
    conda_dep_env = list(yaml_env["dependencies"])

    # Force python version
    conda_dep_env_forced_py = ["python=" + py_version if "python" in dep else dep for dep in conda_dep_env]

    # Optionally, add other dependencies
    if add_deps is not None:
        conda_dep_env_forced_py.extend(add_deps)

    # Copy back to new yaml dict
    yaml_out = yaml_env.copy()
    yaml_out["dependencies"] = conda_dep_env_forced_py

    with open("environment-ci-py" + py_version + ".yml", "w") as outfile:
        yaml.dump(yaml_out, outfile, default_flow_style=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate environment files for CI with fixed python versions.")
    parser.add_argument("fn_env", metavar="fn_env", type=str, help="Path to the generic environment file.")
    parser.add_argument(
        "--pyv",
        dest="py_version",
        default="3.9",
        type=str,
        help="List of Python versions to force.",
    )
    parser.add_argument(
        "--add",
        dest="add_deps",
        default=None,
        type=str,
        help="List of dependencies to add.",
    )
    args = parser.parse_args()
    environment_yml_nopy(fn_env=args.fn_env, py_version=args.py_version, add_deps=args.add_deps.split(","))
