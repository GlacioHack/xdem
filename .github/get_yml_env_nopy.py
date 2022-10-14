import argparse

import yaml  # type: ignore


def environment_yml_nopy(fn_env: str, print_dep: str = "both") -> None:
    """
    List dependencies in environment.yml without python version for setup of continuous integration.

    :param fn_env: Filename path to environment.yml
    :param print_dep: Whether to print conda differences "conda", pip differences "pip" or both.
    """

    # Load the yml as dictionary
    yaml_env = yaml.safe_load(open(fn_env))
    conda_dep_env = list(yaml_env["dependencies"])

    if isinstance(conda_dep_env[-1], dict):
        pip_dep_env = list(conda_dep_env.pop()["pip"])
    else:
        pip_dep_env = ["None"]

    conda_dep_env_without_python = [dep for dep in conda_dep_env if "python" not in dep]

    # Join the lists
    joined_list_conda_dep = " ".join(conda_dep_env_without_python)
    joined_list_pip_dep = " ".join(pip_dep_env)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Get environment list without python version.")
    parser.add_argument("fn_env", metavar="fn_env", type=str, help="Path to the environment file.")
    parser.add_argument(
        "--p",
        dest="print_dep",
        default="both",
        type=str,
        help="Whether to print conda dependencies, pip ones, or both.",
    )

    args = parser.parse_args()
    environment_yml_nopy(fn_env=args.fn_env, print_dep=args.print_dep)
