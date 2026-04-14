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

test for CLI class
"""

import logging
import os

# mypy: disable-error-code=no-untyped-def
from pathlib import Path

import pytest
import yaml  # type: ignore  # noqa

import xdem.cli as cli
from xdem.workflows.schemas import COMPLETE_CONFIG_ACCURACY, COMPLETE_CONFIG_TOPO

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


@pytest.mark.parametrize("help_arg", [[], ["-h"], ["--help"]])
def test_raises_help(capsys, help_arg):
    """Test help"""
    with pytest.raises(SystemExit):
        cli.main(help_arg)

    capsys_log = capsys.readouterr()
    capsys_log_out = capsys_log.out.split("\n")
    assert "usage: xdem" in capsys_log_out[0]
    assert capsys_log_out[3].startswith("CLI tool to run xDEM workflows")

    assert not capsys_log.err


@pytest.mark.parametrize("invalid_arg", ["arg", "-arg", "1"])
def test_invalid_parameters(capsys, invalid_arg):
    """Invalid argument"""
    with pytest.raises(SystemExit):
        cli.main([invalid_arg])

    capsys_log = capsys.readouterr()
    assert not capsys_log.out

    capsys_log_err = capsys_log.err.split("\n")
    assert capsys_log_err[0].startswith("usage: xdem")
    if invalid_arg.startswith("-"):
        assert capsys_log_err[2] == f"xdem: error: unrecognized arguments: {invalid_arg}"
    else:
        assert (
            capsys_log_err[2]
            == f"xdem: error: argument command: invalid choice: '{invalid_arg}' (choose from topo, accuracy)"
        )


@pytest.mark.parametrize("workflow", ["topo", "accuracy"])
def test_missing_param_after_workflow(capsys, workflow):
    """No config file"""
    with pytest.raises(SystemExit):
        cli.main([workflow])

    capsys_log = capsys.readouterr()
    assert not capsys_log.out

    capsys_log_err = capsys_log.err.split("\n")
    assert capsys_log_err[0].startswith("usage: xdem")
    assert capsys_log_err[2] == f"xdem {workflow}: error: one of the arguments --config --template-config is required"


def run_and_check_workflow(workflow, user_config, caplog, capsys, tmp_file):
    # Read working config
    yaml_str = yaml.dump(user_config, allow_unicode=True)
    tmp_file.write_text(yaml_str, encoding="utf-8")

    with caplog.at_level(logging.INFO):
        cli.main([workflow, "--config", str(tmp_file)])

    assert f"root:{workflow}.py" in caplog.text
    assert "End of execution" in caplog.text

    # assert not capsys.readouterr().err # Progress bar in err


def test_run_workflow_topo(get_topo_config_test, tmp_path, capsys, caplog):
    """Run Topo Workflow"""
    user_config = get_topo_config_test
    run_and_check_workflow("topo", user_config, caplog, capsys, Path(tmp_path / "temp_config.yaml"))


def test_run_workflow_accuracy(get_accuracy_config_test, tmp_path, capsys, caplog):
    """Run Accuracy Workflow"""
    user_config = get_accuracy_config_test
    run_and_check_workflow("accuracy", user_config, caplog, capsys, Path(tmp_path / "temp_config.yaml"))


@pytest.mark.parametrize("workflow", ["topo", "accuracy"])
def test_default_config(workflow, tmp_path, caplog):
    """Get default config (print + file)"""
    if workflow == "topo":
        COMPLETE_CONFIG = COMPLETE_CONFIG_TOPO
    else:
        COMPLETE_CONFIG = COMPLETE_CONFIG_ACCURACY

    # Print output
    with caplog.at_level(logging.INFO):
        cli.main([workflow, "--template-config"])
    yaml_part = "\n".join(caplog.text.split("\n")[1:-3])
    dict_from_cli = yaml.safe_load(yaml_part)
    assert dict_from_cli == COMPLETE_CONFIG

    # File output
    with caplog.at_level(logging.INFO):
        cli.main([workflow, "--template-config", str(Path(tmp_path / "temp_config.yaml"))])

    with open(Path(tmp_path / "temp_config.yaml")) as f:
        dict_from_yml = yaml.safe_load(f)
    assert dict_from_yml == COMPLETE_CONFIG


def test_errors_config_file(get_topo_config_test, tmp_path):
    """Config file errors"""
    user_config = get_topo_config_test
    yaml_str = yaml.dump(user_config, allow_unicode=True)
    tmp_file = Path(tmp_path / "temp_config.yaml")
    tmp_file.write_text(yaml_str, encoding="utf-8")

    with pytest.raises(ValueError, match="User configuration invalid"):
        cli.main(["accuracy", "--config", str(tmp_file)])

    file_path = Path(tmp_path / "filename")
    my_file = open(file_path, "w")
    my_file.write("data data")
    my_file.close()

    with pytest.raises(ValueError, match="Unsupported configuration file format"):
        cli.main(["accuracy", "--config", str(file_path)])

    os.rename(str(file_path), str(file_path.with_suffix(".txt")))
    with pytest.raises(ValueError, match="Unsupported configuration file format"):
        cli.main(["accuracy", "--config", str(file_path.with_suffix(".txt"))])

    with pytest.raises(FileNotFoundError):
        cli.main(["accuracy", "--config", str(Path(tmp_path / "path_that_dont_exist.yml"))])
