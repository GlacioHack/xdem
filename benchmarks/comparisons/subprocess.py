"""Small helper function to execute all commands consistently."""

from __future__ import annotations

import os
import subprocess


def execute_command(implementation: str, command: list[str], output_file: str) -> None:
    """Run one implementation command after removing its previous output."""

    if os.path.isfile(output_file):
        os.remove(output_file)
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{implementation} command failed with status {completed.returncode}: {command}\n{completed.stderr}"
        )
