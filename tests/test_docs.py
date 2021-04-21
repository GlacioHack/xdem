import os
import shutil
import subprocess
import sys
import warnings


class TestDocs:
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "docs/")

    def test_example_code(self):
        """Try running each python script in the docs/source/code\
                directory and check that it doesn't raise an error."""
        current_dir = os.getcwd()
        os.chdir(os.path.join(self.docs_dir, "source"))

        # Copy the environment and unset the DISPLAY variable to hide matplotlib plots.
        env = os.environ.copy()
        env["DISPLAY"] = ""

        for filename in os.listdir("code/"):
            if not filename.endswith(".py"):
                continue
            print(f"Running {os.path.join(os.getcwd(), 'code/', filename)}")
            subprocess.run([sys.executable, f"code/{filename}"], check=True, env=env)

        os.chdir(current_dir)

    def test_build(self):
        """Try building the docs and see if it works."""
        current_dir = os.getcwd()
        # Change into the docs directory.
        os.chdir(self.docs_dir)

        # Remove the build directory if it exists.
        if os.path.isdir("build/"):
            shutil.rmtree("build/")

        # Run the makefile
        build_commands = ["make", "html"]
        result = subprocess.run(build_commands, check=True, stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE, encoding="utf-8")

        # Raise an error if the string "error" is in the stderr.
        if "error" in str(result.stderr).lower():
            raise RuntimeError(result.stderr)

        # If "error" is not in the stderr string but it exists, show it as a warning.
        if len(result.stderr) > 0:
            warnings.warn(result.stderr)

        os.chdir(current_dir)
