import os
import shutil
import subprocess
import warnings


class TestDocs:
    current_dir = os.getcwd()

    def test_example_code(self):
        """Try running each python script in the docs/source/code\
                directory and check that it doesn't raise an error."""
        os.chdir(os.path.join(self.current_dir, "docs/source"))

        for filename in os.listdir("code/"):
            if not filename.endswith(".py"):
                continue
            print(f"Running {os.path.join(os.getcwd(), 'code/', filename)}")
            subprocess.run(["python", f"code/{filename}"], check=True)

        os.chdir(self.current_dir)

    def test_build(self):
        """Try building the docs and see if it works."""
        # Change into the docs directory.
        os.chdir(os.path.join(self.current_dir, "docs"))

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

        os.chdir(self.current_dir)
