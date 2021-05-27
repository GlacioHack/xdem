import os
import shutil
import subprocess
import sys
import warnings

import sphinx.cmd.build

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
        # Remove the build directory if it exists.
        if os.path.isdir(os.path.join(self.docs_dir, "build/")):
            shutil.rmtree(os.path.join(self.docs_dir, "build/"))

        sphinx.cmd.build.main([
            os.path.join(self.docs_dir, "source/"),
            os.path.join(self.docs_dir, "build/")
        ])

