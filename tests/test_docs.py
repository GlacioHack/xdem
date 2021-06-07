import os
import concurrent.futures
import shutil
import subprocess
import sys
import warnings

import sphinx.cmd.build


class TestDocs:
    docs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../", "docs/")
    n_threads = os.getenv("N_CPUS")

    def test_example_code(self):
        """Try running each python script in the docs/source/code\
                directory and check that it doesn't raise an error."""
        current_dir = os.getcwd()
        os.chdir(os.path.join(self.docs_dir, "source"))

        def run_code(filename: str) -> None:
            """Run a python script in one thread."""
            with open(filename) as infile:
                # Run everything except plt.show() calls.
                exec(infile.read().replace("plt.show()", ""))

        filenames = [os.path.join("code", filename) for filename in os.listdir("code/") if filename.endswith(".py")]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=int(self.n_threads) if self.n_threads is not None else None
        ) as executor:
            list(executor.map(run_code, filenames))

        os.chdir(current_dir)

    def test_build(self):
        """Try building the docs and see if it works."""
        # Remove the build directory if it exists.
        if os.path.isdir(os.path.join(self.docs_dir, "build/")):
            shutil.rmtree(os.path.join(self.docs_dir, "build/"))

        sphinx.cmd.build.main(
            [
                "-j",
                str(self.n_threads or "auto"),
                os.path.join(self.docs_dir, "source/"),
                os.path.join(self.docs_dir, "build/html"),
            ]
        )
