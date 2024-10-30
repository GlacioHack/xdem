"""Function to test the CLI"""

import subprocess

import xdem


class TestCLI:
    # Define paths to the DEM files using xDEM examples
    ref_dem_path = xdem.examples.get_path("longyearbyen_ref_dem")
    tba_dem_path = xdem.examples.get_path("longyearbyen_tba_dem")

    def test_xdem_cli(self) -> None:
        try:
            # Run the xDEM CLI command with the reference and secondary DEM files
            result = subprocess.run(
                ["xdem", self.ref_dem_path, self.tba_dem_path],
                capture_output=True,
                text=True,
            )
            assert "hello world" in result.stdout
            assert result.returncode == 0

        except FileNotFoundError as e:
            # In case 'xdem' is not found
            raise AssertionError(f"CLI command 'xdem' not found : {e}")

        except Exception as e:
            # Any other errors during subprocess run
            raise AssertionError(f"An error occurred while running the CLI: {e}")
