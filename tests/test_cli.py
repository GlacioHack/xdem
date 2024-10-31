"""Function to test the CLI"""

import os
import subprocess

import rasterio

import xdem
from xdem import dem_coregistration


class TestCLI:
    # Define paths to the DEM files using xDEM examples
    ref_dem_path = xdem.examples.get_path("longyearbyen_ref_dem")
    tba_dem_path = xdem.examples.get_path("longyearbyen_tba_dem")
    aligned_dem_path = "aligned_dem.tiff"
    inlier_mask_path = "inlier_mask.tiff"

    def test_xdem_cli_coreg(self) -> None:
        try:
            # Run the xDEM CLI command with the reference and secondary DEM files
            result = subprocess.run(
                ["xdem", "coregister", self.ref_dem_path, self.tba_dem_path],
                capture_output=True,
                text=True,
            )

            # Assert ClI ran successfully
            assert result.returncode == 0

            # Verify the existence of the output files
            assert os.path.exists(self.aligned_dem_path), f"Aligned DEM not found: {self.aligned_dem_path}"
            assert os.path.exists(self.inlier_mask_path), f"Inlier mask not found: {self.inlier_mask_path}"

            # Retrieve ground truth
            true_coreg_dem, coreg_method, out_stats, true_inlier_mask = dem_coregistration(
                xdem.DEM(self.tba_dem_path), xdem.DEM(self.ref_dem_path), self.aligned_dem_path
            )

            # Load elements processed by the xDEM CLI command
            aligned_dem = xdem.DEM(self.aligned_dem_path)
            with rasterio.open(self.inlier_mask_path) as src:
                inlier_mask = src.read(1)

            # Verify match with ground truth
            assert aligned_dem == true_coreg_dem, "Aligned DEM does not match the ground truth."
            assert inlier_mask.all() == true_inlier_mask.all(), "Inlier mask does not match the ground truth."

            # Erase files
            os.remove(self.aligned_dem_path)
            os.remove(self.inlier_mask_path)

        except FileNotFoundError as e:
            # In case 'xdem' is not found
            raise AssertionError(f"CLI command 'xdem' not found : {e}")

        except Exception as e:
            # Any other errors during subprocess run
            raise AssertionError(f"An error occurred while running the CLI: {e}")
