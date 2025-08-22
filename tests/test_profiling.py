"""Test the xdem.profiling functions."""

from __future__ import annotations

import glob
import os.path as op

import pandas as pd
import pytest

import xdem
from xdem.profiler import Profiler


class TestProfiling:

    # Test that there's no crash when giving profiling configuration
    @pytest.mark.parametrize(
        "profiling_configuration", [(False, False), (True, False), (False, True), (True, True)]
    )  # type: ignore
    @pytest.mark.parametrize("profiling_function", ["load", "attribute", "coreg"])  # type: ignore
    def test_profiling_configuration(self, profiling_configuration, profiling_function, tmp_path) -> None:
        """
        Test the all combinaisons of profiling with three examples of profiled functions.
        """
        s_gr = profiling_configuration[0]
        s_rd = profiling_configuration[1]

        Profiler.enable(save_graphs=s_gr, save_raw_data=s_rd)

        dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
        if profiling_function == "coreg":
            xdem.coreg.VerticalShift().fit_and_apply(dem, dem)
        if profiling_function == "attribute":
            xdem.terrain.slope(dem=dem.data, resolution=dem.res[0])
        Profiler.generate_summary(tmp_path)

        if s_rd or s_gr:
            assert op.isdir(op.join(tmp_path))

            if s_rd:
                assert op.isfile(op.join(tmp_path, "raw_data.pickle"))
                df = pd.read_pickle(op.join(tmp_path, "raw_data.pickle"))
                if profiling_function == "coreg":
                    assert len(df) == 3
                elif profiling_function == "attribute":
                    assert len(df) == 2
                else:
                    assert len(df) == 1

            else:
                assert not op.isfile(op.join(tmp_path, "raw_data.pickle"))

            if s_gr:
                assert op.isfile(op.join(tmp_path, "time_graph.html"))
                assert op.isfile(op.join(tmp_path, "memory_dem.__init__.html"))
                if profiling_function == "coreg":
                    assert op.isfile(op.join(tmp_path, "memory_coreg.base.fit_and_apply.html"))

                if profiling_function == "attribute":
                    assert op.isfile(op.join(tmp_path, "memory_terrain.slope.html"))
            else:
                assert not len(glob.glob(op.join(tmp_path, "*.html")))

            df = Profiler.get_profiling_info()
            if profiling_function == "coreg":
                assert len(df) == 3
            elif profiling_function == "attribute":
                assert len(df) == 2
            else:
                assert len(df) == 1

            Profiler.reset()
            assert len(Profiler.get_profiling_info()) == 0

        else:
            assert not len(glob.glob(op.join(tmp_path, "*")))
