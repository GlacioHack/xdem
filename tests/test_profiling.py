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

        # if profiling is activate
        if s_rd or s_gr:

            # in each case, output dir exist
            assert op.isdir(tmp_path)

            # if save_raw_data:
            if s_rd:
                # check pickle
                assert op.isfile(op.join(tmp_path, "raw_data.pickle"))

                # check data in pickle
                df = pd.read_pickle(op.join(tmp_path, "raw_data.pickle"))
                if profiling_function == "coreg":
                    assert len(df) == 3
                elif profiling_function == "attribute":
                    assert len(df) == 2
                else:
                    assert len(df) == 1

            else:
                assert not op.isfile(op.join(tmp_path, "raw_data.pickle"))

            # if save_graphs:
            if s_gr:
                # check if all output graphs (time_graph + mem graph/profiled function called)
                # are generated
                assert op.isfile(op.join(tmp_path, "time_graph.html"))
                assert op.isfile(op.join(tmp_path, "memory_dem.__init__.html"))
                if profiling_function == "coreg":
                    assert op.isfile(op.join(tmp_path, "memory_coreg.base.fit_and_apply.html"))
                if profiling_function == "attribute":
                    assert op.isfile(op.join(tmp_path, "memory_terrain.slope.html"))
            else:
                assert not len(glob.glob(op.join(tmp_path, "*.html")))

        else:
            # if profiling is deactivated : nothing generated in output dir
            assert not len(glob.glob(op.join(tmp_path, "*")))

    def test_profiling_functions_management(self) -> None:
        """
        Test the management of profiling functions information.
        """
        Profiler.enable(save_graphs=False, save_raw_data=True)

        assert len(Profiler.get_profiling_info()) == 0

        dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
        xdem.coreg.VerticalShift().fit_and_apply(dem, dem)

        assert len(Profiler.get_profiling_info()) == 3
        assert len(Profiler.get_profiling_info(function_name="coreg.base.fit_and_apply")) == 1
        assert len(Profiler.get_profiling_info(function_name="dem.__init__")) == 2
        assert len(Profiler.get_profiling_info(function_name="no_name")) == 0

        Profiler.reset()
        assert len(Profiler.get_profiling_info()) == 0

    def test_selections_functions(self) -> None:
        """
        Test the selection of functions to profile (all or by theirs names).
        """
        Profiler.enable(save_graphs=False, save_raw_data=True)
        Profiler.selection_functions(["terrain.slope", "coreg.base.fit_and_apply"])
        dem = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
        xdem.terrain.slope(dem=dem.data, resolution=dem.res[0])
        xdem.terrain.aspect(dem=dem.data)
        xdem.coreg.VerticalShift().fit_and_apply(dem, dem)
        assert len(Profiler.get_profiling_info()) == 2

        Profiler.selection_functions(["terrain.aspect"])
        xdem.terrain.aspect(dem=dem.data)
        xdem.terrain.slope(dem=dem.data, resolution=dem.res[0])
        assert len(Profiler.get_profiling_info()) == 3

        Profiler.reset_selection_functions()
        xdem.coreg.VerticalShift().fit_and_apply(dem, dem)
        assert len(Profiler.get_profiling_info()) == 5
