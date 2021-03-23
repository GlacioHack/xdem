.. _coregistration:

DEM Coregistration
==================
Coregistration of a DEM is performed when it needs to be compared to a reference, but the DEM does not align with the reference perfectly.
There are many reasons for why this might be, for example: poor georeferencing, unknown coordinate system transforms or vertical datums, and instrument- or processing-induced distortion.

A main principle of all coregistration approaches is the assumption that all or parts of the portrayed terrain are unchanged between the reference and the DEM to be aligned.
This *stable ground* can be extracted by masking out features that are assumed to be unstable.
Then, the DEM to be aligned is translated, rotated and/or bent to fit the stable surfaces of the reference DEM as well as possible.
In mountainous environments, unstable areas could be: glaciers, landslides, vegetation, dead-ice terrain and human settlements.
Unless the entire terrain is assumed to be stable, a mask layer is required.
``xdem`` supports either shapefiles or rasters to specify the masked (unstable) areas.

There are multiple methods for coregistration, and each have their own strengths and weaknesses.
Below is a summary of how each method works, and when it should (and should not) be used.

**Example data**

Examples are given using data close to Longyearbyen on Svalbard. These can be loaded as:


.. code-block:: python

        import geoutils as gu
        import xdem

        # Download the necessary data. This may take a few minutes.
        xdem.examples.download_longyearbyen_examples(overwrite=False)

        # Load a reference DEM from 2009
        reference_dem = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
        # Load a moderately well aligned DEM from 1990
        dem_to_be_aligned = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_tba_dem"])
        # Load glacier outlines from 1990. This will act as the unstable ground.
        glacier_outlines = gu.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])

.. contents:: Contents 
   :local:

Nuth and Kääb (2011)
^^^^^^^^^^^^^^^^^^^^
- **Performs:** translation and bias corrections.
- **Supports weights** (soon)
- **Recommended for:** Noisy data with low rotational differences.

The Nuth and Kääb (`2011 <https:https://doi.org/10.5194/tc-5-271-2011>`_) coregistration approach is named after the paper that first implemented it.
It estimates translation and bias corrections iteratively by solving a cosine equation to model the direction at which the DEM is most likely offset.
First, the DEMs are compared to get a dDEM, and slope/aspect maps are created from the reference DEM.
Together, these three products contain the information about in which direction the offset is.
A cosine function is solved using these products to find the most probable offset direction, and an appropriate horizontal shift is applied to fix it.
This is an iterative process, and cosine functions with suggested shifts are applied in a loop, continuously refining the total offset.
The loop is stopped either when the maximum iteration limit is reached, or when the :ref:`spatial_stats_nmad` between the two products stops improving significantly.

.. plot::

        import xdem
        import geoutils as gu
        import matplotlib.pyplot as plt
        dem_2009 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_ref_dem"])
        dem_1990 = xdem.DEM(xdem.examples.FILEPATHS["longyearbyen_tba_dem"])
        outlines_1990 = gu.Vector(xdem.examples.FILEPATHS["longyearbyen_glacier_outlines"])

        dem_coreg, error = xdem.coreg.coregister(dem_2009, dem_1990, method="nuth_kaab", mask=outlines_1990, max_iterations=5)

        ddem_pre = xdem.spatial_tools.subtract_rasters(dem_2009, dem_1990, resampling_method="nearest")
        ddem_post = xdem.spatial_tools.subtract_rasters(dem_2009, dem_coreg, resampling_method="nearest")

        nmad_pre = xdem.spatial_tools.nmad(ddem_pre.data.data)

        vlim = 20
        plt.figure(figsize=(8, 5))
        plt.subplot2grid((1, 15), (0, 0), colspan=7) 
        plt.title(f"Before coregistration. NMAD={nmad_pre:.1f} m")
        plt.imshow(ddem_pre.data.squeeze(), cmap="coolwarm_r", vmin=-vlim, vmax=vlim)
        plt.axis("off")
        plt.subplot2grid((1, 15), (0, 7), colspan=7) 
        plt.title(f"After coregistration. NMAD={error:.1f} m")
        img = plt.imshow(ddem_post.data.squeeze(), cmap="coolwarm_r", vmin=-vlim, vmax=vlim) 
        plt.axis("off")
        plt.subplot2grid((1, 15), (0, 14), colspan=1) 
        cbar = plt.colorbar(img, fraction=0.4)
        cbar.set_label("Elevation change (m)")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

*Caption: Demonstration of the Nuth and Kääb (2011) approach from Svalbard. Note that large improvements are seen, but nonlinear offsets still exist. The NMAD is calculated from the off-glacier surfaces.*

Limitations
***********
The Nuth and Kääb (2011) coregistation approach does not take rotation into account.
Rotational corrections are often needed on for example satellite derived DEMs, so a complementary tool is required for a perfect fit.
1st or higher degree `Deramping`_ can be used for small rotational corrections.
For large rotations, the Nuth and Kääb (2011) approach will not work properly, and `ICP`_ is recommended instead.

Example
*******
.. code-block:: python

        aligned_dem, error = xdem.coreg.coregister(
                reference_dem,
                dem_to_be_aligned,
                method="nuth_kaab",
                mask=glacier_outlines
        )

Deramping
^^^^^^^^^
- **Performs:** Bias, linear or nonlinear height corrections.
- **Supports weights** (soon)
- **Recommended for:** Data with no horizontal offset and low to moderate rotational differences.

Deramping works by estimating and correcting for an N-degree polynomial over the entire dDEM between a reference and the DEM to be aligned.
This may be useful for correcting small rotations in the dataset, or nonlinear errors that for example often occur in structure-from-motion derived optical DEMs (e.g. Rosnell and Honkavaara `2012 <https://doi.org/10.3390/s120100453>`_; Javernick et al. `2014 <https://doi.org/10.1016/j.geomorph.2014.01.006>`_; Girod et al. `2017 <https://doi.org/10.5194/tc-11827-2017>`_).
Applying a "0 degree deramping" is equivalent to a simple bias correction, and is recommended for e.g. vertical datum corrections.

Limitations
***********
Deramping does not account for horizontal (X/Y) shifts, and should most often be used in conjunction with other methods.

1st order deramping is not perfectly equivalent to a rotational correction: Values are simply corrected in the vertical direction, and therefore includes a horizontal scaling factor, if it would be expressed as a transformation matrix.
For large rotational corrections, `ICP`_ is recommended.

Example
*******
.. code-block:: python

        # Apply a 1st order deramping correction.
        deramped_dem, error = xdem.coreg.coregister(
                reference_dem,
                dem_to_be_aligned,
                method="deramp",
                deramp_degree=1,
                mask=glacier_outlines
        )

ICP
^^^
- **Performs:** Rigid transform correction (translation + rotation).
- **Does not support weights**
- **Recommended for:** Data with low noise and a high relative rotation.

Iterative Closest Point (ICP) coregistration works by iteratively moving the data until it fits the reference as well as possible.
The DEMs are read as point clouds; collections of points with X/Y/Z coordinates, and a nearest neighbour analysis is made between the reference and the data to be aligned.
After the distances are calculated, a rigid transform is estimated to minimise them.
The transform is attempted, and then distances are calculated again.
If the distance is lowered, another rigid transform is estimated, and this is continued in a loop.
The loop stops if it reaches the max iteration limit or if the distances do not improve significantly between iterations.
The opencv implementation of ICP includes outlier removal, since extreme outliers will heavily interfere with the nearest neighbour distances.
This may improve results on noisy data significantly, but care should still be taken, as the risk of landing in `local minima <https://en.wikipedia.org/wiki/Maxima_and_minima>`_ increases.

Limitations
***********
ICP is notoriously bad on noisy data.
TODO: Add references for ICP being bad on noisy data.
The outlier removal functionality of the opencv implementation is a step in the right direction, but it still does not compete with other coregistration approaches when the relative rotation is small.
In cases of high rotation, ICP is the only approach that can account for this properly, but results may need refinement, for example with the `Nuth and Kääb (2011)`_ approach.

Due to the repeated nearest neighbour calculations, ICP is often the slowest coregistration approach out of the alternatives.

Example
*******
.. code-block:: python

        # Use the opencv ICP implementation. For PDAL, use "icp_pdal")
        aligned_dem, error = xdem.coreg.coregister(
                reference_dem,
                dem_to_be_aligned,
                method="icp",
                mask=glacier_outlines
        )


