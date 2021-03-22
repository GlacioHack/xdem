Spatial statistics
==================

How does one calculate the error between two DEMs?
This question is the basis of numerous academic papers, but their conclusions are often hard to grasp as the mathematics behind them can be quite daunting.
In addition, the lack of a simple implementation in a modern programming language makes these methods obscure and used only by those who can program it themselves.
One of the goals of ``xdem`` is to simplify state-of-the-art statistical measures, to allow accurate DEM comparisons for everyone, regardless of one's statistical talent.

.. contents:: Contents 
   :local:

Common statistical measures
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Mean
****
The mean of a dDEM is most often used to calculate the elevation change of terrain, for example of a glacier.
The mean elevation change can then be used to calculate the change in volume by multiplying with the associated area.
A considerable limitation of the mean of unfiltered data is that outliers may severely affect the result.
If you have 100 pixels that generally signify an elevation change of --30 m, but one pixel is a rogue NoData value of -9999, the mean elevation change will be --129.69 m!
If the mean is used, extreme outliers should first be accounted for.

.. code-block:: python

        mean = ddem.data.mean()

Median
******
The median is the most common value in a distribution.
If the values are normally distributed (as a bell-curve), the median lies exactly on top of the curve.
Medians are often used as a more robust value to represent the centre of a distribution than the mean, as it is less affected by outliers.
Going with the same example as above, the median of the 100 pixels with one oulier would be --30 m.
The median is however not always suitable for e.g. volume change, as the value may not be perfectly representative at all times.
For example, the median of a DEM with integer elevations [100, 130, 140, 150, 160] would yield a median of 140 m, while an arguably better value for volume change would be the mean (136 m).

.. code-block:: python
        
        median = np.median(ddem.data)

Standard deviation
******************
The standard deviation (STD) is often used to represent the spread of a distribution of values.
It is theoretically made to represent the spread of a perfect bell-curve, where an STD of ±1 represents 68.2% of all values.
Conversely ±2 STDs represent 95.2% of all values.

.. code-block:: python
        
        std = ddem.data.std()


RMSE
****
The Root Mean Squared Error (RMSE) is a measure of the agreement of the values in a distribution.
It is highly sensitive to outliers, and is often used in photogrammetry where outliers can be detrimental to the relative, internal or external orientation of images.
RMSE's are however unsuitable for e.g. volume change error, as the purposefully exaggerated outliers will not have the same exaggerated effect on the mean. 

.. code-block:: python

        rmse = np.sqrt(np.mean(np.square(ddem.data)))

.. _spatial_stats_nmad:

NMAD
****
The Normalized Median Absolute Deviation (NMAD) is another measure of the spread of a distribution, similar to the RMSE and standard deviation.

TODO: Add a rationale for this approach.

.. code-block:: python

        nmad = xdem.spatial_tools.nmad(ddem.data)

Standard error
**************
The standard error (SE) is a measure of the total integrated uncertainty over a multitude of point values.
For dDEMs, the SE is good for quantifying the effect of stochastic (random) error in mean elevation and volume change calculations.

.. math::

        SE_{dh} = \frac{STD_{dh}}{\sqrt{N}},

where :math:`SE_{dh}` is the standard error of elevation change, :math:`STD_{dh}` is the standard deviation of the samples in the area of interest, and :math:`N` is the number of **independent** observations.

Note that correct use of the SE assumes that the standard deviation represents completely stochastic (independent / random) error.
The SE is therefore useful once all systematic (non-random) errors have been accounted for, e.g. using one or multiple :ref:`coregistration` approaches.

.. code-block:: python

        se = ddem.data.std() / np.sqrt(ddem.data.flatten().shape[0])

Periglacial error
^^^^^^^^^^^^^^^^^
TODO: Add this section


Variograms
^^^^^^^^^^

TODO: Add this section
