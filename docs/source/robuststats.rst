Robust statistics
==================

Digital Elevation Models often contain outliers that hamper further analysis.
In order to deal with outliers, ``xdem`` integrates statistical measures robust to outliers to be used for estimation of the
mean or dispersion of a sample, or more complex function fitting.

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
