Spatial statistics
==================

How does one calculate the error between two DEMs?
This question is the basis of numerous academic papers, but their conclusions are often hard to grasp as the mathematics behind them can be quite daunting.
In addition, the lack of a simple implementation in a modern programming language makes these methods obscure and used only by those who can program it themselves.
One of the goals of ``xdem`` is to simplify state-of-the-art statistical measures, to allow accurate DEM comparisons for everyone, regardless of one's statistical talent.

.. contents:: Contents 
   :local:


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
