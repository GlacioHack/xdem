"""Plot the comparison between a dDEM before and after Nuth and Kääb (2011) coregistration."""
import geoutils as gu
import matplotlib.pyplot as plt
import numpy as np

import xdem

dem_2009 = xdem.DEM(xdem.examples.get_path("longyearbyen_ref_dem"))
dem_1990 = xdem.DEM(xdem.examples.get_path("longyearbyen_tba_dem"))
outlines_1990 = gu.Vector(xdem.examples.get_path("longyearbyen_glacier_outlines"))
inlier_mask = ~outlines_1990.create_mask(dem_2009)

nuth_kaab = xdem.coreg.NuthKaab()
nuth_kaab.fit(dem_2009.data, dem_1990.data, transform=dem_2009.transform, inlier_mask=inlier_mask)
dem_coreg = nuth_kaab.apply(dem_1990.data, transform=dem_1990.transform)

ddem_pre = (dem_2009.data - dem_1990.data).filled(np.nan).squeeze()
ddem_post = (dem_2009.data - dem_coreg).filled(np.nan).squeeze()

nmad_pre = xdem.spatial_tools.nmad(ddem_pre[inlier_mask.squeeze()])
nmad_post = xdem.spatial_tools.nmad(ddem_post[inlier_mask.squeeze()])

vlim = 20
plt.figure(figsize=(8, 5))
plt.subplot2grid((1, 15), (0, 0), colspan=7)
plt.title(f"Before coregistration. NMAD={nmad_pre:.1f} m")
plt.imshow(ddem_pre, cmap="coolwarm_r", vmin=-vlim, vmax=vlim)
plt.axis("off")
plt.subplot2grid((1, 15), (0, 7), colspan=7)
plt.title(f"After coregistration. NMAD={nmad_post:.1f} m")
img = plt.imshow(ddem_post, cmap="coolwarm_r", vmin=-vlim, vmax=vlim)
plt.axis("off")
plt.subplot2grid((1, 15), (0, 14), colspan=1)
cbar = plt.colorbar(img, fraction=0.4, ax=plt.gca())
cbar.set_label("Elevation change (m)")
plt.axis("off")

plt.tight_layout()
plt.show()
