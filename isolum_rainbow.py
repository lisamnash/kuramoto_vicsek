#!/usr/bin/env python

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.pyplot import register_cmap
from scipy import interpolate
import numpy as np

#Isoluminate color map from Kindlmann et al., Face-based Luminance Matching for Perceptual Colormap Generation
#Modified the blue and yellow a little to make them more pronounced
RYGCBM = np.array([
    [0.847, 0.057, 0.057],
#    [0.527, 0.527, 0.000],
    # [0.700, 0.700, 0.000],
    [0.650, 0.650, 0.000],
    [0.000, 0.592, 0.000],
    [0.000, 0.559, 0.559],
#    [0.316, 0.316, 0.991],
    # [0.250, 0.250, 0.991],
    [0.200, 0.200, 1.0],
    [0.718, 0.000, 0.718],
])

ROYGBV = []
gamma = 1.5 #Theoretically, 2.2, but this exagerates the saturation a bit
for i, spacing in enumerate([120, 60, 30, 30, 60, 60]):
    x = (np.arange(spacing, dtype='f') / spacing)[:, np.newaxis]
    ROYGBV.append((1-x) * RYGCBM[i]**gamma + x * RYGCBM[(i+1)%6]**gamma)
    
ROYGBV = np.vstack(ROYGBV)
ROYGBV /= ROYGBV.max()
ROYGBV = np.vstack([ROYGBV, ROYGBV[:1]])**(1/gamma)


ROYGBV_i = interpolate.interp1d(np.arange(361)/360., ROYGBV, axis=0)

# isolum_rainbow = LinearSegmentedColormap.from_list('isolum_rainbow', ROYGBV_i(np.linspace(0, 1, 256)))
# register_cmap(cmap=isolum_rainbow)

isolum_rainbow_brighter = LinearSegmentedColormap.from_list('isolum_rainbow', np.clip(ROYGBV_i(np.linspace(0, 1, 256))*1.4, 0, 1))
register_cmap(cmap=isolum_rainbow_brighter)

if __name__ == '__main__':
    import pylab as P
    
    x, y = np.mgrid[-20:20, -20:20]

    # P.subplot(211)
    P.imshow(np.arctan2(y, x), clim=(-np.pi, np.pi), cmap='isolum_rainbow')

    # P.subplot(212)
    # P.imshow(np.arctan2(y, x), clim=(-np.pi, np.pi), cmap='isolum_rainbow_brighter')

    P.show()