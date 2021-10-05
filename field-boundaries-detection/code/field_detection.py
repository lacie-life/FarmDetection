import ee
import os
import rasterio
import affine
import matplotlib.pyplot as plt
import cv2
import numpy as np
import fiona
import sys
import itertools as it
import matplotlib as mPlt
from matplotlib.collections import PatchCollection

import shapely

from osgeo import gdal

from skimage.filters import sobel
from skimage import segmentation

from skimage import feature
from skimage import data
from skimage import filters
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage.filters.rank import entropy
from skimage.morphology import disk
from skimage import (transform, exposure, segmentation, filters, feature, morphology)
from skimage.color import label2rgb
from scipy import ndimage as ndi
from PIL import Image
from rasterio import plot
from rasterio.plot import show, show_hist
from rasterio.features import shapes as polygonize
from matplotlib import pyplot
from rasterio.plot import show, reshape_as_raster, reshape_as_image
from rasterio.mask import mask as msk
import matplotlib as mpl
from descartes import PolygonPatch

import numpy.ma as ma
import copy
import geopandas as gp

with rasterio.open('/media/lacie-life/Data/datasets/mapirHN/odm_orthophoto.tif', masked=True) as src:
    ndvi = src.read()

show_hist(ndvi, 30)

ndvim = copy.copy(ndvi)
ndvim[ndvi == 0] = np.nan_to_num(np.nan)

veg = rasterio.open('/media/lacie-life/Data/datasets/mapirHN/odm_orthophoto.tif', masked=True)
aff = veg.meta['transform']
veg_nd = veg.read(1)
veg_nd[veg_nd == 0] = np.nan_to_num(np.nan)
veg_nd[veg_nd < 0] = 0

fig, ax = plt.subplots(figsize=(8, 8),
                       sharex=True, sharey=True)
img0 = ax.imshow(veg_nd, cmap='RdYlGn')
ax.set_title("NDVI")

ax.axis("off")
fig.colorbar(img0, ax=ax, fraction=0.04)
fig.tight_layout()

plt.show()

# Create figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 15))

# Display the image
show((veg_nd), ax=ax1, cmap='RdYlGn', title='NDVI of Study Area')

# Display the area of interest
show((veg, 1), ax=ax2, cmap='RdYlGn', title='Area of Interest')

# Create a Rectangle patch
with fiona.open("/media/lacie-life/Data/datasets/mapirHN/fakeGeometry.shp", "r") as shapefile:
    features = [feature["geometry"] for feature in shapefile]

with rasterio.open("/media/lacie-life/Data/datasets/mapirHN/odm_orthophoto.tif") as src:
    out_image, out_transform = msk(src, features, crop=True)
    out_meta = src.meta.copy()
    # out_image [out_image == 0] = np.nan

out_meta.update({"driver": "GTiff",
                 "height": out_image.shape[1],
                 "width": out_image.shape[2],
                 "transform": out_transform,
                 "affine": veg.meta['transform']
                 })
out_image.astype(rasterio.uint8)

with rasterio.open("/media/lacie-life/Data/datasets/mapirHN/AoiPlots.tif", "w", **out_meta) as dest:
    dest.write(out_image)

with rasterio.open('/media/lacie-life/Data/datasets/mapirHN/AoiPlots.tif', masked=True) as src:
    aoi = src.read(1)
    aoi[aoi == -10000] = np.nan_to_num(np.nan)
    aoi[aoi < 0] = 0

fig, ax1 = plt.subplots(figsize=(10, 10), sharex=True, sharey=True)
img0 = ax1.imshow(aoi, cmap='RdYlGn')
ax1.set_title("NDVI")
ax1.axis('off')
fig.colorbar(img0, ax=ax1, fraction=0.03)
fig.tight_layout()

plt.show()

selem = disk(16)

with rasterio.open('/media/lacie-life/Data/datasets/mapirHN/AoiPlots.tif', masked=True) as src:
    aoi1 = src.read(1)
    aoi1[aoi1 == -10000] = np.nan_to_num(np.nan)
    aoi1[aoi1 < 0] = 0
    aoi1[aoi1 > 1] = 1

percentile_result = rank.mean_percentile(aoi1, selem=selem, p0=0.1, p1=0.9)
bilateral_result = rank.mean_bilateral(aoi1, selem=selem, s0=2, s1=2)
normal_result = rank.mean(aoi1, selem=selem)

fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(10, 10),
                         sharex=True, sharey=True)
ax = axes.ravel()

titles = ['Original', 'Percentile mean', 'Bilateral mean', 'Local mean']
imgs = [aoi1, percentile_result, bilateral_result, normal_result]
for n in range(0, len(imgs)):
    img = ax[n].imshow(imgs[n], cmap='RdYlGn')
    ax[n].set_title(titles[n])
    ax[n].axis('off')
    # fig.colorbar(img,ax[n])

# plt.tight_layout()
# plt.show()

edges1 = sobel(percentile_result)
edges2 = sobel(normal_result)
bounds1 = edges1
bounds2 = edges2

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 15))
# ax.axis('off')
show(bounds1, ax=ax1, cmap=plt.cm.gray, interpolation='nearest', title='Percentile Mean')
show(bounds2, ax=ax2, cmap=plt.cm.gray, interpolation='nearest', title='Local Mean')

# fig,(ax1,ax2)=plt.subplots(nrows=1,ncols=2,figsize=(15,15))
show_hist(bounds1, 10, title='percentile mean')
show_hist(bounds2, 10, title='local mean')

bounds2 = edges2
bounds2 = bounds2.astype(rasterio.float64)
bounds2[bounds2 > 0.001] = 0

show(bounds2)

markers = np.zeros_like(bounds2)
markers[bounds2 == 0] = 0
markers[bounds2 < 0.015] = 1
markers[bounds2 > 0.015] = 2

# markers[percentile_result < 100] = 3
# markers[percentile_result < 50] = 3

fig, ax = plt.subplots(figsize=(5, 5))
ax.imshow(markers, cmap=plt.cm.Spectral, interpolation='nearest')
ax.set_title('markers')
ax.axis('off')
# ax.set_adjustable('box-forced')

from skimage import morphology
from skimage.color import label2rgb
from scipy import ndimage as ndi

segmentation = morphology.watershed(bounds2, markers)

# fig, ax = plt.subplots(figsize=(10, 10))
# ax.imshow(segmentation, cmap=plt.cm.gray, interpolation='nearest')
# ax.set_title('segmentation')
# ax.axis('off')
# ax.set_adjustable('box-forced')

segmentation = ndi.binary_fill_holes(segmentation - 1)
labeled_plots, _ = ndi.label(segmentation)
plot_overlay = label2rgb(labeled_plots, image=aoi)

type(labeled_plots)

titles = 'Contours'
fig, axes = plt.subplots(figsize=(10, 10))
axes.imshow(aoi, cmap=plt.cm.gray, interpolation='nearest')
axes.contour(segmentation, [0.4], linewidths=1, colors='r')
axes.set_title(titles)
axes.axis('off')

plt.tight_layout()

titles = 'Detected Plots'
fig, axes = plt.subplots(figsize=(15, 15))
axes.imshow(plot_overlay, interpolation='nearest')
axes.set_title(titles)
axes.axis('off')

plt.tight_layout()
plt.savefig('/media/lacie-life/Data/datasets/mapirHN/bounds.png')

borders = plt.contour(segmentation, [0.4], linewidths=1, colors='r', origin='image')
bounds = borders.collections

len(borders.collections)
# type(segmentation)

if __name__ == "__main__":
    print("Hello Python")
