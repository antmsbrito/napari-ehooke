"""
TODO
"""
from typing import TYPE_CHECKING

from magicgui import magic_factory

if TYPE_CHECKING:
    import napari

import numpy as np
from scipy import ndimage, signal
from skimage.filters import threshold_isodata, threshold_local
from skimage.morphology import binary_closing, binary_dilation
from skimage.transform import EuclideanTransform, warp

@magic_factory(algorithm={"choices":["Isodata","Local Average"]})
def compute_mask(image:"napari.types.ImageData",algorithm="Isodata",blocksize:int=151,offset:float=0.02,closing:int=1,dilation:int=0,fillholes:bool=False)->"napari.types.LabelsData":
    """
    TODO    
    """

    if algorithm == "Isodata":
        
        mask = image > threshold_isodata(image)
        mask = mask.astype(int)
        mask = 1 - mask

    elif algorithm == "Local Average":
        if blocksize%2==0:
            blocksize += 1
        mask = image > threshold_local(image, block_size=blocksize, method="gaussian", offset=offset)
        mask = mask.astype(int)
        mask = 1 - mask
    
    if closing > 0:
        # removes small white spots and then small dark spots
        closing_matrix = np.ones((int(closing), int(closing)))
        mask = binary_closing(mask, closing_matrix)
        mask = 1 - binary_closing(1 - mask, closing_matrix)

    for f in range(dilation):
        mask = binary_dilation(mask, np.ones((3, 3)))

    if fillholes:
        mask = ndimage.binary_fill_holes(mask)

    return mask


@magic_factory
def align_mask(mask:"napari.types.LabelsData", fluor:"napari.types.ImageData")->"napari.types.ImageData":
    """
    TODO
    """

    corr = signal.fftconvolve(mask,fluor[::-1,::-1])
    deviation = np.unravel_index(np.argmax(corr),corr.shape)
    cm = ndimage.center_of_mass(np.ones(corr.shape))
    
    dy,dx = np.subtract(deviation,cm)
    matrix = EuclideanTransform(rotation=0, translation=(dx,dy))

    return warp(fluor, matrix.inverse, preserve_range=True) # TODO check if fluor intensity values stay the same

    