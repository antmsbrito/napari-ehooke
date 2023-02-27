"""
TODO
"""

import typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from magicgui import magic_factory

import numpy as np
from scipy import ndimage, signal
from skimage.filters import threshold_isodata, threshold_local
from skimage.morphology import binary_closing, binary_dilation
from skimage.transform import EuclideanTransform, warp

@magic_factory(algorithm={"choices":["Isodata","Local Average"]})
def compute_mask(Base:"napari.types.ImageData",Fluor:"napari.types.ImageData",algorithm="Isodata",blocksize:int=151,offset:float=0.02,closing:int=1,dilation:int=0,fillholes:bool=False,autoalign:bool=False)->typing.List["napari.types.LayerDataTuple"]:
    """
    TODO    
    """
    # TODO MOVE ALL LOGIC TO EHOOKE SUBFOLDER
    if algorithm == "Isodata":

        mask = Base > threshold_isodata(Base)
        mask = mask.astype(int)
        mask = 1 - mask

    elif algorithm == "Local Average":
        if blocksize%2==0:
            blocksize += 1
        mask = Base > threshold_local(Base, block_size=blocksize, method="gaussian", offset=offset)
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


    if autoalign:
        corr = signal.fftconvolve(mask,Fluor[::-1,::-1])
        deviation = np.unravel_index(np.argmax(corr),corr.shape)
        cm = ndimage.center_of_mass(np.ones(corr.shape))
        
        dy,dx = np.subtract(deviation,cm)
        matrix = EuclideanTransform(rotation=0, translation=(dx,dy))

        aligned_fluor = warp(Fluor, matrix.inverse, preserve_range=True) # TODO check if fluor intensity values stay the same

        return [(mask, {'name': 'Mask'}, 'Labels'), (aligned_fluor,{'name':'Aligned fluor'}, 'Image')]
    else:
        return [(mask, {'name': 'Mask'}, 'Labels'),]