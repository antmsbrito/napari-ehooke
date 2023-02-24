"""
TODO
"""

import typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from magicgui import magic_factory

from .ehooke.segments import SegmentsManager

@magic_factory
def compute_features(Mask:"napari.types.LabelsData",Peak_min_distance:int=5,Peak_min_height:int=5,Peak_min_margin:int=10,Max_peaks:int=10000)->typing.List["napari.types.LayerDataTuple"]:

    pars = {'peak_min_distance_from_edge':Peak_min_margin,'peak_min_distance':Peak_min_distance,'peak_min_height':Peak_min_height,'max_peaks':Max_peaks}

    seg_man = SegmentsManager()
    seg_man.compute_segments(pars, Mask)

    return [(seg_man.base_w_features, {'name': 'Features'}, 'Labels'), (seg_man.labels,{'name':'Labels'}, 'Labels')]


    