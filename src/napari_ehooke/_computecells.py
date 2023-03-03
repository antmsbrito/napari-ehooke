"""
TODO
"""

import typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from magicgui import magic_factory

from napari_skimage_regionprops import add_table

from .ehooke.cells import CellManager

@magic_factory(Septum_algorithm={"choices":["Isodata","Box"]},Microscope={"choices":["Epi","SIM"]})
def compute_cells(Viewer:"napari.Viewer",
                  Label_Image:"napari.layers.Labels",
                  Fluor_Image:"napari.layers.Image",
                  DNA_Image:"napari.layers.Image",
                  Pixel_size:float=1,
                  Inner_mask_thickness:int=4,
                  Septum_algorithm="Isodata",
                  Baseline_margin:int=30,
                  Find_septum:bool=False,
                  Find_open_septum:bool=False,
                  Classify_cell_cycle:bool=False,
                  Microscope:str="Epi"
                  ):

    params = {"pixel_size":Pixel_size,
              "inner_mask_thickness":Inner_mask_thickness,
              "septum_algorithm":Septum_algorithm,
              "baseline_margin":Baseline_margin,
              "find_septum":Find_septum,
              "find_openseptum":Find_open_septum,
              "classify_cell_cycle":Classify_cell_cycle,
              "microscope":Microscope,
              }

    cell_man = CellManager(label_img=Label_Image.data, fluor=Fluor_Image.data, optional=DNA_Image.data, params=params)
    cell_man.compute_cell_properties()
    
    Label_Image.properties = cell_man.properties

    add_table(Label_Image, Viewer)
    

