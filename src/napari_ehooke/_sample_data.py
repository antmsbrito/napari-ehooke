"""
TODO
"""
from __future__ import annotations

import numpy
from skimage.io import imread

def phase_example():
    return [(imread("https://github.com/antmsbrito/napari-ehooke/raw/main/docs/test_phase.tif",{"name":"Example S.aureus phase contrast"}, "image")),]

def membrane_example():
    return [(imread("https://github.com/antmsbrito/napari-ehooke/raw/main/docs/test_membrane.tif",{"name":"Example S.aureus labeled with membrane dye"}, "image")),]

def dna_example():
    return [(imread("https://github.com/antmsbrito/napari-ehooke/raw/main/docs/test_dna.tif",{"name":"Example S.aureus labeled with DNA dye"}, "image")),]