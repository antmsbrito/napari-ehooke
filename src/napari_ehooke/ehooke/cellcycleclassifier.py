import numpy as np
from skimage.util import img_as_float
from skimage.exposure import rescale_intensity
from skimage.transform import resize as skresize
from keras.utils import get_file
from keras.models import load_model

import tensorflow as tf

import os

# force classification to happen on CPU to avoid CUDA problems
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Remove some extraneous log outputs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class CellCycleClassifier:

    def __init__(self, fluor_fov, optional_fov, microscope):

        cnnmodel = get_file("model","https://github.com/antmsbrito/napari-ehooke/blob/main/docs/cellcycle_cnn_model?raw=true")
        self.model = load_model(cnnmodel)

        self.fluor_fov = fluor_fov
        self.optional_fov = optional_fov

        self.microscope = microscope

    def preprocess_image(self, image):

        if self.microscope == "Epi":
            max_dim = 50

        elif self.microscope == "SIM":
            max_dim = 100

        h, w = image.shape

        max_h, max_w = max_dim, max_dim

        lines_to_add = max_h - h
        columns_to_add = max_w - w

        if lines_to_add > 0:
            if lines_to_add % 2 == 0:
                new_line = np.zeros((int(lines_to_add / 2), w))
                image = np.concatenate((new_line, image, new_line), axis=0)
            else:
                new_line_top = np.zeros((int(lines_to_add / 2) + 1, w))
                new_line_bot = np.zeros((int(lines_to_add / 2), w))
                image = np.concatenate((new_line_top, image, new_line_bot), axis=0)

        elif lines_to_add < 0:
            if (lines_to_add * -1) % 2 == 0:
                cutsize = int((lines_to_add * -1) / 2)
                image = image[cutsize:h - cutsize, :]
            else:
                cutsize = int((lines_to_add * -1) / 2)
                image = image[cutsize:h - cutsize - 1, :]

        if columns_to_add > 0:
            if columns_to_add % 2 == 0:
                columns_to_add = np.zeros((max_dim, int(columns_to_add / 2)))
                image = np.concatenate((columns_to_add, image, columns_to_add), axis=1)
            else:
                columns_to_add_left = np.zeros((max_dim, int(columns_to_add / 2) + 1))
                columns_to_add_right = np.zeros((max_dim, int(columns_to_add / 2)))
                image = np.concatenate((columns_to_add_left, image, columns_to_add_right), axis=1)

        elif columns_to_add < 0:
            if (columns_to_add * -1) % 2 == 0:
                cutsize = int((columns_to_add * -1) / 2)
                image = image[:, cutsize:w - cutsize]
            else:
                cutsize = int((columns_to_add * -1) / 2)
                image = image[:, cutsize:w - cutsize - 1]

        image = img_as_float(image)
        image = image.reshape(max_dim, max_dim, 1)

        return image

    def classify_cell(self, cell_object):
        
        x0, y0, x1, y1 = cell_object.box

        fluor = rescale_intensity(self.fluor_fov[x0:x1 + 1, y0:y1 + 1] * cell_object.cell_mask)
        optional = rescale_intensity(self.optional_fov[x0:x1 + 1, y0:y1 + 1] * cell_object.cell_mask)

        fluor_img = skresize(self.preprocess_image(fluor),
                             (100, 100),
                             order=0,
                             preserve_range=True,
                             anti_aliasing=False,
                             anti_aliasing_sigma=None)

        optional_img = skresize(self.preprocess_image(optional),
                                (100, 100),
                                order=0,
                                preserve_range=True,
                                anti_aliasing=False,
                                anti_aliasing_sigma=None)

        pred = self.model.predict(np.concatenate((fluor_img, optional_img), axis=1).reshape(-1, 100, 200, 1), verbose=0)
        
        return np.argmax(pred,axis=-1)[0] + 1
