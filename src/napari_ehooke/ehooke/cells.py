"""Module containg the classes needed for the computation of each
cell, their stats and the selection/filterion of the computed cells.
Contains a class Cell that works as the template for each cell object
and a CellManager class that controls the different steps of the cell
processing."""

from collections import OrderedDict
import numpy as np
import matplotlib as plt

from copy import deepcopy
from skimage.draw import line
from skimage.measure import label, regionprops
from skimage.filters import threshold_isodata
from skimage.util import img_as_float, img_as_int
from skimage import morphology, color, exposure


class Cell(object):
    """Template for each cell object."""

    def __init__(self, cell_id):
        self.label = cell_id

        self.marked_as_noise = "No"

        self.box = None
        self.box_margin = 5

        self.outline = []

        self.cell_mask = None
        self.perim_mask = None
        self.sept_mask = None
        self.cyto_mask = None
        self.membsept_mask = None


        self.stats = OrderedDict([("Area", 0),
                                  ("Perimeter", 0),
                                  ("Orientation", 0),
                                  ("Length", 0),
                                  ("Width", 0),
                                  ("Eccentricity", 0),
                                  ("Irregularity", 0),
                                  ("Baseline", 0),
                                  ("Cell Median", 0),
                                  ("Membrane Median", 0),
                                  ("Septum Median", 0),
                                  ("Cytoplasm Median", 0),
                                  ("Fluor Ratio", 0),
                                  ("Fluor Ratio 75%", 0),
                                  ("Fluor Ratio 25%", 0),
                                  ("Fluor Ratio 10%", 0),
                                  ("Cell Cycle Phase", 0)])

        self.selection_state = 1

    def clean_cell(self):
        """Resets the cell to an empty instance.
        Can be used to mark the cell to discard"""
        self.label = 0

        self.marked_as_noise = "No"

        self.box = None
        self.box_margin = 5

        self.outline = []
        
        self.cell_mask = None
        self.perim_mask = None
        self.sept_mask = None
        self.cyto_mask = None
        self.membsept_mask = None


        self.stats = OrderedDict([("Area", 0),
                                  ("Perimeter", 0),
                                  ("Orientation", 0),
                                  ("Length", 0),
                                  ("Width", 0),
                                  ("Eccentricity", 0),
                                  ("Irregularity", 0),
                                  ("Baseline", 0),
                                  ("Cell Median", 0),
                                  ("Membrane Median", 0),
                                  ("Septum Median", 0),
                                  ("Cytoplasm Median", 0),
                                  ("Fluor Ratio", 0),
                                  ("Fluor Ratio 75%", 0),
                                  ("Fluor Ratio 25%", 0),
                                  ("Fluor Ratio 10%", 0),
                                  ("Cell Cycle Phase", 0)])

        self.selection_state = 1

    def fluor_box(self, fluor):
        """ returns box of flurescence from fluor image """

        x0, y0, x1, y1 = self.box
        try:
            return fluor[x0:x1 + 1, y0:y1 + 1]
        except TypeError:
            return None

    def compute_cell_mask(self):
        x0, y0, x1, y1 = self.box
        mask = np.zeros((x1 - x0 + 1, y1 - y0 + 1))
        for lin in self.lines:
            y, st, en = lin
            mask[st - x0:en - x0 + 1, y - y0] = 1.0
        return mask

    def compute_perim_mask(self, mask, thick):
        """returns mask for perimeter
        needs cell mask
        """
        # create mask

        eroded = morphology.binary_erosion(mask, np.ones(
            (thick * 2 - 1, thick - 1))).astype(float)
        perim = mask - eroded

        return perim

    def compute_sept_mask(self, mask, thick, septum_base, septum_opt, algorithm):
        """ returns mask for axis.
        needs cell mask
        """

        if algorithm == "Isodata":
            return self.compute_sept_isodata(mask, thick, septum_base, septum_opt)

        elif algorithm == "Box":
            return self.compute_sept_box(mask, thick)

        else:
            print("Not a a valid algorithm")

    def compute_opensept_mask(self, mask, thick, septum_base, septum_opt, algorithm):
        """ returns mask for axis.
        needs cell mask
        """

        if algorithm == "Isodata":
            return self.compute_opensept_isodata(mask, thick, septum_base, septum_opt)

        elif algorithm == "Box":
            return self.compute_sept_box(mask, thick)

        else:
            print("Not a a valid algorithm")

    def compute_sept_isodata(self, mask, thick):
        """Method used to create the cell sept_mask using the threshold_isodata
        to separate the cytoplasm from the septum"""
        cell_mask = mask

        fluor_box = self.fluor
        perim_mask = self.compute_perim_mask(cell_mask, thick)
        inner_mask = cell_mask - perim_mask
        inner_fluor = (inner_mask > 0) * fluor_box

        threshold = threshold_isodata(inner_fluor[inner_fluor > 0])
        interest_matrix = inner_mask * (inner_fluor > threshold)

        label_matrix = label(interest_matrix, connectivity=2)
        interest_label = 0
        interest_label_sum = 0

        for l in range(np.max(label_matrix)):
            if np.sum(img_as_float(label_matrix == l + 1)) > interest_label_sum:
                interest_label = l + 1
                interest_label_sum = np.sum(
                    img_as_float(label_matrix == l + 1))

        return img_as_float(label_matrix == interest_label)

    def compute_opensept_isodata(self, mask, thick):
        """Method used to create the cell sept_mask using the threshold_isodata
        to separate the cytoplasm from the septum"""
        cell_mask = mask
        fluor_box = self.fluor
        perim_mask = self.compute_perim_mask(cell_mask, thick)
        inner_mask = cell_mask - perim_mask
        inner_fluor = (inner_mask > 0) * fluor_box

        threshold = threshold_isodata(inner_fluor[inner_fluor > 0])
        interest_matrix = inner_mask * (inner_fluor > threshold)

        label_matrix = label(interest_matrix, connectivity=2)
        label_sums = []

        for l in range(np.max(label_matrix)):
            label_sums.append(np.sum(img_as_float(label_matrix == l + 1)))

        print(label_sums)

        sorted_label_sums = sorted(label_sums)

        first_label = 0
        second_label = 0

        for i in range(len(label_sums)):
            if label_sums[i] == sorted_label_sums[-1]:
                first_label = i + 1
                label_sums.pop(i)
                break

        for i in range(len(label_sums)):
            if label_sums[i] == sorted_label_sums[-2]:
                second_label = i + 2
                label_sums.pop(i)
                break

        if second_label != 0:
            return img_as_float((label_matrix == first_label) + (label_matrix == second_label))
        else:
            return img_as_float((label_matrix == first_label))

    def compute_sept_box(self, mask, thick):
        """Method used to create a mask of the septum based on creating a box
        around the cell and then defining the septum as being the dilated short
        axis of the box."""
        x0, y0, x1, y1 = self.box
        lx0, ly0 = self.short_axis[0]
        lx1, ly1 = self.short_axis[1]
        x, y = line(lx0 - x0, ly0 - y0, lx1 - x0, ly1 - y0)

        linmask = np.zeros((x1 - x0 + 1, y1 - y0 + 1))
        linmask[x, y] = 1
        linmask = morphology.binary_dilation(
            linmask, np.ones((thick, thick))).astype(float)

        if mask is not None:
            linmask = mask * linmask

        return linmask

    def get_outline_points(self, data):
        """Method used to obtain the outline pixels of the septum"""
        outline = []
        for x in range(0, len(data)):
            for y in range(0, len(data[x])):
                if data[x, y] == 1:
                    if x == 0 and y == 0:
                        neighs_sum = data[x, y] + data[x + 1, y] + \
                                     data[x + 1, y + 1] + data[x, y + 1]
                    elif x == len(data) - 1 and y == len(data[x]) - 1:
                        neighs_sum = data[x, y] + data[x, y - 1] + \
                                     data[x - 1, y - 1] + data[x - 1, y]
                    elif x == 0 and y == len(data[x]) - 1:
                        neighs_sum = data[x, y] + data[x, y - 1] + \
                                     data[x + 1, y - 1] + data[x + 1, y]
                    elif x == len(data) - 1 and y == 0:
                        neighs_sum = data[x, y] + data[x - 1, y] + \
                                     data[x - 1, y + 1] + data[x, y + 1]
                    elif x == 0:
                        neighs_sum = data[x, y] + data[x, y - 1] + data[x, y + 1] + \
                                     data[x + 1, y - 1] + \
                                     data[x + 1, y] + data[x + 1, y + 1]
                    elif x == len(data) - 1:
                        neighs_sum = data[x, y] + data[x, y - 1] + data[x, y + 1] + \
                                     data[x - 1, y - 1] + \
                                     data[x - 1, y] + data[x - 1, y + 1]
                    elif y == 0:
                        neighs_sum = data[x, y] + data[x - 1, y] + data[x + 1, y] + \
                                     data[x - 1, y + 1] + \
                                     data[x, y + 1] + data[x + 1, y + 1]
                    elif y == len(data[x]) - 1:
                        neighs_sum = data[x, y] + data[x - 1, y] + data[x + 1, y] + \
                                     data[x - 1, y - 1] + \
                                     data[x, y - 1] + data[x + 1, y - 1]
                    else:
                        neighs_sum = data[x, y] + data[x - 1, y] + data[x + 1, y] + data[x - 1, y - 1] + data[
                            x, y - 1] + data[x + 1, y - 1] + data[x - 1, y + 1] + data[x, y + 1] + data[x + 1, y + 1]
                    if neighs_sum != 9:
                        outline.append((x, y))
        return outline

    def compute_sept_box_fix(self, outline, maskshape):
        """Method used to create a box aroung the septum, so that the short
        axis of this box can be used to choose the pixels of the membrane
        mask that need to be removed"""
        points = np.asarray(outline)  # in two columns, x, y
        bm = self.box_margin
        w, h = maskshape
        box = (max(min(points[:, 0]) - bm, 0),
               max(min(points[:, 1]) - bm, 0),
               min(max(points[:, 0]) + bm, w - 1),
               min(max(points[:, 1]) + bm, h - 1))

        return box

    def remove_sept_from_membrane(self, maskshape):
        """Method used to remove the pixels of the septum that were still in
        the membrane"""

        # get outline points of septum mask
        septum_outline = []
        septum_mask = self.sept_mask
        septum_outline = self.get_outline_points(septum_mask)

        # compute box of the septum
        septum_box = self.compute_sept_box_fix(septum_outline, maskshape)

        # compute axis of the septum
        rotations = cp.rotation_matrices(5)
        points = np.asarray(septum_outline)  # in two columns, x, y
        width = len(points) + 1

        # no need to do more rotations, due to symmetry
        for rix in range(int(len(rotations) / 2) + 1):
            r = rotations[rix]
            nx0, ny0, nx1, ny1, nwidth = cp.bound_rectangle(
                np.asarray(np.dot(points, r)))

            if nwidth < width:
                width = nwidth
                x0 = nx0
                x1 = nx1
                y0 = ny0
                y1 = ny1
                angle = rix

        rotation = rotations[angle]

        # midpoints
        mx = (x1 + x0) / 2
        my = (y1 + y0) / 2

        # assumes long is X. This duplicates rotations but simplifies
        # using different algorithms such as brightness
        long = [[x0, my], [x1, my]]
        short = [[mx, y0], [mx, y1]]
        short = np.asarray(np.dot(short, rotation.T), dtype=np.int32)
        long = np.asarray(np.dot(long, rotation.T), dtype=np.int32)

        # check if axis fall outside area due to rounding errors
        bx0, by0, bx1, by1 = septum_box
        short[0] = cp.bounded_point(bx0, bx1, by0, by1, short[0])
        short[1] = cp.bounded_point(bx0, bx1, by0, by1, short[1])
        long[0] = cp.bounded_point(bx0, bx1, by0, by1, long[0])
        long[1] = cp.bounded_point(bx0, bx1, by0, by1, long[1])

        length = np.linalg.norm(long[1] - long[0])
        width = np.linalg.norm(short[1] - short[0])

        if length < width:
            dum = length
            length = width
            width = dum
            dum = short
            short = long
            long = dum

        # expand long axis to create a linmask
        bx0, by0 = long[0]
        bx1, by1 = long[1]

        h, w = self.sept_mask.shape
        linmask = np.zeros((h, w))

        h, w = self.sept_mask.shape[0] - 2, self.sept_mask.shape[1] - 2
        bin_factor = int(width)

        if bx1 - bx0 == 0:
            x, y = line(bx0, 0, bx0, w)
            linmask[x, y] = 1
            try:
                linmask = morphology.binary_dilation(
                    linmask, np.ones((bin_factor, bin_factor))).astype(float)
            except RuntimeError:
                bin_factor = 4
                linmask = morphology.binary_dilation(
                    linmask, np.ones((bin_factor, bin_factor))).astype(float)

        else:
            m = ((by1 - by0) / (bx1 - bx0))
            b = by0 - m * bx0

            if b < 0:
                l_y0 = 0
                l_x0 = int(-b / m)

                if h * m + b > w:
                    l_y1 = w
                    l_x1 = int((w - b) / m)

                else:
                    l_x1 = h
                    l_y1 = int(h * m + b)

            elif b > w:
                l_y0 = w
                l_x0 = int((w - b) / m)

                if h * m + b < 0:
                    l_y1 = 0
                    l_x1 = int(-b / m)

                else:
                    l_x1 = h
                    l_y1 = int((h - b) / m)

            else:
                l_x0 = 0
                l_y0 = int(b)

                if m > 0:
                    if h * m + b > w:
                        l_y1 = w
                        l_x1 = int((w - b) / m)
                    else:
                        l_x1 = h
                        l_y1 = int(h * m + b)

                elif m < 0:
                    if h * m + b < 0:
                        l_y1 = 0
                        l_x1 = int(-b / m)
                    else:
                        l_x1 = h
                        l_y1 = int(h * m + b)

                else:
                    l_x1 = h
                    l_y1 = int(b)

            x, y = line(l_x0, l_y0, l_x1, l_y1)
            linmask[x, y] = 1
            try:
                linmask = morphology.binary_dilation(
                    linmask, np.ones((bin_factor, bin_factor))).astype(float)
            except RuntimeError:
                bin_factor = 4
                linmask = morphology.binary_dilation(
                    linmask, np.ones((bin_factor, bin_factor))).astype(float)
        return img_as_float(linmask)

    def recursive_compute_sept(self, cell_mask, inner_mask_thickness,algorithm):
        try:
            self.sept_mask = self.compute_sept_mask(cell_mask, inner_mask_thickness, algorithm)
        except IndexError:
            try:
                self.recursive_compute_sept(cell_mask, inner_mask_thickness - 1, algorithm)
            except RuntimeError:
                self.recursive_compute_sept(cell_mask, inner_mask_thickness - 1, "Box")

    def recursive_compute_opensept(self, cell_mask, inner_mask_thickness, algorithm):
        try:
            self.sept_mask = self.compute_opensept_mask(cell_mask, inner_mask_thickness, algorithm)
        except IndexError:
            try:
                self.recursive_compute_opensept(cell_mask, inner_mask_thickness - 1, algorithm)
            except RuntimeError:
                self.recursive_compute_opensept(cell_mask, inner_mask_thickness - 1, "Box")

    def compute_regions(self, params, fluor_image):
        """Computes each different region of the cell (whole cell, membrane,
        septum, cytoplasm) and creates their respectives masks."""

        self.fluor = self.fluor_box(fluor_image)
        self.cell_mask = self.compute_cell_mask()

        if params.find_septum:
            self.recursive_compute_sept(self.cell_mask, params["inner_mask_thickness"],params["septum_algorithm"])

            if params["septum_algorithm"] == "Isodata":

                self.perim_mask = self.compute_perim_mask(self.cell_mask, params["inner_mask_thickness"])
                self.membsept_mask = (self.perim_mask + self.sept_mask) > 0
                linmask = self.remove_sept_from_membrane(fluor_image.shape)
                self.cyto_mask = (self.cell_mask - self.perim_mask - self.sept_mask) > 0 
                if linmask is not None:
                    old_membrane = self.perim_mask
                    self.perim_mask = (old_membrane - linmask) > 0

            else:

                self.perim_mask = (self.compute_perim_mask(self.cell_mask, params["inner_mask_thickness"]) - self.sept_mask) > 0
                self.membsept_mask = (self.perim_mask + self.sept_mask) > 0
                self.cyto_mask = (self.cell_mask - self.perim_mask - self.sept_mask) > 0

        elif params.find_openseptum:
            self.recursive_compute_opensept(self.cell_mask, params["inner_mask_thickness"], params["septum_algorithm"])

            if params.septum_algorithm == "Isodata":
                self.perim_mask = self.compute_perim_mask(self.cell_mask, params["inner_mask_thickness"])
                self.membsept_mask = (self.perim_mask + self.sept_mask) > 0
                linmask = self.remove_sept_from_membrane(fluor_image.shape)
                self.cyto_mask = (self.cell_mask - self.perim_mask - self.sept_mask) > 0

                if linmask is not None:
                    old_membrane = self.perim_mask
                    self.perim_mask = (old_membrane - linmask) > 0

            else:

                self.perim_mask = (self.compute_perim_mask(self.cell_mask,params["inner_mask_thickness"]) - self.sept_mask) > 0
                self.membsept_mask = (self.perim_mask + self.sept_mask) > 0
                self.cyto_mask = (self.cell_mask - self.perim_mask - self.sept_mask) > 0

        else:
            self.sept_mask = None
            self.perim_mask = self.compute_perim_mask(self.cell_mask, params["inner_mask_thickness"])
            self.cyto_mask = (self.cell_mask - self.perim_mask) > 0

    def compute_fluor_baseline(self, mask, fluor, margin):
        """mask and fluor are the global images
           NOTE: mask is 0 (black) at cells and 1 (white) outside
        """
        
        # TODO 
        mask = 1 - mask

        x0, y0, x1, y1 = self.box
        wid, hei = mask.shape
        x0 = max(x0 - margin, 0)
        y0 = max(y0 - margin, 0)
        x1 = min(x1 + margin, wid - 1)
        y1 = min(y1 + margin, hei - 1)
        mask_box = mask[x0:x1, y0:y1]

        count = 0

        inverted_mask_box = 1 - mask_box

        while count < 5:
            inverted_mask_box = morphology.binary_dilation(inverted_mask_box)
            count += 1

        mask_box = 1 - inverted_mask_box

        fluor_box = fluor[x0:x1, y0:y1]
        self.stats["Baseline"] = np.median(
            mask_box[mask_box > 0] * fluor_box[mask_box > 0])

    def measure_fluor(self, fluorbox, roi, fraction=1.0):
        """returns the median and std of  fluorescence in roi
        fluorbox has the same dimensions as the roi mask
        """
        fluorbox = fluorbox
        if roi is not None:
            bright = fluorbox * roi
            bright = bright[roi > 0.5]
            # check if not enough points

            if (len(bright) * fraction) < 1.0:
                return 0.0

            if fraction < 1:
                sortvals = np.sort(bright, axis=None)[::-1]
                sortvals = sortvals[np.nonzero(sortvals)]
                sortvals = sortvals[:int(len(sortvals) * fraction)]
                return np.median(sortvals)

            else:
                return np.median(bright)
        else:
            return 0

    def compute_fluor_stats(self, params, mask, fluor):
        """Computes the cell stats related to the fluorescence"""
        self.compute_fluor_baseline(mask, fluor, params["baseline_margin"])
        fluorbox = self.fluor_box(fluor)

        self.stats["Cell Median"] = \
            self.measure_fluor(fluorbox, self.cell_mask) - \
            self.stats["Baseline"]

        self.stats["Membrane Median"] = \
            self.measure_fluor(fluorbox, self.perim_mask) - \
            self.stats["Baseline"]

        self.stats["Cytoplasm Median"] = \
            self.measure_fluor(fluorbox, self.cyto_mask) - \
            self.stats["Baseline"]

        if params["find_septum"] or params["find_openseptum"]:
            self.stats["Septum Median"] = self.measure_fluor(
                fluorbox, self.sept_mask) - self.stats["Baseline"]

            self.stats["Fluor Ratio"] = (self.measure_fluor(fluorbox, self.sept_mask) - self.stats[
                "Baseline"]) / (self.measure_fluor(fluorbox, self.perim_mask) - self.stats["Baseline"])

            self.stats["Fluor Ratio 75%"] = (self.measure_fluor(fluorbox, self.sept_mask, 0.75) - self.stats[
                "Baseline"]) / (self.measure_fluor(fluorbox, self.perim_mask) - self.stats["Baseline"])

            self.stats["Fluor Ratio 25%"] = (self.measure_fluor(fluorbox, self.sept_mask, 0.25) - self.stats[
                "Baseline"]) / (self.measure_fluor(fluorbox, self.perim_mask) - self.stats["Baseline"])

            self.stats["Fluor Ratio 10%"] = (self.measure_fluor(fluorbox, self.sept_mask, 0.10) - self.stats[
                "Baseline"]) / (self.measure_fluor(fluorbox, self.perim_mask) - self.stats["Baseline"])
            self.stats["Memb+Sept Median"] = self.measure_fluor(fluorbox, self.membsept_mask) - self.stats["Baseline"]

        else:
            self.stats["Septum Median"] = 0

            self.stats["Fluor Ratio"] = 0

            self.stats["Fluor Ratio 75%"] = 0

            self.stats["Fluor Ratio 25%"] = 0

            self.stats["Fluor Ratio 10%"] = 0

            self.stats["Memb+Sept Median"] = 0

    def get_cell_image(self, fluor_image):
        x0, y0, x1, y1 = self.box
        img = fluor_image[x1:x0 - 1, y1:y0 - 1] * self.cell_mask

        return img


class CellManager:
    """Main class of the module. Should be used to interact with the rest of
    the modules."""

    def __init__(self, label, fluor):
        self.cells = {}
        self.original_cells = {}

        self.label_img = label
        self.fluor_img = fluor
        self.mask = (label>0).astype(int)

        spmap = plt.cm.get_cmap("hsv", 10)
        self.cell_colors = spmap(np.arange(10))

    def clean_empty_cells(self):
        """Removes empty cell objects from the cells dict"""
        newcells = {}
        for k in self.cells.keys():
            if self.cells[k].stats["Area"] > 0:
                newcells[k] = self.cells[k]

        self.cells = newcells

    def cell_regions_from_labels(self):
        """
        Creates a dictionary of N cells assuming self.labels has consecutive values from 1 to N. Each cell is at key label-1
        """
        
        properties = regionprops(self.label_img)

        cells = {}
        for f in np.unique(labels):
            proto_cell = Cell(f)

            proto_cell.box = properties[f].bbox
            proto_cell.stats["Area"] = properties[f].area
            proto_cell.stats["Perimeter"] = properties[f].perimeter
            proto_cell.stats["Orientation"] = properties[f].orientation
            proto_cell.stats["Length"] = properties[f].axis_major_length
            proto_cell.stats["Width"] = properties[f].axis_minor_length
            proto_cell.stats["Eccentricity"] = properties[f].eccentricity
            proto_cell.stats["Irregularity"] = properties[f].perimeter / (properties[f].area**0.5)
            

            cells[str(int(f))] = proto_cell

        self.cells = cells
        self.original_cells = deepcopy(self.cells)

    def mark_cell_as_noise(self, label_c1, is_noise):
        """Used to change the selection_state of a cell to 0 (noise)
        or to revert that change if the optional param "is_noise" is marked as
        false."""
        if is_noise:
            self.cells[str(label_c1)].selection_state = 0
            self.cells[str(label_c1)].marked_as_noise = "Yes"
        else:
            self.cells[str(label_c1)].selection_state = 1
            self.cells[str(label_c1)].marked_as_noise = "No"

    def process_cells(self, params):
        """Method used to compute the individual regions of each cell and the
        computation of the stats related to the fluorescence"""
        for k in list(self.cells.keys()):
            try:
                self.cells[k].compute_regions(params, self.fluor)
                self.cells[k].compute_fluor_stats(params, self.mask, self.fluor_img)
            except TypeError:
                del self.cells[k]