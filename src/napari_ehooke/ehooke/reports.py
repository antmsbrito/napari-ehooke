"""Module used to create the report of the cell identification"""
import matplotlib as mpl
from skimage.io import imsave
from skimage.util import img_as_float, img_as_uint
from skimage.filters import threshold_isodata
from skimage.color import gray2rgb
from decimal import Decimal
import numpy as np
import os
from tifffile import imwrite

from .cellprocessing import stats_format

class ReportManager:

    def __init__(self, parameters, cells):
        self.cells = cells
        self.params = parameters
        self.keys = stats_format(parameters)

        self.cell_data_filename = None

    def html_report(self, filename):
        params = self.params
        cells = self.cells
        """generates an html report with the all the cell stats from the
        selected cells"""

        HTML_HEADER = """<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN"
                        "http://www.w3.org/TR/html4/strict.dtd">
                    <html lang="en">
                      <head>
                        <meta http-equiv="content-type" content="text/html; charset=utf-8">
                        <title>title</title>
                        <link rel="stylesheet" type="text/css" href="style.css">
                        <script type="text/javascript" src="script.js"></script>
                      </head>
                      <body>\n"""

        report = [HTML_HEADER]

        if len(cells) > 0:
            header = '<table>\n<th>Cell ID</th><th>Images'
            for k in self.keys:
                label, digits = k
                header = header + '</th><th>' + label
            header += '</th>\n'
            selects = ['\n<h1>Selected cells:</h1>\n' + header + '\n']
            rejects = ['\n<h1>Rejected cells:</h1>\n' + header + '\n']

            count = 0
            count2 = 0

            print("Total Cells: " + str(len(cells)))

            for cell in cells:
                if cell.selection_state == 1:
                    cellid = str(int(cell.label))
                    img = img_as_float(cell.image)

                    """
                    x0, y0, x1, y1 = cell.box
                    memb_img = cell.fluor[x0:x1 + 1, y0:y1 + 1]
                    dna_img = cell.optional[x0:x1 + 1, y0:y1 + 1]
                    imwrite(filename + "/_images/membrane" + os.sep + cellid + '.tif', memb_img)
                    imwrite(filename + "/_images/dna" + os.sep + cellid + '.tif', dna_img)
                    imsave(filename + "/_images/crops" + os.sep + cellid + '.png', img)
                    """
                    
                    imsave(filename + "/_images" +
                           os.sep + cellid + '.png', img)
                    lin = '<tr><td>' + cellid + '</td><td><img src="./' + '_images/' + \
                          cellid + '.png" alt="pic" width="200"/></td>'

                    count += 1

                    for stat in self.keys:
                        lbl, digits = stat
                        number = ("{0:." + str(digits) + "f}").format(cell.stats[lbl])
                        number = str(Decimal(number))
                        number = number.rstrip("0").rstrip(".") if "." in number else number
                        lin = lin + '</td><td>' + number

                    lin += '</td></tr>\n'
                    selects.append(lin)

                elif cell.selection_state == 0:
                    cellid = str(int(cell.label))
                    img = img_as_float(cell.image)
                    imsave(filename + "/_rejected_images" +
                           os.sep + cellid + '.png', img)
                    lin = '<tr><td>' + cellid + '</td><td><img src="./' + '_rejected_images/' + \
                          cellid + '.png" alt="pic" width="200"/></td>'

                    count2 += 1

                    for stat in self.keys:
                        lbl, digits = stat
                        number = ("{0:." + str(digits) + "f}").format(cell.stats[lbl])
                        number = str(Decimal(number))
                        number = number.rstrip("0").rstrip(".") if "." in number else number
                        lin = lin + '</td><td>' + number

                    lin += '</td></tr>\n'
                    rejects.append(lin)

            print("Selected Cells: " + str(count))
            print("Rejected Cells: " + str(count2))

            report.append(
                "\n<h1>napari-eHooke Report - <a href='TODO' target='_blank'>wiki</a></h1>")

            report.append("\n<h3>Total cells: " + str(count + count2) + "</h3>")
            report.append("\n<h3>Selected cells: " + str(count) + "</h3>")
            report.append("\n<h3>Rejected cells: " + str(count2) + "</h3>")

            if params['classify_cell_cycle']:
                p1count = 0
                p2count = 0
                p3count = 0

                for cell in cells:

                    if cell.selection_state == 1:
                        if cell.stats["Cell Cycle Phase"] == 1:
                            p1count += 1
                        elif cell.stats["Cell Cycle Phase"] == 2:
                            p2count += 1
                        elif cell.stats["Cell Cycle Phase"] == 3:
                            p3count += 1

                report.append("\n<h3>Phase 1 cells: " + str(p1count) + "</h3>")
                report.append("\n<h3>Phase 2 cells: " + str(p2count) + "</h3>")
                report.append("\n<h3>Phase 3 cells: " + str(p3count) + "</h3>")
            
            params['units'] = 'um' # TODO
            if params['units'] == "um":
                report.append(
                    "\n<h3>Pixel size: " + str(params['pixel_size']) + " x " + str(params['pixel_size']) + " " + "\u03BC" + "m" + "</h3>")
            else:
                report.append(
                    "\n<h3>Pixel size: " + str(params['pixel_size']) + " x " + str(params['pixel_size']) + " " + params['units'] + "</h3>")

            if len(selects) > 1:
                report.extend(selects)
                report.append('</table>\n')

            if len(rejects) > 1:
                report.extend(rejects)
                report.append('</table>\n')

            report.append('</body>\n</html>')

        open(filename + '/html_report_' + '.html', 'w', encoding="utf-16").writelines(report)

    def check_filename(self, filename):
        if os.path.exists(filename):
            tmp = ""
            split_path = filename.split("_")
            tmp = "_".join(split_path[:len(split_path) - 1])
            tmp += "_" + str(int(split_path[-1]) + 1)
            return self.check_filename(tmp)

        else:
            return filename

    def generate_report(self, path, report_id=None):
        if report_id is None:
            filename = path + "/Report_1"
            filename = self.check_filename(filename)
            self.cell_data_filename = filename

            if not os.path.exists(filename + "/_images"):
                os.makedirs(filename + "/_images")
                os.makedirs(filename + "/_images/membrane")
                os.makedirs(filename + "/_images/dna")
                os.makedirs(filename + "/_images/crops")
            if not os.path.exists(filename + "/_rejected_images"):
                os.makedirs(filename + "/_rejected_images")
        else:
            filename = path + "/Report_" + report_id + "_1"
            filename = self.check_filename(filename)
            self.cell_data_filename = filename

            if not os.path.exists(filename + "/_images"):
                os.makedirs(filename + "/_images")
                os.makedirs(filename + "/_images/membrane")
                os.makedirs(filename + "/_images/dna")
                os.makedirs(filename + "/_images/crops")
            if not os.path.exists(filename + "/_rejected_images"):
                os.makedirs(filename + "/_rejected_images")

        self.html_report(filename)

        # TODO add view of selected cells
        # TODO SAVE PARS
