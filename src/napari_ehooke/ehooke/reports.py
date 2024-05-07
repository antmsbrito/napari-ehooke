"""Module used to create the report of the cell identification"""
import pandas as pd
import matplotlib as mpl
from skimage.io import imsave
from skimage.util import img_as_float, img_as_uint, img_as_ubyte
from skimage.filters import threshold_isodata
from skimage.color import gray2rgb
from decimal import Decimal
import numpy as np
import os
from tifffile import imwrite

from .cellprocessing import stats_format

class ReportManager:

    def __init__(self, parameters,properties,allcells):
        self.cells = allcells
        self.properties = properties
        self.params = parameters
        self.keys = stats_format(parameters)

        self.cell_data_filename = None

    def html_report(self, filename):
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

            print("Total Cells: " + str(len(cells)))

            for cell in cells:
                
                cellid = str(int(cell.label))

                img = img_as_ubyte(cell.image)

                imsave(filename+"/_images"+os.sep+cellid+'.png',img)

                lin = '<tr><td>' + cellid + '</td><td><img src="./_images/'+cellid+'.png" alt="pic" width="200"/></td>'


                for stat in self.keys:
                    lbl, digits = stat
                    number = ("{0:." + str(digits) + "f}").format(cell.stats[lbl])
                    number = str(Decimal(number))
                    number = number.rstrip("0").rstrip(".") if "." in number else number
                    lin = lin + '</td><td>' + number

                lin += '</td></tr>\n'
                selects.append(lin)


            report.append(
                "\n<h1>napari-eHooke Report - <a href='TODO' target='_blank'>wiki</a></h1>")

            report.append("\n<h3>Total cells: " + str(len(self.properties['label'])) + "</h3>")

            if self.params['classify_cell_cycle']:
                _,pcounts=np.unique(self.properties['Cell Cycle Phase'], return_counts=True)

                report.append("\n<h3>Phase 1 cells: " + str(pcounts[0]) + "</h3>")
                report.append("\n<h3>Phase 2 cells: " + str(pcounts[1]) + "</h3>")
                report.append("\n<h3>Phase 3 cells: " + str(pcounts[2]) + "</h3>")
            
            if len(selects) > 1:
                report.extend(selects)
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
                #os.makedirs(filename + "/_images/membrane")
                #os.makedirs(filename + "/_images/dna")
                #os.makedirs(filename + "/_images/crops")
        else:
            filename = path + "/Report_" + report_id + "_1"
            filename = self.check_filename(filename)
            self.cell_data_filename = filename

            if not os.path.exists(filename + "/_images"):
                os.makedirs(filename + "/_images")
                #os.makedirs(filename + "/_images/membrane")
                #os.makedirs(filename + "/_images/dna")
                #os.makedirs(filename + "/_images/crops")


        self.html_report(filename)

        df = pd.DataFrame(self.properties)
        df.to_csv(os.path.join(filename,f"Analysis.csv"))


        # TODO add view of selected cells
        # TODO SAVE PARS
