﻿"""Module used to create the report of the cell identification"""
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

        self.max_shape = np.max([cell.shape for cell in self.cells], axis=0)

        paddiffx = [(self.max_shape[0] - cell.shape[0]) for cell in self.cells]
        paddiffy = [(self.max_shape[1] - cell.shape[1]) for cell in self.cells]

        padx = [(p//2,p-p//2) for p in paddiffx]
        #pady = [(p//2,p-p//2) for p in paddiffy]

        padded_cells = [np.pad(cell, [(padx[idx][0],padx[idx][1]),(0,paddiffy[idx])], mode='constant',constant_values=1) for idx,cell in enumerate(self.cells)]
        self.cells = padded_cells

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

            imsave(filename+"/_images"+os.sep+'all_cells.png',img_as_ubyte(np.concatenate(cells,axis=0)))

            for idx,cell in enumerate(cells):

                lin = '<tr><td>' + str(self.properties['label'][idx]) + '</td><td><div style="width: '+str(self.max_shape[1])+'px; height: '+str(self.max_shape[0])+'px; overflow: hidden;"><img src="./_images/'+'all_cells'+'.png" alt="pic" style="width: '+str(self.max_shape[1])+'; height: auto; transform: translateY(-'+str(idx*self.max_shape[0])+'px);"></div></td>'


                for stat in self.keys:
                    lbl, digits = stat
                    number = ("{0:." + str(digits) + "f}").format(self.properties[lbl][idx])
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
