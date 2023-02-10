import numpy as np
from osgeo import gdal

# encoding for two-class confusion matrix agreement maps
two_class_confusion_matrix_encoding = { 'TP': 1, 'FP': 2, 'TN': 3, 'FN': 4, 'NDV': 0,
                                        'np_dtype': np.uint8, 'gdal_dtype': gdal.GDT_UInt16 }
