"""Python function for showboxes()

If no plot shows up after calling this function, try:

>>> import pylab
>>> showboxes(...)
>>> pylab.show()
"""

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np


def showboxes(image, boxes, output_figure_path=None):
  """Draw bounding boxes on top of an image.

  Args:
    image:               PIL.Image object
    boxes:               A N * 4 matrix for box coordinate.
    output_figure_path:  String or None. The figure will be saved to
                         output_figure_path if not None.
  """
  figure = plt.figure()
  axis = figure.add_subplot(111, aspect='equal')
  plt.imshow(image)
  for box in boxes:
    axis.add_patch(patches.Rectangle(box[:2],
                                     box[2] - box[0],
                                     box[3] - box[1],
                                     fill=None,
                                     ec=np.asarray([240, 39, 40]) / 255.,
                                     lw=3))

  if output_figure_path is not None:
    plt.savefig(output_figure_path)
