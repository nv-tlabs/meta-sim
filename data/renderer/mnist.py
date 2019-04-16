"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import json
import struct
import os
import os.path as osp
import numpy as np
from PIL import Image

class MNISTRenderer(object):
  """
  Domain specific renderer definition
  Main purpose is to be able to take a scene
  graph and return its corresponding rendered image
  """
  def __init__(self, config):
    self.config = config
    self._init_data()

  def _init_data(self):
    asset_dir = self.config['attributes']['asset_dir']
    self.data = {}

    with open(osp.join(asset_dir,'labels'), 'rb') as flbl:
      _, size = struct.unpack('>II', flbl.read(8))
      lbls = np.fromfile(flbl, dtype=np.int8)

    with open(osp.join(asset_dir,'images'), 'rb') as fimg:
      _, size, rows, cols = struct.unpack('>IIII', fimg.read(16))
      imgs = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbls), rows, cols)

    for lbl in np.unique(lbls):
      idxs = (lbls == lbl)
      self.data[lbl] = imgs[idxs]

    return

  def render(self, graphs):
    """
    Render a batch of graphs to their 
    corresponding images
    """
    if not isinstance(graphs, list): 
      graphs = [graphs]
    
    # should use multiprocessing here
    # but then won't be able to use renderer
    # objects from inside another class
    return [self._render(g) for g in graphs]

  def _render(self, graph):
    """
    Render a single graph into its 
    corresponding image
    """
    # vars
    labels = []
    size = self.config['attributes']['size']
    bg_idx = list(filter(
      lambda i: graph.node[i]['cls'] ==  'Background',
      range(len(graph.nodes))))
    bg = graph.node[bg_idx[0]]['attr']['texture']
    out_img = np.zeros(size, dtype=np.uint8)

    # HACK: Should be a better way to find if a node
    # is a digit
    digit_idxs = list(filter(
      lambda i: graph.node[i]['cls'] in range(10),
      range(len(graph.nodes))))

    # Add bg
    out_img[out_img == 0] = max(bg, 0)

    # Add digits
    for i in digit_idxs:
      x = int(graph.node[i]['attr']['loc_x'])
      y = int(graph.node[i]['attr']['loc_y'])
      digit_cls = graph.node[i]['cls']
      texture = graph.node[i]['attr']['texture']

      # rotate the object
      img = Image.fromarray(texture)
      yaw = graph.node[i]['attr']['yaw'] % 360
      rimg = img.rotate(yaw, expand=True)

      img = np.array(rimg)
      width, height = img.shape[1], img.shape[0]
      xmin, ymin = x - width // 2, y - height // 2
      xmax, ymax = xmin + width, ymin + height

      # out of bounds
      if xmax <= 0 or xmin >= size[1] or ymax <= 0 or ymin >= size[0]:
          labels.append([-1, -1, -1, -1, -1])
      else:
          if (xmin < 0):
              img = img[:, -xmin:]
              xmin = 0
          if (ymin < 0):
              img = img[-ymin:, :]
              ymin = 0

          if (xmax > size[1]):
              img = img[:, 0:size[1] - xmin]
              xmax = size[1]
          if (ymax > size[0]):
              img = img[0:size[0] - ymin, :]
              ymax = size[0]

          labels.append({
              'obj_class': digit_cls,
              'yaw': yaw,
              'bbox': [xmin, ymin, xmax, ymax]
          })

          out_img[ymin:ymax, xmin:xmax] = img | out_img[ymin:ymax, xmin:xmax]

    return out_img, labels

if __name__ == '__main__':
  config = json.load(open('data/generator/config/mnist.json', 'r'))
  re = Renderer(config)