"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import torch
from models.layers.render.render import RenderTorch

class RenderLayer(object):
  """
  Wrapper for the rendering layer
  for easy API
  """
  def __init__(self, generator, device):
    self.generator = generator
    self.device = device
    self.layer = RenderTorch()

  def render(self, graphs, features, masks):
    return self.layer.apply(graphs, features, masks, 
      self.generator, self.device)