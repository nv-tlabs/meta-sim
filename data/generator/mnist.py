"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import json
import random

from data.generator.generator import Generator
from data.renderer.mnist import MNISTRenderer
from data.features.mnist import MNISTFeatures

class MNISTGenerator(Generator):
  """
  Domain specific generator definition
  Inherits the generic Generator and overrides
  core functions as required
  """
  def __init__(self, config):
    super(MNISTGenerator, self).__init__(config, 
      renderer=True,
      features=True
    )

  def _init_features(self):
    self.features = MNISTFeatures(self.config)

  def _init_renderer(self):
    self.renderer = MNISTRenderer(self.config)

  def _sample_attributes(self, attr, node_class):
    """
    Overriding to ensure textures are properly done
    since each class has multiple possible textures
    and we will need to sample one
    """
    out = super(MNISTGenerator, self)._sample_attributes(attr, node_class)
    # First run to get standard attributes

    # Handle special case of node textures
    if node_class in range(10):
      # HACK: if it is a digit -- find better way of figuring
      # this out

      # Textures are available in the renderer class 
      # for this case

      if out['texture'] == 'Random':
        out['texture'] = random.choice(
          self.renderer.data[node_class])
      
      elif out['texture'] == 'Deterministic':
        out['texture'] = self.renderer.data[node_class][0]

      else:
        raise NotImplementedError

    return out

if __name__ == '__main__':
  import skimage.io as sio
  config = json.load(open('data/generator/config/mnist.json', 'r'))
  g = MNISTGenerator(config)
  x = g.sample()
  f = g.encode(x)
  a = g.adjacency_matrix(x)
  f[0][0][2, -1] = 0.8
  x = g.update(x, f)