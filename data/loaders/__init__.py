"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

from data.loaders.mnist import MNISTLoader
from data.loaders.scene_graph import SceneGraphLoader

def get_loader(name):
  if name == 'mnist':
    return MNISTLoader

def get_scene_graph_loader(name):
  """
  Change here if implementing custom
  SceneGraphLoaders
  """
  return SceneGraphLoader
