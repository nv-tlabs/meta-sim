"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import json
import random
import numpy as np
import networkx as nx

class Generator(object):
  """
  Generic generator class
  """
  def __init__(self, config, 
    renderer=False,
    features=False):
    self.config = config
    self.has_renderer = renderer
    self.has_features = features
    if renderer:
      self._init_renderer()
    if features:
      self._init_features()

  def sample(self):
    """
    Samples a graph according to the config
    and returns a networkX graph
    """
    cfg = self.config
    graph = nx.Graph()

    # current front to be explored
    # [(parent_idx, config)]
    front = [(None, cfg)]

    while len(front) > 0:
      # pop and define vars
      curr = front.pop(-1)
      curr_idx = len(graph.nodes)
      curr_parent = curr[0]
      curr_cfg = curr[1]

      # sample class
      curr_class = self._sample_class(curr_cfg['class'])
      # sample attributes (can depend on class) and add node
      curr_attr = self._sample_attributes(curr_cfg['attributes'], curr_class)
      graph.add_node(curr_idx, cls=curr_class, attr=curr_attr)

      # add edge to parent
      if curr_parent is not None:
        graph.add_edge(curr_parent, curr_idx)

      # add children to front
      children = self._sample_children(curr_cfg, curr_idx)
      front.extend(children)

    return graph

  def render(self, graphs):
    """
    graphs: a graph or a list of graphs to render
    returns: list of img, label tuples [(img, label)]
    """
    assert self.has_renderer,\
      'Need to set renderer to True if calling render'
    return self.renderer.render(graphs)
      
  def encode(self, graphs):
    """
    Encode a batch of graphs to their 
    corresponding features

    graphs: scene graphs

    Returns: [(feature, mask)], where the
    mask represents where the feature is mutable
    """

    assert self.has_features,\
      'Need to set features to True if calling encode'
    return self.features.encode(graphs)

  def adjacency_matrix(self, graphs, dense=True):
    """
    graphs: a graph or a list of graphs to render

    returns: list of adjacency matrices
    """
    assert self.has_features,\
      'Need to set features to True if calling adjacency matrix'
    return self.features.adjacency_matrix(graphs, dense)

  def update(self, graphs, features, masks):
    """
    Update features in a graph
    
    graphs: list of networkx scene graphs to be
    updated in place
    features: encoded scene graphs
    masks: indicates whether the features of a given node is mutable

    Returns [graphs]
    A set of new graphs with updated features
    """
    assert self.has_features,\
      'Need to set features to True if calling update'
    return self.features.update(graphs, features, masks)

  def _init_renderer(self):
    """
    To be overridden

    Must initialize self.renderer if the user 
    wants to conveniently call render()
    from an object of this class
    """
    raise NotImplementedError

  def _init_features(self):
    """
    To be overridden

    Must initialize self.features if the user 
    wants to conveniently call encode()
    from an object of this class
    """
    raise NotImplementedError

  def _sample_attributes(self, attr, node_class):
    """
    Can be overridden if non-standard
    attributes need to be handled
    """
    out = {}
    types = ['Deterministic', 'Gaussian', 'Uniform']

    for k in attr:
      if isinstance(attr[k], list) and attr[k][0] in types:
        t = attr[k][0]
        v = attr[k][1:]
        # v = [mean, std, max]

        if t == 'Deterministic':
          val = v[0]

        elif t == 'Gaussian':
          val = np.random.normal(v[0], v[1])

        elif t == 'Uniform':
          val = np.random.uniform(v[0], v[1])

        if k == 'yaw':
          val = val % v[2]

        out[k] = val
        out[k + '_max'] = v[2]

      else:
        out[k] = attr[k]
  
    return out

  @staticmethod
  def _sample_children(cfg, idx):
    """
    Simple function that samples number
    of children for a node given the config
    """
    if cfg['num_child'][0] == 'Deterministic':
      return [(idx, c) for c in cfg['child']]

    else:
      raise NotImplementedError

  @staticmethod
  def _sample_class(cfg):
    """
    Simple function to sample class of the node
    """
    if cfg[0] == 'Deterministic':
      return cfg[1]

    elif cfg[0] == 'Random':
      return random.choice(cfg[1])

    else:
      raise NotImplementedError

if __name__ == '__main__':
  config = json.load(open('data/generator/config/mnist.json', 'r'))
  g = Generator(config)
  x = g.sample()