"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import networkx as nx
import numpy as np
import warnings

class Features(object):
  """
  Generic features class definition
  Override any function for special cases
  """
  def __init__(self, config):
    self.config = config
    if 'class' in config['attributes']['features']:
      self.classes = list(set(self._get_all_classes(config)))
    else:
      warnings.warn('Classes are not in features!')
      self.classes = []

    # TODO: this is not generic enough to put in __init__

  def encode(self, graphs):
    """
    Encode a batch of graphs to their 
    corresponding features

    graphs: scene graphs 

    Returns: [(feature, mask)], where the
    mask represents where the feature is mutable
    """
    if not isinstance(graphs, list): 
      graphs = [graphs]
    
    # should use multiprocessing here
    return [self._encode(g) for g in graphs]

  def update(self, graphs, features, masks):
    """
    Update features in a graph
    
    graphs: list of networkx graphs to be 
    updated in place

    features: encoded input scene graphs
    masks: indicates whether the features of a given node is mutable

    Returns [graphs]
    A set of new graphs with updated features
    """
    if not isinstance(graphs, list):
      graphs = [graphs]
      features = [features]
      masks = [masks]

    return [self._update(graphs[i], features[i], masks[i])\
      for i in range(len(graphs))]

  def adjacency_matrix(self, graphs, dense=True):
    """
    Return the adjacency matrix of a batch of 
    graphs

    Returns: [adj_matrix]
    """
    if not isinstance(graphs, list): 
      graphs = [graphs]

    adj = [nx.adjacency_matrix(g) for g in graphs]

    if dense:
      adj = [a.toarray() for a in adj]

    return adj

  def _update(self, graph, feature, mask):
    """
    Update a single graph with new features
    """
    for i in graph.nodes:
      n = graph.node[i]

      idx = 0
      if 'class' in self.config['attributes']['features']:
        idx += len(self.classes)

      for f in self.config['attributes']['features']:
        if f == 'class':
          continue
        else:
          if mask[i, idx]:
            # if mask is not 0
            n['attr'][f] = feature[i, idx] *\
              n['attr'][f+'_max']

          idx += 1

    return graph

  def _encode(self, graph):
    """
    Encode a single graph into its features
    """
    features = []
    mask = []
    for n in graph.nodes:
      n = graph.node[n]
      node_features = []
      node_mask = []

      if 'class' in self.config['attributes']['features']:
        tmp = [0] * len(self.classes)
        if 'class' in self.config['attributes']['mutable']:
          raise NotImplementedError

        tmp_mask = [0] * len(self.classes)
        # in this implementation classes are immutable
        idx = self.classes.index(n['cls'])
        tmp[idx] = 1
        
        node_mask.extend(tmp_mask)
        node_features.extend(tmp)    

      for f in self.config['attributes']['features']:
        if f == 'class':
          continue
        else:
          if n['attr']['immutable']:
            # if node is immutable
            tmp_mask = 0
          elif f in self.config['attributes']['mutable']:
            # if attribute is immutable
            # easy to extend to per node by adding a 
            # mutable field to every node's attributes and 
            # changing this to n['attr']['mutable']
            tmp_mask = 1
          else:
            tmp_mask = 0
          
          tmp = n['attr'][f] / n['attr'][f+'_max']

          node_mask.append(tmp_mask)
          node_features.append(tmp)

      mask.append(node_mask)
      features.append(node_features)

    return np.array(features, dtype=np.float32),\
      np.array(mask, dtype=np.float32)

  def _get_all_classes(self, config):
    """
    Recursive function to get all classes in the config
    """
    classes = []
    cs = config["class"][1]

    if isinstance(cs, list):
      classes.extend(cs)
    elif isinstance(cs, (str, int)):
      classes.append(cs)
    else:
      raise NotImplementedError

    for c in config["child"]:
      classes.extend(self._get_all_classes(c))

    return classes

if __name__ == '__main__':
  import json
  config = json.load(open('data/generator/config/mnist.json', 'r'))
  re = Features(config)