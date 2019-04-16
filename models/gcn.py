"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.layers.gcn import GraphConvolution

class GCN(nn.Module):
  def __init__(self, dims, dropout=None):
    """
    Build a GCN with L layers, with feature sizes defined by dims
    with optional dropout.

    dims[0] is the size of the input, dims[L] is the size of the output
    """
    super(GCN, self).__init__()
    self.dims = dims
    self.n_layers = len(dims) - 1
    assert self.n_layers > 0

    layers = []
    for l in range(self.n_layers):
        layers.append(GraphConvolution(dims[l], dims[l+1]))

    self.layers = nn.ModuleList(layers)
    self.dropout = dropout

  def forward(self, x, adj):
    for l in range(self.n_layers - 1):
      x = self.layers[l](x, adj)
      x = F.relu(x)
      if self.dropout is not None:
          x = F.dropout(x, self.dropout, training=self.training)

    # Final layer
    x = self.layers[-1](x, adj)

    return x

  def __repr__(self):
    dstr =  self.__class__.__name__ 
    for l in self.layers:
      dstr += '\n' + str(l) + f"\t Dropout: {str(self.dropout)}"

    return dstr

if __name__ == '__main__':
  model = GCN([50, 100, 200, 100, 50])
  print(model)
