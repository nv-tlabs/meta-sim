"""
Adapted from https://github.com/tkipf/pygcn with MIT License

The MIT License

Copyright (c) 2017 Thomas Kipf

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import math
import torch
import torch.nn as nn

class GraphConvolution(nn.Module):
    """
    Adapted from Thomas Kipf's pygcn

    GCN layer that works on a directed graph
    Uses different layers for forward edges (along digraph edges),
    for backward edges (reversing the digraph edges) and
    for self connections (adjacency matrix does not have self loops)

    Additionally scales each of these differently using
    learned weights
    """
    def __init__(self, in_features, out_features, bias=True):
      super(GraphConvolution, self).__init__()
      self.in_features = in_features
      self.out_features = out_features
      
      self.self_connection = nn.Linear(in_features, 
        out_features, bias=bias)
      self.scale_self_conn = nn.Linear(in_features, 
        1, bias=True)

      self.forward_edge = nn.Linear(in_features, 
        out_features, bias=bias)
      self.scale_forward = nn.Linear(in_features, 
        1, bias=True)

      self.back_edge = nn.Linear(in_features, 
        out_features, bias=bias)
      self.scale_back = nn.Linear(in_features, 
        1, bias=True)

      self.reset_parameters()

    @staticmethod
    def _get_in_degrees(adj, add_identity=False):
      deg = torch.transpose(adj, 1, 2).sum(2).unsqueeze(2)
      if add_identity:
        deg += torch.ones_like(deg)
      return deg

    def reset_parameters(self):
      for m in self.modules():
        if isinstance(m, nn.Linear):
          stdv = 1. / math.sqrt(m.weight.size(1))
          nn.init.uniform_(m.weight, -stdv, stdv)
          if m.bias is not None:
            nn.init.uniform_(m.bias, -stdv, stdv)

    def forward(self, inp, adj):
      """
      inp: [batch_size, num_nodes, feature_length]
      adj: [batch_size, num_nodes, num_nodes]
      """
      support = self.self_connection(inp)
      output = self.forward_edge(inp)
      rev = self.back_edge(inp)
      
      # Edge wise gates. Should result in a single scalar per edge. 
      scale_self_conn = torch.sigmoid(self.scale_self_conn(inp))
      scale_forward = torch.sigmoid(self.scale_forward(inp))
      scale_back = torch.sigmoid(self.scale_back(inp))

      scaled_support = torch.mul(scale_self_conn, support)
      scaled_output = torch.mul(scale_forward, output)
      scaled_rev = torch.mul(scale_back, rev)
        
      out_neighbor_features = torch.bmm(adj, scaled_output)
      in_neighbor_features = torch.bmm(torch.transpose(adj, 1, 2), scaled_rev)

      # Normalize by in-degree, when in-degree is > 1
      in_degree_adj = self._get_in_degrees(adj)
      in_degree_adj = torch.where(in_degree_adj > 0, 
        in_degree_adj, torch.ones_like(in_degree_adj))
      in_degree_adj_transpose = self._get_in_degrees(torch.transpose(adj, 1, 2))
      in_degree_adj_transpose = torch.where(in_degree_adj_transpose > 0, 
        in_degree_adj_transpose, torch.ones_like(in_degree_adj))

      normalized_out_neighbor_features = torch.div(out_neighbor_features, 
        in_degree_adj)
      normalized_in_neighbor_features = torch.div(in_neighbor_features, 
        in_degree_adj_transpose)

      return scaled_support + normalized_out_neighbor_features + \
        normalized_in_neighbor_features

    def __repr__(self):
      return self.__class__.__name__ + ' (' \
        + str(self.in_features) + ' -> ' \
        + str(self.out_features) + ')'