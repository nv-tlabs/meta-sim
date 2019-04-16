"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import torch
import numpy as np
import torch.utils.data as data

class SceneGraphLoader(data.Dataset):
  def __init__(self, generator,
    length=1000):
    self.generator = generator
    self.length = 1000

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    x = self.generator.sample()
    f = self.generator.encode(x)[0]
    adj = self.generator.adjacency_matrix(x)[0]

    return x, f[0], f[1], adj
    # graph, features, mask, adjacency matrix

  @staticmethod
  def collate_fn(inp):
    out = []
    num_out = len(inp[0])
    bs = len(inp)

    for i in range(num_out):
      if isinstance(inp[0][i], np.ndarray):
        this_out = torch.from_numpy(np.stack(
          [inp[b][i] for b in range(bs)],
          axis=0))

      else:
        this_out = [inp[b][i] for b in range(bs)]
        
      out.append(this_out)

    return tuple(out)