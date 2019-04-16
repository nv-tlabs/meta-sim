"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import random
import torch
import numpy as np

def set_seeds(seed):
  np.random.seed(seed)
  random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False

  return [
    np.random.rand(1),
    torch.randn(1)
  ]