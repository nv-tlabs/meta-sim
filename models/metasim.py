"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import torch
import torch.nn as nn
import numpy as np

from utils.io import read_json
from models.gcn import GCN
from data.generator import get_generator

class MetaSim(nn.Module):
  def __init__(self, opts):
    super(MetaSim, self).__init__()
    self.opts = opts
    self.cfg = read_json(opts['config'])
    
    g = get_generator(opts['dataset'])
    self.generator = g(self.cfg)

    self.init_model()

  def init_model(self):
    self.get_feature_length()

    self.encoder = GCN([self.in_feature_len, 30, 18])
    self.decoder = GCN([18, 30, self.in_feature_len])

  def get_feature_length(self):
    """
    Samples a graph and encodes it to get the 
    feature length
    """
    x = self.generator.sample()
    f = self.generator.encode(x)

    self.num_classes = len(self.generator.features.classes)
    self.in_feature_len = f[0][0].shape[-1]
    # first 0 for batch, second 0 for features
    # f[.][1] is the mask denoting mutability

  def freeze_encoder(self):
    for n,p in self.encoder.named_parameters():
      p.requires_grad = False

  def forward(self, x, adj, masks=None, sample=False):
    enc = self.encoder(x, adj)
    dec = self.decoder(enc, adj)

    # apply sigmoid on all continuous params
    # softmax on classes
    dec_act = torch.cat((
      torch.softmax(dec[..., :self.num_classes], dim=-1),
      torch.sigmoid(dec[..., self.num_classes:])
    ), dim=-1)

    if sample:
      assert masks is not None, 'Need masks for sampling'
      dec_act, log_probs = self.sample(dec_act, masks)

      return dec, dec_act, log_probs

    return dec, dec_act

  def sample(self, features, masks, sigma=None):
    """
    Takes in predicted features (dec_act), masks
    for mutable nodes

    Returns new features and a log_prob for
    each batch element
    """
    if sigma is None:
      sigma = torch.tensor(0.02).to(self.opts['device'])
      
    m = masks.cpu().numpy()
    log_probs = torch.zeros(features.shape[0], device=features.device,
      dtype=torch.float32)
    
    for b in range(features.shape[0]):
      lp = 0
      idxs = np.array(np.nonzero(m[b])).transpose()
      for idx in idxs:
        mu = features[b, idx[0], idx[1]]
        n = torch.distributions.normal.Normal(mu, sigma)
        sample = n.rsample()
        features[b, idx[0], idx[1]] = sample
        lp += n.log_prob(sample)

      log_probs[b] = lp

    return features, log_probs