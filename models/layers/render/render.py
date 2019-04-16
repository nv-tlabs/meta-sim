"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import torch
import numpy as np

class RenderTorch(torch.autograd.Function):
  @staticmethod
  def forward(ctx, graphs, features, masks, generator, device):
    features = features.cpu().numpy()
    masks = masks.cpu().numpy()

    # update in place
    graphs = generator.update(graphs, features, masks)
    images = [r[0] for r in generator.render(graphs)]

    images = torch.tensor(images, dtype=torch.float32,
      device=device)

    # Save in ctx for backward
    ctx.graphs = graphs
    ctx.features = features
    ctx.masks = masks
    ctx.generator = generator
    ctx.device = device

    return images

  @staticmethod
  def backward(ctx, output_grad):
    graphs = ctx.graphs
    features = ctx.features
    masks = ctx.masks
    generator = ctx.generator
    device = ctx.device

    out_grad = torch.zeros(features.shape, dtype=torch.float32,
      device=device)

    delta = 0.03

    for b in range(features.shape[0]):
      idxs = np.array(np.nonzero(masks[b])).transpose()
      
      for idx in idxs:
        # f(x+delta)
        features[b, idx[0], idx[1]] += delta
        tmp_graph = generator.update(graphs[b], features[b],
          masks[b])
        img_plus = generator.render(tmp_graph)[0][0] / 255.0

        # f(x-delta)
        features[b, idx[0], idx[1]] -= 2*delta
        tmp_graph = generator.update(graphs[b], features[b],
          masks[b])
        img_minus = generator.render(tmp_graph)[0][0] / 255.0

        grad = ((img_plus - img_minus) / 2*delta).astype(np.float32)
        grad = torch.from_numpy(grad).to(device)

        out_grad[b, idx[0], idx[1]] = (output_grad * grad).sum()

        # back to normal
        features[b, idx[0], idx[1]] += delta
        tmp_graph = generator.update(graphs[b], features[b],
          masks[b])

    return None, out_grad, None, None, None
