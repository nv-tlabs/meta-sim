"""
Parts of code licensed under Apache 2.0 License from https://github.com/napsternxg/pytorch-practice/

Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.
"""
import torch
import torch.nn as nn
from models.layers.mmd.inception_v3 import InceptionV3

class MMDInception(nn.Module):
  def __init__(self, device='cuda', resize_input=True, 
    include_image=False, dims=[2048]):
    super(MMDInception, self).__init__()
    
    block_idx = []
    self.dims = dims
    self.include_image = include_image

    for dim in dims:
      assert dim in InceptionV3.BLOCK_INDEX_BY_DIM.keys()
      block_idx.append(InceptionV3.BLOCK_INDEX_BY_DIM[dim])

    print(f'Creating InceptionV3 features of dim {dims}, '
          f'with input resize {resize_input} for MMD on {device}')
    self.model = InceptionV3(
      output_blocks=block_idx, 
      resize_input=resize_input
    ).to(device)

  def get_features(self, images, batch_size=64):
    self.model.eval()
    
    n_images = images.size(0)
    n_batches = 1 + n_images // batch_size

    out_feats = []
    for i,d in enumerate(self.dims):
      out_feats.append([])

    for i in range(n_batches):
      minibatch = images[i*batch_size : min((i+1)*batch_size, n_images), ...]
      pred = self.model(minibatch)

      for i,d in enumerate(self.dims):
        p = pred[i]
        out_feats[i].append(p.view(p.size(0), -1))

    for i,d in enumerate(self.dims):
      out_feats[i] = torch.cat(out_feats[i], dim=0)
    
    if self.include_image:
      out_feats.append(images.contiguous().view(images.size(0), -1))

    return out_feats

  @staticmethod
  def compute_kernel(x, y):
    x_size = x.size(0)
    y_size = y.size(0)
    dim = x.size(1)
    x = x.unsqueeze(1) # (x_size, 1, dim)
    y = y.unsqueeze(0) # (1, y_size, dim)
    tiled_x = x.expand(x_size, y_size, dim)
    tiled_y = y.expand(x_size, y_size, dim)
    # expand does not allocate new memory
    kernel_input = (tiled_x - tiled_y).pow(2).mean(2)
      
    return torch.exp(-kernel_input) # (x_size, y_size)

  def forward(self, real, gen, is_feat=False):
    if is_feat:
      # If we cache features on a huge set and
      # want mmd on the bigger set
      x = real
      y = gen
    else:
      with torch.no_grad():
        x = self.get_features(real)
      y = self.get_features(gen)

    mmd2 = 0.0
    for i in range(len(x)):
      x_kernel = self.compute_kernel(x[i], x[i])
      y_kernel = self.compute_kernel(y[i], y[i])
      xy_kernel = self.compute_kernel(x[i], y[i])

      # shouldn't compute x_kernel for speed
      mmd2 += y_kernel.mean() - 2*xy_kernel.mean()\
        + x_kernel.mean()

    return mmd2