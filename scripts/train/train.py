"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import argparse
from tqdm import tqdm

import utils
import utils.io as io
from data.loaders import get_loader, get_scene_graph_loader
from models.tasknet import get_tasknet
from models.metasim import MetaSim
from models.layers.render import RenderLayer
from models.layers.mmd import MMDInception

class Trainer(object):
  def __init__(self, opts):
    self.opts = opts
    self.device = opts['device']

    # Logdir
    self.logdir = os.path.join(opts['logdir'],
      opts['exp_name'], opts['variant_name'])
    io.makedirs(self.logdir)

    # Set seeds
    rn = utils.set_seeds(opts['seed'])

    self.model = MetaSim(opts).to(self.device)
    self.generator = self.model.generator

    tasknet_class = get_tasknet(opts['dataset'])
    self.tasknet = tasknet_class(opts['task']).to(
      self.opts['task']['device'])

    # Data
    sgl = get_scene_graph_loader(opts['dataset']) 
    self.scene_graph_dataset = sgl(
      self.generator, 
      self.opts['epoch_length'])

    # Rendering layer
    self.renderer = RenderLayer(self.generator, 
      self.device)

    # MMD
    self.mmd = MMDInception(device=self.device, 
      resize_input=self.opts['mmd_resize_input'], 
      include_image=False, dims=self.opts['mmd_dims'])

    dl = get_loader(opts['dataset'])
    self.target_dataset = dl(self.opts['task']['val_root'])
    # In the paper, this is different
    # than the data used to get task net acc.
    # Keeping it the same here for simplicity to 
    # reduce memory overhead. To do this correctly,
    # generate another copy of the target data
    # and use it for MMD computation. 

    # Optimizer
    self.optimizer = torch.optim.Adam(
      self.model.parameters(),
      lr = opts['optim']['lr'],
      weight_decay = opts['optim']['weight_decay']
    )

    # LR scheduler
    self.lr_sched = torch.optim.lr_scheduler.StepLR(
      self.optimizer,
      step_size = opts['optim']['lr_decay'],
      gamma = opts['optim']['lr_decay_gamma']
    )

  def train_reconstruction(self):
    loader = torch.utils.data.DataLoader(self.scene_graph_dataset, 
      opts['batch_size'], num_workers=0,
      collate_fn=self.scene_graph_dataset.collate_fn)

    for e in range(self.opts['reconstruction_epochs']):
      for idx, (g, x, m, adj) in enumerate(loader):
        # g: scene graph, x: encoded features, m : mutability mask, adj: adjacency matrix
        x, m, adj = x.float().to(self.device), m.float().to(self.device),\
          adj.float().to(self.device)
        dec, dec_act = self.model(x, adj) 
        # no sampling here

        cls_log_prob = F.log_softmax(dec[..., :self.model.num_classes], dim=-1)
        cls_loss = -torch.mean(torch.sum(
          cls_log_prob * x[..., :self.model.num_classes], 
          dim=-1))
        cls_loss *= self.opts['weight']['class']
        # negative log likelihood

        cont_loss = F.mse_loss(dec_act[..., self.model.num_classes:], 
          x[..., self.model.num_classes:])

        loss = cls_loss + cont_loss

        if idx % 50 == 0: 
          print(f'[Reconstruction] Epoch{e:4d}, Batch{idx:4d}, '
                f'Class Loss {cls_loss.item():0.5f}, Cont Loss '
                f'{cont_loss.item():0.5f}')

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    del(loader)
    return

  def train(self):
    if self.opts['train_reconstruction']:
      self.train_reconstruction()
    
    if self.opts['freeze_encoder']:
      self.model.freeze_encoder()

    loader = torch.utils.data.DataLoader(self.scene_graph_dataset, 
      opts['batch_size'], num_workers=0,
      collate_fn=self.scene_graph_dataset.collate_fn)

    # baseline for moving average
    baseline = 0.
    alpha = self.opts['moving_avg_alpha']

    for e in range(self.opts['max_epochs']):
      # Set seeds for epoch
      rn = utils.set_seeds(e)  

      with torch.no_grad():
        # Generate this epoch's data for task net
        i = 0
        
        # datadir
        out_dir = os.path.join(self.logdir, 'datagen')
        io.makedirs(out_dir)

        for idx, (g, x, m, adj) in tqdm(enumerate(loader), desc='Generating Data'):
          x, adj = x.float().to(self.device), adj.float().to(self.device)
          # no sampling here

          dec, dec_act = self.model(x, adj)
          f = dec_act.cpu().numpy()
          m = m.cpu().numpy()
          g = self.generator.update(g, f, m)
          r = self.generator.render(g)

          for k in range(len(g)):
            img, lbl = r[k]
            out_img = os.path.join(out_dir, f'{str(i).zfill(6)}.jpg')        
            out_lbl = os.path.join(out_dir, f'{str(i).zfill(6)}.json')
            io.write_img(img, out_img)
            io.write_json(lbl, out_lbl)
            i+=1

      # task accuracy
      acc = self.tasknet.train_from_dir(out_dir)
      # compute moving average
      if e > 0:
        baseline = alpha * acc + (1-alpha) * baseline
      else:
        # initialize baseline to acc
        baseline = acc

      # Reset seeds to get exact same outputs
      rn2 = utils.set_seeds(e)
      for i in range(len(rn)):
        assert rn[i] == rn2[i], 'Random numbers generated are different'

      # zero out gradients for first step
      self.optimizer.zero_grad()

      # Train dist matching and task loss
      for idx, (g, x, m, adj) in enumerate(loader):
        x, m, adj = (x.float().to(self.device), m.float().to(self.device),
          adj.float().to(self.device))

        dec, dec_act, log_probs = self.model(x, adj, m, sample=True)
        # sample here
        
        # get real images
        im_real = torch.from_numpy(self.target_dataset.get_bunch_images(
          self.opts['num_real_images'])).to(self.device)

        # get fake images
        im = self.renderer.render(g, dec_act, m)
        # different from generator.render, this 
        # has a backward pass implemented and 
        # it calls the generator.render function in 
        # the forward pass

        if self.opts['dataset'] == 'mnist':
          # add channel dimension and repeat 3 times for MNIST
          im = im.unsqueeze(1).repeat(1,3,1,1) / 255.
          im_real = im_real.permute(0,3,1,2).repeat(1,3,1,1) / 255.

        mmd = self.mmd(im_real, im) * self.opts['weight']['dist_mmd']

        if self.opts['use_task_loss']:
          task_loss = -1 * torch.mean((acc - baseline) * log_probs)
          loss = mmd + task_loss # weighting is already done
          loss.backward()
        else:
          mmd.backward()
          self.optimizer.step()
          self.optimizer.zero_grad()

        if idx % self.opts['print_freq'] == 0:
          print(f'[Dist] Step: {idx} MMD: {mmd.item()}')
          if self.opts['use_task_loss']:
            print(f'[Task] Reward: {acc}, Baseline: {baseline}')
          # debug information
          print(f'[Feat] Step: {idx} {dec_act[0, 2, 15:].tolist()} {x[0, 2, 15:].tolist()}')
          # To debug, this index is the loc_x, loc_y, yaw of the 
          # digit in MNIST

      if self.opts['use_task_loss']:
        self.optimizer.step()
        self.optimizer.zero_grad()

      # LR scheduler step
      self.lr_sched.step()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp', required=True,
    type=str)
  opts = parser.parse_args()
  opts = io.read_yaml(opts.exp)

  trainer = Trainer(opts)
  trainer.train()
