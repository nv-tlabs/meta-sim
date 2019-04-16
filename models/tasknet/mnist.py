"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from data.loaders import get_loader

class MNISTModel(nn.Module):
  def __init__(self, opts):
    super(MNISTModel, self).__init__()
    self.opts = opts
    self.device = opts['device']

    self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=5//2)
    self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=5//2)
    self.conv2_drop = nn.Dropout2d()
    self.flat_size = 64*opts['input_dim'][0]*opts['input_dim'][1]//16
    self.fc1 = nn.Linear(self.flat_size,256)
    self.fc2 = nn.Linear(256, 100)
    self.fc3 = nn.Linear(100, 10)

    self.weights_init()
    self.optimizer = optim.SGD(self.parameters(), lr=0.01, momentum=0.5)

    self.dataset_class = get_loader(opts['dataset'])

    self.val_dataset = self.dataset_class(opts['val_root'])
    self.val_loader = torch.utils.data.DataLoader(self.val_dataset, 
      opts['batch_size'], shuffle=True, num_workers=0)

    if 'reload' in opts.keys():
        self.reload(opts['reload'])

  def weights_init(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
      if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)

  def forward(self, x):
    x = F.relu(F.max_pool2d(self.conv1(x), 2))
    x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
    x = x.view(-1, self.flat_size)
    x = F.relu(self.fc1(x))
    x = F.dropout(x, training=self.training)
    x = F.relu(self.fc2(x))
    x = F.dropout(x, training=self.training)
    x = self.fc3(x)
    return F.log_softmax(x, dim=1)

  def train_from_dir(self, root):
    self.train()

    dataset = self.dataset_class(root)
    data_loader = torch.utils.data.DataLoader(dataset,
      self.opts['batch_size'], shuffle=True, num_workers=0)

    for e in range(self.opts['epochs']):
      for idx, (img, lbl) in enumerate(data_loader):
        img = img.float().to(self.device)
        out = self.forward(img)
        loss = F.nll_loss(out, lbl.to(self.device))      
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if idx % self.opts['print_freq'] == 0:
          print(f'[TaskNet] Epoch{e:4d}, Batch{idx:4d}, '
                f'Loss {loss.item():0.5f}')

    # Call destructors
    del(dataset, data_loader)

    acc = self.test()
    return acc

  def test(self):
    self.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
      for img, target in self.val_loader:
        img, target = img.to(self.device), target.to(self.device)
        out = self.forward(img)
        pred = out.max(1, keepdim=True)[1]  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    acc = 100. * correct / len(self.val_dataset)
    print('###############################')
    print(f'[TaskNet] Accuracy: {acc:3.2f}')
    print('###############################')
    
    return acc

  def save(self, fname, acc):
    print(f'Saving task network at {fname}')
    save_state = {
        'state_dict': self.state_dict(),
        'acc': acc
    }

    torch.save(save_state, fname)

  def reload(self, fname):
    print(f'Reloading task network from {fname}')
    state_dict = torch.load(fname, map_location=lambda storage, loc: storage)
    self.load_state_dict(state_dict['state_dict'])