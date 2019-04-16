"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import glob
import torch
import cv2
import yaml
import os.path as osp
import numpy as np
import torch.utils.data as data

import utils.io as io

class MNISTLoader(data.Dataset):
  def __init__(self, root):
    self.root = root
    self.files = glob.glob(osp.join(self.root, '*.jpg'))
    self.files = [''.join(f.split('.')[:-1]) for f in self.files]

  def __getitem__(self, index):
    img = self.pull_image(index)
    target = self.pull_anno(index)

    img = img.transpose((2, 0, 1))  # convert to HWC
    img = (torch.FloatTensor(img) / 255.0)

    return img, target

  def __len__(self):
    return len(self.files)

  def pull_image(self, index):
    data_id = self.files[index]
    img = cv2.imread(data_id + '.jpg', cv2.IMREAD_COLOR)
    height, width, channels = img.shape
    assert channels == 3

    return np.expand_dims(img[:,:,0], axis=2)

  def pull_anno(self, index):
    data_id = self.files[index]
    target = io.read_json(data_id + '.json')
    return target[0]['obj_class']

  def get_bunch_images(self, num):
    assert (num < len(self),
      'Asked for more images than size of data')
    
    idxs = np.random.choice(
      range(len(self)),
      num,
      replace=False,
    )

    return np.array([self.pull_image(idx) for idx in idxs],
      dtype=np.float32)