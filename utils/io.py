"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import os
import json
import yaml
import warnings
import shutil
import skimage.io as sio
from tqdm import tqdm

def generate_data(generator, out_dir, num_samples):
  """
  Generate data and save to an output directory

  generator: an object of the generator class
  out_dir: target to save data to
  num_samples: number of samples to generate
  """
  for i in tqdm(range(num_samples), desc='Generating data'):
    out_img = os.path.join(out_dir, f'{str(i).zfill(6)}.jpg')        
    out_lbl = os.path.join(out_dir, f'{str(i).zfill(6)}.json')
    # are you really making more than .zfill(6) can handle!
    g = generator.sample()
    r = generator.render(g)
    img, lbl = r[0]
    
    # write to disk
    write_img(img, out_img)
    write_json(lbl, out_lbl)

def makedirs(out_dir):
  if os.path.isdir(out_dir):
    warnings.warn(f'Directory {out_dir} exists. Deleting!')
    shutil.rmtree(out_dir)

  os.makedirs(out_dir)

def read_json(fname):
  with open(fname, 'r') as f:
    config = json.load(f)

  return config

def write_json(config, fname):
  with open(fname, 'w') as f:
    json.dump(config, f)

def read_yaml(fname):
  with open(fname, 'r') as f:
    config = yaml.safe_load(f)

  return config

def write_img(img, fname):
  sio.imsave(fname, img)