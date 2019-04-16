"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE.md at https://github.com/nv-tlabs/meta-sim.
Authors: Amlan Kar, Aayush Prakash, Ming-Yu Liu, Eric Cameracci, Justin Yuan, Matt Rusiniak, David Acuna, Antonio Torralba and Sanja Fidler
"""

import os
import json
import argparse

import utils.io as io
from data.generator import get_generator

def get_args():
  parser = argparse.ArgumentParser()

  parser.add_argument('--config', type=str, 
    required=True)

  return parser.parse_args()

def generate_data(config):
  attr = config['attributes']
  generator_class = get_generator(attr['dataset'])
  generator = generator_class(config)

  # vars and housekeeping
  out_dir = attr['output_dir'] 
  n_samples = attr['num_samples']
  
  # out directory
  io.makedirs(out_dir)
  io.write_json(config, os.path.join(out_dir, 'config.json'))

  # generate
  io.generate_data(generator, out_dir, n_samples)

if __name__ == '__main__':
  args = get_args()
  config = io.read_json(args.config)
  generate_data(config)