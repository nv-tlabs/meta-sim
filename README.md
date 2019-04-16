# Meta-Sim: Learning to Generate Synthetic Datasets

PyTorch code for Meta-Sim (ICCV 2019). For technical details, please refer to:

**Meta-Sim: Learning to Generate Synthetic Datasets**  
[Amlan Kar](http://www.cs.toronto.edu/~amlan/), [Aayush Prakash](https://www.linkedin.com/in/aayush-prakash-0798142b/), [Ming-Yu Liu](http://mingyuliu.net/), [Eric Cameracci](https://www.linkedin.com/in/eric-cameracci-b926505a), [Justin Yuan](https://www.linkedin.com/in/justin-yuan), [Matt Rusiniak](https://www.linkedin.com/in/mrusiniak), [David Acuna](http://www.cs.toronto.edu/~davidj/), [Antonio Torralba](http://web.mit.edu/torralba/www/), [Sanja Fidler](http://www.cs.toronto.edu/~fidler/)\
ICCV, 2019 (Oral)\
**[[Paper](https://arxiv.org/abs/1904.11621)] [[Video](./docs/resources/meta-sim-video.mp4)] [[Project Page](https://nv-tlabs.github.io/meta-sim/)]**

**Abstract:**
Training models to high-end performance requires availability of large labeled datasets, which are expensive to get. The goal of our work is to automatically synthesize labeled datasets that are relevant for a downstream task. We propose Meta-Sim, which learns a generative model of synthetic scenes, and obtain images as well as its corresponding ground-truth via a graphics engine. We parametrize our dataset generator with a neural network, which learns to modify attributes of scene graphs obtained from probabilistic scene grammars, so as to minimize the distribution gap between its rendered outputs and target data. If the real dataset comes with a small labeled validation set, we additionally aim to optimize a meta-objective, i.e. downstream task performance. Experiments show that the proposed method can greatly improve content generation quality over a human-engineered probabilistic scene grammar, both qualitatively and quantitatively as measured by performance on a downstream task. 

**Note:** This codebase is a reimplementation of Meta-Sim, and currently has the MNIST experiments from the paper. Some practices (eg: testing by generating a static final dataset and training task network offline, creating separate validation data (used by task network) and testing data (used to report numbers) for the target distribution) are omitted for simplicity of code use and understanding. Comments are provided at appropriate locations for interested users, and the changes required should be simple. 

### Citation
If you use this code, please cite:
```
@inproceedings{kar2019metasim,
title={Meta-Sim: Learning to Generate Synthetic Datasets},
author={Kar, Amlan and Prakash, Aayush and Liu, Ming-Yu and Cameracci, Eric and Yuan, Justin and Rusiniak, Matt and Acuna, David and Torralba, Antonio and Fidler, Sanja},
booktitle={ICCV},
year={2019}
}
```

### Environment Setup
All the code has been run and tested on Ubuntu 16.04, Python 3.7 with NVIDIA Titan V GPUs

- Clone repository
```
git clone git@github.com:nv-tlabs/meta-sim.git
cd meta-sim
```

- Setup python environment
```
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
export PYTHONPATH=$PWD:$PYTHONPATH
```

- Download assets
```
./scripts/data/download_assets.sh
```

- Create target data
```
python scripts/data/generate_dataset.py --config data/generator/config/mnist_val.json
python scripts/data/generate_dataset.py --config data/generator/config/bigmnist_val.json
```

### Training
First, define an experiment file, such as [mnist_rot.yaml](experiments/mnist_rot.yaml). Then, run train.py as,
```
# For MNIST rotation of digits experiment
python scripts/train/train.py --exp experiments/mnist_rot.yaml
```

Synthetic images generated for a training epoch for the task net should be available in the {logdir} inside the appropriate experiment directory. The model should slowly learn to rotate digits and they look like this across time:
<img src = "./docs/mnist-rot.gif" width="50%" margin="auto"/>

**Getting Started:** To get your hands dirty, [train.py](scripts/train/train.py) is the appropriate location.

**Tips:** 

- Training with the task-loss is slow, with one gradient update for a lot of computation. For larger experiments, we train with just MMD first, and finetune with the task loss. Here, both are set to be on by default. Depending on initialization, sometimes training might take a long time to converge, but in our experience, it eventually always converges.
- Sometimes, it is important to have enough target data for distribution matching to work properly. Here, for example we generate 1000 examples synthetically to use as target data, which sometimes might be not enough due to randomness in how diverse the generated data is. Try increasing the size if you face issues by modifying the appropriate config file used by the data generation script. 