echo "Creating data/assets/mnist"
mkdir -p data/assets/mnist/train
mkdir -p data/assets/mnist/test

wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz -O data/assets/mnist/train/images.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz -O data/assets/mnist/train/labels.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz -O data/assets/mnist/test/images.gz
wget http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz -O data/assets/mnist/test/labels.gz

echo "Unzipping"
gunzip data/assets/mnist/train/images.gz
gunzip data/assets/mnist/train/labels.gz
gunzip data/assets/mnist/test/images.gz
gunzip data/assets/mnist/test/labels.gz