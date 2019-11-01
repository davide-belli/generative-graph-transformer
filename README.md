
# Generative Graph Transformer
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
PyTorch implementation of <i>Image-Conditioned Graph Generation for Road Network Extraction</i> (https://arxiv.org/abs/1910.14388)


## Overview
This library contains a PyTorch implementation of the Generative Graph Transformer (GGT): an autoregressive, attention-based model for image-to-graph generation as presented in [[1]](#citation)(https://arxiv.org/abs/1910.14388), in addition to other baselines discussed in the paper.
Find out more about this project in our [blog post](https://davide-belli.github.io/generative-graph-transformer).

## Dependencies

See [`requirements.txt`](https://github.com/davide-belli/generative-graph-transformer/tree/master/requirements.txt)

* **scipy==1.2.1**
* **scikit_image==0.15.0**
* **numpy==1.14.2**
* **seaborn==0.9.0**
* **networkx==1.11**
* **torch==1.1.0**
* **matplotlib==2.2.2**
* **Pillow==6.1.0**
* **skimage==0.0**
* **tensorboardX==1.8**
* **torchvision==0.4.0**

## Structure
* [`data/`](https://github.com/davide-belli/generative-graph-transformer/tree/master/data): Should contain the Toulouse Road Network dataset. If you run `download_dataset.sh` the script will download the dataset introduced in our paper (Toulouse Road Network dataset).
* [`models/`](https://github.com/davide-belli/generative-graph-transformer/tree/master/models): Contains the implementation of encoder and decoder models and baselines discussed in the paper, including but not only: GGT, GraphRNN extended to node features, simple RNN, simple MLP.
* [`metrics/`](https://github.com/davide-belli/generative-graph-transformer/tree/master/metrics): Class for StreetMover distance in : `streetmover_distance.py`. Also contains different methods to compute statistics for the evaluation the models.
* [`utils/`](https://github.com/davide-belli/generative-graph-transformer/tree/master/utils): Contains hyper-parameter configuration for the different models, the dataset class and other utils.
* [`main.py`](https://github.com/davide-belli/generative-graph-transformer/blob/master/main.py): Main script for training and testing of all the models.
* [`arguments.py`](https://github.com/davide-belli/generative-graph-transformer/blob/master/arguments.py): Configuration specifying which model, experiment and hyper-parameter setting to be used in `main.py`.
* [`pretrain_encoder.py`](https://github.com/davide-belli/generative-graph-transformer/blob/master/pretrain_encoder.py): Script to pre-train the CNN encoder for image reconstruction as part of an auto-encoder.

## Usage
- First download Toulouse Road Network dataset using [`data/download_dataset.sh`](https://github.com/davide-belli/generative-graph-transformer/blob/master/data/download_dataset.sh).
- Then, configure [`arguments.py`](https://github.com/davide-belli/generative-graph-transformer/blob/master/arguments.py) to choose which model to train/test and finally run [`main.py`](https://github.com/davide-belli/generative-graph-transformer/blob/master/main.py).
- The output plots, logs, tensorboard and statistics will be automatically generated in `output_graph/`

Find out more about this project in our [blog post](https://davide-belli.github.io/generative-graph-transformer).
Please cite [[1](#citation)] in your work when using this library in your experiments.

## Feedback
For questions and comments, feel free to contact [Davide Belli](mailto:davidebelli95@gmail.com).

## Citation
```
[1] Belli, Davide and Kipf, Thomas (2019). Image-Conditioned Graph Generation for Road Network Extraction. NeurIPS 2019 workshop on Graph Representation Learning.
```

BibTeX format:
```
@article{belli2019image,
  title={Image-Conditioned Graph Generation for Road Network Extraction},
  author={Belli, Davide and Kipf, Thomas},
  journal={NeurIPS 2019 workshop on Graph Representation Learning},
  year={2019}
}

```

## Copyright

Copyright © 2019 Davide Belli.

This project is distributed under the <a href="LICENSE">MIT license</a>. This was developed as part of a master thesis supervised by [Thomas Kipf](https://tkipf.github.io/) at the University of Amsterdam, and presented as a paper at the [Graph Representation Learning workshop in NeurIPS 2019](https://grlearning.github.io/papers/), Vancouver, Canada.
