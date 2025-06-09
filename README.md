# Introduction

Code for replication of experiments in the paper:

Inference on Strongly Identified Functionals of Weakly Identified Functions

Andrew Bennett, Nathan Kallus, Xiaojie Mao, Whitney Newey, Vasilis Syrgkanis, Masatoshi Uehara

https://arxiv.org/abs/2208.08291

# Installation

Before running any script you need to install the mliv package by running:

```
python setup.py develop
```

The main dependencies of the package are, cvxopt, scikit-learn, numpy, pytorch.

cvxopt is installed by running the setup.py command above. The rest need to be installed independently.

# Replication

To replicate the tables in the paper related to neural network experiments run:
```
cd notebooks
python all_experiments.py
```
Once the code finishes, then execute the jupyter notebook `notebooks\Postprocess.ipynb`


To replicate the tables in the paper related to RKHS experiments run the notebook: `notebooks\RKHS.ipynb`