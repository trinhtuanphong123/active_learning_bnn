# Active Learning for Bayesian Neural Networks with Gaussian Processes

This directory contains code for an active learning framework for Bayesian Neural Networks applied to regression tasks, based on a paper by Tsymbalov et al. [1]. It also includes a generalization with faster runtime (the Fast GPA Sampler).

## (very short) Introduction

As example data, the housing dataset from sklearn is used, which is a well-known regression problem. The aim is to train a Bayesian Neural Network with low RMSE while minimizing the number of training data points.

After a Bayesian Neural Network is trained on an initial set of points, Active Learning iterations are performed. In each iteration, a Sampler selects points from a pool (without having access to the labels of these points) which then get added to the training data before the training process is resumed.

For further details please consider the paper [1].

Two example scripts are provided which showcase:

1. That the GPA Sampler from [1] is superior to randomly selecting additional points.
2. That the Fast GPA Sampler computes the same points as the GPA Sampler from [1] while reducing the runtime.

## Usage

To run the experiments, it is recommended to set up a virtualenv with python3.8 and install the requirements via

```
virtualenv --python=python3.8 venv
source venv/bin/activate
pip install -r requirements.txt
```

Afterwards the two example scripts can be called via

```
python source/compare_gpa_rand.py
python source/compare_fastgpa_batchgpa.py
```

The first script takes about 10min to run, the second less than 2min. Both create a results directory which contains a logfile, experiment configurations as well as a plot with the most important metrics.

The first script compares the GPA Sampler from [1] to random point selection. The results depend on the chosen random seed but in most cases using the GPA Sampler leads to a faster and more stable convergence.

The second script compares the Fast GPA Sampler to the GPA Sampler (from [1]). Both Samplers in theory compute the same posterior variance, however, sometimes differences in the convergence occur from rounding errors. In general, the Fast version of the sampler does the same job in shorter time as it avoids the repeated inversion of the posterior covariance matrix.

To change the experimental setup, consider changing the parameter values (in particular random seed and train/pool/test sizes) in the python scripts and the .yaml configuration files in `configs/`.

## Example Results from source/compare_gpa_rand.py

![Example Results](images/result_script1.png)

## Example Results from source/compare_fastgpa_batchgpa.py

![Example Results](images/result_script2.png)

### References

- [1] E. Tsymbalov, S. Makarychev, A. Shapeev, M. Panov, *Deeper Connections between Neural Networks and Gaussian Processes Speed-up Active Learning*, 2019, <https://arxiv.org/abs/1902.10350>

- the code contained in `source/models/` is modified from:

    [2] D. Hafner, D. Tran, T. Lillicrap, A. Irpan, J. Davidson, *Noise Contrastive Priors for Functional Uncertainty*, arXiv eprint 1807.09289, 2018, <https://github.com/brain-research/ncp>

- the AttrDict class in `source/classes/attrdict.py` is taken from:

    Danijar Hafner, *Patterns for Fast Prototyping with TensorFlow*, blog post, <https://danijar.com/patterns-for-fast-prototyping-with-tensorflow/>

## Original Project

This repository is adapted from the original project by Lukas Erlenbach: <https://github.com/LukasErlenbach/active_learning_bnn>