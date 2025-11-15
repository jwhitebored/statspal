# statspal
Neural net made in TensorFlow to determine the statistical distribution governing a dataset. Intended use is to better understand the relevant statistics of a dataset, and give an informed choice of function to fit data to in standard curve fitting algorithms (non-linear regression) like SciPy's opt.curve_fit().
Intended users include students and researchers who are not *yet* enthusiastic about the statistics relevant to their field of research.

# How to Install
Create a virtual environment with Python version 3.11 (the onnx runtime requires 3.8 =< Python < 3.12). If you use conda, run

conda create -n statspal python=3.11 pip

Then activate your virtual env and in your python console, run

pip install git+https://github.com/jwhitebored/statspal.git

# Model's Accuracy 

Model attained 75% evaluation accuracy when classifying the data belonging to the following scipy.stats discrete distributions:

['bernoulli', 'betabinom', 'betanbinom', 'binom', 'boltzmann', 'dlaplace', 'geom', 'hypergeom', 'logser', 'nbinom', 'nchypergeom_fisher', 'nchypergeom_wallenius', 'nhypergeom', 'planck', 'poisson', 'poisson_binom', 'randint', 'skellam', 'yulesimon', 'zipf', 'zipfian']

Please see the confusion_matrix.png image for an illustration of the evaluation.

#Notes on model's shortcomings:

1. The model has hard time distinguishing between the geometric, boltzmann (not to be confused with the maxwell-boltzmann distribution), and planck distributions. Qualitatively this makes sense, as they all exhibit monotone decreasing curves with a shape that looks roughly like a decreasing exponential. Analytically, based on the following functional forms, it is apparent that the distributions share nearly identical pdfs:

Geom: f(k) = ((1-p)^(k-1))p
Boltzmann: f(k) = (1-exp(-L))exp(-Lk)/(1-extp(-LN))
Planck: f(k) = (1-exp(-L))exp(-Lk)

Scipy's documentation for the Planck distribution even notes "planck takes L as shape parameter. The Planck distribution can be written as a geometric distribution (geom) with p = 1-exp(-L) shifted by loc = -1."

2. The model also struggles to distinguish variations on the hypergeometric distributions, namely:

hypergeom
nchypergeom_fisher
nchypergeom_wallenius

Qualitatively they are similar in shape, and analytically their pdfs are similar. See the scipy documentaion for their pdfs.

# How to use (example code): 

#Have some data with 1-dimensional shape, for example,

from statspal.predict import predict, predict_max, keys

data = scipy.stats.poisson.rvs(mu=3, size=1000)

output = predict.predict(data) #returns the activations of the output layer of the neural network

predicted_distribution = predict.predict_max(a, True) #returns the most activated output

keys = predict.keys() #shows which outputs correspond to which distribution

#alternatively you can plot the prediction

import matplotlib.pyplot as plt

plt.bar([i for i in range(21), output)

# Interpreting the result 

output gives an array of shape (21), where the ith entry corresponds to the relative likelihood that distribution i (according to the distribution order listed above, and given by the keys() function) models your data best out of all 21 discrete distributions. For example, if output[0] has a greater value than any other output[i] value, then your data is best modeled by a 'bernoulli' distribution. Similary output[1] stands for 'betabinom', output[2] stands for 'betanbinom', etc.

Note: the .onnx model (a neural network) takes numpy arrays of shape (1024), so any other array size is down-sampled or up-sampled to 1024 before being fed into the network. This re-sampling is done in a manner to preserve the statistics of the original dataset, and the methods to do so can be inspected in the source code's _downsample_data and _upsample_data methods.

# Future Updates
1. Model will train on both scipy's continuous and discrete distributions, and thus have a broader classification ability.

Please note this project is ongoing and in its early stages.
