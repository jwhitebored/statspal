# statspal
Neural net made in TensorFlow to determine the statistical distribution governing a dataset. Intended use is to better understand the relevant statistics of a dataset, and give an informed choice of function to fit data to in standard curve fitting algorithms (non-linear regression) like SciPy's opt.curve_fit().

##################################### Model's Accuracy ####################################

Model attained 75% evaluation accuracy when classifying the data belonging to the following scipy.stats discrete distributions:

['bernoulli', 'betabinom', 'betanbinom', 'binom', 'boltzmann', 'dlaplace', 'geom', 'hypergeom', 'logser', 'nbinom', 'nchypergeom_fisher', 'nchypergeom_wallenius', 'nhypergeom', 'planck', 'poisson', 'poisson_binom', 'randint', 'skellam', 'yulesimon', 'zipf', 'zipfian']

#Notes on model's shortcomings:
The model has hard time distinguishing between the geometric, boltzmann (not to be confused with the maxwell-boltzmann distribution), and planck distributions. Qualitatively this makes sense, as they all exhibit monotone decreasing curves with a shape that looks roughly like a decreasing exponential. Analytically, based on the following functional forms, it is apparent that the distributions share nearly identical pdfs:

Geom: f(k) = ((1-p)^(k-1))p
Boltzmann: f(k) = (1-exp(-L))exp(-Lk)/(1-extp(-LN))
Planck: f(k) = (1-exp(-L))exp(-Lk)

Scipy's documentation for the Planck distribution even notes "planck takes L as shape parameter. The Planck distribution can be written as a geometric distribution (geom) with p = 1-exp(-L) shifted by loc = -1."

################################ How to use (example code): ###############################

#Have some data with shape (1024), for example,

data = scipy.stats.poisson.rvs(mu=3, size=1024)

data = data.astype(np.float32) #make dataset correct data type

data = data.reshape(1024, 1) #reshape the data to fit keras model input

data = np.expand_dims(data, axis=0) #reshape that data again to fit keras model input

model = tf.keras.models.load_model('file_path_and_name.keras') #load keras model into Python

prediction = model.predict(data) #predict the statistical distribution of you dataset

print(prediction[0]) #This gives the result

#alternatively you can plot the prediction

import matplotlib.pyplot as plt

plt.bar([i for i in range(21), prediction[0])

Interpreting the result:
prediction[0] gives an array of shape (21), where the ith entry corresponds to the relative likelihood that distribution i (according to the distribution order listed above) models your data best out of all 21 discrete distributions. For example, if prediction[0][0] has a greater value than any other prediction[0][i] value, then your data is best modeled by a 'bernoulli' distribution. Similary prediction[0][1] stands for 'betabinom', prediction[0][2] stands for 'betanbinom', etc.

Note: to downsample or upsample your dataset to fit the input size of the keras model (1024 data points), please see the downsampling and upsampling scripts in this repository.

Future updates:
1. Model will train on both scipy's continuous and discrete distributions, and thus have a broader classification ability.
2. Model architecture/weights will be converted to depend only on numpy, not tensorflow.
3. Model will be integrated into a Python package to easily extend scipy's stats and optimization packages.

Please note this project is ongoing and in its early stages.
