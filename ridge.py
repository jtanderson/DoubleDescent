import logging
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook

####################### Setup ###########################
opts = {
    # Number of process threads we're allowed to use.
    "n_processes": 10,
    # number of times to run a given n_samples
    "n_trials": 10,
    # Dimensionality to generate for our data. For plotability, keep it at 1.
    "dimensions": 10,
    # Range of samples to pull from. This determines the number of estimators we use.
    "samples": np.arange(100, 1000, 100),
    "lambda_scalars": [2 ** i for i in range(-8, 4, 2)],
    # σ^2 is the distance from truth to sample from in our default noise function.
    "sigma": 0.5,
    # e is our noise function. None gives us the default sigma^2
    "e": None
    }
########################################################
######### The Math #####################################
def gen_dist(n_samples: int, sigma: float, dimensions: int, e: Callable = None, **kwargs):
  """Generates a sample distribution given some noise function e. 
    
  The resulting input/covariate should be 

  x ∈ R^d and sampled from N(0, Identity(d))
  y = dot(x, β*) + ε

  Default ε is ε ~ N(0, σ^2)
  """
  # We'll hide this definition close to where we use it. 
  def sigma_sq_e(n, sigma):
    """Default noise function.
    
    e ~ N(0, sigma^2)
    """
    # Default noise function is the normal distribution N(0, σ^2)
    e = np.array([np.random.normal(0, sigma ** 2) for _ in range(n_samples)])
    e = e.reshape(e.shape[0], 1)
    return e

  # If no noise is supplied, we will just use our default.
  if(e is None):
    e = sigma_sq_e

  # Generate n x values uniformly distributed from 0 to n.
  # We want to generate n_samples of a multivariate normal distribution, each with a mean 0 and covariance matrix of Id
  x = np.array([np.random.multivariate_normal(np.zeros(dimensions), np.identity(dimensions)) for _ in range(n_samples)])

  Beta = np.ones(dimensions)
  Beta = Beta / np.linalg.norm(Beta)
  Beta = Beta.reshape(Beta.shape[0], 1)
  y = np.dot(x, Beta) + e(n_samples, sigma)

  return (x, y)


def get_beta(x, y, dimensions: int, lmbda: float, **kwargs):
  """Applies ridge regression onto the given input/covariate and returns the estimated beta β"""
  # regularized least-squares estimator for a given lambda > 0
  # Beta_(n,lambda) := argmin ||X*Beta-y||(2 2) + lambda||Beta||(2 2) = 
  # (X^T * X + lambda * I_d)^-1 * X^T * y
  I = np.identity(dimensions)
  inner = np.dot(x.T, x) + lmbda * I
  inv = np.linalg.inv(inner)
  return np.dot(np.dot(inv, x.T), y)

def get_error(x, y, beta):
  """Calculates and returns the Risk/Error of some Beta compared to the actual y.

  mse = np.average((np.dot(x, beta) - y) ** 2)
  """
  return np.average((np.dot(x, beta) - y) ** 2)

def calc_opt_lmbda(dimensions: int, sigma, **kwargs):
  """Calculates optimal lambda based on lemma 2 of https://arxiv.org/pdf/2003.01897.pdf"""
  Beta = np.ones(dimensions)
  return (dimensions * sigma ** 2) / np.linalg.norm(Beta) ** 2
  
##############################################################
################# Utility ####################################
def show(x, y, beta, dimensions: int, **kwargs):
  """Shows the matplotlib plot for x,y,beta given that our dimensions is set to 1"""
  assert dimensions == 1, "We can't plot any dimensionality greater than 1"
  plt.scatter(x,y)
  plt.plot(x, beta * x, c='red')
  plt.show()
##############################################################
############# Data generation ################################

def gen_sample_batch(samples, n_processes, n_trials, **kwargs):
  """Generates a list of (n_samples, beta_error)"""
  n_sample_error = []
  for n_samples in samples:
    # Training data
    (x, y) = gen_dist(n_samples=n_samples, **opts)
    # estimate beta
    Bhat = get_beta(x, y, **opts)
    # Generates n_trials worth of distributions of n_samples. Then calculates the mse of all of the errors in those test datasets.
    mse = np.average([get_error(*gen_dist(n_samples, **opts), Bhat) for _ in range(n_trials)])
    n_sample_error.append((n_samples, mse))
  return n_sample_error


def gen_lambda_batch():
  """Generates an excel worksheet containing data generated over a given lambda scalar range in opts["lambda_scalars"]"""

  # Generate our workbook.
  wb = Workbook()
  headers = ["n_samples"]

  sheet = wb.create_sheet(title="Double Descent")
  samples = []
  errors = {}
  opt_lmbda = calc_opt_lmbda(**opts)

  for lmbda in opts["lambda_scalars"]:
    errors[lmbda] = []

    opts["lmbda"] = opt_lmbda * lmbda
    headers.append("Lambda {:f}".format(lmbda * opt_lmbda))
    for n_samples, error in gen_sample_batch(**opts):
      if(n_samples not in samples):
        samples.append(n_samples)
      errors[lmbda].append(error)
  sheet.append(headers)
  for i in range(len(samples)):
    line = [samples[i]]
    for val in errors.values():
      line.append(val[i])
    sheet.append(line)
  print(errors)
  wb.save(filename="ridge_error.xlsx")
  wb.close()

if __name__ == "__main__":
  logging.basicConfig(level=logging.DEBUG)
  gen_lambda_batch()
