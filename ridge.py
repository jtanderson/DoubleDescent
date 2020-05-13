import logging
import sys
from copy import deepcopy
from typing import Callable
import math

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import Workbook
import multiprocessing
from scipy.optimize import fmin_cg as gradient_descent

####################### Setup ###########################
opts = {
    # Number of process threads we're allowed to use.
    "n_processes": 20,
    # number of times to run a given n_samples
    "n_trials": 10,
    # Dimensionality to generate for our data. For plotability, keep it at 1.
    "dimensions": 20,
    # Range of samples to pull from. This determines the number of estimators we use.
    "samples": np.arange(100, 2000, 50),
    "lambda_scalars": [2 ** i for i in range(-8, 20, 1)],
    # σ^2 is the distance from truth to sample from in our default noise function.
    "sigma": 0.5,
    # e is our noise function. None gives us the default sigma^2
    "e": None
    }
########################################################
######### The Math #####################################
def gen_dist(n_samples: int, sigma: float, beta, dimensions: int, e: Callable = None):
  """Generates a sample distribution given some noise function e. 
    
  The resulting input/covariate should be 

  x ∈ R^d and sampled from N(0, Identity(d))
  y = dot(x, β*) + ε

  Default ε is ε ~ N(0, σ^2)
  """
  # We'll hide this definition close to where we use it. 
  def sigma_sq_e(data, n, sigma):
    """Default noise function.
    
    e ~ N(0, sigma^2)
    """
    # Default noise function is the normal distribution N(0, σ^2)
    e = np.array([np.random.normal(0, sigma ** 2) for _ in range(n_samples)])
    e = e.reshape(e.shape[0], 1)
    return y+e

  # If no noise is supplied, we will just use our default.
  if(e is None):
    e = sigma_sq_e

  # Generate n x values uniformly distributed from 0 to n.
  # We want to generate n_samples of a multivariate normal distribution, each with a mean 0 and covariance matrix of Id
  x = np.array([np.random.multivariate_normal(np.zeros(dimensions), np.identity(dimensions)) for _ in range(n_samples)])

  Beta = beta.reshape(beta.shape[0], 1)
  y = np.dot(x, Beta)
  y = e(y, n_samples, sigma)

  return (x, y)


def get_beta(x, y, dimensions: int, lmbda: float):
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

def calc_opt_lmbda(dimensions: int, sigma, beta):
  """Calculates optimal lambda based on lemma 2 of https://arxiv.org/pdf/2003.01897.pdf"""
  return (dimensions * sigma ** 2) / (np.linalg.norm(beta)) ** 2
  
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

def gen_sample_batch(samples, n_trials, sigma, dimensions, lmbda, beta, noise, **kwargs):
  """Generates a list of (n_samples, beta_error)"""
  n_sample_error = []
  for n_samples in samples:
    # Training data
    (x, y) = gen_dist(n_samples=n_samples, sigma=sigma, dimensions=dimensions, beta=beta, e=noise)
    # estimate beta
    Bhat = get_beta(x, y, dimensions, lmbda)
    # Generates n_trials worth of distributions of n_samples. Then calculates the mse of all of the errors in those test datasets.
    mse = np.average([get_error(*gen_dist(n_samples=n_samples, sigma=sigma, dimensions=dimensions, beta=beta, e=noise), Bhat) for _ in range(n_trials)])
    n_sample_error.append((n_samples, mse))
  return n_sample_error

def worker_wrapper(arg):
  logging.debug("Worker received opts %s" % str(arg))
  return gen_sample_batch(**arg)

def gen_beta(dimensions: int):
  """Generates a beta to use, given some dimensional argument.
  The paper states the argument should have an l2 norm of 1. therefor ours will be the ones array / the norm.
  """
  beta = np.ones(dimensions)
  beta /= np.linalg.norm(beta)
  return beta

def generate_expected_risk(dimensions, samples, beta, opt_lmbda):
  # X = ???
  # yv = ??
  # X = 0
  # yv = 0
  # lhs = np.linalg.norm(X*beta - yv)
  # rhs = np.linalg.norm(beta)**2
  pass

def custom_noise(y, n, sigma):
  """We decided our noise function would be R{-1, 1}
  Sigma passed to fit the noise function template.
  """
  fc = 0.2 # Flip chance.
  return y*np.random.choice((-1,1), p=(fc, 1-fc), size=n)

def gen_lambda_batch(filename: str, noise=None, lmbda=None):
  """Generates an excel worksheet containing data generated over a given lambda scalar range in opts["lambda_scalars"]"""

  # Generate our workbook.
  wb = Workbook()
  headers = ["n_samples"]

  sheet = wb.create_sheet(title="Double Descent")
  samples = []
  errors = {}
  beta = gen_beta(dimensions=opts["dimensions"])
  if(lmbda is None):
    opt_lmbda = calc_opt_lmbda(dimensions=opts["dimensions"], sigma=opts["sigma"], beta=beta)
  else:
    opt_lmbda = lmbda
  print(opts)
  logging.info(opt_lmbda)

  # Calculate args for multiprocessing pool & create spreadsheet headers
  args = []

  dopts = deepcopy(opts)
  dopts["beta"] = beta
  for lmbda in opts["lambda_scalars"]:
    errors[lmbda] = []
    # Create the header in the sheet for this lambda column
    headers.append("Lambda 2^{:f}".format(math.log2(lmbda)))
    dopts["lmbda"] = opt_lmbda * lmbda
    print(opt_lmbda*lmbda)
    dopts["noise"] = noise
    args.append((deepcopy(dopts)))

  # Begin multiprocessing pool
  pool = multiprocessing.Pool(processes=opts["n_processes"])
  results = pool.map(worker_wrapper, args)

  # Use pool results to fill out the error & sample columns
  for lmbda, result in zip(opts["lambda_scalars"], results):
    for n_samples, error in result:
      if(n_samples not in samples):
        samples.append(n_samples)
      errors[lmbda].append(error)

  sheet.append(headers)
  for i in range(len(samples)):
    line = [samples[i]]
    for val in errors.values():
      line.append(val[i])
    sheet.append(line)
  wb.save(filename=filename)
  wb.close()

def grad_wrapper(lmbda, samples, trials, sigma, dims, beta, noise):
  """Wrapper around gen_sample_batch to organize the args to be compatible with gradient descent"""
  return np.average(np.array(gen_sample_batch(samples, trials, sigma, dims, lmbda, beta, noise))[:,1])

def find_opt_lambda(noise=None):
  """Generates an excel worksheet containing data generated over a given lambda scalar range in opts["lambda_scalars"]"""
  beta = gen_beta(dimensions=opts["dimensions"])
  args = (opts["samples"],opts["n_trials"], opts["sigma"], opts["dimensions"], beta, noise )
  start = calc_opt_lmbda(dimensions=opts["dimensions"], sigma=opts["sigma"], beta=beta)
  print(start)
  print(args)
  res = gradient_descent(grad_wrapper, start, args=args, callback=lambda xk : print(xk))
  print(res)

  # learning_rate = 20
  # decay_rate = 0.8
  # lmbda = 40
  # trials = 1

  # eps = 0.0005
  # while(learning_rate >= eps):
  #   print("LR: {:f} | DR: {:f} | lmbda: {:f}".format(learning_rate,1-decay_rate, lmbda))
  #   e = 0
  #   e1 = 0
  #   for i in range(trials):
  #     e += np.average(np.array(gen_sample_batch(
  #       opts["samples"],
  #       opts["n_trials"],
  #       opts["sigma"],
  #       opts["dimensions"],
  #       lmbda,
  #       beta,
  #       noise))[:,1]
  #     )
  #     print(".",end='',flush=True)
  #     e1 += np.average(np.array(gen_sample_batch(
  #       opts["samples"],
  #       opts["n_trials"],
  #       opts["sigma"],
  #       opts["dimensions"],
  #       lmbda+learning_rate,
  #       beta,
  #       noise))[:,1]
  #     )
  #     print(".",end='',flush=True)

  #   e/=trials
  #   e1/=trials
  #   print("e: {:f} e1: {:f}".format(e,e1))
  #   if(e < e1): # error gets larger as we increase by learning rate
  #     lmbda-=learning_rate
  #   else:
  #     lmbda+=learning_rate
  #   learning_rate*=decay_rate
  lmbda = res
  print("Optimal lambda detected: {:f}".format(lmbda))
  return lmbda

if __name__ == "__main__":
  if(len(sys.argv) > 1):
    filename = sys.argv[1]
  else:
    filename = "ridge_error.xlsx"
  logging.basicConfig(level=logging.DEBUG)
  noise = custom_noise
  lmbda = find_opt_lambda(noise=None)
  gen_lambda_batch(filename, noise=noise, lmbda=lmbda)
