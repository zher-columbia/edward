#!/usr/bin/env python
"""
Bayesian linear regression using mean-field variational inference.

Probability model:
    Bayesian linear model
    Prior: Normal
    Likelihood: Normal
Variational model
    Likelihood: Mean-field Normal
"""
import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Variational, Normal
from edward.stats import norm

class MatrixFactorization:
    """
    Matrix Factorization x \in R^{NxM}

    p(x,a,b) =    [\prod_{ij} Normal(x_{ij} ; a_i^Tb_j, lik_variance)] *
                  [\prod_{i}  Normal(a_i | 0, prior_variance*I)] *
                  [\prod_{i}  Normal(b_i | 0, prior_variance*I)] 

    where a \in R^{NxK}, b \in R^{MxK}, and lik_variance and prior_variance fixed.

    Parameters
    ----------
    N : int
        number of rows
    M: int, optional
        number of columns, by default M=N
    K : int
        embedding dimension
    lik_variance : float, optional
        Variance of the normal likelihood; aka noise parameter,
        homoscedastic variance, scale parameter.
    prior_variance : float, optional
        Variance of the normal prior on weights; aka L2
        regularization parameter, ridge penalty, scale parameter.
    """
    def __init__(self, N, M=None, K=10, lik_variance=0.01, prior_variance=0.01):
        self.N = N
        if not M:
            M = N
        self.M = M
        self.K = K
        self.lik_variance = lik_variance
        self.prior_variance = prior_variance
        self.num_vars = 0
        self.num_local_vars = M*K + N*K

    def map_data_indices(self, data_indices):
        """
        Input
        --------------------------------
        data_indices: list of integers


        Output
        --------------------------------
        indices: list of integers

        the NxM matrix entries are  flattened 
        [[0*M + 0, 0*M + 1, ...., 0*M + M-1],
         [1*M + 0, 1*M + 1, ....
           .
           .        ... i*M + j ...
           .
         [(N-2)*M + 0, .... , (N-2)*M + M-1],
         [(N-1)*M + 0, .... , (N-1)*M + M-1]]

        each index l in data_indices corresponds 
        to row i, column j such that l = i*M + j


        The model parameters are a flattened NxK array,
        followed by a flattened MxK array.

        [[0*K + 0, 0*K + 1, ...., 0*K + K-1],
         [1*K + 0, 1*K + 1, ....
           .
           .        ... i*K + [0:K-1] ...
           .
         [(N-1)*K + 0, .... , (N-1)*K + K-1]]
         [(N+0)*K + 0, .... , (N+0)*K + K-1],
           .
           .        ... (N+j)*K + x ...
           .
         [(N+M-2)*K + 0, .... , (N+M-2)*K + K-1],
         [(N+M-1)*K + 0, .... , (N+M-1)*K + K-1]]
        
        For each row i we add the parameters indices
        [i*K : i*K + K-1]
        and for each column j we add the parameters
        [(N+j)*K : (N+j)*K + K-1]
        

        """
        indices = []
        for l in data_indices:
            i = l/self.M
            j = l%self.M
            indices = indices + list(range(i*self.K,(i+1)*self.K))
            indices = indices + list(range((N+j)*self.K,(N+j+1)*self.K))
        return indices

    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs, 1)
        # broadcasting to do (x*W) + b (n_data x n_minibatch - n_minibatch)
        x = tf.expand_dims(x, 1)
        W = tf.expand_dims(zs[:, 0], 0)
        b = zs[:, 1]
        mus = tf.matmul(x, W) + b
        # broadcasting to do mus - y (n_data x n_minibatch - n_data)
        y = tf.expand_dims(y, 1)
        log_lik = -tf.reduce_sum(tf.pow(mus - y, 2), 0) / self.lik_variance
        return log_lik + log_prior

def build_toy_dataset(n_data=40, noise_std=0.1):
    ed.set_seed(0)
    x  = np.concatenate([np.linspace(0, 2, num=n_data/2),
                         np.linspace(6, 8, num=n_data/2)])
    y = 0.075*x + norm.rvs(0, noise_std, size=n_data)
    x = (x - 4.0) / 4.0
    x = x.reshape((n_data, 1))
    y = y.reshape((n_data, 1))
    data = np.concatenate((y, x), axis=1) # n_data x 2
    data = tf.constant(data, dtype=tf.float32)
    return ed.Data(data)

ed.set_seed(42)
model = LinearModel()
variational = Variational()
variational.add(Normal(model.num_vars))
data = build_toy_dataset()

inference = ed.MFVI(model, variational, data)
inference.run(n_iter=250, n_minibatch=5, n_print=10)
