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
        
        Note: there could be duplicates in the returned indices
        """
        a_indices = []
        b_indices = []
        for l in data_indices:
            i = l/self.M
            j = l%self.M
            a_indices = a_indices + list(range(i*self.K,(i+1)*self.K))
            b_indices = b_indices + list(range((self.N+j)*self.K,(self.N+j+1)*self.K))
        return a_indices + b_indices

    def log_prob(self, xs, zs, n_minibatch):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs)
        # reshaping the latent variable
        a = tf.reshape(zs[:,:n_minibatch*self.K],[n_minibatch,self.K])
        b = tf.reshape(zs[:,n_minibatch*self.K:],[n_minibatch,self.K])
        mus = tf.matmul(tf.mul(a,b),tf.ones([self.K,1]))
        # broadcasting to do mus - y (n_data x n_minibatch - n_data)
        log_lik = -tf.reduce_sum(tf.pow(mus - xs, 2), [0,1]) / self.lik_variance*self.N/n_minibatch
        return log_lik + log_prior

def build_toy_dataset(N=10,K=2, noise_std=0.1):
    ed.set_seed(0)
    a = tf.random_normal([N, K], mean=0, stddev=1)
    b = tf.random_normal([N, K], mean=0, stddev=1)
    noise = tf.random_normal([N, N], mean=0, stddev=noise_std)
    data = tf.matmul(a,b,transpose_b=True) + noise
    return ed.Data(data), a, b

ed.set_seed(42)
N = 1000
K = 100
data, a, b = build_toy_dataset(N,K)
model = MatrixFactorization(N,K)

inference = ed.MAP(model, data, n_minibatch =100)
inference.run(n_iter=500, n_print=10)
