
import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Variational
from edward.stats import norm, poisson


class LogbilinearNetworkModel():
    """
    p(x, z) = [ prod_{i=1}^N prod_{j=1}^N Poi(Y_{ij}; \exp(s_iTt_j) ) ]
              [ prod_{i=1}^N N(s_i; 0, var) N(t_i; 0, var) ]
              
    where z = {s,t}.
    """
    
    def __init__(self,n_rows,n_cols,K,var=0.01):
        self.n_vars = (n_rows+n_cols) * K
        self.n_rows = n_rows
        self.n_cols = n_cols
        self.K = K
        self.prior_variance = var


    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs)
        # reshaping the latent variable
        s = tf.reshape(zs[:,:self.n_rows*self.K],[self.n_rows,self.K])
        t = tf.reshape(zs[:,self.n_cols*self.K:],[self.n_cols,self.K])
        xp = tf.reshape(tf.exp(tf.matmul(s,t,transpose_b = True)), [N*N,1])
        log_lik = tf.reduce_sum( poisson.logpmf(xs,xp))
        return log_lik + log_prior

    def n_log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs)
        # reshaping the latent variable
        s = tf.reshape(zs[:,:self.n_rows*self.K],[self.n_rows,self.K])
        t = tf.reshape(zs[:,self.n_cols*self.K:],[self.n_cols,self.K])
        xp = tf.reshape(tf.matmul(s,t,transpose_b = True), [self.n_rows*self.n_cols,1])
        log_lik = tf.reduce_sum( norm.logpdf(xs,xp))
        return log_lik + log_prior


def build_toy_poisson_dataset(N=10,K=2, noise_std=0.1):
    ed.set_seed(0)
    a = tf.constant(np.random.randn(N, K))
    b = tf.constant(np.random.randn(N, K))
    noise = tf.constant(np.random.randn(N,N)*0.1)
    data = tf.round(tf.reshape(tf.exp(tf.matmul(a,b,transpose_b=True)) + tf.exp(noise), [N*N,1]))
    return ed.Data(data), a, b

def build_toy_dataset(N=10,K=2, noise_std=0.1):
    ed.set_seed(0)
    a = tf.constant(np.random.randn(N, K))
    b = tf.constant(np.random.randn(N, K))
    noise = tf.constant(np.random.randn(N,N)*0.1)
    data = tf.round(tf.reshape(tf.matmul(a,b,transpose_b=True) + noise, [N*N,1]))
    return ed.Data(data), a, b

ed.set_seed(42)
N = 300
K = 2
#data, a, b = build_toy_poisson_dataset(N,K)
data, a, b = build_toy_poisson_dataset(N,K)
model = LogbilinearNetworkModel(N,N,K)

inference = ed.MAP(model, data)

var = inference.run(n_iter=500, n_print=100)
