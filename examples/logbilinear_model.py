
import edward as ed
import tensorflow as tf
import numpy as np

from edward.models import Variational
from edward.stats import norm, poisson


class LogbilinearNetworkModel:
    """
    p(x, z) = [ prod_{i=1}^N prod_{j=1}^N Poi(Y_{ij}; \exp(s_iTt_j) ) ]
              [ prod_{i=1}^N N(s_i; 0, var) N(t_i; 0, var) ]
              
    where z = {s,t}.
    """
    
    def __init__(self,num_rows,num_cols,K,var):
        self.num_vars = (num_rows+num_cols) * K
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.K = K
        self.prior_variance = var


    def log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs)
        # reshaping the latent variable
        s = tf.reshape(zs[:,:self.num_rows*self.K],[self.num_rows,self.K])
        t = tf.reshape(zs[:,self.num_cols*self.K:],[self.num_cols,self.K])
        xp = tf.reshape(tf.exp(tf.matmul(s,t,transpose_b = True)), [N*N,1])
        log_lik = tf.reduce_sum( poisson.logpmf(xs,xp))
        return log_lik + log_prior

    def norm_log_prob(self, xs, zs):
        """Returns a vector [log p(xs, zs[1,:]), ..., log p(xs, zs[S,:])]."""
        log_prior = -self.prior_variance * tf.reduce_sum(zs*zs)
        # reshaping the latent variable
        s = tf.reshape(zs[:,:self.num_rows*self.K],[self.num_rows,self.K])
        t = tf.reshape(zs[:,self.num_cols*self.K:],[self.num_cols,self.K])
        xp = tf.reshape(tf.matmul(s,t,transpose_b = True), [N*N,1])
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
K = 5
data, a, b = build_toy_poisson_dataset(N,K)
model = LogbilinearNetworkModel(N,N,K,0.0001)

inference = ed.MAP(model, data, n_minibatch =1 , n_data_samples=N*N)

var = inference.run(n_iter=5000, n_print=10, optimizer='gradient descent',learning_rate = 0.00000001)
