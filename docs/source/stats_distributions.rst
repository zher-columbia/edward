[This is not built in the API for now.]
Statistics Library
------------------

The ``edward.stats`` library provides a collection of primitive
distribution methods for use in TensorFlow.

[api doc for ``edward.stats.distributions`` goes here]

.. code:: python

  class Distribution(object):
    """Template for all distributions."""
    def rvs(self, size=1):
      """
      Parameters
      ----------
      size : int, list of int, or tuple of int, optional
          Number of samples, in a particular shape if specified in a
          list or tuple with more than one element.

      params : float or np.ndarray

      Returns
      -------
      np.ndarray
          np.ndarray of dimension (size x shape), where shape is the
          shape of its parameter argument. For multivariate
          distributions, shape may correspond to only one of the
          parameter arguments, e.g., alpha in Dirichlet, p in
          Multinomial, mean in Multivariate_Normal.

      Notes
      -----
      This is written in NumPy/SciPy, as TensorFlow does not support
      many distributions for random number generation. It follows
      SciPy's naming and argument conventions. It does not support
      taking in tf.Tensors as input.

      The equivalent method in SciPy is not guaranteed to be
      supported with a batch of parameter inputs, e.g., a vector of
      location parameters in a normal distribution, or a matrix of
      concentration parameters in a Dirichlet. This is.

      This does not follow SciPy's behavior, e.g., the number (or
      shape) of the draws will always be denoted by its outer
      dimension(s).

      params as a 2-D or higher tensor is not guaranteed to be
      supported (for either univariate or multivariate
      distribution).

      size as a list or tuple of more than one element is not
      guaranteed to be supported.

      For most distributions, the parameters must be of the same
      shape and type, e.g., n and p in Binomial must be np.arrays()
      of same shape or both floats. For some, they may differ by one
      dimension, e.g., n and p in Multinomial can be float and
      np.array(), or both np.arrays, and n always has one less
      dimension.
      """
      raise NotImplementedError()
