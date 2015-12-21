from dora.active_sampling import Sampler, random_sample, grid_sample
import numpy as np
import dora.regressors.gp as gp
import scipy.stats as stats
from dora.active_sampling.util import ArrayBuffer


class GaussianProcess(Sampler):
    """
    GaussianProcess Class

    Inherits from the Sampler class and augments pick and update with the
    mechanics of the GP method

    Attributes
    ----------
    kerneldef : function
        Kernel function definition. See the 'gp' module.
    n_min : int
        Number of training samples required before sampler can be trained
    acq_name : str
        A string specifying the type of acquisition function used
    explore_priority : float, optional
        The priority of exploration against exploitation
    hyperparams : numpy.ndarray
        The hyperparameters of the Gaussian Process Inference Model
    regressors : list
        List of regressor objects. See 'gp.types.RegressionParams'
    y_mean : float
        Mean of the training target outputs
    n_tasks : int
        Number of tasks or, equivalently, number of target outputs

    See Also
    --------
    Sampler : Base Class
    """
    def __init__(self, lower, upper, kerneldef=None, n_min=None,
                 acq_name='var_sum', explore_priority=0.0001):
        """
        Initialises the GaussianProcess class

        .. note:: Currently only supports rectangular type restrictions on the
        parameter space

        Parameters
        ----------
        lower : array_like
            Lower or minimum bounds for the parameter space
        upper : array_like
            Upper or maximum bounds for the parameter space
        kerneldef : function
            Kernel function definition. See the 'gp' module.
        n_min : int
            Number of training samples required before sampler can be trained
        acq_name : str
            A string specifying the type of acquisition function used
        explore_priority : float, optional
            The priority of exploration against exploitation
        """
        Sampler.__init__(self, lower, upper)

        if kerneldef is None:
            onez = np.ones(self.dims)
            self.kerneldef = lambda h, k: \
                h(1e-3, 1e+2, 1) * k('matern3on2',
                                     h(1e-2 * onez, 1e+3 * onez, 1e+0 * onez))
        else:
            self.kerneldef = kerneldef

        self.n_min = n_min if n_min is not None else (4 ** self.dims)
        self.acq_name = acq_name
        self.explore_priority = explore_priority
        self.hyperparams = None
        self.regressors = None
        self.y_mean = None
        self.n_tasks = None

    def update_y_mean(self):
        """
        Update the mean of the target outputs

        .. note :: At anytime, 'y_mean' should be the mean of all the output
        targets including the virtual ones, since that is what we are training
        upon

        .. note :: [Properties Modified]
                    y
                    y_mean
        """
        if not self.y:
            return
        self.y = atleast_2d(self.y)
        self.y_mean = self.y().mean(axis=0) if len(self.y) else None

    def learn_hyperparams(self):
        """
        Learns the kernel hyperparameters from the data collected so far
        Equivalent to training the Gaussian process used for the sampler
        The training result is summarised by the hyperparameters of the kernel


        .. note :: Learns common hyperparameters between all tasks

        .. note :: [Properties Modified]
                    (None)

        Returns
        -------
        list
            A list of hyperparameters with each element being the
            hyperparameters of each corresponding task
        """
        # Compose the kernel and setup the optimiser
        kernel = gp.compose(self.kerneldef)
        opt_config = gp.OptConfig()
        opt_config.sigma = gp.auto_range(self.kerneldef)
        opt_config.noise = gp.Range([0.0001], [0.5], [0.05])
        opt_config.walltime = 50.0
        opt_config.global_opt = False

        # Make sure the number of stacks recorded is consistent
        if self.n_tasks is None:
            self.n_tasks = self.y_mean.shape[0]
        else:
            assert self.n_tasks == self.y_mean.shape[0]

        # We need to train a regressor for each of the stacks
        # Each regressor will use the same hyperparameters!
        # We will use folds to do this
        folds = gp.Folds(self.n_tasks, [], [], [])
        for i_task in range(self.n_tasks):
            folds.X.append(self.X())
            folds.flat_y.append(self.y[:, i_task] - self.y_mean[i_task])
        hyperparams = gp.train.learn_folds(folds, kernel, opt_config)

        # Use the same hyperparameters for each of the stacks
        return [hyperparams for i_task in range(self.n_tasks)]

    def update_regressors(self):
        """
        Update the regressors of the Gaussian process model
        Only makes sense to do this after hyperparameters are learned

        .. note :: [Properties Modified]
                    regressors

        .. note :: [Further Work] Use Cholesky Update here correctly to cache
                    regressors and improve efficiency
        """
        if self.hyperparams is None:
            return
            # raise ValueError('Hyperparameters are not learned yet.' +
            #                  'Regressors cannot be updated.')

        # Create the regressors if it hasn't already been
        # if self.regressors is None:
        kernel = gp.compose(self.kerneldef)
        self.regressors = []
        for i_task in range(self.n_tasks):
            self.regressors.append(
                gp.condition(self.X(), self.y()[:, i_task]
                             - self.y_mean[i_task],
                             kernel, self.hyperparams[i_task]))

        # # Otherwise, simply update the regressors
        # else:
        #     for i_task, regressor in enumerate(self.regressors):
        #         regressor.y = y[:, i_task] - self.y_mean[i_task]
        #         regressor.alpha = gp.predict.alpha(regressor.y, regressor.L)

    def train(self):
        """
        Trains the Gaussian process model
        A wrapper function that learns the hyperparameters and updates the
        regressors, which is equivalent to a fully trained model that is
        ready to perform Inference

        .. note :: [Properties Modified]
                    hyperparameters
                    regressors
        """
        # Learn hyperparameters
        self.hyperparams = self.learn_hyperparams()

        # Update the regressors
        self.update_regressors()

    def update(self, uid, y_true):
        """
        Updates a job with its observed value

        .. note :: [Properties Modified]
                    y
                    virtual_flag
                    y_mean
                    regressors

        Parameters
        ----------
        uid : str
            A hexadecimal ID that identifies the job to be updated
        y_true : float
            The observed value corresponding to the job identified by 'uid'

        Returns
        -------
        int
            Index location in the data buffer 'GaussianProcess.X' and
            'GaussianProcess.y' corresponding to the job being updated
        """
        y_true = atleast_1d(y_true)
        assert y_true.ndim == 1
        ind = self._update(uid, y_true)
        self.update_y_mean()
        self.update_regressors()
        return ind

    def pick(self, n_test=500, train=False):
        """
        Picks the next location in parameter space for the next observation
        to be taken, with a Gaussian process model

        .. note :: [Properties Modified]
                    X
                    y
                    virtual_flag
                    pending_results
                    y_mean
                    hyperparameters
                    regressors

        Parameters
        ----------
        n_test : int, optional
            The number of random query points across the search space to pick
            from
        train : bool, optional
            To train the model or not before picking, if allowed

        Returns
        -------
        numpy.ndarray
            Location in the parameter space for the next observation to be
            taken
        str
            A random hexadecimal ID to identify the corresponding job
        """
        n = len(self.X)
        n_corners = 2 ** self.dims

        self.update_y_mean()

        # If we do not have enough samples yet, randomly sample for more!
        if n < self.n_min:

            xq = random_sample(self.lower, self.upper, 1)[0]
            yq_exp = self.y_mean  # Note: Can be 'None' initially

        # Bootstrap with a regular sampling strategy to get it started
        elif n < n_corners + 1:
            xq = grid_sample(self.lower, self.upper, n)
            yq_exp = self.y_mean  # Note: Can be 'None' initially
            # Counter note (Al): none can't be inserted into a numpy array...
        else:

            if train or self.regressors is None:
                self.train()

            # Randomly sample the volume for test points
            Xq = random_sample(self.lower, self.upper, n_test)

            # Generate cached predictors for those test points
            predictors = [gp.query(Xq, r) for r in self.regressors]

            # Compute the posterior distributions at those points
            # Note: No covariance information implemented at this stage
            Yq_exp = self.y_mean + np.asarray([gp.mean(r, q) for r, q in
                                               zip(self.regressors,
                                                   predictors)]).T
            Yq_var = np.asarray([gp.variance(r, q) for r, q in
                                 zip(self.regressors, predictors)]).T

            # Aquisition Functions
            acq_defs_current = acq_defs(y_mean=self.y_mean,
                                        explore_priority=self.explore_priority)

            # Compute the acquisition levels at those test points
            yq_acq = acq_defs_current[self.acq_name](Yq_exp, Yq_var)

            # Find the test point with the highest acquisition level
            iq_acq = np.argmax(yq_acq)
            xq = Xq[iq_acq, :]
            yq_exp = Yq_exp[iq_acq, :]

        # Place a virtual observation...
        if yq_exp is None:
            yq_exp = np.zeros(self.n_tasks)  # can't insert None
        uid = Sampler._assign(self, xq, atleast_1d(yq_exp))

        return xq, uid

    def predict(self, Xq, real=True):
        """
        Infers the mean and variance of the Gaussian process at given locations
        using the data collected so far

        .. note :: [Properties Modified]
                    (None)

        Parameters
        ----------
        Xq : numpy.ndarray
            Query points
        real : bool
            To use only the real observations or also the virtual observations

        Returns
        -------
        numpy.ndarray
            Expectance of the prediction at the given locations
        numpy.ndarray
            Variance of the prediction at the given locations
        """
        assert self.hyperparams, "Sampler is not trained yet. " \
                                 "Possibly not enough observations provided."

        # To use only the real data, extract the real data and compute the
        # regressors using only the real data
        if real:
            X_real, y_real = self.get_real_data()
            kernel = gp.compose(self.kerneldef)
            regressors = [gp.condition(X_real, y_real[:, i_task]
                          - self.y_mean[i_task],
                          kernel, self.hyperparams[i_task])
                          for i_task in range(self.n_tasks)]

        # Otherwise, just use the regressors we already have
        else:
            regressors = self.regressors

        # Compute using the standard predictor sequence
        predictors = [gp.query(Xq, r) for r in regressors]
        yq_exp = [gp.mean(r, p)
                  for r, p in zip(regressors, predictors)]
        yq_var = [gp.variance(r, p)
                  for r, p in zip(regressors, predictors)]

        return np.asarray(yq_exp).T + self.y_mean, np.asarray(yq_var).T

    def set_kerneldef(self, kerneldef):
        assert callable(kerneldef)
        self.kerneldef = kerneldef

    def get_kerneldef(self):
        return self.kerneldef

    def print_kernel(self, kerneldef):
        # TO DO: Use the printer method to print the current kernel!
        pass

    def set_hyperparams(self, hyperparams):
        if isinstance(hyperparams, list):
            self.hyperparams = hyperparams
        else:
            self.hyperparams = [hyperparams for i in range(self.n_tasks)]

        self.update_regressors()

    def get_hyperparams(self):
        return self.hyperparams

    def set_acq_name(self, acq_name):
        assert type(acq_name) is str
        self.acq_name = acq_name

    def get_acq_func(self):
        return acq_defs(y_mean=self.y_mean,
                        explore_priority=self.explore_priority)[self.acq_name]

    def set_explore_priority(self, explore_priority):
        self.explore_priority = explore_priority

    def get_explore_priority(self):
        return self.explore_priority

    def set_min_training_size(self, n_min):
        self.n_min = n_min

    def get_min_training_size(self):
        return self.n_min

    def get_real_data(self):

        assert self.X
        assert self.y

        real_flag = ~self.virtual_flag()
        return self.X()[real_flag], self.y()[real_flag]


def atleast_2d(y):
    """
    Make sure the input data is two dimensional, either in the form of a list
    of vectors, or a matrix as a 'numpy.ndarray' of two dimensions.

    ..note : This currently only accepts homogenous lists or arrays. It will
    NOT raise errors if the list is non-homogenous as it only uses the
    first element for checking.

    Parameters
    ----------
    y : numpy.ndarray or list
        The array or list to make 2D

    Returns
    -------
    numpy.ndarray or list
        The 2D array or list
    """
    if isinstance(y, list):
        if type(y[0]) is not np.ndarray:
            return [np.array([y_i]) for y_i in y]
        elif y[0].ndim == 1:
            return y
        else:
            raise ValueError("List element already has more than 1 dimension")
    elif isinstance(y, np.ndarray):
        if y.ndim == 1:
            return y[:, np.newaxis]
        elif y.ndim == 2:
            return y
        else:
            raise ValueError("Object already has more than 2 dimensions")
    elif isinstance(y, ArrayBuffer):
        return y
    else:
        raise ValueError('Object is not a list or an array')


def atleast_1d(y_obs):
    """
    Make sure the input comes in a standard vector form as a 'numpy.ndarray'
    type of one dimension.

    Parameters
    ----------
    y : numpy.ndarray or list
        The array or list to make 1D

    Returns
    -------
    numpy.ndarray or list
        The 1D array or list
    """
    if type(y_obs) is np.ndarray:
        if y_obs.ndim == 0:
            return y_obs.flatten()
        elif y_obs.ndim > 1:
            raise ValueError('Target output is not vector valued!')
        else:
            return y_obs
    else:
        if type(y_obs) is list:
            return np.array(y_obs)
        elif np.isscalar(y_obs):
            return np.array([y_obs])
        else:
            raise ValueError('Unexpected target output type: %s'
                             % str(type(y_obs)))


def acq_defs(y_mean=0, explore_priority=0.0001):
    """
    Generates a dictionary of acquisition functions

    Parameters
    ----------
    y_mean : int or np.ndarray
        The mean of the target outputs
    explore_priority : float
        Exploration priority against exploitation

    Returns
    -------
    dict
        A dictionary of acquisition functions to be used for the GP Sampler
    """
    # Aquisition Functions
    # u: Mean matrix (n x n_tasks)
    # v: Variance matrix (n x n_tasks)
    # Returns an array of n values
    return {
        'var_sum': lambda u, v: np.sum(v, axis=1),
        'pred_max': lambda u, v: np.max(u + 3 * np.sqrt(v), axis=1),
        'prod_max': lambda u, v: np.max((u + (y_mean +
                                        explore_priority / 3.0)) *
                                        np.sqrt(v), axis=1),
        'prob_tail': lambda u, v: np.max((1 - stats.norm.cdf(
                                         explore_priority *
                                         np.ones(u.shape), u,
                                         np.sqrt(v))), axis=1),
        'sigmoid': lambda u, v: np.abs(stats.logistic.cdf(u + np.sqrt(v),
                                       loc=0.5,
                                       scale=explore_priority) -
                                       stats.logistic.cdf(u - np.sqrt(v),
                                       loc=0.5,
                                       scale=explore_priority)).sum(axis=1)
    }