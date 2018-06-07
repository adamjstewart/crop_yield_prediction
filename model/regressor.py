"""Functions for initializing the regressor."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def get_regressor(model, alpha=1.0, c=1.0, epsilon=0.1, verbose=0, jobs=-1):
    """Initializes a new regressor.

    Parameters:
        model (str): the regression model
        alpha (float): the regularization strength
        c (float): SVR penalty parameter
        epsilon (float): SVR epsilon
        verbose (int): the verbosity level
        jobs (int): the number of jobs to run in parallel

    Returns:
        regressor: the regressor
    """
    if model == 'lasso':
        return Lasso(
            alpha=alpha, fit_intercept=False, normalize=False, copy_X=False)
    elif model == 'linear':
        return LinearRegression(
            fit_intercept=False, normalize=False, copy_X=False, n_jobs=jobs)
    elif model == 'mlp':
        return MLPRegressor(
            hidden_layer_sizes=[50, 50], verbose=verbose)
    elif model == 'ridge':
        return Ridge(
            alpha=alpha, fit_intercept=False, normalize=False, copy_X=False)
    elif model == 'random-forest':
        return RandomForestRegressor(n_jobs=jobs, verbose=verbose)
    elif model == 'svr':
        return SVR(C=c, epsilon=epsilon, verbose=verbose)
    else:
        msg = "Unsupported regression model: '{}'"
        raise ValueError(msg.format(model))
