"""Functions for initializing the regressor."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def get_regressor(model,
                  ridge_lasso_alpha=1.0,
                  svr_c=1.0, svr_epsilon=0.1, svr_kernel='rbf',
                  verbose=0, jobs=-1):
    """Initializes a new regressor.

    Parameters:
        model (str): the regression model
        ridge_lasso_alpha (float): the regularization strength
        svr_c (float): SVR penalty parameter C of the error term
        svr_epsilon (float): epsilon in the epsilon-SVR model
        svr_kernel (str): SVR kernel type
        verbose (int): the verbosity level
        jobs (int): the number of jobs to run in parallel

    Returns:
        regressor: the regressor
    """
    if model == 'linear':
        return LinearRegression(
            fit_intercept=False, normalize=False, copy_X=False, n_jobs=jobs)
    elif model == 'ridge':
        return Ridge(
            alpha=ridge_lasso_alpha, fit_intercept=False, normalize=False,
            copy_X=False)
    elif model == 'lasso':
        return Lasso(
            alpha=ridge_lasso_alpha, fit_intercept=False, normalize=False,
            copy_X=False)
    elif model == 'svr':
        return SVR(
            C=svr_c, epsilon=svr_epsilon, kernel=svr_kernel, verbose=verbose,
            max_iter=-1)
    elif model == 'random-forest':
        return RandomForestRegressor(n_jobs=jobs, verbose=verbose)
    elif model == 'mlp':
        return MLPRegressor(
            hidden_layer_sizes=[50, 50], verbose=verbose)
    else:
        msg = "Unsupported regression model: '{}'"
        raise ValueError(msg.format(model))
