"""Functions for initializing the regressor."""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR


def get_regressor(model, ridge_lasso_alpha=1.0, svr_kernel='rbf',
                  svr_gamma='auto', svr_c=1.0, svr_epsilon=0.1, verbose=0,
                  jobs=-1):
    """Initializes a new regressor.

    Parameters:
        model (str): the regression model
        ridge_lasso_alpha (float): the regularization strength
        svr_kernel (str): SVR kernel type
        svr_gamma (float or 'auto'): SVR kernel coefficient
        svr_c (float): SVR penalty parameter C of the error term
        svr_epsilon (float): epsilon in the epsilon-SVR model
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
            kernel=svr_kernel, gamma=svr_gamma, C=svr_c, epsilon=svr_epsilon,
            verbose=verbose, max_iter=-1)
    elif model == 'random-forest':
        return RandomForestRegressor(n_jobs=jobs, verbose=verbose)
    elif model == 'mlp':
        return MLPRegressor(
            hidden_layer_sizes=[50, 50], verbose=verbose)
    else:
        msg = "Unsupported regression model: '{}'"
        raise ValueError(msg.format(model))
