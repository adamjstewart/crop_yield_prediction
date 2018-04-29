"""Functions for initializing the classifier."""

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


def get_classifier(classifier, jobs=-1, verbose=0):
    """Initializes a new classifier.

    Args:
        classifier (str): the classification model
        jobs (int): the number of jobs to run in parallel
        verbose (int): the verbosity level

    Returns:
        classifier: the classifier
    """
    if classifier == 'logistic':
        return LogisticRegression(
            solver='sag', multi_class='multinomial',
            verbose=verbose, n_jobs=jobs)
    elif classifier == 'svm':
        return LinearSVC(multi_class='ovr', verbose=verbose)
    elif classifier == 'random_forest':
        return RandomForestClassifier(verbose=verbose, n_jobs=jobs)
