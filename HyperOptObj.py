# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 11:37:00 2025

@author: Paul
"""

# =============================================================================
# Class for Hyper Opt
# =============================================================================

# Packages


from hyperopt import fmin, tpe, STATUS_OK  # hyper parameter tuning
from hyperopt import space_eval

# Class
'''
Makes it a little bit easier to manage models using different datasets...
'''


class HyperOptObj:
    def __init__(
            self, X_train, X_test, y_train, y_test, model, func_score):

        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.model = model
        self.func_score = func_score
        self.space = None

    def objective(self, params):
        model = self.model(**params)

        model = model.fit(
            self.X_train,
            self.y_train,
            )

        score = self.func_score(self, model)

        # return negative of the score
        return {'loss': -score, 'status': STATUS_OK}

    def default_score(self, model):
        return 1

    def run_optimization(self, space, evals=50):

        self.space = space

        best = fmin(
            fn=self.objective,
            space=space,
            algo=tpe.suggest,
            max_evals=evals
            )

        print("Best hyperparameters:", best)

        return space_eval(self.space, best)

    def set_objective(self, new_func):
        self.objective = new_func.__get__(self)


# =============================================================================
# Examples
# =============================================================================
'''
# Create the object
hyperopt_obj = HyperOptObj(X_train, X_test, y_train, y_test, model, func_score)

# Overwrite with a new function
hyperopt_obj.set_objective(new_objective)
'''
