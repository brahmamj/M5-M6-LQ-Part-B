import sys
from pathlib import Path
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

from typing import List
import sys
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

class TemporalVariableTransformer(BaseEstimator, TransformerMixin):
    # Transformer for calculating the elapsed time between a reference variable and specified temporal variables.

    def __init__(self, variables, reference_variable):
        # Initialize the transformer with the list of temporal variables and a reference variable.

        # Check that 'variables' is a list; if not, raise a ValueError.
        if not isinstance(variables, list):
            raise ValueError('variables should be a list')

        self.variables = variables  # Store the list of temporal variables.
        self.reference_variable = reference_variable  # Store the reference variable.

    def fit(self, X, y=None):
        # Fit method required for scikit-learn pipeline compatibility.
        # This method does not need to perform any operations, so it just returns self.
        return self

    def transform(self, X):
        # Transform the input DataFrame to calculate elapsed time.

        # Create a copy of the DataFrame to avoid modifying the original data.
        X = X.copy()

        # Loop through each temporal variable and calculate the difference between the reference variable and the temporal variable.
        for feature in self.variables:
            X[feature] = X[self.reference_variable] - X[feature]  # Calculate elapsed time.

        return X  # Return the modified DataFrame with updated temporal variables.