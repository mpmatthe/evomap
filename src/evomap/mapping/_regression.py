# from sklearn.isotonic import IsotonicRegression
import numpy as np
from scipy import interpolate
from sklearn._isotonic import _inplace_contiguous_isotonic_regression, _make_unique

class IsotonicRegression():
    def __init__(self, y_min=None, y_max=None, increasing=True):
        self.y_min = y_min
        self.y_max = y_max
        self.increasing = increasing

    def fit(self, X, y, sample_weight=None):
        # Check for NaN values
        if np.isnan(X).any() or np.isnan(y).any():
            raise ValueError("Input X and y cannot contain NaN values.")

        # Sort X and y using lexsort (by X, then y)
        order = np.lexsort((y, X))
        X = X[order]
        y = y[order]

        # Handle sample weights
        if sample_weight is not None:
            sample_weight = sample_weight[order]
        else:
            sample_weight = np.ones_like(y, dtype=np.float64)

        # Handle increasing=False by negating and reversing X, reversing y and sample_weight
        if not self.increasing:
            X = -X[::-1].copy()
            y = y[::-1].copy()
            sample_weight = sample_weight[::-1].copy()

        # Remove duplicates
        X_unique, y_unique, sample_weight_unique = _make_unique(X, y, sample_weight)

        # Perform isotonic regression in-place
        _inplace_contiguous_isotonic_regression(y_unique, sample_weight_unique)

        # Apply y_min and y_max constraints
        if self.y_min is not None or self.y_max is not None:
            np.clip(y_unique, self.y_min, self.y_max, out=y_unique)

        # Handle increasing=False by reversing transformations
        if not self.increasing:
            X_unique = -X_unique[::-1].copy()
            y_unique = y_unique[::-1].copy()

        self.X_thresholds_ = X_unique
        self.y_thresholds_ = y_unique

        # Create interpolation function
        self.f_ = interpolate.interp1d(
            self.X_thresholds_,
            self.y_thresholds_,
            kind='linear',
            bounds_error=False,
            fill_value=(self.y_thresholds_[0], self.y_thresholds_[-1]),
            assume_sorted=True
        )

        return self

    def predict(self, X_new):
        return self.f_(X_new)

    def transform(self, X_new):
        return self.predict(X_new)

    def fit_transform(self, X, y, sample_weight=None):
        self.fit(X, y, sample_weight)
        return self.transform(X)
