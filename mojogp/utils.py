"""Utility classes for MojoGP.

Provides data preprocessing utilities like StandardScaler for normalizing
inputs and targets before GP training.
"""

import numpy as np
from typing import Optional


class StandardScaler:
    """Zero-mean, unit-variance scaler for GP inputs and targets.

    Normalizes data to have zero mean and unit variance before training,
    and unnormalizes predictions back to the original scale.

    Example:
        >>> from mojogp.utils import StandardScaler
        >>> from mojogp import MojoGP
        >>>
        >>> scaler = StandardScaler()
        >>> X_scaled, y_scaled = scaler.fit_transform(X_train, y_train)
        >>>
        >>> gp = MojoGP()
        >>> gp.fit(X_scaled, y_scaled)
        >>>
        >>> X_test_scaled = scaler.transform_X(X_test)
        >>> mean_scaled, std_scaled = gp.predict(X_test_scaled, return_std=True)
        >>> mean, std = scaler.inverse_transform_y(mean_scaled, std_scaled)
    """

    def __init__(self):
        self.X_mean_: Optional[np.ndarray] = None
        self.X_std_: Optional[np.ndarray] = None
        self.y_mean_: Optional[float] = None
        self.y_std_: Optional[float] = None
        self._is_fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> "StandardScaler":
        """Compute mean and std from training data.

        Args:
            X: Training inputs [n, d]
            y: Training targets [n] (optional)

        Returns:
            self (for chaining)
        """
        X = np.asarray(X, dtype=np.float64)

        self.X_mean_ = np.mean(X, axis=0)
        self.X_std_ = np.std(X, axis=0)
        # Avoid division by zero for constant features
        self.X_std_ = np.where(self.X_std_ < 1e-12, 1.0, self.X_std_)

        if y is not None:
            y = np.asarray(y, dtype=np.float64)
            self.y_mean_ = float(np.mean(y))
            self.y_std_ = float(np.std(y))
            if self.y_std_ < 1e-12:
                self.y_std_ = 1.0

        self._is_fitted = True
        return self

    def transform_X(self, X: np.ndarray) -> np.ndarray:
        """Normalize input features.

        Args:
            X: Input data [n, d]

        Returns:
            Normalized X with zero mean and unit variance per feature
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() or fit_transform() first")
        X = np.asarray(X, dtype=np.float64)
        return ((X - self.X_mean_) / self.X_std_).astype(np.float32)

    def transform_y(self, y: np.ndarray) -> np.ndarray:
        """Normalize targets.

        Args:
            y: Target values [n]

        Returns:
            Normalized y with zero mean and unit variance
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() or fit_transform() first")
        if self.y_mean_ is None:
            raise RuntimeError("Scaler was not fitted with y data")
        y = np.asarray(y, dtype=np.float64)
        return ((y - self.y_mean_) / self.y_std_).astype(np.float32)

    def fit_transform(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> tuple:
        """Fit and transform in one step.

        Args:
            X: Training inputs [n, d]
            y: Training targets [n] (optional)

        Returns:
            (X_scaled,) if y is None, or (X_scaled, y_scaled) if y is provided
        """
        self.fit(X, y)
        X_scaled = self.transform_X(X)
        if y is not None:
            y_scaled = self.transform_y(y)
            return X_scaled, y_scaled
        return (X_scaled,)

    def inverse_transform_y(
        self,
        mean: np.ndarray,
        std: Optional[np.ndarray] = None,
        variance: Optional[np.ndarray] = None,
    ) -> tuple:
        """Transform predictions back to original scale.

        The mean is unnormalized as: mean_orig = mean * y_std + y_mean
        The std is unnormalized as: std_orig = std * y_std
        The variance is unnormalized as: var_orig = variance * y_std^2

        Args:
            mean: Predicted mean in normalized space
            std: Predicted std in normalized space (optional)
            variance: Predicted variance in normalized space (optional)

        Returns:
            (mean_orig,) or (mean_orig, std_orig) or (mean_orig, var_orig)
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() or fit_transform() first")
        if self.y_mean_ is None:
            raise RuntimeError("Scaler was not fitted with y data")

        mean = np.asarray(mean, dtype=np.float64)
        mean_orig = (mean * self.y_std_ + self.y_mean_).astype(np.float32)

        if std is not None:
            std = np.asarray(std, dtype=np.float64)
            std_orig = (std * self.y_std_).astype(np.float32)
            return mean_orig, std_orig

        if variance is not None:
            variance = np.asarray(variance, dtype=np.float64)
            var_orig = (variance * self.y_std_**2).astype(np.float32)
            return mean_orig, var_orig

        return (mean_orig,)

    def inverse_transform_X(self, X: np.ndarray) -> np.ndarray:
        """Transform inputs back to original scale.

        Args:
            X: Normalized inputs [n, d]

        Returns:
            Original-scale inputs
        """
        if not self._is_fitted:
            raise RuntimeError("Call fit() or fit_transform() first")
        X = np.asarray(X, dtype=np.float64)
        return (X * self.X_std_ + self.X_mean_).astype(np.float32)
