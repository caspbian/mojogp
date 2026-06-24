"""Composable categorical kernel trait and implementations.

Provides a CategoricalKernel trait that wraps the existing math functions in
categorical_kernel.mojo with a uniform, compile-time-dispatchable interface.

Each struct wraps the existing per-variant functions (compute_gd_correlation, etc.)
without modifying them. The trait enables compile-time dispatch in
TypedCategoricalState, replacing the runtime integer dispatch in
CategoricalCorrelationState.

Reference: Saves et al. (2023), arXiv:2211.08262v4
"""

from math import exp as math_exp, log, sqrt, sin, cos
from memory import UnsafePointer
from .constants import PI
from .categorical_kernel import (
    compute_gd_correlation,
    compute_gd_gradient,
    compute_cr_correlation,
    compute_cr_gradient,
    compute_ehh_correlation,
    compute_ehh_gradient,
    compute_hh_correlation,
    compute_hh_gradient,
    compute_fe_correlation,
    compute_fe_gradient,
)


# ============================================================================
# Trait definition
# ============================================================================


trait CategoricalKernel:
    """Compile-time interface for categorical kernel variants.

    Each variant computes an L x L correlation matrix R from unconstrained
    parameters theta, entirely on CPU. The correlation matrix is then
    uploaded to GPU for lookup-table evaluation.

    Methods:
        num_params: How many learnable parameters for L levels.
        compute_correlation: theta -> R (L x L matrix).
        compute_gradient: theta -> dR/d(theta[param_index]).
        default_raw_param: Initial unconstrained parameter value.
        constrain: Raw -> constrained space (e.g., softplus, sigmoid*pi).
        constrain_derivative: Chain rule factor d(constrained)/d(raw).
    """

    @staticmethod
    fn num_params(L: Int) -> Int:
        """Number of learnable parameters for a variable with L levels."""
        ...

    @staticmethod
    fn compute_correlation(
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        """Compute the L x L correlation matrix R from constrained parameters theta.

        Args:
            R_ptr: Output buffer, L*L floats, row-major.
            theta_ptr: Constrained parameter values.
            L: Number of categorical levels.
            work_ptr: Workspace buffer (size depends on variant, >= 2*L*L is safe).
        """
        ...

    @staticmethod
    fn compute_gradient(
        dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        param_index: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        """Compute dR/d(theta[param_index]).

        Args:
            dR_ptr: Output buffer, L*L floats, row-major.
            theta_ptr: Constrained parameter values.
            R_ptr: Pre-computed correlation matrix (needed by EHH/FE, ignored by GD/CR/HH).
            L: Number of categorical levels.
            param_index: Which parameter to differentiate with respect to.
            work_ptr: Workspace buffer (>= 2*L*L is safe).
        """
        ...

    @staticmethod
    fn default_raw_param(param_index: Int, L: Int) -> Float32:
        """Default raw (unconstrained) parameter initialization."""
        ...

    @staticmethod
    fn constrain(raw: Float32, param_index: Int, L: Int) -> Float32:
        """Raw -> constrained space."""
        ...

    @staticmethod
    fn constrain_derivative(raw: Float32, param_index: Int, L: Int) -> Float32:
        """Chain rule derivative: d(constrained)/d(raw)."""
        ...


# ============================================================================
# Helper math functions
# ============================================================================


@always_inline
fn _softplus(x: Float32) -> Float32:
    """Softplus: log(1 + exp(x)). Numerically stable."""
    if x > Float32(20.0):
        return x
    return log(Float32(1.0) + math_exp(x))


@always_inline
fn _softplus_derivative(x: Float32) -> Float32:
    """Derivative of softplus: sigmoid(x) = 1/(1+exp(-x))."""
    if x > Float32(20.0):
        return Float32(1.0)
    var ex = math_exp(x)
    return ex / (Float32(1.0) + ex)


@always_inline
fn _sigmoid(x: Float32) -> Float32:
    """Sigmoid: 1 / (1 + exp(-x))."""
    if x > Float32(20.0):
        return Float32(1.0)
    if x < Float32(-20.0):
        return Float32(0.0)
    return Float32(1.0) / (Float32(1.0) + math_exp(-x))


@always_inline
fn _sigmoid_derivative(x: Float32) -> Float32:
    """Derivative of sigmoid: sigmoid(x) * (1 - sigmoid(x))."""
    var s = _sigmoid(x)
    return s * (Float32(1.0) - s)


# ============================================================================
# GD (Gower Distance) — 1 param, softplus constraint
# ============================================================================


struct GDKernel(CategoricalKernel):
    """Gower Distance categorical kernel.

    One parameter controlling uniform off-diagonal correlation.
    R[i,j] = exp(-theta) if i != j, else 1.0.
    """

    @staticmethod
    fn num_params(L: Int) -> Int:
        return 1

    @staticmethod
    fn compute_correlation(
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        compute_gd_correlation(R_ptr, theta_ptr[0], L)

    @staticmethod
    fn compute_gradient(
        dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        param_index: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        compute_gd_gradient(dR_ptr, theta_ptr[0], L)

    @staticmethod
    fn default_raw_param(param_index: Int, L: Int) -> Float32:
        # inv_softplus(0.5) = log(exp(0.5) - 1)
        return log(math_exp(Float32(0.5)) - Float32(1.0))

    @staticmethod
    fn constrain(raw: Float32, param_index: Int, L: Int) -> Float32:
        return _softplus(raw)

    @staticmethod
    fn constrain_derivative(raw: Float32, param_index: Int, L: Int) -> Float32:
        return _softplus_derivative(raw)


# ============================================================================
# CR (Continuous Relaxation) — L params, softplus constraint
# ============================================================================


struct CRKernel(CategoricalKernel):
    """Continuous Relaxation categorical kernel.

    L parameters, one per level. Per-level radial correlation.
    """

    @staticmethod
    fn num_params(L: Int) -> Int:
        return L

    @staticmethod
    fn compute_correlation(
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        compute_cr_correlation(R_ptr, theta_ptr, L)

    @staticmethod
    fn compute_gradient(
        dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        param_index: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        compute_cr_gradient(dR_ptr, theta_ptr, L, param_index)

    @staticmethod
    fn default_raw_param(param_index: Int, L: Int) -> Float32:
        # inv_softplus(0.3) = log(exp(0.3) - 1)
        return log(math_exp(Float32(0.3)) - Float32(1.0))

    @staticmethod
    fn constrain(raw: Float32, param_index: Int, L: Int) -> Float32:
        return _softplus(raw)

    @staticmethod
    fn constrain_derivative(raw: Float32, param_index: Int, L: Int) -> Float32:
        return _softplus_derivative(raw)


# ============================================================================
# EHH (Exponential Homoscedastic Hypersphere) — L(L-1)/2 params, sigmoid*pi
# ============================================================================


struct EHHKernel(CategoricalKernel):
    """Exponential Homoscedastic Hypersphere categorical kernel.

    L*(L-1)/2 angle parameters. Full pairwise correlation via hypersphere
    decomposition with exponential transform ensuring R is positive semi-definite
    with all non-negative entries.
    """

    @staticmethod
    fn num_params(L: Int) -> Int:
        return L * (L - 1) // 2

    @staticmethod
    fn compute_correlation(
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        compute_ehh_correlation(R_ptr, theta_ptr, L, work_ptr)

    @staticmethod
    fn compute_gradient(
        dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        param_index: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        compute_ehh_gradient(dR_ptr, theta_ptr, R_ptr, L, param_index, work_ptr)

    @staticmethod
    fn default_raw_param(param_index: Int, L: Int) -> Float32:
        # inv_sigmoid(0.25) = log(0.25 / 0.75)
        return log(Float32(0.25) / Float32(0.75))

    @staticmethod
    fn constrain(raw: Float32, param_index: Int, L: Int) -> Float32:
        return _sigmoid(raw) * PI

    @staticmethod
    fn constrain_derivative(raw: Float32, param_index: Int, L: Int) -> Float32:
        return _sigmoid_derivative(raw) * PI


# ============================================================================
# HH (Homoscedastic Hypersphere) — L(L-1)/2 params, sigmoid*pi
# ============================================================================


struct HHKernel(CategoricalKernel):
    """Homoscedastic Hypersphere categorical kernel.

    L*(L-1)/2 angle parameters. Like EHH but without the exponential transform,
    allowing negative correlations.
    """

    @staticmethod
    fn num_params(L: Int) -> Int:
        return L * (L - 1) // 2

    @staticmethod
    fn compute_correlation(
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        compute_hh_correlation(R_ptr, theta_ptr, L, work_ptr)

    @staticmethod
    fn compute_gradient(
        dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        param_index: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        # HH does not need R_ptr for gradient computation
        compute_hh_gradient(dR_ptr, theta_ptr, L, param_index, work_ptr)

    @staticmethod
    fn default_raw_param(param_index: Int, L: Int) -> Float32:
        return log(Float32(0.25) / Float32(0.75))

    @staticmethod
    fn constrain(raw: Float32, param_index: Int, L: Int) -> Float32:
        return _sigmoid(raw) * PI

    @staticmethod
    fn constrain_derivative(raw: Float32, param_index: Int, L: Int) -> Float32:
        return _sigmoid_derivative(raw) * PI


# ============================================================================
# FE (Fully Exponential) — L(L+1)/2 params, mixed constraints
# ============================================================================


struct FEKernel(CategoricalKernel):
    """Fully Exponential categorical kernel.

    L*(L-1)/2 angle parameters (sigmoid*pi) + L diagonal parameters (softplus).
    Most flexible variant.
    """

    @staticmethod
    fn num_params(L: Int) -> Int:
        return L * (L + 1) // 2

    @staticmethod
    fn compute_correlation(
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        compute_fe_correlation(R_ptr, theta_ptr, L, work_ptr)

    @staticmethod
    fn compute_gradient(
        dR_ptr: UnsafePointer[Float32, MutAnyOrigin],
        theta_ptr: UnsafePointer[Float32, MutAnyOrigin],
        R_ptr: UnsafePointer[Float32, MutAnyOrigin],
        L: Int,
        param_index: Int,
        work_ptr: UnsafePointer[Float32, MutAnyOrigin],
    ) -> None:
        compute_fe_gradient(dR_ptr, theta_ptr, R_ptr, L, param_index, work_ptr)

    @staticmethod
    fn default_raw_param(param_index: Int, L: Int) -> Float32:
        # First L*(L-1)/2 params are angles (sigmoid*pi), rest are diagonal (softplus)
        var n_angles = L * (L - 1) // 2
        if param_index < n_angles:
            return log(Float32(0.25) / Float32(0.75))
        else:
            return log(math_exp(Float32(0.3)) - Float32(1.0))

    @staticmethod
    fn constrain(raw: Float32, param_index: Int, L: Int) -> Float32:
        var n_angles = L * (L - 1) // 2
        if param_index < n_angles:
            return _sigmoid(raw) * PI
        else:
            return _softplus(raw)

    @staticmethod
    fn constrain_derivative(raw: Float32, param_index: Int, L: Int) -> Float32:
        var n_angles = L * (L - 1) // 2
        if param_index < n_angles:
            return _sigmoid_derivative(raw) * PI
        else:
            return _softplus_derivative(raw)
