"""Shared pathwise prior feature construction for posterior sampling.

This module builds finite feature maps for the current supported pathwise
posterior samplers. The correction step remains exact-GP conditioning through
the JIT backend; only the prior draw is represented through explicit features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .kernel import KernelNode, KernelType


_CAT_PARAM_COUNT_FNS = {
    KernelType.GD: lambda L: 1,
    KernelType.CR: lambda L: L,
    KernelType.EHH: lambda L: L * (L - 1) // 2,
    KernelType.HH: lambda L: L * (L - 1) // 2,
    KernelType.FE: lambda L: L * (L + 1) // 2,
}


@dataclass(frozen=True)
class PathwiseFeatureMap:
    """Finite feature map used to draw approximate or exact prior samples."""

    size: int
    is_exact: bool
    evaluate: Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]


def sample_prior_values(
    feature_map: PathwiseFeatureMap,
    X_cont: np.ndarray,
    C: Optional[np.ndarray],
    weights: np.ndarray,
) -> np.ndarray:
    """Evaluate one or more prior draws from a feature map."""
    features = feature_map.evaluate(X_cont, C).astype(np.float64, copy=False)
    return (weights.astype(np.float64, copy=False) @ features.T).astype(np.float32)


def build_feature_weights(
    feature_map: PathwiseFeatureMap,
    n_draws: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample standard normal weights for a feature map."""
    return rng.standard_normal((n_draws, feature_map.size)).astype(np.float32)


def build_pathwise_feature_map(
    kernel: KernelNode,
    cont_params: np.ndarray,
    *,
    input_dim: int,
    n_features: int,
    rng: np.random.Generator,
    cat_params: Optional[np.ndarray] = None,
    cat_col_map: Optional[Dict[int, int]] = None,
    feature_cap: int = 65536,
) -> PathwiseFeatureMap:
    """Build a finite feature map for a kernel tree."""
    cont_params = np.asarray(cont_params, dtype=np.float64)
    cat_params = np.asarray(
        np.zeros(0, dtype=np.float32) if cat_params is None else cat_params,
        dtype=np.float64,
    )
    col_map = {} if cat_col_map is None else dict(cat_col_map)
    fmap, cont_off, cat_off = _build_feature_map_recursive(
        kernel,
        cont_params,
        0,
        cat_params,
        0,
        n_features,
        rng,
        col_map,
        feature_cap,
        input_dim_override=input_dim,
    )
    if cont_off != len(cont_params):
        raise ValueError(
            f"Unused continuous params while building feature map: consumed {cont_off} of {len(cont_params)}"
        )
    if cat_off != len(cat_params):
        raise ValueError(
            f"Unused categorical params while building feature map: consumed {cat_off} of {len(cat_params)}"
        )
    return fmap


def _slice_active_dims(X: np.ndarray, node: KernelNode) -> np.ndarray:
    if node.active_dims is None:
        return X
    return X[:, list(node.active_dims)]


def _extract_lengthscales(
    p: np.ndarray, node: KernelNode, d: int
) -> Tuple[np.ndarray, int]:
    if node.ard_dim is not None and node.kernel_type not in (
        KernelType.LINEAR,
        KernelType.POLYNOMIAL,
    ):
        lengthscales = np.asarray(p[: node.ard_dim], dtype=np.float64)
        if lengthscales.size != d:
            raise ValueError(
                f"Expected {d} ARD lengthscales for {node.kernel_type}, got {lengthscales.size}"
            )
        return lengthscales, node.ard_dim

    if node.kernel_type in (KernelType.LINEAR, KernelType.POLYNOMIAL):
        return np.ones(d, dtype=np.float64), 0

    return np.full(d, float(p[0]), dtype=np.float64), 1


def _stationary_base_features(
    kernel_type: KernelType,
    X: np.ndarray,
    lengthscales: np.ndarray,
    params: np.ndarray,
    extra_idx: int,
    n_features: int,
    rng: np.random.Generator,
) -> np.ndarray:
    X_scaled = X.astype(np.float64) / lengthscales[np.newaxis, :]
    d = X_scaled.shape[1]

    if kernel_type == KernelType.RBF:
        W = rng.standard_normal((n_features, d))
    elif kernel_type == KernelType.MATERN12:
        W = rng.standard_t(df=1, size=(n_features, d))
    elif kernel_type == KernelType.MATERN32:
        W = rng.standard_t(df=3, size=(n_features, d))
    elif kernel_type == KernelType.MATERN52:
        W = rng.standard_t(df=5, size=(n_features, d))
    elif kernel_type == KernelType.RQ:
        rq_alpha = float(params[extra_idx]) if extra_idx < len(params) else 1.0
        W = rng.standard_t(df=max(2.0 * rq_alpha, 0.5), size=(n_features, d))
    else:
        raise ValueError(
            f"Unsupported stationary kernel for pathwise features: {kernel_type}"
        )

    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)

    def evaluate(X_eval: np.ndarray, _C_unused: Optional[np.ndarray]) -> np.ndarray:
        X_eval = X_eval.astype(np.float64, copy=False) / lengthscales[np.newaxis, :]
        return np.sqrt(2.0 / n_features) * np.cos(X_eval @ W.T + phase[np.newaxis, :])

    return evaluate


def _periodic_base_features(
    X: np.ndarray,
    lengthscales: np.ndarray,
    period: float,
    n_features: int,
    rng: np.random.Generator,
) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
    d = X.shape[1]
    W = rng.standard_normal((n_features, 2 * d))
    phase = rng.uniform(0.0, 2.0 * np.pi, size=n_features)

    def _embed(X_eval: np.ndarray) -> np.ndarray:
        theta = (2.0 * np.pi / period) * X_eval.astype(np.float64, copy=False)
        cos_part = np.cos(theta) / lengthscales[np.newaxis, :]
        sin_part = np.sin(theta) / lengthscales[np.newaxis, :]
        return np.concatenate([cos_part, sin_part], axis=1)

    def evaluate(X_eval: np.ndarray, _C_unused: Optional[np.ndarray]) -> np.ndarray:
        embedded = _embed(X_eval)
        return np.sqrt(2.0 / n_features) * np.cos(embedded @ W.T + phase[np.newaxis, :])

    return evaluate


def _polynomial_base_features(
    degree_value: float,
    offset: float,
    outputscale: float,
    input_dim: int,
    feature_cap: int,
) -> PathwiseFeatureMap:
    degree = int(round(degree_value))
    if abs(float(degree) - float(degree_value)) > 1e-6 or degree < 1:
        raise NotImplementedError(
            "Polynomial pathwise sampling requires a fixed positive integer degree."
        )
    if offset < 0.0:
        raise NotImplementedError(
            "Polynomial pathwise sampling requires a non-negative offset."
        )
    if outputscale < 0.0:
        raise NotImplementedError(
            "Polynomial pathwise sampling requires a non-negative outputscale."
        )

    total_features = sum(input_dim**r for r in range(degree + 1))
    if total_features > feature_cap:
        raise NotImplementedError(
            "Polynomial pathwise feature map exceeds the current cap of "
            f"{feature_cap} features. Reduce degree/input dimension or use 'diagonal'."
        )

    scale = math.sqrt(max(outputscale, 0.0))

    def evaluate(X_cont: np.ndarray, _C_unused: Optional[np.ndarray]) -> np.ndarray:
        X_eval = X_cont.astype(np.float64, copy=False)
        blocks = []
        for order in range(degree + 1):
            coeff = math.comb(degree, order) * (offset ** (degree - order))
            block_scale = scale * math.sqrt(max(coeff, 0.0))
            if order == 0:
                block = np.full((X_eval.shape[0], 1), block_scale, dtype=np.float64)
            else:
                block = X_eval
                for _ in range(1, order):
                    block = (block[:, :, None] * X_eval[:, None, :]).reshape(
                        X_eval.shape[0], -1
                    )
                block = block_scale * block
            blocks.append(block)
        return np.concatenate(blocks, axis=1)

    return PathwiseFeatureMap(size=total_features, is_exact=True, evaluate=evaluate)


def _lower_tri_index(row: int, col: int) -> int:
    return row * (row - 1) // 2 + col


def _compute_cholesky_factor(theta: np.ndarray, levels: int) -> np.ndarray:
    C = np.zeros((levels, levels), dtype=np.float64)
    C[0, 0] = 1.0
    for row in range(1, levels):
        theta_val = float(theta[_lower_tri_index(row, 0)])
        C[row, 0] = np.cos(theta_val)
        sin_prod = np.sin(theta_val)
        for col in range(1, row):
            theta_val = float(theta[_lower_tri_index(row, col)])
            C[row, col] = np.cos(theta_val) * sin_prod
            sin_prod *= np.sin(theta_val)
        C[row, row] = sin_prod
    return C


def _categorical_corr_matrix(
    kernel_type: KernelType, levels: int, params: np.ndarray
) -> np.ndarray:
    params = np.asarray(params, dtype=np.float64)
    R = np.eye(levels, dtype=np.float64)
    if kernel_type == KernelType.GD:
        off_diag = np.exp(-float(params[0]))
        R.fill(off_diag)
        np.fill_diagonal(R, 1.0)
        return R
    if kernel_type == KernelType.CR:
        theta = params[:levels]
        R = np.exp(-(theta[:, None] + theta[None, :]))
        np.fill_diagonal(R, 1.0)
        return R
    if kernel_type == KernelType.EHH:
        C = _compute_cholesky_factor(params, levels)
        dot = C @ C.T
        log_eps_half = -13.815510558
        R = np.exp(log_eps_half * (1.0 - dot))
        np.fill_diagonal(R, 1.0)
        return R
    if kernel_type == KernelType.HH:
        C = _compute_cholesky_factor(params, levels)
        return C @ C.T
    if kernel_type == KernelType.FE:
        num_angles = levels * (levels - 1) // 2
        C = _compute_cholesky_factor(params[:num_angles], levels)
        diag_params = params[num_angles : num_angles + levels]
        dot = C @ C.T
        log_eps_half = -13.815510558
        for i in range(levels):
            for j in range(levels):
                if i == j:
                    R[i, j] = 1.0
                else:
                    phi = (
                        diag_params[i]
                        + diag_params[j]
                        + log_eps_half * (dot[i, j] - 1.0)
                    )
                    R[i, j] = np.exp(-phi)
        return R
    raise ValueError(f"Unsupported categorical kernel type: {kernel_type}")


def _build_categorical_feature_map(
    node: KernelNode,
    cat_params: np.ndarray,
    cat_offset: int,
    cat_col_map: Dict[int, int],
) -> Tuple[PathwiseFeatureMap, int]:
    if node.levels is None:
        raise ValueError(f"Categorical kernel {node.kernel_type} requires levels")
    if node.active_dims is None or len(node.active_dims) != 1:
        raise ValueError(
            "Categorical pathwise leaves require exactly one active dimension"
        )
    original_col = int(node.active_dims[0])
    if original_col not in cat_col_map:
        raise ValueError(
            f"No categorical column mapping found for active dim {original_col}"
        )
    cat_col = cat_col_map[original_col]
    n_params = _CAT_PARAM_COUNT_FNS[node.kernel_type](node.levels)
    theta = np.asarray(cat_params[cat_offset : cat_offset + n_params], dtype=np.float64)
    corr = _categorical_corr_matrix(node.kernel_type, node.levels, theta)
    jitter = float(np.abs(np.diag(corr)).mean()) * 1e-6 + 1e-8
    factor = np.linalg.cholesky(corr + jitter * np.eye(node.levels, dtype=np.float64))

    def evaluate(_X_unused: np.ndarray, C: Optional[np.ndarray]) -> np.ndarray:
        if C is None:
            raise ValueError("Categorical feature map requires categorical inputs")
        levels_idx = np.asarray(C[:, cat_col], dtype=np.int64)
        return factor[levels_idx]

    return PathwiseFeatureMap(
        size=node.levels, is_exact=True, evaluate=evaluate
    ), cat_offset + n_params


def _combine_sum(
    left: PathwiseFeatureMap, right: PathwiseFeatureMap
) -> PathwiseFeatureMap:
    def evaluate(X_cont: np.ndarray, C: Optional[np.ndarray]) -> np.ndarray:
        left_feat = left.evaluate(X_cont, C)
        right_feat = right.evaluate(X_cont, C)
        return np.concatenate([left_feat, right_feat], axis=1)

    return PathwiseFeatureMap(
        size=left.size + right.size,
        is_exact=left.is_exact and right.is_exact,
        evaluate=evaluate,
    )


def _combine_product(
    left: PathwiseFeatureMap,
    right: PathwiseFeatureMap,
    feature_cap: int,
) -> PathwiseFeatureMap:
    total_features = left.size * right.size
    if total_features > feature_cap:
        raise NotImplementedError(
            "Pathwise posterior sampling feature product exceeds the current cap of "
            f"{feature_cap} features. Reduce kernel complexity or use 'diagonal'."
        )

    def evaluate(X_cont: np.ndarray, C: Optional[np.ndarray]) -> np.ndarray:
        left_feat = left.evaluate(X_cont, C).astype(np.float64, copy=False)
        right_feat = right.evaluate(X_cont, C).astype(np.float64, copy=False)
        return (left_feat[:, :, None] * right_feat[:, None, :]).reshape(
            X_cont.shape[0], total_features
        )

    return PathwiseFeatureMap(
        size=total_features,
        is_exact=left.is_exact and right.is_exact,
        evaluate=evaluate,
    )


def _build_feature_map_recursive(
    node: KernelNode,
    cont_params: np.ndarray,
    cont_offset: int,
    cat_params: np.ndarray,
    cat_offset: int,
    n_features: int,
    rng: np.random.Generator,
    cat_col_map: Dict[int, int],
    feature_cap: int,
    input_dim_override: Optional[int] = None,
) -> Tuple[PathwiseFeatureMap, int, int]:
    if (
        node.active_dims is not None
        and node.kernel_type is not None
        and node.is_continuous()
    ):
        import copy

        inner = copy.deepcopy(node)
        inner.active_dims = None
        fmap, cont_offset, cat_offset = _build_feature_map_recursive(
            inner,
            cont_params,
            cont_offset,
            cat_params,
            cat_offset,
            n_features,
            rng,
            cat_col_map,
            feature_cap,
            input_dim_override=len(node.active_dims),
        )

        def evaluate(X_cont: np.ndarray, C: Optional[np.ndarray]) -> np.ndarray:
            return fmap.evaluate(_slice_active_dims(X_cont, node), C)

        return (
            PathwiseFeatureMap(fmap.size, fmap.is_exact, evaluate),
            cont_offset,
            cat_offset,
        )

    if node.kernel_type is not None:
        if node.is_categorical():
            fmap, next_cat = _build_categorical_feature_map(
                node, cat_params, cat_offset, cat_col_map
            )
            return fmap, cont_offset, next_cat

        X_dummy_dim = (
            input_dim_override
            if input_dim_override is not None
            else (
                len(node.active_dims)
                if node.active_dims is not None
                else (node.ard_dim or 1)
            )
        )
        n_public = node.num_params()
        p = np.asarray(
            cont_params[cont_offset : cont_offset + n_public], dtype=np.float64
        )

        if node.kernel_type == KernelType.LINEAR:
            variance = float(p[0])
            outputscale = float(p[1])

            def evaluate(
                X_cont: np.ndarray, _C_unused: Optional[np.ndarray]
            ) -> np.ndarray:
                X_eval = X_cont.astype(np.float64, copy=False)
                return np.sqrt(max(variance * outputscale, 0.0)) * X_eval

            return (
                PathwiseFeatureMap(size=X_dummy_dim, is_exact=True, evaluate=evaluate),
                cont_offset + n_public,
                cat_offset,
            )

        if node.kernel_type == KernelType.POLYNOMIAL:
            if node.ard_dim is not None:
                raise NotImplementedError(
                    "Pathwise posterior sampling does not yet support ARD polynomial kernels."
                )
            degree, poly_offset, outputscale = p[0], p[1], p[2]
            return (
                _polynomial_base_features(
                    float(degree),
                    float(poly_offset),
                    float(outputscale),
                    X_dummy_dim,
                    feature_cap,
                ),
                cont_offset + n_public,
                cat_offset,
            )

        d = X_dummy_dim
        lengthscales, extra_idx = _extract_lengthscales(p, node, d)
        output_idx = n_public - 1
        outputscale = float(p[output_idx]) if output_idx < len(p) else 1.0

        if node.kernel_type == KernelType.PERIODIC:
            period = float(p[extra_idx])
            evaluator = _periodic_base_features(
                np.zeros((1, d), dtype=np.float32),
                lengthscales,
                period,
                n_features,
                rng,
            )
            base_size = n_features
        else:
            evaluator = _stationary_base_features(
                node.kernel_type,
                np.zeros((1, d), dtype=np.float32),
                lengthscales,
                p,
                extra_idx,
                n_features,
                rng,
            )
            base_size = n_features

        def evaluate(X_cont: np.ndarray, _C_unused: Optional[np.ndarray]) -> np.ndarray:
            return np.sqrt(max(outputscale, 0.0)) * evaluator(
                X_cont.astype(np.float64, copy=False), None
            )

        return (
            PathwiseFeatureMap(size=base_size, is_exact=False, evaluate=evaluate),
            cont_offset + n_public,
            cat_offset,
        )

    if node.operator == "sum":
        left, mid_cont, mid_cat = _build_feature_map_recursive(
            node.left,
            cont_params,
            cont_offset,
            cat_params,
            cat_offset,
            n_features,
            rng,
            cat_col_map,
            feature_cap,
            input_dim_override=input_dim_override,
        )
        right, end_cont, end_cat = _build_feature_map_recursive(
            node.right,
            cont_params,
            mid_cont,
            cat_params,
            mid_cat,
            n_features,
            rng,
            cat_col_map,
            feature_cap,
            input_dim_override=input_dim_override,
        )
        return _combine_sum(left, right), end_cont, end_cat

    if node.operator == "product":
        left, mid_cont, mid_cat = _build_feature_map_recursive(
            node.left,
            cont_params,
            cont_offset,
            cat_params,
            cat_offset,
            n_features,
            rng,
            cat_col_map,
            feature_cap,
            input_dim_override=input_dim_override,
        )
        right, end_cont, end_cat = _build_feature_map_recursive(
            node.right,
            cont_params,
            mid_cont,
            cat_params,
            mid_cat,
            n_features,
            rng,
            cat_col_map,
            feature_cap,
            input_dim_override=input_dim_override,
        )
        return _combine_product(left, right, feature_cap), end_cont, end_cat

    if node.operator == "scale":
        inner, mid_cont, mid_cat = _build_feature_map_recursive(
            node.left,
            cont_params,
            cont_offset,
            cat_params,
            cat_offset,
            n_features,
            rng,
            cat_col_map,
            feature_cap,
            input_dim_override=input_dim_override,
        )
        scale = float(cont_params[mid_cont])

        def evaluate(X_cont: np.ndarray, C: Optional[np.ndarray]) -> np.ndarray:
            return np.sqrt(max(scale, 0.0)) * inner.evaluate(X_cont, C)

        return (
            PathwiseFeatureMap(
                size=inner.size, is_exact=inner.is_exact, evaluate=evaluate
            ),
            mid_cont + 1,
            mid_cat,
        )

    raise ValueError(f"Unknown kernel operator: {node.operator}")
